import abc
import functools
import math
import warnings
from typing import Union, Optional, Dict, Any

import torch

from falkon import sparse
from falkon.sparse import SparseTensor

from falkon.kernels import KeopsKernelMixin, Kernel
from falkon.la_helpers.square_norm_fn import square_norm_diff
from falkon.options import FalkonOptions


SQRT3 = 1.7320508075688772
SQRT5 = 2.23606797749979


def validate_sigma(sigma: Union[float, torch.Tensor]) -> torch.Tensor:
    if isinstance(sigma, torch.Tensor):
        # Sigma is a 1-item tensor ('single')
        try:
            sigma.item()
            return sigma
        except ValueError:
            pass
        # Sigma is a vector ('diag')
        if sigma.dim() == 1 or sigma.shape[1] == 1:
            return sigma.reshape(-1)
        else:
            # TODO: Better error
            raise ValueError("sigma must be a scalar or a vector.")
    else:
        try:
            return torch.tensor([float(sigma)], dtype=torch.float64)
        except TypeError:
            raise TypeError("Sigma must be a scalar or a tensor.")


def validate_diff_scalar(num: Union[float, torch.Tensor], param_name: str) -> torch.Tensor:
    if isinstance(num, torch.Tensor):
        # Sigma is a 1-item tensor ('single')
        try:
            num.item()
            return num
        except ValueError:
            raise ValueError(f"Parameter {param_name} must be a scalar.")
    else:
        try:
            return torch.tensor([float(num)], dtype=torch.float64)
        except TypeError:
            raise TypeError(f"Parameter {param_name} must be a scalar or a tensor.")


def calc_grads(bwd, saved_tensors, needs_input_grad):
    inputs = []
    for i in range(len(needs_input_grad)):
        if needs_input_grad[i]:
            inputs.append(saved_tensors[i])
    grads = torch.autograd.grad(bwd, inputs)
    j = 0
    results = []
    for i in range(len(needs_input_grad)):
        if needs_input_grad[i]:
            results.append(grads[j])
            j += 1
        else:
            results.append(None)
    return tuple(results)


def _addmm_wrap(mat1, mat2, norm_mat1, norm_mat2, out: Optional[torch.Tensor]) -> torch.Tensor:
    if mat1.dim() == 3:
        if out is None:
            out = torch.baddbmm(norm_mat1, mat1, mat2.transpose(-2, -1), alpha=-2, beta=1)  # b*n*m
        else:
            out = torch.baddbmm(norm_mat1, mat1, mat2.transpose(-2, -1), alpha=-2, beta=1, out=out)  # b*n*m
    else:
        if out is None:
            out = torch.addmm(norm_mat1, mat1, mat2.transpose(-2, -1), alpha=-2, beta=1)  # n*m
        else:
            out = torch.addmm(norm_mat1, mat1, mat2.transpose(-2, -1), alpha=-2, beta=1, out=out)  # n*m
    out.add_(norm_mat2.transpose(-2, -1))
    out.clamp_min_(1e-20)
    return out


def rbf_core(mat1, mat2, out: Optional[torch.Tensor], sigma):
    """
    Note 1: if out is None, then this function will be differentiable wrt all three remaining inputs.
    Note 2: this function can deal with batched inputs

    Parameters
    ----------
    sigma
    mat1
    mat2
    out

    Returns
    -------

    """
    mat1_div_sig = mat1 / sigma
    mat2_div_sig = mat2 / sigma
    norm_sq_mat1 = square_norm_diff(mat1_div_sig, -1, True)  # b*n*1 or n*1
    norm_sq_mat2 = square_norm_diff(mat2_div_sig, -1, True)  # b*m*1 or m*1

    out = _addmm_wrap(mat1_div_sig, mat2_div_sig, norm_sq_mat1, norm_sq_mat2, out)
    out.mul_(-0.5)
    out.exp_()
    return out


def laplacian_core(mat1, mat2, out: Optional[torch.Tensor], sigma):
    mat1_div_sig = mat1 / sigma
    mat2_div_sig = mat2 / sigma
    norm_sq_mat1 = square_norm_diff(mat1_div_sig, -1, True)  # b*n*1
    norm_sq_mat2 = square_norm_diff(mat2_div_sig, -1, True)  # b*m*1
    orig_out = out
    out = _addmm_wrap(mat1_div_sig, mat2_div_sig, norm_sq_mat1, norm_sq_mat2, out)
    out.sqrt_()  # Laplacian: sqrt of squared-difference
    # The gradient calculation needs the output of sqrt_
    if orig_out is None:  # TODO: We could be more explicit in the parameters about whether the gradient is or isn't needed
        out = out.neg()
    else:
        out.neg_()
    out.exp_()
    return out


def matern_core(mat1, mat2, out: Optional[torch.Tensor], sigma, nu):
    if nu == 0.5:
        return laplacian_core(mat1, mat2, out, sigma)
    elif nu == float('inf'):
        return rbf_core(mat1, mat2, out, sigma)
    orig_out = out
    mat1_div_sig = mat1 / sigma
    mat2_div_sig = mat2 / sigma
    norm_sq_mat1 = square_norm_diff(mat1_div_sig, -1, True)  # b*n*1
    norm_sq_mat2 = square_norm_diff(mat2_div_sig, -1, True)  # b*m*1

    out = _addmm_wrap(mat1_div_sig, mat2_div_sig, norm_sq_mat1, norm_sq_mat2, out)
    if nu == 1.5:
        # (1 + sqrt(3)*D) * exp(-sqrt(3)*D))
        out.sqrt_()
        if orig_out is None:  # TODO: We could be more explicit in the parameters about whether the gradient is or isn't needed
            out = out.mul(SQRT3)
        else:
            out.mul_(SQRT3)
        out_neg = torch.neg(out)  # extra n*m block
        out_neg.exp_()
        out.add_(1.0).mul_(out_neg)
    elif nu == 2.5:
        # (1 + sqrt(5)*D + (sqrt(5)*D)^2 / 3 ) * exp(-sqrt(5)*D)
        out_sqrt = torch.sqrt(out)
        if orig_out is None:  # TODO: We could be more explicit in the parameters about whether the gradient is or isn't needed
            out_sqrt = out_sqrt.mul(SQRT5)
        else:
            out_sqrt.mul_(SQRT5)
        out.mul_(5.0 / 3.0).add_(out_sqrt).add_(1.0)
        out_sqrt.neg_().exp_()
        out.mul_(out_sqrt)

    return out


def linear_core(mat1, mat2, out: Optional[torch.Tensor], beta, gamma):
    if out is None:
        out = torch.mm(mat1, mat2.T)
    else:
        out = torch.mm(mat1, mat2.T, out=out)
    out.mul_(gamma)
    out.add_(beta)
    return out


def polynomial_core(mat1, mat2, out: Optional[torch.Tensor], beta, gamma, degree):
    if out is None:
        out = torch.mm(mat1, mat2.T)
    else:
        out = torch.mm(mat1, mat2.T, out=out)
    out.mul_(gamma)
    out.add_(beta)
    out.pow_(degree)
    return out


def sigmoid_core(mat1, mat2, out: Optional[torch.Tensor], beta, gamma):
    if out is None:
        out = torch.mm(mat1, mat2.T)
    else:
        out = torch.mm(mat1, mat2.T, out=out)
    out.mul_(gamma)
    out.add_(beta)
    out.tanh_()
    return out


def sparse_dist_core(X1_csr: SparseTensor, X2_csr: SparseTensor,
                     X1: SparseTensor, X2: SparseTensor,
                     out: torch.Tensor) -> torch.Tensor:
    sq1 = torch.empty(X1_csr.size(0), dtype=X1_csr.dtype, device=X1_csr.device)
    sparse.sparse_square_norm(X1_csr, sq1)  # TODO: This must be implemented for CUDA tensors
    sq1 = sq1.reshape(-1, 1)
    sq2 = torch.empty(X2_csr.size(0), dtype=X2_csr.dtype, device=X2_csr.device)
    sparse.sparse_square_norm(X2_csr, sq2)
    sq2 = sq2.reshape(-1, 1)
    sparse.sparse_matmul(X1, X2, out)
    out.mul_(-2.0)
    out.add_(sq1.to(device=X1.device))
    out.add_(sq2.to(device=X2.device).t())
    out.clamp_min_(1e-20)
    return out


class DistanceKernel(Kernel, KeopsKernelMixin, abc.ABC):
    def __init__(self, name, options, core_fn, **kernel_params):
        super(DistanceKernel, self).__init__(name=name, kernel_type="distance", opt=options)
        self.core_fn = core_fn
        self._tensor_params = {k: v for k, v in kernel_params.items() if isinstance(v, torch.Tensor)}
        self._other_params = {k: v for k, v in kernel_params.items() if not isinstance(v, torch.Tensor)}

    @property
    def diff_params(self) -> Dict[str, torch.Tensor]:
        return self._tensor_params

    @property
    def nondiff_params(self) -> Dict[str, Any]:
        return self._other_params

    @abc.abstractmethod
    def _keops_mmv_impl(self, X1, X2, v, kernel, out, opt: FalkonOptions):
        pass

    def _decide_mmv_impl(self, X1, X2, v, opt: FalkonOptions):
        if self.keops_can_handle_mmv(X1, X2, v, opt):
            return self._keops_mmv_impl
        else:
            return super()._decide_mmv_impl(X1, X2, v, opt)

    def _decide_dmmv_impl(self, X1, X2, v, w, opt: FalkonOptions):
        if self.keops_can_handle_dmmv(X1, X2, v, w, opt):
            return functools.partial(self.keops_dmmv_helper, mmv_fn=self._keops_mmv_impl)
        else:
            return super()._decide_dmmv_impl(X1, X2, v, w, opt)

    def _move_kernel_params(self, to_mat):
        new_kernel_params = {}
        for k, v in self._tensor_params.items():
            new_kernel_params[k] = v.to(device=to_mat.device, dtype=to_mat.dtype)
        return new_kernel_params

    def _detach_params(self):
        detached_tensor_params = {k: v.detach() for k, v in self._tensor_params.items()}
        detached_tensor_params.update(**self._other_params)
        return detached_tensor_params

    def compute(self, X1, X2, out):
        dev_kernel_tensor_params = self._move_kernel_params(X1)
        return self.core_fn(X1, X2, out, **dev_kernel_tensor_params, **self._other_params)

    def compute_diff(self, X1, X2):
        dev_kernel_tensor_params = self._move_kernel_params(X1)
        return self.core_fn(X1, X2, out=None, **dev_kernel_tensor_params, **self._other_params)


class GaussianKernel(DistanceKernel):
    r"""Class for computing the Gaussian kernel and related kernel-vector products

    The Gaussian kernel is one of the most common and effective kernel embeddings
    since it is infinite dimensional, and governed by a single parameter. The kernel length-scale
    determines the width of the Gaussian distribution which is placed on top of each point.
    A larger sigma corresponds to a wide Gaussian, so that the relative influence of far away
    points will be high for computing the kernel at a given datum.
    On the opposite side of the spectrum, a small sigma means that only nearby points will
    influence the kernel.

    Parameters
    -----------
    sigma
        The length-scale of the kernel.
        This can be a scalar, and then it corresponds to the standard deviation
        of the Gaussian distribution from which the kernel is derived.
        If `sigma` is a vector of size `d` (where `d` is the dimensionality of the data), it is
        interpreted as the diagonal standard deviation of the Gaussian distribution.
        It can also be a matrix of  size `d*d` where `d`, in which case sigma will be the precision
        matrix (inverse covariance).
    opt
        Additional options to be forwarded to the matrix-vector multiplication
        routines.

    Examples
    --------
    Creating a Gaussian kernel with a single length-scale. Operations on this kernel will not
    use KeOps.

    >>> K = GaussianKernel(sigma=3.0, opt=FalkonOptions(keops_active="no"))

    Creating a Gaussian kernel with a different length-scale per dimension

    >>> K = GaussianKernel(sigma=torch.tensor([1.0, 3.5, 7.0]))

    Creating a Gaussian kernel object with full covariance matrix (randomly chosen)

    >>> mat = torch.randn(3, 3, dtype=torch.float64)
    >>> sym_mat = mat @ mat.T
    >>> K = GaussianKernel(sigma=sym_mat)
    >>> K
    GaussianKernel(sigma=tensor([[ 2.0909,  0.0253, -0.2490],
            [ 0.0253,  0.3399, -0.5158],
            [-0.2490, -0.5158,  4.4922]], dtype=torch.float64))  #random


    Notes
    -----
    The Gaussian kernel with a single length-scale follows

    .. math::

        k(x, x') = \exp{-\dfrac{\lVert x - x' \rVert^2}{2\sigma^2}}


    When the length-scales are specified as a matrix, the RBF kernel is determined by

    .. math::

        k(x, x') = \exp{-\dfrac{1}{2}x\Sigma x'}


    In both cases, the actual computation follows a different path, working on the expanded
    norm.
    """
    kernel_name = "gaussian"
    core_fn = rbf_core

    def __init__(self, sigma: Union[float, torch.Tensor], opt: Optional[FalkonOptions] = None):
        self.sigma = validate_sigma(sigma)

        super().__init__(self.kernel_name, opt, core_fn=GaussianKernel.core_fn, sigma=self.sigma)

    def _keops_mmv_impl(self, X1, X2, v, kernel, out, opt: FalkonOptions):
        formula = 'Exp(SqDist(x1 / g, x2 / g) * IntInv(-2)) * v'
        aliases = [
            'x1 = Vi(%d)' % (X1.shape[1]),
            'x2 = Vj(%d)' % (X2.shape[1]),
            'v = Vj(%d)' % (v.shape[1]),
            'g = Pm(%d)' % (self.sigma.shape[0])
        ]
        other_vars = [self.sigma.to(device=X1.device, dtype=X1.dtype)]

        return self.keops_mmv(X1, X2, v, out, formula, aliases, other_vars, opt)

    def extra_mem(self) -> Dict[str, float]:
        return {
            # Data-matrix / sigma in prepare + Data-matrix / sigma in apply
            'nd': 2,
            'md': 1,
            # Norm results in prepare
            'm': 1,
            'n': 1,
        }

    def detach(self) -> 'GaussianKernel':
        detached_params = self._detach_params()
        return GaussianKernel(detached_params["sigma"], opt=self.params)

    # noinspection PyMethodOverriding
    def compute_sparse(self, X1: SparseTensor, X2: SparseTensor, out: torch.Tensor,
                       X1_csr: SparseTensor, X2_csr: SparseTensor) -> torch.Tensor:
        if len(self.sigma) > 1:
            raise NotImplementedError("Sparse kernel is only implemented for scalar sigmas.")
        dev_kernel_tensor_params = self._move_kernel_params(X1)
        gamma = 0.5 / (dev_kernel_tensor_params["sigma"] ** 2)
        out = sparse_dist_core(X1_csr=X1_csr, X2_csr=X2_csr, X1=X1, X2=X2, out=out)
        out.mul_(-gamma)
        out.exp_()
        return out

    def __repr__(self):
        return f"GaussianKernel(sigma={self.sigma})"

    def __str__(self):
        return f"Gaussian kernel<{self.sigma}>"


class LaplacianKernel(DistanceKernel):
    r"""Class for computing the Laplacian kernel, and related kernel-vector products.

    The Laplacian kernel is similar to the Gaussian kernel, but less sensitive to changes
    in the parameter `sigma`.

    Parameters
    ----------
    sigma
        The length-scale of the Laplacian kernel

    Notes
    -----
    The Laplacian kernel is determined by the following formula

    .. math::

        k(x, x') = \exp{-\frac{\lVert x - x' \rVert}{\sigma}}

    """
    kernel_name = "laplacian"

    def __init__(self, sigma: Union[float, torch.Tensor], opt: Optional[FalkonOptions] = None):
        self.sigma = validate_sigma(sigma)

        super().__init__(self.kernel_name, opt, core_fn=laplacian_core, sigma=self.sigma)

    def _keops_mmv_impl(self, X1, X2, v, kernel, out, opt: FalkonOptions):
        formula = 'Exp(-Sqrt(SqDist(x1 / g, x2 / g))) * v'
        aliases = [
            'x1 = Vi(%d)' % (X1.shape[1]),
            'x2 = Vj(%d)' % (X2.shape[1]),
            'v = Vj(%d)' % (v.shape[1]),
            'g = Pm(%d)' % (self.sigma.shape[0])
        ]
        other_vars = [self.sigma.to(device=X1.device, dtype=X1.dtype)]

        return self.keops_mmv(X1, X2, v, out, formula, aliases, other_vars, opt)

    def extra_mem(self) -> Dict[str, float]:
        return {
            # Data-matrix / sigma in prepare + Data-matrix / sigma in apply
            'nd': 2,
            'md': 1,
            # Norm results in prepare
            'm': 1,
            'n': 1,
        }

    def detach(self) -> 'LaplacianKernel':
        detached_params = self._detach_params()
        return LaplacianKernel(detached_params["sigma"], opt=self.params)

    # noinspection PyMethodOverriding
    def compute_sparse(self, X1: SparseTensor, X2: SparseTensor, out: torch.Tensor,
                       X1_csr: SparseTensor, X2_csr: SparseTensor) -> torch.Tensor:
        if len(self.sigma) > 1:
            raise NotImplementedError("Sparse kernel is only implemented for scalar sigmas.")
        dev_kernel_tensor_params = self._move_kernel_params(X1)
        gamma = 1 / dev_kernel_tensor_params["sigma"]
        out = sparse_dist_core(X1_csr=X1_csr, X2_csr=X2_csr, X1=X1, X2=X2, out=out)
        out.sqrt_()
        out.mul_(-gamma)
        out.exp_()
        return out

    def __repr__(self):
        return f"LaplacianKernel(sigma={self.sigma})"

    def __str__(self):
        return f"Laplaciankernel<{self.sigma}>"


class MaternKernel(DistanceKernel):
    r"""Class for computing the Matern kernel, and related kernel-vector products.

    The Matern kernels define a generic class of kernel functions which includes the
    Laplacian and Gaussian kernels. The class is parametrized by 'nu'. When `nu = 0.5`
    this kernel is equivalent to the Laplacian kernel, when `nu = float('inf')`, the
    Matern kernel is equivalent to the Gaussian kernel.

    This class implements the Matern kernel only for the values of nu which have a closed
    form solution, which are 0.5, 1.5, 2.5, and infinity.

    Parameters
    ----------
    sigma
        The length-scale of the Matern kernel. The length-scale can be either a scalar
        or a vector. Matrix-valued length-scales are not allowed for the Matern kernel.
    nu
        The parameter of the Matern kernel. It should be one of `0.5`, `1.5`, `2.5` or
        `inf`.

    Notes
    -----
    While for `nu = float('inf')` this kernel is equivalent to the :class:`GaussianKernel`,
    the implementation is more general and using the :class:`GaussianKernel` directly
    may be computationally more efficient.

    """
    _valid_nu_values = frozenset({0.5, 1.5, 2.5, float('inf')})

    def __init__(self,
                 sigma: Union[float, torch.Tensor],
                 nu: Union[float, torch.Tensor],
                 opt: Optional[FalkonOptions] = None):
        self.sigma = validate_sigma(sigma)
        self.nu = self.validate_nu(nu)
        self.kernel_name = f"{self.nu:.1f}-matern"
        super().__init__(self.kernel_name, opt, core_fn=matern_core, sigma=self.sigma, nu=self.nu)

    def _keops_mmv_impl(self, X1, X2, v, kernel, out, opt: FalkonOptions):
        if self.nu == 0.5:
            formula = 'Exp(-Norm2(x1 / s - x2 / s)) * v'
        elif self.nu == 1.5:
            formula = ('(IntCst(1) + Sqrt(IntCst(3)) * Norm2(x1 / s - x2 / s)) * '
                       '(Exp(-Sqrt(IntCst(3)) * Norm2(x1 / s - x2 / s)) * v)')
        elif self.nu == 2.5:
            formula = ('(IntCst(1) + Sqrt(IntCst(5)) * Norm2(x1 / s - x2 / s) + '
                       '(IntInv(3) * IntCst(5)) * SqNorm2(x1 / s - x2 / s)) * '
                       '(Exp(-Sqrt(IntCst(5)) * Norm2(x1 / s - x2 / s)) * v)')
        elif self.nu == float('inf'):
            formula = 'Exp(IntInv(-2) * SqDist(x1 / s, x2 / s)) * v'
        else:
            raise RuntimeError(f"Unrecognized value of nu ({self.nu}). "
                               f"The onnly allowed values are 0.5, 1.5, 2.5, inf.")
        aliases = [
            'x1 = Vi(%d)' % (X1.shape[1]),
            'x2 = Vj(%d)' % (X2.shape[1]),
            'v = Vj(%d)' % (v.shape[1]),
            's = Pm(%d)' % (self.sigma.shape[0])
        ]
        other_vars = [self.sigma.to(device=X1.device, dtype=X1.dtype)]

        return self.keops_mmv(X1, X2, v, out, formula, aliases, other_vars, opt)

    def extra_mem(self) -> Dict[str, float]:
        extra_mem = {
            # Data-matrix / sigma
            'nd': 1,
            'md': 1,
            # Norm results in prepare
            'm': 1,
            'n': 1,
        }
        if self.nu in {1.5, 2.5}:
            # Extra kernel block in transform
            extra_mem['nm'] = 1
        return extra_mem

    def detach(self) -> 'MaternKernel':
        detached_params = self._detach_params()
        return MaternKernel(detached_params["sigma"], detached_params["nu"], opt=self.params)

    @staticmethod
    def validate_nu(nu: Union[torch.Tensor, float]) -> float:
        if isinstance(nu, torch.Tensor):
            if nu.requires_grad:
                raise ValueError("The nu parameter of the Matern kernel is not differentiable, "
                                 "and must not require gradients.")
            try:
                out_nu = round(nu.item(), ndigits=2)
            except ValueError:
                raise ValueError("nu=%s is not convertible to a scalar." % (nu))
        elif isinstance(nu, float):
            out_nu = round(nu, ndigits=2)
        else:
            raise TypeError(f"nu must be a float or a tensor, not a {type(nu)}")
        if out_nu not in MaternKernel._valid_nu_values:
            raise ValueError(f"The given value of nu = {out_nu} can only take "
                             f"values {MaternKernel._valid_nu_values}.")
        return out_nu

    # noinspection PyMethodOverriding
    def compute_sparse(self, X1: SparseTensor, X2: SparseTensor, out: torch.Tensor,
                       X1_csr: SparseTensor, X2_csr: SparseTensor) -> torch.Tensor:
        if len(self.sigma) > 1:
            raise NotImplementedError("Sparse kernel is only implemented for scalar sigmas.")
        dev_kernel_tensor_params = self._move_kernel_params(X1)
        gamma = 1 / (dev_kernel_tensor_params["sigma"] ** 2)
        out = sparse_dist_core(X1_csr=X1_csr, X2_csr=X2_csr, X1=X1, X2=X2, out=out)
        out.mul_(gamma)

        # For certain nu = 1.5, 2.5 we will need an extra n*m block
        if self.nu == 0.5:
            out.sqrt_()
            out.neg_()  # *= -1
            out.exp_()
        elif self.nu == 1.5:
            # (1 + sqrt(3)*D) * exp(-sqrt(3)*D))
            out.sqrt_()
            out.mul_(math.sqrt(3))

            out_neg = torch.neg(out)
            out_neg.exp_()

            out.add_(1.0)
            out.mul_(out_neg)
        elif self.nu == 2.5:
            # (1 + sqrt(5)*D + (sqrt(5)*D)^2 / 3 ) * exp(-sqrt(5)*D)
            out_sqrt = torch.sqrt(out)
            out_sqrt.mul_(math.sqrt(5))

            out.mul_(5 / 3)
            out.add_(out_sqrt)
            out.add_(1)

            out_sqrt.neg_()
            out_sqrt.exp_()

            out.mul_(out_sqrt)
        elif self.nu == float('inf'):
            out.mul_(-0.5)
            out.exp_()
        return out

    def __repr__(self):
        return f"MaternKernel(sigma={self.sigma}, nu={self.nu:.1f})"

    def __str__(self):
        return f"Matern kernel<{self.sigma}, {self.nu:.1f}>"


class LinearKernel(DistanceKernel):
    """Linear Kernel with optional scaling and translation parameters.

    The kernel implemented here is the covariance function in the original
    input space (i.e. `X @ X.T`) with optional parameters to translate
    and scale the kernel: `beta + 1/(sigma**2) * X @ X.T`

    Parameters
    -----------
    beta : float-like
        Additive constant for the kernel, default: 0.0
    gamma : float-like
        Multiplicative constant for the kernel. The kernel will
        be multiplied by the inverse of sigma squared. Default: 1.0
    opt : Optional[FalkonOptions]
        Options which will be used in downstream kernel operations.

    Examples
    --------
    >>> k = LinearKernel(beta=0.0, sigma=2.0)
    >>> X = torch.randn(100, 3)  # 100 samples in 3 dimensions
    >>> kernel_matrix = k(X, X)
    >>> torch.testing.assert_allclose(kernel_matrix, X @ X.T * (1/2**2))
    """
    def __init__(self,
                 beta: Union[float, torch.Tensor] = 0.0,
                 gamma: Union[float, torch.Tensor] = 1.0,
                 opt: Optional[FalkonOptions] = None):
        self.beta = validate_diff_scalar(beta, param_name="beta")
        self.gamma = validate_diff_scalar(gamma, param_name="gamma")
        super().__init__("Linear", opt, linear_core, beta=self.beta, gamma=self.gamma)

    def _keops_mmv_impl(self, X1, X2, v, kernel, out, opt):
        formula = '(gamma * (X | Y) + beta) * v'
        aliases = [
            'X = Vi(%d)' % (X1.shape[1]),
            'Y = Vj(%d)' % (X2.shape[1]),
            'v = Vj(%d)' % (v.shape[1]),
            'gamma = Pm(1)',
            'beta = Pm(1)'
        ]
        other_vars = [
            torch.tensor([self.gamma]).to(dtype=X1.dtype, device=X1.device),
            torch.tensor([self.beta]).to(dtype=X1.dtype, device=X1.device)
        ]
        return self.keops_mmv(X1, X2, v, out, formula, aliases, other_vars, opt)

    def extra_mem(self) -> Dict[str, float]:
        return {}

    def detach(self) -> 'LinearKernel':
        detached_params = self._detach_params()
        return LinearKernel(beta=detached_params["beta"], gamma=detached_params["gamma"],
                            opt=self.params)

    def compute_sparse(self, X1: SparseTensor, X2: SparseTensor, out: torch.Tensor, **kwargs) -> torch.Tensor:
        dev_kernel_tensor_params = self._move_kernel_params(X1)

        sparse.sparse_matmul(X1, X2, out)
        out.mul_(dev_kernel_tensor_params["gamma"])
        out.add_(dev_kernel_tensor_params["beta"])
        return out

    def __str__(self):
        return f"LinearKernel(beta={self.beta}, gamma={self.gamma})"

    def __repr__(self):
        return self.__str__()


class PolynomialKernel(DistanceKernel):
    r"""Polynomial kernel with multiplicative and additive constants.

    Follows the formula

    .. math::

        (\alpha * X_1^\top X_2 + \beta)^{\mathrm{degree}}

    Where all operations apart from the matrix multiplication are taken element-wise.

    Parameters
    ----------
    beta : float-like
        Additive constant
    gamma : float-like
        Multiplicative constant
    degree : float-like
        Power of the polynomial kernel
    opt : Optional[FalkonOptions]
        Options which will be used in downstream kernel operations.
    """
    def __init__(self,
                 beta: Union[float, torch.Tensor],
                 gamma: Union[float, torch.Tensor],
                 degree: Union[float, torch.Tensor],
                 opt: Optional[FalkonOptions] = None):
        self.beta = validate_diff_scalar(beta, param_name="beta")
        self.gamma = validate_diff_scalar(gamma, param_name="gamma")
        self.degree = validate_diff_scalar(degree, param_name="degree")
        super().__init__("Polynomial", opt, polynomial_core, beta=self.beta, gamma=self.gamma,
                         degree=self.degree)

    def _keops_mmv_impl(self, X1, X2, v, kernel, out, opt):
        formula = 'Powf((alpha * (X | Y) + beta), degree) * v'
        aliases = [
            'X = Vi(%d)' % (X1.shape[1]),
            'Y = Vj(%d)' % (X2.shape[1]),
            'v = Vj(%d)' % (v.shape[1]),
            'alpha = Pm(1)',
            'beta = Pm(1)',
            'degree = Pm(1)',
        ]
        other_vars = [
            torch.tensor([self.gamma]).to(dtype=X1.dtype, device=X1.device),
            torch.tensor([self.beta]).to(dtype=X1.dtype, device=X1.device),
            torch.tensor([self.degree]).to(dtype=X1.dtype, device=X1.device)
        ]

        return self.keops_mmv(X1, X2, v, out, formula, aliases, other_vars, opt)

    def extra_mem(self) -> Dict[str, float]:
        return {}

    def detach(self) -> 'PolynomialKernel':
        detached_params = self._detach_params()
        return PolynomialKernel(beta=detached_params["beta"], gamma=detached_params["gamma"],
                                degree=detached_params["degree"], opt=self.params)

    def compute_sparse(self, X1: SparseTensor, X2: SparseTensor, out: torch.Tensor, **kwargs) -> torch.Tensor:
        dev_kernel_tensor_params = self._move_kernel_params(X1)

        sparse.sparse_matmul(X1, X2, out)
        out.mul_(dev_kernel_tensor_params["gamma"])
        out.add_(dev_kernel_tensor_params["beta"])
        out.pow_(dev_kernel_tensor_params["degree"])
        return out

    def __str__(self):
        return f"PolynomialKernel(beta={self.beta}, gamma={self.gamma}, degree={self.degree})"

    def __repr__(self):
        return self.__str__()


class SigmoidKernel(DistanceKernel):
    r"""Sigmoid (or hyperbolic tangent) kernel function, with additive and multiplicative constants.

    Follows the formula

    .. math::

        k(x, y) = \tanh(\alpha x^\top y + \beta)

    Parameters
    ----------
    beta : float-like
        Multiplicative constant
    gamma : float-like
        Multiplicative constant
    opt : Optional[FalkonOptions]
        Options which will be used in downstream kernel operations.


    """

    def __init__(self,
                 beta: Union[float, torch.Tensor],
                 gamma: Union[float, torch.Tensor],
                 opt: Optional[FalkonOptions] = None):
        self.beta = validate_diff_scalar(beta, param_name="beta")
        self.gamma = validate_diff_scalar(gamma, param_name="gamma")
        super().__init__("Sigmoid", opt, sigmoid_core, beta=self.beta, gamma=self.gamma)

    def _keops_mmv_impl(self, X1, X2, v, kernel, out, opt: FalkonOptions):
        return RuntimeError("SigmoidKernel is not implemented in KeOps")

    def _decide_mmv_impl(self, X1, X2, v, opt):
        if self.keops_can_handle_mmv(X1, X2, v, opt):
            warnings.warn("KeOps MMV implementation for %s kernel is not available. "
                          "Falling back to matrix-multiplication based implementation."
                          % (self.name))
        return super()._decide_mmv_impl(X1, X2, v, opt)

    def _decide_dmmv_impl(self, X1, X2, v, w, opt):
        if self.keops_can_handle_dmmv(X1, X2, v, w, opt):
            warnings.warn("KeOps dMMV implementation for %s kernel is not available. "
                          "Falling back to matrix-multiplication based implementation."
                          % (self.name))
        return super()._decide_dmmv_impl(X1, X2, v, w, opt)

    def extra_mem(self) -> Dict[str, float]:
        return {}

    def detach(self) -> 'SigmoidKernel':
        detached_params = self._detach_params()
        return SigmoidKernel(beta=detached_params["beta"], gamma=detached_params["gamma"],
                             opt=self.params)

    def compute_sparse(self, X1: SparseTensor, X2: SparseTensor, out: torch.Tensor, **kwargs) -> torch.Tensor:
        dev_kernel_tensor_params = self._move_kernel_params(X1)

        sparse.sparse_matmul(X1, X2, out)
        out.mul_(dev_kernel_tensor_params["gamma"])
        out.add_(dev_kernel_tensor_params["beta"])
        out.tanh_()
        return out

    def __str__(self):
        return f"SigmoidKernel(beta={self.beta}, gamma={self.gamma})"

    def __repr__(self):
        return self.__str__()
