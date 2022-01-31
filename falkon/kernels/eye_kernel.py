from typing import Union, Optional, Dict

import torch

from falkon import sparse
from falkon.kernels.diff_kernel import DiffKernel
from falkon.la_helpers.square_norm_fn import square_norm_diff
from falkon.options import FalkonOptions
from falkon.sparse import SparseTensor


def eye_core(mat1, mat2, out: Optional[torch.Tensor], sigma):
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

    # out = _sq_dist(mat1_div_sig, mat2_div_sig, norm_sq_mat1, norm_sq_mat2, out)
    out.mul_(-0.5)
    out.exp_()
    return out

class GaussianKernel(DiffKernel):
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
        return rbf_core_sparse(X1, X2, X1_csr, X2_csr, out, dev_kernel_tensor_params["sigma"])

    def __repr__(self):
        return f"GaussianKernel(sigma={self.sigma})"

    def __str__(self):
        return f"Gaussian kernel<{self.sigma}>"