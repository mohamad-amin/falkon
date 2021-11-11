import functools
import math
import warnings
from typing import Union, Optional

import torch
from dataclasses import dataclass

from falkon import FalkonOptions
from falkon.kernels import GaussianKernel
from falkon.kernels.distance_kernel import DistKerContainer
from falkon.kernels.tiling_red import TilingGenred
from falkon.mmv_ops.keops import _decide_backend, _keops_dtype
from falkon.utils.helpers import check_same_device


class SimpleKernel():
    def __init__(self, sigma: float):
        self.sigma = sigma
        self.gamma = 1/(2*sigma**2)

    @staticmethod
    def _get_sigma_kt(sigma: torch.Tensor):
        return sigma, "single"

    def _sigma2gamma(self, sigma: torch.Tensor):
        return sigma

    def mmv_batch(self, mat1_b, mat2_b, mat_out_b, vec_b, vec_out_b):
        """
        Splitting over n, m

        mat1_b : b_n * d
        mat2_b : M * d
        mat_out_b: b_n * M
        vec_b : M * T
        vec_out_b : b_n * T
        """
        norm_sq_mat1 = torch.norm(mat1_b, p=2, dim=1, keepdim=True).pow_(2)
        norm_sq_mat2 = torch.norm(mat2_b, p=2, dim=1, keepdim=True).pow_(2)
        torch.mm(mat1_b, mat2_b.T, out=mat_out_b)
        mat_out_b.mul_(-2.0)
        mat_out_b.add_(norm_sq_mat1)
        mat_out_b.add_(norm_sq_mat2.T)

        mat_out_b.clamp_min_(0)
        mat_out_b.mul_(self.gamma)
        mat_out_b.exp_()
        vec_out_b.addmm_(mat_out_b, vec_b)

    def dmmv_batch(self, mat1_b, mat2, mat_out_b, vec1, vec2_b, temp_vec_out_b, vec_out):
        """
        Splitting over n ONLY

        Parameters
        ----------
        mat1_b
        mat2
        mat_out_b
        vec
        vec_out_b

        Returns
        -------

        """
        norm_sq_mat1 = torch.norm(mat1_b, p=2, dim=1, keepdim=True).pow_(2)
        norm_sq_mat2 = torch.norm(mat2, p=2, dim=1, keepdim=True).pow_(2)
        torch.mm(mat1_b, mat2.T, out=mat_out_b)
        mat_out_b.mul_(-2.0)
        mat_out_b.add_(norm_sq_mat1)
        mat_out_b.add_(norm_sq_mat2.T)
        mat_out_b.clamp_min_(0)
        mat_out_b.mul_(self.gamma)
        mat_out_b.exp_()

        torch.mm(mat_out_b, vec1, out=temp_vec_out_b)
        if vec2_b is not None:
            temp_vec_out_b.add_(vec2_b)

        vec_out.addmm_(mat_out_b.T, temp_vec_out_b)

    def grad_batch(self, mat1_b, mat2_b, mat_out_b, vec_b, vec_out_b):
        # Gradient of
        raise RuntimeError("")


@torch.jit.script
def rbf_core(mat1, mat2, normsq1, normsq2, mat_out):
    return torch.mm(mat1, mat2.T, out=mat_out).mul_(-2.0).add_(normsq1).add_(normsq2.T).exp_()


class MmvRbf(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sigmas, mat1_b, mat2_b, vec_b, mat_out_b, vec_out_b):
        """Run vec_out_b = k(mat1_b, mat2_b) @ vec_b

        Using tensor `mat_out_b` for temporary storage

        Parameters
        ----------
        ctx
        sigmas : torch.Tensor
            Scalar or d-dimensional tensor. The RBF kernel length-scales
        mat1_b : torch.Tensor
            n * D data tensor
        mat2_b : torch.Tensor
            m * D data tensor
        vec_b : torch.Tensor
            m * T data tensor
        mat_out_b : torch.Tensor
            n * m tensor, used as scratch space. After this function, the buffer can be
            overwritten.
        vec_out_b : torch.Tensor
            n * T tensor. At the end of this function this will contain kernel-vector product.
            It will share the data with the return value of the function itself.

        Returns
        -------

        Notes
        -----
        Additional memory allocated by this function:
         - m * d
         - n * d
         - n + m
        """
        mat1_div_sig = mat1_b / sigmas
        mat2_div_sig = mat2_b / sigmas

        norm_sq_mat1 = torch.norm(mat1_div_sig, p=2, dim=1, keepdim=True).pow_(2)  # n * 1
        norm_sq_mat2 = torch.norm(mat2_div_sig, p=2, dim=1, keepdim=True).pow_(2)  # m * 1
        mat_out_b = rbf_core(mat1_div_sig, mat2_div_sig, norm_sq_mat1, norm_sq_mat2, mat_out_b)

        torch.mm(mat_out_b, vec_b, out=vec_out_b)  # (n * m) @ (m * 1) => (n * 1)

        # Save the large buffer to reuse ITS MEMORY in the backward pass. The data of mat_out_b
        # is not used. The use-case is to launch this function for several data-tiles, all
        # using the same n*m buffer. During the backward pass, as long as the tiles are processed
        # one at a time, the same buffer will be reused.
        # NOTE that the backward pass removes references to the buffer (via `del`)
        ctx.big_mat = mat_out_b
        ctx.save_for_backward(sigmas, mat1_b, mat2_b, vec_b)
        return vec_out_b

    @staticmethod
    def backward(ctx, grad_outputs):
        """
        Notes
        -----
        Additional memory allocated by this function
         - n * m (large buffer)
         - n * d * 2
         - m * d * 2
         - n * 3
         - m * 3
         - d * 5
        """
        sigmas, mat1, mat2, vec = ctx.saved_tensors
        sigmas_arg = BackwardArg(sigmas, needs_grad=ctx.needs_input_grad[0])
        mat1_arg = BackwardArg(mat1, needs_grad=ctx.needs_input_grad[1])
        mat2_arg = BackwardArg(mat2, needs_grad=ctx.needs_input_grad[2])

        try:
            big_mat = ctx.big_mat
        except AttributeError:  # Can happen when doing gradcheck
            big_mat = torch.empty(mat1.shape[0], mat2.shape[0],
                                  dtype=mat1.dtype, device=mat1.device)

        mmv_rbf_bwd(grad_outputs, sigmas_arg, mat1_arg, mat2_arg, vec, big_mat)

        try:
            del big_mat, ctx.big_mat
        except AttributeError:
            pass
        return sigmas_arg.grad, mat1_arg.grad, mat2_arg.grad, None, None, None


@dataclass
class BackwardArg:
    val: torch.Tensor
    needs_grad: bool
    grad: Optional[torch.Tensor] = None


def mmv_rbf_bwd(grad_outputs: torch.Tensor,
                sigmas: BackwardArg,
                mat1: BackwardArg,
                mat2: BackwardArg,
                vec: torch.Tensor,
                temp_nm: torch.Tensor):
    n, m, d = mat1.val.shape[0], mat2.val.shape[0], mat1.val.shape[1]
    dev, dt = mat1.val.device, mat1.val.dtype

    if sigmas.needs_grad and sigmas.grad is None:
        sigmas.grad = torch.zeros(d, dtype=dt, device=dev)
    if mat1.needs_grad and mat1.grad is None:
        mat1.grad = torch.zeros(n, d, dtype=dt, device=dev)
    if mat2.needs_grad and mat2.grad is None:
        mat2.grad = torch.zeros(m, d, dtype=dt, device=dev)
    if not (sigmas.needs_grad or mat1.needs_grad or mat2.needs_grad):
        return

    temp_nd, temp_md = None, None
    if sigmas.needs_grad or mat1.needs_grad:
        temp_nd = torch.empty(n, d, dtype=dt, device=dev)
    if sigmas.needs_grad or mat2.needs_grad:
        temp_md = torch.empty(m, d, dtype=dt, device=dev)

    # Replicate forward pass to get past the `exp` derivative.
    mat1_div_sig = mat1.val / sigmas.val  # n * d
    mat2_div_sig = mat2.val / sigmas.val  # m * d
    norm_mat1 = torch.norm(mat1_div_sig, p=2, dim=1, keepdim=True)  # copy n
    norm_mat2 = torch.norm(mat2_div_sig, p=2, dim=1, keepdim=True)  # copy m
    sqnorm_mat1 = norm_mat1.square()  # copy n * 1
    sqnorm_mat2 = norm_mat2.square()  # copy m * 1
    temp_nm = rbf_core(mat1_div_sig, mat2_div_sig, sqnorm_mat1, sqnorm_mat2, temp_nm)

    # PART 1: Backwards w.r.t. first mat-mul
    # MM & EXP combined
    tmp2 = temp_nm.mul_(grad_outputs).mul_(vec.T)  # n * m
    # Save the row and col sums for parts 2, 3
    tmp2_1 = tmp2.sum(1, keepdim=True)  # copy n * 1
    tmp3_1 = tmp2.sum(0, keepdim=True)  # copy 1 * m
    # Mul
    tmp3 = tmp2.mul_(-2.0)  # n * m
    if mat2.needs_grad:
        # MM (mat1/sigma^2 @ mat2.T) wrt mat2. Transpose is included
        mat_div_sigsq = torch.div(mat1_div_sig, sigmas.val, out=temp_nd)  # n * d
        mat2.grad.addmm_(tmp3.T, mat_div_sigsq)  # (m * n) @ (n * d) => (m * d)
    if sigmas.needs_grad or mat1.needs_grad:
        # MM (mat1/sigma^2 @ mat2.T) wrt mat1/sigma^2
        tmp4 = torch.mm(tmp3, mat2.val, out=temp_nd)  # (n * m) @ (m * d) => (n * d)
        # (div_(sigmas**2) first to help grad wrt mat2).
        tmp4.div_(sigmas.val**2)
        if mat1.needs_grad:
            # Div (mat1/sigma^2) wrt mat1
            mat1.grad.add_(tmp4)
        if sigmas.needs_grad:
            # Div (mat1/sigma^2) wrt sigma^2
            tmp5 = tmp4.neg_().mul_(mat1_div_sig).div_(sigmas.val)  # n * d + (copy d)
            tmp6 = tmp5.sum(0)  # copy d
            # Pow (sigma^2)
            tmp6.mul_(sigmas.val * 2)  # copy d
            sigmas.grad.add_(tmp6)

    # PART 2: Backwards w.r.t. norm(mat1/sigma)^2
    if sigmas.needs_grad or mat1.needs_grad:
        # Squaring of the norm
        tmp2_2 = tmp2_1.mul_(2).mul_(norm_mat1)  # n * 1
        # Norm itself
        scale_v = tmp2_2.div_(norm_mat1)  # n * 1
        scale_v.masked_fill_(norm_mat1 == 0, 0)
        tmp2_3 = torch.mul(mat1_div_sig, scale_v, out=temp_nd)  # n * d
        # (div_(sigmas) first to help grad wrt mat1).
        tmp2_3.div_(sigmas.val)
        if mat1.needs_grad:
            # Div (mat1/sigma) wrt mat1
            mat1.grad.add_(tmp2_3)
        if sigmas.needs_grad:
            # Div (mat1/sigma) wrt sigma
            tmp2_4 = tmp2_3.neg_().mul_(mat1_div_sig)  # n * d
            sigmas.grad.add_(tmp2_4.sum(0))  # copy d

    # PART 3: Backwards w.r.t. (norm(mat2/sigma)^2).T
    if sigmas.needs_grad or mat2.needs_grad:
        # Permute
        tmp3_2 = tmp3_1.T  # m * 1
        # Squaring of the norm
        tmp3_3 = tmp3_2.mul_(2).mul_(norm_mat2)  # m * 1
        # Norm
        scale_v = tmp3_3.div_(norm_mat2)  # m * 1
        scale_v.masked_fill_(norm_mat2 == 0, 0)  # m * 1
        tmp3_4 = torch.mul(mat2_div_sig, scale_v, out=temp_md)  # m * d
        # Div (mat2/sigma) wrt sigma (div_(sigmas) first to help grad wrt mat2).
        tmp3_4.div_(sigmas.val)
        if mat2.needs_grad:
            # Div (mat2/sigma) wrt mat2
            mat2.grad.add_(tmp3_4)
        if sigmas.needs_grad:
            tmp3_5 = tmp3_4.neg_().mul_(mat2_div_sig)  # m * d
            sigmas.grad.add_(tmp3_5.sum(0))  # copy d



def mmv_h2d(hbuf, dbuf):
    dbuf['mat2'].copy_(hbuf['mat2'], non_blocking=True)
    dbuf['vec'].copy_(hbuf['vec'], non_blocking=True)


def mmv_comp(dbuf):
    MmvRbf.apply(
        dbuf['sigmas'],
        dbuf['mat1'],
        dbuf['mat2'],
        dbuf['vec'],
        dbuf['nm_temp'],
        dbuf['out'],
    )


def mmv_d2h(dbuf, hbuf):
    hbuf['out'].copy_(dbuf['out'], non_blocking=True)


def double_buffer(hbuf1, hbuf2, dbuf1, dbuf2, h2d, d2h, comp):
    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()

    with torch.cuda.stream(s1):
        h2d(hbuf=hbuf1, dbuf=dbuf1)
        comp(dbuf=dbuf1)
        d2h(dbuf=dbuf1, hbuf=hbuf1)
    if hbuf2 is None:
        s1.synchronize()
        return
    with torch.cuda.stream(s2):
        h2d(hbuf=hbuf2, dbuf=dbuf2)
        comp(dbuf=dbuf2)
        d2h(dbuf=dbuf2, hbuf=hbuf2)
    s1.synchronize()
    s2.synchronize()


class MmvRun(torch.autograd.Function):
    NUM_NONDIFF_ARGS = 5
    MIN_DOUBLEBUF_LEN = 10_000

    @staticmethod
    def forward(ctx, sigmas, mat1, mat2, vec, vec_out, b1, b2, dev):
        N, D = mat1.shape
        M = mat2.shape[0]
        T = vec.shape[1]
        b1 = min(b1, N)
        b2 = min(b2, M)

        # Initialize GPU buffers
        dev_sigmas = sigmas.to(device=dev)
        # large n_b*m_b buffer
        dev_mat_temp = torch.empty(b1, b2, dtype=mat1.dtype, device=dev)
        # Buffer for the output (probably UNNECESSARY?)
        dev_out_temp = torch.zeros(2, b1, T, dtype=mat1.dtype, device=dev)
        host_out_temp = torch.zeros(2, b1, T, dtype=mat1.dtype, device=mat1.device)
        # Device-side buffers for the data
        dev_mat1 = torch.empty(b1, D, dtype=mat1.dtype, device=dev)
        dev_mat2 = torch.empty(b2, D, dtype=mat1.dtype, device=dev)
        dev_vec = torch.empty(b2, T, dtype=mat1.dtype, device=dev)

        for i in range(0, N, b1):
            leni = min(b1, N - i)
            # Populate GPU buffer with mat1
            dev_mat1[:leni].copy_(mat1[i: i + leni])
            for j in range(0, M, b2):
                host_out_temp.fill_(0.0)
                full_lenj = min(b2, M - j)
                if full_lenj < MmvRun.MIN_DOUBLEBUF_LEN:
                    j1, lenj1 = j, full_lenj
                    j2, lenj2 = None, None
                else:
                    j1, lenj1 = j, full_lenj // 2
                    j2, lenj2 = j1 + lenj1, full_lenj - lenj1

                hbuf1 = {
                    'mat1': mat1[i: i + leni],
                    'mat2': mat2[j1: j1 + lenj1],
                    'vec': vec[j1: j1 + lenj1],
                    'sigmas': sigmas,
                    'out': host_out_temp[0, :leni],
                }
                dbuf1 = {
                    'mat1': dev_mat1[:leni],
                    'mat2': dev_mat2[:lenj1],
                    'vec': dev_vec[:lenj1],
                    'nm_temp': dev_mat_temp[:leni, :lenj1],
                    'sigmas': dev_sigmas,
                    'out': dev_out_temp[0, :leni],
                }
                if j2 is not None:
                    hbuf2 = {
                        'mat1': mat1[i: i + leni],
                        'mat2': mat2[lenj1: lenj1 + lenj2],
                        'vec': vec[lenj1: lenj1 + lenj2],
                        'sigmas': sigmas,
                        'out': host_out_temp[1, :leni],
                    }
                    dbuf2 = {
                        'mat1': dev_mat1[:leni],
                        'mat2': dev_mat2[lenj1: lenj1 + lenj2],
                        'vec': dev_vec[lenj1: lenj1 + lenj2],
                        'nm_temp': dev_mat_temp[:leni, lenj1: lenj1 + lenj2],
                        'sigmas': dev_sigmas,
                        'out': dev_out_temp[1, :leni],
                    }
                else:
                    hbuf2, dbuf2 = None, None

                double_buffer(hbuf1=hbuf1, hbuf2=hbuf2, dbuf1=dbuf1, dbuf2=dbuf2,
                              h2d=mmv_h2d, d2h=mmv_d2h, comp=mmv_comp)
                # Aggregate outputs across j-dim
                vec_out[i: i + leni].add_(host_out_temp[:, :leni].sum(0))

        ctx.save_for_backward(sigmas, mat1, mat2, vec)
        ctx.b1 = b1
        ctx.b2 = b2
        ctx.dev = dev
        ctx.big_mat = dev_mat_temp
        return vec_out

    @staticmethod
    def backward(ctx, grad_output):
        sigmas, mat1, mat2, vec = ctx.saved_tensors
        host = mat1.device
        dtype = mat1.dtype
        b1, b2, dev = ctx.b1, ctx.b2, ctx.dev
        N, D = mat1.shape
        M = mat2.shape[0]
        T = vec.shape[1]

        # Initialize GPU buffers
        dev_grad_output = grad_output.to(device=dev)
        dev_sigmas = sigmas.to(device=dev)
        try:
            dev_mat_temp = ctx.big_mat
        except AttributeError:  # Can happen when doing gradcheck
            dev_mat_temp = torch.empty(b1.shape[0], b2.shape[0], dtype=dtype, device=dev)
        dev_mat1_buf = torch.empty(b1, D, dtype=dtype, device=dev)
        dev_mat2_buf = torch.empty(b2, D, dtype=dtype, device=dev)
        dev_vec_buf = torch.empty(b2, T, dtype=dtype, device=dev)

        need_sigma_grad, need_m1_grad, need_m2_grad = ctx.needs_input_grad[0], ctx.needs_input_grad[1], ctx.needs_input_grad[2]

        host_sigma_grad, host_mat1_grad, host_mat2_grad = None, None, None

        sigmas_arg = BackwardArg(dev_sigmas, needs_grad=ctx.needs_input_grad[0])
        if need_sigma_grad:
            dev_sigma_grad = torch.zeros(D, dtype=dtype, device=dev)
            host_sigma_grad = torch.zeros(D, dtype=dtype, device=host)
            sigmas_arg.grad = dev_sigma_grad
        if need_m1_grad:
            dev_mat1_grad = torch.empty(b1, D, dtype=dtype, device=dev)
            host_mat1_grad = torch.zeros(N, D, dtype=dtype, device=host)
        if need_m2_grad:
            dev_mat2_grad = torch.empty(b2, D, dtype=dtype, device=dev)
            host_mat2_grad = torch.zeros(M, D, dtype=dtype, device=host)

        for i in range(0, N, b1):
            leni = min(b1, N - i)
            ji = 0
            # Populate GPU buffers
            dev_mat1_buf[:leni].copy_(mat1[i: i + leni])
            mat1_arg = BackwardArg(dev_mat1_buf[:leni], needs_grad=need_m1_grad)
            if need_m1_grad:
                dev_mat1_grad[:leni].fill_(0.0)
                mat1_arg.grad = dev_mat1_grad[:leni]
            for j in range(0, M, b2):
                lenj = min(b2, M - j)
                # Populate GPU buffers
                dev_mat2_buf[:lenj].copy_(mat2[j: j + lenj])
                dev_vec_buf[:lenj].copy_(vec[j: j + lenj])
                mat2_arg = BackwardArg(dev_mat2_buf[:lenj], needs_grad=need_m2_grad)
                if need_m2_grad:
                    dev_mat2_grad[:lenj].fill_(0.0)
                    mat2_arg.grad = dev_mat2_grad[:lenj]

                mmv_rbf_bwd(
                    dev_grad_output[i: i + leni],
                    sigmas_arg,
                    mat1_arg,
                    mat2_arg,
                    dev_vec_buf[:lenj],
                    dev_mat_temp[:leni, :lenj])
                if need_m2_grad:
                    host_mat2_grad[j: j + lenj].add_(mat2_arg.grad.to(device=host))
            if need_m1_grad:
                host_mat1_grad[i: i + leni].add_(mat1_arg.grad.to(device=host))
        if need_sigma_grad:
            host_sigma_grad.copy_(sigmas_arg.grad)

        try:
            del dev_mat_temp, ctx.big_mat
        except AttributeError:
            pass

        out = [host_sigma_grad, host_mat1_grad, host_mat2_grad] + [None] * MmvRun.NUM_NONDIFF_ARGS
        return tuple(out)



class DiffGaussianKernel(GaussianKernel):
    def __init__(self, sigma: Union[float, torch.Tensor], opt: Optional[FalkonOptions] = None):
        super().__init__(sigma, opt)

    @staticmethod
    def _get_sigma_kt(sigma: torch.Tensor):
        return sigma, "single"

    def _sigma2gamma(self, sigma: torch.Tensor):
        return sigma

    def _keops_mmv_impl(self, X1, X2, v, kernel, out, opt: FalkonOptions):
        formula = 'Exp(SqDist(x1 / g, x2 / g) * IntInv(-2)) * v'
        aliases = [
            'x1 = Vi(%d)' % (X1.shape[1]),
            'x2 = Vj(%d)' % (X2.shape[1]),
            'v = Vj(%d)' % (v.shape[1]),
            'g = Pm(%d)' % (self.sigma.shape[0]),
        ]
        other_vars = [self.sigma.to(device=X1.device, dtype=X1.dtype)]

        # Choose backend
        N, D = X1.shape
        backend = _decide_backend(opt, D)
        dtype = _keops_dtype(X1.dtype)
        device = X1.device

        if not check_same_device(X1, X2, v, out, *other_vars):
            raise RuntimeError("All input tensors must be on the same device.")
        if (device.type == 'cuda') and (not backend.startswith("GPU")):
            warnings.warn("KeOps backend was chosen to be CPU, but GPU input tensors found. "
                          "Defaulting to 'GPU_1D' backend. To force usage of the CPU backend, "
                          "please pass CPU tensors; to avoid this warning if the GPU backend is "
                          "desired, check your options (i.e. set 'use_cpu=False').")
            backend = "GPU_1D"

        func = TilingGenred(formula, aliases, reduction_op='Sum', axis=1, dtype=dtype,
                            dtype_acc="auto", sum_scheme="auto", opt=opt)
        return func(X1, X2, v, *other_vars, out=out, backend=backend)

    def _decide_mmv_impl(self, X1, X2, v, opt: FalkonOptions):
        return self._keops_mmv_impl

    def _decide_dmmv_impl(self, X1, X2, v, w, opt: FalkonOptions):
        return functools.partial(self.keops_dmmv_helper, mmv_fn=self._keops_mmv_impl)

    def _prepare(self, X1, X2):
        if self.gaussian_type == "full":
            raise NotImplementedError("DiffRbfKernel doesn't work with full covariance")

        sigma = self.sigma.to(X1)
        X1 = X1.div(sigma)
        X2 = X2.div(sigma)
        return DistKerContainer(
            sq1=torch.norm(X1, p=2, dim=1, keepdim=True).square(),
            sq2=torch.norm(X2, p=2, dim=1, keepdim=True).square()
        )

    def _apply(self, X1, X2, out):
        if self.gaussian_type == "full":
            raise NotImplementedError("DiffRbfKernel doesn't work with full covariance")

        sigma = self.sigma.to(X1)
        out.addmm_(X1.div(sigma.square()), X2)

    def _transform(self, A) -> torch.Tensor:
        A.mul_(-0.5)
        A.exp_()
        return A

    def __repr__(self):
        return f"DiffGaussianKernel(sigma={self.sigma})"

    def __str__(self):
        return f"DiffGaussian kernel<{self.sigma}>"
