import time
import dataclasses
from contextlib import ExitStack
from typing import Dict, Optional, Tuple, Sequence

import numpy as np
import scipy.linalg
import torch

from falkon import FalkonOptions
from falkon.hypergrad.common import full_rbf_kernel
from falkon.kernels import GaussianKernel
from falkon.kernels.diff_rbf_kernel import DiffGaussianKernel
from falkon.optim import FalkonConjugateGradient
from falkon.preconditioner import FalkonPreconditioner
from falkon.utils.helpers import sizeof_dtype, select_dim_over_n
from falkon.la_helpers import trsm
from falkon.utils.tictoc import Timer, TicToc

__all__ = (
    "NoRegLossAndDeff",
    "GCV",
    "RegLossAndDeffv2",
    "creg_plainfit",
    "gcv",
    "creg_penfit",
    "validation_loss",
    "ValidationLoss",
)

EPS = 5e-5


class NoRegLossAndDeffCtx():
    __slots__ = ("_flk_solve_zy", "_flk_solve_zy_prec",
                 "_kmn_zy",
                 "_flk_solve_ytilde", "_flk_solve_ytilde_prec",
                 "_len_z",
                 "_knm_solve_zy", "_kmm_solve_zy", "_nys_trace", "_nys_deff",
                 "_nys_d_eff",
                 "_nys_data_fit",
                 "_flk_solve_y",
                 "_kmn_y",
                 "_z",
                 )

    def __init__(self,
                 len_z: int,
                 ):
        self._len_z = len_z

        self._flk_solve_zy_prec = None
        self._flk_solve_zy = None
        self._flk_solve_y = None
        self._kmn_zy = None
        self._kmn_y = None
        self._flk_solve_ytilde = None
        self._flk_solve_ytilde_prec = None
        self._knm_solve_zy = None
        self._kmm_solve_zy = None
        self._nys_trace = None
        self._nys_deff = None
        self._nys_d_eff = None
        self._nys_data_fit = None
        self._z = None

    @property
    def z(self):
        return self._z

    @z.setter
    def z(self, val):
        self._z = val

    @property
    def solve_zy_prec(self):
        return self._flk_solve_zy_prec

    @solve_zy_prec.setter
    def solve_zy_prec(self, val):
        self._flk_solve_zy_prec = val

    @property
    def solve_zy(self):
        return self._flk_solve_zy

    @solve_zy.setter
    def solve_zy(self, val):
        self._flk_solve_zy = val

    @property
    def solve_z(self):
        return self._flk_solve_zy[:, :self._len_z]

    @property
    def solve_y(self):
        if self._flk_solve_y is None:
            if self._flk_solve_zy is None:
                return None
            return self._flk_solve_zy[:, self._len_z:].contiguous()
        return self._flk_solve_y

    @solve_y.setter
    def solve_y(self, val):
        self._flk_solve_y = val

    @property
    def kmn_zy(self):
        return self._kmn_zy

    @kmn_zy.setter
    def kmn_zy(self, val):
        self._kmn_zy = val

    @property
    def kmn_z(self):
        return self._kmn_zy[:, :self._len_z]

    @property
    def kmn_y(self):
        if self._kmn_y is None:
            if self._kmn_zy is None:
                return None
            return self._kmn_zy[:, self._len_z:]
        return self._kmn_y

    @kmn_y.setter
    def kmn_y(self, val):
        self._kmn_y = val

    @property
    def solve_ytilde_prec(self):
        return self._flk_solve_ytilde_prec

    @solve_ytilde_prec.setter
    def solve_ytilde_prec(self, val):
        self._flk_solve_ytilde_prec = val

    @property
    def solve_ytilde(self):
        return self._flk_solve_ytilde

    @solve_ytilde.setter
    def solve_ytilde(self, val):
        self._flk_solve_ytilde = val

    @property
    def y_tilde(self):
        return self.knm_solve_y

    @property
    def knm_solve_zy(self):
        return self._knm_solve_zy

    @knm_solve_zy.setter
    def knm_solve_zy(self, val):
        self._knm_solve_zy = val

    @property
    def knm_solve_z(self):
        return self._knm_solve_zy[:, :self._len_z]

    @property
    def knm_solve_y(self):
        return self._knm_solve_zy[:, self._len_z:]

    @property
    def kmm_solve_zy(self):
        return self._kmm_solve_zy

    @kmm_solve_zy.setter
    def kmm_solve_zy(self, val):
        self._kmm_solve_zy = val

    @property
    def kmm_solve_z(self):
        return self._kmm_solve_zy[:, :self._len_z]

    @property
    def kmm_solve_y(self):
        if self._kmm_solve_zy is None:
            return None
        return self._kmm_solve_zy[:, self._len_z:]

    @property
    def nys_trace(self):
        return self._nys_trace

    @nys_trace.setter
    def nys_trace(self, val):
        self._nys_trace = val

    @property
    def nys_deff(self):
        return self._nys_deff

    @nys_deff.setter
    def nys_deff(self, val):
        self._nys_deff = val

    @property
    def nys_d_eff(self):
        return self._nys_d_eff

    @nys_d_eff.setter
    def nys_d_eff(self, val):
        self._nys_d_eff = val

    @property
    def data_fit(self):
        return self._nys_data_fit

    @data_fit.setter
    def data_fit(self, val):
        self._nys_data_fit = val


def tri_inverse(T, lower):
    if T.dtype == torch.float32:
        Tinv = scipy.linalg.lapack.strtri(T.cpu().detach().numpy(), lower=lower, unitdiag=0,
                                          overwrite_c=0)
    elif T.dtype == torch.float64:
        Tinv = scipy.linalg.lapack.dtrtri(T.cpu().detach().numpy(), lower=lower, unitdiag=0,
                                          overwrite_c=0)
    else:
        raise TypeError("Dtype %s invalid" % (T.dtype))
    if Tinv[1] != 0:
        raise RuntimeError("Trtri failed ", Tinv[1])
    return torch.from_numpy(Tinv[0]).to(device=T.device)


def init_random_vecs(n, t, dtype, device, gaussian_random: bool):
    if gaussian_random:
        Z = torch.randn(n, t, dtype=dtype, device=device)
    else:
        Z = torch.empty(n, t, dtype=dtype, device=device).bernoulli_().mul_(2).sub_(1)
    return Z


def solve_falkon(X, centers, penalty, rhs, kernel_args, solve_options, solve_maxiter,
                 init_sol=None):
    penalty = penalty.item()
    M_ = centers.detach()
    kernel_args_ = kernel_args.detach()
    K = GaussianKernel(kernel_args_, opt=solve_options)  # here which opt doesnt matter
    precond = FalkonPreconditioner(penalty, K, solve_options)  # here which opt doesnt matter
    precond.init(M_)
    optim = FalkonConjugateGradient(K, precond, solve_options)
    beta = optim.solve(
        X, M_, rhs, penalty,
        initial_solution=init_sol,
        max_iter=solve_maxiter,
    )
    num_iters = optim.optimizer.num_iter
    sol_full = precond.apply(beta)  # eta, alpha
    return sol_full, beta, num_iters


def calc_grads_tensors(inputs: Sequence[torch.Tensor],
                       inputs_need_grad: Sequence[bool],
                       backward: torch.Tensor,
                       retain_graph: bool,
                       allow_unused: bool) -> Tuple[Optional[torch.Tensor], ...]:
    assert len(inputs) <= len(inputs_need_grad)
    needs_grad = []
    for i in range(len(inputs)):
        if inputs_need_grad[i]:
            needs_grad.append(inputs[i])
    grads = torch.autograd.grad(
        backward, needs_grad, retain_graph=retain_graph, allow_unused=allow_unused)
    j = 0
    results = []
    for i in range(len(inputs_need_grad)):
        if inputs_need_grad[i]:
            results.append(grads[j])
            j += 1
        else:
            results.append(None)
    return tuple(results)


def calc_grads(ctx, backward, num_diff_args):
    return calc_grads_tensors(ctx.saved_tensors, ctx.needs_input_grad, backward,
                              retain_graph=True, allow_unused=True)


""" Nystrom Kernel Trace (2 methods) """


def nystrom_trace_fwd(kernel_args, M, X, kmn_z=None, use_ste=False):
    return nystrom_trace_frotrsm_fwd(kernel_args, M, X)
    if use_ste:
        if kmn_z is None:
            raise RuntimeError(
                "Cannot calculate nystrom-kernel trace with STE if kmn_z is not specified.")
        return nystrom_trace_hutch_fwd(kernel_args, M, kmn_z=kmn_z, n=X.shape[0])
    else:
        return nystrom_trace_trinv_fwd(kernel_args, M, X)


def nystrom_trace_bwd(ctx: Dict[str, torch.Tensor], use_ste=False):
    return nystrom_trace_frotrsm_bwd(ctx)
    if use_ste:
        return nystrom_trace_hutch_bwd(**ctx)
    return nystrom_trace_trinv_bwd(**ctx)


def nystrom_trace_hutch_fwd(kernel_args, M, kmn_z, n):
    mm_eye = torch.eye(M.shape[0], device=M.device, dtype=M.dtype) * EPS
    with torch.autograd.enable_grad():
        kmm = full_rbf_kernel(M, M, kernel_args)
        kmm_chol, info = torch.linalg.cholesky_ex(kmm + mm_eye, check_errors=True)

    with torch.autograd.no_grad():
        l_solve_1 = torch.triangular_solve(kmn_z, kmm_chol, upper=False,
                                           transpose=False).solution  # m * t
        l_solve_2 = torch.triangular_solve(l_solve_1, kmm_chol, upper=False,
                                           transpose=True).solution.contiguous()  # m * t
        output = torch.square(l_solve_1).sum(0).mean()
        output = 1.0 - output / n

    ctx = dict(kmn_z=kmn_z, kmm_chol=kmm_chol, l_solve_1=l_solve_1, l_solve_2=l_solve_2, n=n)

    return output, ctx


def nystrom_trace_hutch_bwd(kmn_z, kmm_chol, l_solve_1, l_solve_2, n):
    with torch.autograd.enable_grad():
        bg = - (
                2 * (kmn_z * l_solve_2).sum(0).mean()
                - 2 * (l_solve_2 * (kmm_chol @ l_solve_1)).sum(0).mean()
        ) / n
    return bg


def nystrom_trace_frotrsm_fwd(kernel_args, M, X):
    opt = FalkonOptions(no_single_kernel=False, use_cpu=not torch.cuda.is_available())
    if X.is_cuda:
        from falkon.mmv_ops.utils import _get_gpu_info
        gpu_info = _get_gpu_info(opt, slack=0.9)
        single_gpu_info = [g for g in gpu_info if g.Id == X.device.index][0]
        avail_mem = single_gpu_info.usable_memory / sizeof_dtype(X.dtype)
        device = torch.device("cuda:%d" % (single_gpu_info.Id))
    elif not opt.use_cpu:
        from falkon.mmv_ops.utils import _get_gpu_info
        gpu_info = _get_gpu_info(opt, slack=0.9)[0]  # TODO: Splitting across gpus
        avail_mem = gpu_info.usable_memory / sizeof_dtype(X.dtype)
        device = torch.device("cuda:%d" % (gpu_info.Id))
    else:
        avail_mem = opt.max_cpu_mem / sizeof_dtype(X.dtype)
        device = torch.device("cpu")

    coef_nm = 10
    blk_n = select_dim_over_n(max_n=X.shape[0], m=M.shape[0], d=X.shape[1], max_mem=avail_mem,
                              coef_nm=coef_nm, coef_nd=1, coef_md=1, coef_n=0,
                              coef_m=0, coef_d=0, rest=0)
    M_dev = M.to(device).requires_grad_()
    kernel_args_dev = kernel_args.to(device).requires_grad_()

    mm_eye = torch.eye(M_dev.shape[0], device=M_dev.device, dtype=M_dev.dtype) * EPS
    with torch.autograd.enable_grad():
        kmm = full_rbf_kernel(M_dev, M_dev, kernel_args_dev)
        kmm_chol, info = torch.linalg.cholesky_ex(kmm + mm_eye, check_errors=True)

    grad_wrt = [arg for arg in [kernel_args_dev, M_dev] if arg.requires_grad]
    fwd = torch.tensor(0.0, dtype=X.dtype, device=X.device)
    # print(f"Starting trace calc. blk_n={blk_n}")
    # print(torch.cuda.memory_summary())
    grads = None
    with ExitStack() as stack:
        for i in range(0, X.shape[0], blk_n):
            leni = min(blk_n, X.shape[0] - i)
            c_X = X[i: i + leni, :].to(device)
            with torch.autograd.enable_grad():
                k_mn = full_rbf_kernel(M_dev, c_X, kernel_args_dev)

            # Forward
            with torch.autograd.no_grad():
                solve1 = trsm(k_mn, kmm_chol, 1.0, lower=True, transpose=False)
                solve2 = trsm(solve1, kmm_chol, 1.0, lower=True, transpose=True)
                fwd += solve1.square().sum().to(X.device)  # TODO: make inplace?

            with torch.autograd.enable_grad():
                bwd = 2 * (k_mn.mul(solve2)).sum().to(X.device)  # .sum(0).mean()
                bwd -= 2 * ((kmm_chol @ solve1).mul(solve2)).sum().to(X.device)  # .sum(0).mean()
                bwd = (- bwd)
            new_grads = torch.autograd.grad(bwd, grad_wrt, retain_graph=True, allow_unused=False)
            if grads is None:
                grads = [g.to(device=X.device) for g in new_grads]
            else:
                for gi in range(len(grads)):
                    grads[gi] += new_grads[gi].to(X.device)
            # print(f"Iteration {i}")
            # print(torch.cuda.memory_summary())
    with torch.autograd.no_grad():
        fwd = (X.shape[0] - fwd)
    return fwd, grads


def nystrom_trace_frotrsm_bwd(bwd):
    return bwd


def nystrom_trace_trinv_fwd(kernel_args, M, X):
    diff_kernel = GaussianKernel(kernel_args)
    mm_eye = torch.eye(M.shape[0], device=M.device, dtype=M.dtype) * EPS
    with torch.autograd.enable_grad():
        kmm = full_rbf_kernel(M, M, kernel_args)
        kmm_chol, info = torch.linalg.cholesky_ex(kmm + mm_eye, check_errors=True)
    with torch.autograd.no_grad():
        linv = tri_inverse(kmm_chol, lower=1)
    with torch.autograd.enable_grad():
        k_linv = diff_kernel.mmv(X, M, linv.T)  # n * m (problematic)

    tr_fwd = k_linv.square().sum()
    tr_fwd = 1.0 - tr_fwd / X.shape[0]
    return tr_fwd, dict(k_linv=k_linv, kmm_chol=kmm_chol, linv=linv)


def nystrom_trace_trinv_bwd(linv, k_linv, kmm_chol):
    with torch.autograd.no_grad():
        k_linv_linv = k_linv @ linv  # n * m
    with torch.autograd.enable_grad():
        bg = - (
                2 * (k_linv * k_linv.detach()).sum() -
                2 * (k_linv.detach() @ kmm_chol.T * k_linv_linv).sum()
        ) / k_linv.shape[0]
    return bg


""" Nystrom Effective Dimension """


def nystrom_deff_fwd(kernel_args, penalty, M, X, Y, solve_opt, solve_maxiter,
                     last_solve_zy: Optional[torch.Tensor], data: NoRegLossAndDeffCtx):
    diff_kernel = DiffGaussianKernel(kernel_args, opt=solve_opt)

    ZY = torch.cat((data.z, Y), dim=1)
    if data.kmn_zy is None:
        with torch.autograd.enable_grad():
            data.kmn_zy = diff_kernel.mmv(M, X, ZY)
    if data.solve_zy is None:
        with torch.autograd.no_grad():
            data.solve_zy, data.solve_zy_prec, _ = solve_falkon(
                X, M, penalty, ZY, kernel_args, solve_opt, solve_maxiter, init_sol=last_solve_zy)

    with torch.autograd.no_grad():
        d_eff = (data.kmn_z * data.solve_z).sum(0).mean()

    return d_eff, data


def nystrom_deff_bwd(kernel_args, penalty, M, X, data):
    diff_kernel = DiffGaussianKernel(kernel_args)
    with torch.autograd.enable_grad():
        if data.knm_solve_zy is None:
            data.knm_solve_zy = diff_kernel.mmv(X, M,
                                                data.solve_zy)  # k_nm @ alpha  and  k_nm @ eta
        if data.kmm_solve_zy is None:
            data.kmm_solve_zy = diff_kernel.mmv(M, M, data.solve_zy)

        pen_n = penalty * X.shape[0]
        # Effective dimension
        deff_bg = (
                2 * (data.kmn_z * data.solve_z).sum(0).mean()
                - (data.knm_solve_z.square().sum(0).mean() +
                   pen_n * (data.solve_z * data.kmm_solve_z).sum(0).mean())
        )
    return deff_bg, data


""" Nystrom Datafit (no penalty) """


def datafit_fwd(kernel_args, penalty, M, X, Y, solve_opt, solve_maxiter,
                last_solve_zy: Optional[torch.Tensor],
                last_solve_ytilde: Optional[torch.Tensor],
                data: NoRegLossAndDeffCtx):
    diff_kernel = GaussianKernel(kernel_args, opt=solve_opt)

    ZY = torch.cat((data.z, Y), dim=1)
    if data.kmn_zy is None:
        with torch.autograd.enable_grad():
            data.kmn_zy = diff_kernel.mmv(M, X, ZY)
    if data.solve_zy is None:
        with torch.autograd.no_grad():
            # Solve Falkon part 1
            data.solve_zy, data.solve_zy_prec, _ = solve_falkon(
                X, M, penalty, ZY, kernel_args, solve_opt, solve_maxiter, init_sol=last_solve_zy)
    if data.knm_solve_zy is None:
        with torch.autograd.enable_grad():
            # Note that knm @ alpha = y_tilde. This is handled by the data-class, hence we need to run this in fwd.
            data.knm_solve_zy = diff_kernel.mmv(X, M, data.solve_zy)
    if data.solve_ytilde is None:
        with torch.autograd.no_grad():
            # Solve Falkon part 2 (alpha_tilde = H^{-1} @ k_nm.T @ y_tilde
            data.solve_ytilde, data.solve_ytilde_prec, _ = solve_falkon(
                X, M, penalty, data.y_tilde, kernel_args, solve_opt, solve_maxiter,
                init_sol=last_solve_ytilde)

    with torch.autograd.no_grad():
        # Loss = Y.T @ Y - 2 Y.T @ KNM @ alpha + alpha.T @ KNM.T @ KNM @ alpha
        loss = Y.square().sum()
        loss -= 2 * (data.kmn_y * data.solve_y).sum(0).mean()
        loss += data.y_tilde.square().sum(0).mean()
        loss /= X.shape[0]

    return loss, data


def datafit_bwd(kernel_args, penalty, M, X, data):
    diff_kernel = GaussianKernel(kernel_args)
    with torch.autograd.enable_grad():
        pen_n = penalty * X.shape[0]
        if data.kmm_solve_y is None:
            data.kmm_solve_zy = diff_kernel.mmv(M, M, data.solve_zy)

        # Loss without regularization
        loss_bg = (
                          -4 * (data.kmn_y * data.solve_y).sum(
                      0).mean()  # -4 * Y.T @ g(k_nm) @ alpha
                          + 2 * (data.knm_solve_y.square().sum(0).mean() +
                                 pen_n * (data.kmm_solve_y * data.solve_y).sum(
                              0).mean())  # 2 * alpha.T @ g(H) @ alpha
                          + 2 * (data.kmn_y * data.solve_ytilde).sum(
                      0).mean()  # 2 * Y.T @ g(k_nm) @ alpha_tilde
                          + 2 * (data.y_tilde * data.y_tilde.detach()).sum(
                      0).mean()  # 2 * alpha.T @ g(k_nm.T) @ y_tilde
                          - 2 * ((data.knm_solve_y * diff_kernel.mmv(X, M, data.solve_ytilde)).sum(
                      0) +
                                 pen_n * (data.kmm_solve_y * data.solve_ytilde).sum(0)).mean()
                      # -2 alpha @ g(H) @ alpha
                  ) / X.shape[0]
    return loss_bg, data


""" Nystrom Datafit (with penalty) """


def penalized_datafit_fwd(kernel_args, penalty, M, X, Y, solve_opt, solve_maxiter,
                          data: NoRegLossAndDeffCtx):
    diff_kernel = GaussianKernel(kernel_args, opt=solve_opt)
    if data.solve_y is None:
        with torch.autograd.no_grad():
            data.solve_y, _, _ = solve_falkon(X, M, penalty, Y, kernel_args, solve_opt, solve_maxiter)
    if data.kmn_y is None:
        with torch.autograd.enable_grad():
            data.kmn_y = diff_kernel.mmv(M, X, Y)

    with torch.autograd.no_grad():
        loss = Y.square().sum()
        loss -= (data.kmn_y * data.solve_y).sum(0).mean()

    return loss, data


def penalized_datafit_bwd(kernel_args, penalty, M, X, data: NoRegLossAndDeffCtx):
    diff_kernel = GaussianKernel(kernel_args)
    with torch.autograd.enable_grad():
        pen_n = penalty * X.shape[0]
        if data.kmm_solve_y is None:
            kmm_solve_y = diff_kernel.mmv(M, M, data.solve_y)
        else:
            kmm_solve_y = data.kmm_solve_y
        if data.knm_solve_y is None:
            knm_solve_y = diff_kernel.mmv(X, M, data.solve_y)
        else:
            knm_solve_y = data.knm_solve_y

        loss_bg = (
                -2 * (data.kmn_y * data.solve_y).sum(0).mean()
                + (knm_solve_y.square().sum(0).mean() +
                   pen_n * (kmm_solve_y * data.solve_y).sum(0).mean())
        )
    return loss_bg, data


def creg_plainfit(kernel_args, penalty, centers, X, Y, num_estimators, deterministic, solve_options,
                  solve_maxiter, gaussian_random, use_stoch_trace, warm_start: bool = True):
    return NoRegLossAndDeff.apply(
        kernel_args, penalty, centers, X, Y, num_estimators, deterministic, solve_options,
        solve_maxiter, gaussian_random, use_stoch_trace, warm_start
    )


# noinspection PyMethodOverriding
class NoRegLossAndDeff(torch.autograd.Function):
    NUM_DIFF_ARGS = 3
    _last_solve_zy = None
    _last_solve_ytilde = None
    last_alpha = None
    _last_t = None

    t_deff_fwd = []
    t_fit_fwd = []
    t_tr_fwd = []
    t_deff_bwd = []
    t_fit_bwd = []
    t_tr_bwd = []
    t_grad = []

    @staticmethod
    def forward(
            ctx,
            kernel_args: torch.Tensor,
            penalty: torch.Tensor,
            M: torch.Tensor,
            X: torch.Tensor,
            Y: torch.Tensor,
            t: int,
            deterministic: bool,
            solve_options: FalkonOptions,
            solve_maxiter: int,
            gaussian_random: bool,
            use_stoch_trace: bool,
            warm_start: bool,
    ):
        if NoRegLossAndDeff._last_t is not None and NoRegLossAndDeff._last_t != t:
            NoRegLossAndDeff._last_solve_zy = None
            NoRegLossAndDeff._last_solve_ytilde = None
            NoRegLossAndDeff.last_alpha = None
        NoRegLossAndDeff._last_t = t

        data = NoRegLossAndDeffCtx(t)
        if deterministic:
            torch.manual_seed(12)
        data.z = init_random_vecs(X.shape[0], t, dtype=X.dtype, device=X.device,
                                  gaussian_random=gaussian_random)

        t_s = time.time()
        d_eff, data = nystrom_deff_fwd(
            kernel_args=kernel_args, penalty=penalty, M=M, X=X, Y=Y,
            solve_opt=solve_options, solve_maxiter=solve_maxiter,
            last_solve_zy=NoRegLossAndDeff._last_solve_zy, data=data)
        NoRegLossAndDeff.t_deff_fwd.append(time.time() - t_s)
        t_s = time.time()
        datafit, data = datafit_fwd(
            kernel_args=kernel_args, penalty=penalty, M=M, X=X, Y=Y,
            solve_maxiter=solve_maxiter, solve_opt=solve_options,
            last_solve_zy=NoRegLossAndDeff._last_solve_zy,
            last_solve_ytilde=NoRegLossAndDeff._last_solve_ytilde,
            data=data)
        NoRegLossAndDeff.t_fit_fwd.append(time.time() - t_s)
        t_s = time.time()
        trace, tr_ctx = nystrom_trace_frotrsm_fwd(kernel_args, M, X)
        ## This is the `old` version
        # with torch.autograd.enable_grad():
        #     kmn_z = data.kmn_z  # Need to diff through the slicing
        # trace2, tr_ctx2 = nystrom_trace_fwd(
        #    kernel_args=kernel_args, M=M, X=X, kmn_z=kmn_z, use_ste=use_stoch_trace)
        # print(f"Trace new: {trace:.6f}, old {trace2:.6f}", flush=True)
        ## Uncomment when running in NO-TRACE mode
        # tr_ctx = None
        NoRegLossAndDeff.t_tr_fwd.append(time.time() - t_s)

        if warm_start:
            NoRegLossAndDeff._last_solve_zy = data.solve_zy_prec.detach()
            NoRegLossAndDeff._last_solve_ytilde = data.solve_ytilde_prec.detach()
        NoRegLossAndDeff.last_alpha = data.solve_y.detach()
        ctx.save_for_backward(kernel_args, penalty, M)
        ctx.data, ctx.tr_ctx, ctx.X, ctx.use_stoch_trace = data, tr_ctx, X, use_stoch_trace
        ctx.deff = d_eff.detach()
        ctx.dfit = datafit.detach()
        ctx.trace = trace.detach()
        print(
            f"stochastic creg-no-pen-fit - D-eff={d_eff:5.3e} DFit={datafit:5.3e} Trace={trace:5.3e}")
        return d_eff + datafit + trace

    @staticmethod
    def backward(ctx, out):
        kernel_args, penalty, M = ctx.saved_tensors
        data = ctx.data

        t_s = time.time()
        deff_bwd, data = nystrom_deff_bwd(kernel_args=kernel_args, penalty=penalty, M=M, X=ctx.X,
                                          data=data)
        NoRegLossAndDeff.t_deff_bwd.append(time.time() - t_s)
        t_s = time.time()
        dfit_bwd, data = datafit_bwd(kernel_args=kernel_args, penalty=penalty, M=M, X=ctx.X,
                                     data=data)
        NoRegLossAndDeff.t_fit_bwd.append(time.time() - t_s)
        t_s = time.time()
        tr_bwd = nystrom_trace_frotrsm_bwd(ctx.tr_ctx)
        ## `old` version
        # tr_bwd = nystrom_trace_bwd(ctx.tr_ctx, use_ste=ctx.use_stoch_trace)
        NoRegLossAndDeff.t_tr_bwd.append(time.time() - t_s)

        t_s = time.time()
        grads_deff = list(calc_grads(ctx, deff_bwd, NoRegLossAndDeff.NUM_DIFF_ARGS))
        grads_dfit = list(calc_grads(ctx, dfit_bwd, NoRegLossAndDeff.NUM_DIFF_ARGS))
        grads_trace = list(calc_grads(ctx, tr_bwd, NoRegLossAndDeff.NUM_DIFF_ARGS))

        # print(f"Grads before everything: deff={grads_deff[1]:.2e} dfit={grads_dfit[1]:.2e} ")

        pen_n = penalty * ctx.X.shape[0]
        # if grads_deff[1] is not None:
        #    grads_deff[1] *= penalty#(grads_deff[1] / ctx.X.shape[0] - ctx.deff)
        # if ctx.needs_input_grad[1]:
        #    grads_trace[1] = -(ctx.trace / penalty)
        # if grads_dfit[1] is not None:
        #    grads_dfit[1] *= penalty
        # if grads_deff[0] is not None:
        #    grads_deff[0] /= pen_n
        # if grads_trace[0] is not None:
        #    grads_trace[0] /= pen_n
        # if grads_deff[2] is not None:
        #    grads_deff[2] /= pen_n
        # if grads_trace[2] is not None:
        #    grads_trace[2] /= pen_n
        NoRegLossAndDeff.t_grad.append(time.time() - t_s)

        # print(f"Grads after division: deff={grads_deff[1]:.2e} dfit={grads_dfit[1]:.2e} trace={grads_trace[1]:.2e}")
        # print(f"Lambda grads: deff={grads_deff[1]:.2e} - dfit={grads_dfit[1]:.2e} - trace={grads_trace[1]:.2e}")
        grads = []
        for i in range(len(grads_deff)):
            grad, any_not_none = 0.0, False
            for g in (grads_deff[i], grads_dfit[i], grads_trace[i]):
                if g is not None:
                    grad += g
                    any_not_none = True
            grads.append(grad * out if any_not_none else None)
        return tuple(grads)

    @staticmethod
    def grad_check():
        torch.manual_seed(4)
        X = torch.randn(50, 6, dtype=torch.float64)
        w = torch.randn(X.shape[1], 1, dtype=torch.float64)
        Y = X @ w
        M = X[:10].clone().detach().requires_grad_()
        s = torch.tensor([10.0], dtype=X.dtype).requires_grad_()
        p = torch.tensor(1e-2, dtype=X.dtype).requires_grad_()

        torch.autograd.gradcheck(
            lambda sigma, pen, centers:
            NoRegLossAndDeff.apply(sigma, pen, centers, X, Y, 20, True, FalkonOptions(), 30, False,
                                   False, False),
            (s, p, M))


def gcv(kernel_args, penalty, centers, X, Y, num_estimators, deterministic, solve_options,
        solve_maxiter, gaussian_random, warm_start: bool = True):
    return GCV.apply(
        kernel_args, penalty, centers, X, Y, num_estimators, deterministic, solve_options,
        solve_maxiter, gaussian_random, warm_start
    )


# noinspection PyMethodOverriding
class GCV(torch.autograd.Function):
    """
    Numerator: Exactly the data-fit term of NoRegLossAndDeff
    Denominator: Similar to Nystrom Effective Dim.
    """

    NUM_DIFF_ARGS = 3
    _last_solve_zy = None
    _last_solve_ytilde = None
    last_alpha = None
    _last_t = None

    @staticmethod
    def forward(ctx,
                kernel_args: torch.Tensor,
                penalty: torch.Tensor,
                M: torch.Tensor,
                X: torch.Tensor,
                Y: torch.Tensor,
                t: int,
                deterministic: bool,
                solve_options: FalkonOptions,
                solve_maxiter: int,
                gaussian_random: bool,
                warm_start: bool, ):
        if GCV._last_t is not None and GCV._last_t != t:
            GCV._last_solve_zy = None
            GCV._last_solve_ytilde = None
            GCV.last_alpha = None
        GCV._last_t = t
        data = NoRegLossAndDeffCtx(t)
        if deterministic:
            torch.manual_seed(12)
        data.z = init_random_vecs(X.shape[0], t, dtype=X.dtype, device=X.device,
                                  gaussian_random=gaussian_random)

        d_eff, data = nystrom_deff_fwd(
            kernel_args=kernel_args, penalty=penalty, M=M, X=X, Y=Y,
            solve_opt=solve_options, solve_maxiter=solve_maxiter,
            last_solve_zy=GCV._last_solve_zy, data=data)
        datafit, data = datafit_fwd(
            kernel_args=kernel_args, penalty=penalty, M=M, X=X, Y=Y,
            solve_maxiter=solve_maxiter, solve_opt=solve_options,
            last_solve_zy=GCV._last_solve_zy, last_solve_ytilde=GCV._last_solve_ytilde,
            data=data)

        if warm_start:
            GCV._last_solve_zy = data.solve_zy_prec.detach()
            GCV._last_solve_ytilde = data.solve_ytilde_prec.detach()
        GCV.last_alpha = data.solve_y.detach()
        ctx.save_for_backward(kernel_args, penalty, M)
        ctx.data = data
        ctx.X = X
        ctx.d_eff = d_eff
        ctx.datafit = datafit
        return datafit / torch.square((1.0 - d_eff / X.shape[0]))

    @staticmethod
    def backward(ctx, out):
        kernel_args, penalty, M = ctx.saved_tensors
        data = ctx.data
        n = ctx.X.shape[0]

        denominator = torch.square((1.0 - ctx.d_eff / n))

        deff_bwd, data = nystrom_deff_bwd(kernel_args=kernel_args, penalty=penalty, M=M, X=ctx.X,
                                          data=data)
        dfit_bwd, data = datafit_bwd(kernel_args=kernel_args, penalty=penalty, M=M, X=ctx.X,
                                     data=data)
        with torch.autograd.enable_grad():
            bg = out * (dfit_bwd * denominator - ctx.datafit * (
                    - 2 / n + 2 * ctx.d_eff / n ** 2) * deff_bwd) / torch.square(denominator)
        return calc_grads(ctx, bg, GCV.NUM_DIFF_ARGS)

    @staticmethod
    def grad_check():
        torch.manual_seed(3)
        X = torch.randn(50, 6, dtype=torch.float64)
        w = torch.randn(X.shape[1], 1, dtype=torch.float64)
        Y = X @ w
        M = X[:10].clone().detach().requires_grad_()
        s = torch.tensor([10.0], dtype=X.dtype).requires_grad_()
        p = torch.tensor(1e-2, dtype=X.dtype).requires_grad_()

        torch.autograd.gradcheck(
            lambda sigma, pen, centers:
            GCV.apply(sigma, pen, centers, X, Y, 20, True, FalkonOptions(), 30, False, False),
            (s, p, M))


def creg_penfit(kernel_args, penalty, centers, X, Y, num_estimators, deterministic, solve_options,
                solve_maxiter, gaussian_random, use_stoch_trace, warm_start=True):
    return RegLossAndDeffv2.apply(
        kernel_args, penalty, centers, X, Y, num_estimators, deterministic, solve_options,
        solve_maxiter, gaussian_random, use_stoch_trace, warm_start
    )


# noinspection PyMethodOverriding
class RegLossAndDeffv2(torch.autograd.Function):
    _last_solve_z = None
    _last_solve_y = None
    _last_solve_zy = None
    last_alpha = None
    _last_t = None
    iter_prep_times, fwd_times, bwd_times, solve_times, kmm_times, grad_times = [], [], [], [], [], []
    iter_times, num_flk_iters = [], []
    solve_together = False
    use_direct_for_stoch = False
    print(f"Initialized class RegLossAndDeffv2. solve_together={solve_together}, "
          f"use_direct_for_stoch={use_direct_for_stoch}")


    @staticmethod
    def print_times():
        num_times = len(RegLossAndDeffv2.iter_times)
        print(
            f"Timings: Preparation {np.sum(RegLossAndDeffv2.iter_prep_times) / num_times:.2f} "
            f"Falkon solve {np.sum(RegLossAndDeffv2.solve_times) / num_times:.2f} "
            f"(in {np.sum(RegLossAndDeffv2.num_flk_iters) / num_times:.1f} iters) "
            f"KMM (toCUDA) {np.sum(RegLossAndDeffv2.kmm_times) / num_times:.2f} "
            f"Forward {np.sum(RegLossAndDeffv2.fwd_times) / num_times:.2f} "
            f"Backward {np.sum(RegLossAndDeffv2.bwd_times) / num_times:.2f} "
            f"Grad {np.sum(RegLossAndDeffv2.grad_times) / num_times:.2f} "
            f"\n\tTotal {np.sum(RegLossAndDeffv2.iter_times) / num_times:.2f}"
        )
        (RegLossAndDeffv2.iter_prep_times, RegLossAndDeffv2.fwd_times, RegLossAndDeffv2.bwd_times,
         RegLossAndDeffv2.solve_times, RegLossAndDeffv2.kmm_times, RegLossAndDeffv2.grad_times,
         RegLossAndDeffv2.iter_times, RegLossAndDeffv2.num_flk_iters) = [], [], [], [], [], [], [], []

    @staticmethod
    def trace_bwd(k_mn, k_mn_zy, solve2, kmm, use_stoch_trace, t):
        if use_stoch_trace:
            if k_mn_zy is None or t is None or t <= 0:
                raise ValueError("Using stochastic trace but k_mn_zy is None.")
        else:
            if k_mn is None:
                raise ValueError("Not using stochastic trace but k_mn is None.")
        if use_stoch_trace:
            return -(
                    2 * (k_mn_zy[:, :t].mul(solve2)).sum(0).mean() -
                    (solve2 * (kmm @ solve2)).sum(0).mean()
            )
        else:
            return -(
                    2 * (k_mn.mul(solve2)).sum() -
                    (solve2 * (kmm @ solve2)).sum()
            )

    @staticmethod
    def deff_bwd(zy_knm_solve_zy, zy_solve_knm_knm_solve_zy, zy_solve_kmm_solve_zy, pen_n, t,
                 include_kmm_term):
        deff_bwd = (
                2 * zy_knm_solve_zy[:t].mean() -
                zy_solve_knm_knm_solve_zy[:t].mean()
        )
        if include_kmm_term:
            deff_bwd -= pen_n * zy_solve_kmm_solve_zy[:t].mean()
        return deff_bwd

    @staticmethod
    def dfit_bwd(zy_knm_solve_zy, zy_solve_knm_knm_solve_zy, zy_solve_kmm_solve_zy, pen_n, t,
                 include_kmm_term):
        dfit_bwd = -(
                2 * zy_knm_solve_zy[t:].mean() -
                zy_solve_knm_knm_solve_zy[t:].mean()
        )
        if include_kmm_term:
            dfit_bwd += pen_n * zy_solve_kmm_solve_zy[t:].mean()
        return dfit_bwd

    @staticmethod
    def trace_fwd(trace_fwd, k_mn, k_mn_zy, kmm_chol, use_stoch_trace, t):
        """ Nystrom kernel trace forward """
        if use_stoch_trace:
            solve1 = torch.triangular_solve(k_mn_zy[:, :t], kmm_chol, upper=False,
                                            transpose=False).solution  # m * t
            solve2 = torch.triangular_solve(solve1, kmm_chol, upper=False,
                                            transpose=True).solution.contiguous()  # m * t
            trace_fwd -= solve1.square_().sum(0).mean()
        else:
            solve1 = trsm(k_mn, kmm_chol, 1.0, lower=True, transpose=False)  # (M*N)
            solve2 = trsm(solve1, kmm_chol, 1.0, lower=True, transpose=True)  # (M*N)
            trace_fwd -= solve1.square_().sum()
        return trace_fwd, solve2

    @staticmethod
    def direct_nosplit(X, M, Y, penalty, kmm, kmm_chol, zy, solve_zy, zy_solve_kmm_solve_zy, kernel,
                       t):
        with Timer(RegLossAndDeffv2.iter_prep_times), torch.autograd.enable_grad():
            k_mn_zy = kernel.mmv(M, X, zy)
            zy_knm_solve_zy = k_mn_zy.mul(solve_zy).sum(0)  # T+1

        # Forward
        dfit_fwd = Y.square().sum()
        deff_fwd = torch.tensor(0, dtype=X.dtype)
        _trace_fwd = torch.tensor(X.shape[0], dtype=X.dtype)
        with Timer(RegLossAndDeffv2.fwd_times), torch.autograd.no_grad():
            _trace_fwd, solve2 = RegLossAndDeffv2.trace_fwd(
                _trace_fwd, k_mn=None, k_mn_zy=k_mn_zy, kmm_chol=kmm_chol, use_stoch_trace=True,
                t=t)
            # Nystrom effective dimension forward
            deff_fwd += zy_knm_solve_zy[:t].mean()
            # Data-fit forward
            dfit_fwd -= zy_knm_solve_zy[t:].mean()

        # Backward
        with Timer(RegLossAndDeffv2.bwd_times), torch.autograd.enable_grad():
            zy_solve_knm_knm_solve_zy = kernel.mmv(X, M, solve_zy).square().sum(0)  # T+1
            pen_n = penalty * X.shape[0]
            # Nystrom kernel trace backward
            trace_bwd = RegLossAndDeffv2.trace_bwd(
                k_mn=None, k_mn_zy=k_mn_zy, solve2=solve2, kmm=kmm, use_stoch_trace=True, t=t)
            # Nystrom effective dimension backward
            deff_bwd = RegLossAndDeffv2.deff_bwd(
                zy_knm_solve_zy, zy_solve_knm_knm_solve_zy, zy_solve_kmm_solve_zy, pen_n, t,
                include_kmm_term=True)
            # Data-fit backward
            dfit_bwd = RegLossAndDeffv2.dfit_bwd(
                zy_knm_solve_zy, zy_solve_knm_knm_solve_zy, zy_solve_kmm_solve_zy, pen_n, t,
                include_kmm_term=True)
            bwd = deff_bwd + dfit_bwd + trace_bwd
        return (deff_fwd, dfit_fwd, _trace_fwd), bwd

    @staticmethod
    def choose_device_mem(data_dev: torch.device, dtype: torch.dtype,
                          solve_options: FalkonOptions) -> Tuple[torch.device, float]:
        if data_dev.type == 'cuda':  # CUDA in-core
            from falkon.mmv_ops.utils import _get_gpu_info
            gpu_info = _get_gpu_info(solve_options, slack=0.9)
            single_gpu_info = [g for g in gpu_info if g.Id == data_dev.index][0]
            avail_mem = single_gpu_info.usable_memory / sizeof_dtype(dtype)
            device = torch.device("cuda:%d" % (single_gpu_info.Id))
        elif not solve_options.use_cpu and torch.cuda.is_available():  # CUDA out-of-core
            from falkon.mmv_ops.utils import _get_gpu_info
            gpu_info = _get_gpu_info(solve_options, slack=0.9)[0]  # TODO: Splitting across gpus
            avail_mem = gpu_info.usable_memory / sizeof_dtype(dtype)
            device = torch.device("cuda:%d" % (gpu_info.Id))
        else:  # CPU in-core
            avail_mem = solve_options.max_cpu_mem / sizeof_dtype(dtype)
            device = torch.device("cpu")

        return device, avail_mem

    @staticmethod
    def solve_flk(X, M, Y, Z, ZY, penalty, kernel_args, solve_options, solve_maxiter, warm_start):
        t = Z.shape[1]
        solve_together = RegLossAndDeffv2.solve_together
        solve_opt_precise = solve_options
        solve_maxiter_precise = solve_maxiter

        kernel_args_ = kernel_args.detach()
        penalty_ = penalty.item()
        M_ = M.detach()

        #solve_opt_precise = dataclasses.replace(solve_opt_precise, keops_active="no")
        K = GaussianKernel(kernel_args_, opt=solve_opt_precise)
        precond = FalkonPreconditioner(penalty_, K, solve_opt_precise)
        precond.init(M_)

        if solve_together:
            optim = FalkonConjugateGradient(K, precond, solve_opt_precise)
            solve_zy_prec = optim.solve(
                X, M_, ZY, penalty_,
                initial_solution=RegLossAndDeffv2._last_solve_zy,
                max_iter=solve_maxiter,
            )
            solve_zy = precond.apply(solve_zy_prec)
            if warm_start:
                RegLossAndDeffv2._last_solve_zy = solve_zy_prec.detach().clone()
            RegLossAndDeffv2.last_alpha = solve_zy[:, t:].detach().clone()
            num_iters = optim.optimizer.num_iter
        else:
            optim_y = FalkonConjugateGradient(K, precond, solve_opt_precise)
            solve_y_prec = optim_y.solve(X, M_, Y, penalty_,
                                         initial_solution=RegLossAndDeffv2._last_solve_y,
                                         max_iter=solve_maxiter_precise)
            optim_z = FalkonConjugateGradient(K, precond, solve_opt_precise)
            solve_z_prec = optim_z.solve(X, M_, Z, penalty_,
                                         initial_solution=RegLossAndDeffv2._last_solve_z,
                                         max_iter=solve_maxiter_precise)
            solve_z = precond.apply(solve_z_prec)
            solve_y = precond.apply(solve_y_prec)
            solve_zy = torch.cat((solve_z, solve_y), dim=1)
            if warm_start:
                RegLossAndDeffv2._last_solve_y = solve_y_prec.detach().clone()
                RegLossAndDeffv2._last_solve_z = solve_z_prec.detach().clone()
            RegLossAndDeffv2.last_alpha = solve_y.detach().clone()
            num_iters = optim_z.optimizer.num_iter
        return solve_zy, num_iters

    @staticmethod
    def direct_wsplit(X, M, Y, penalty, kernel_args, kmm, kmm_chol, zy, solve_zy,
                      zy_solve_kmm_solve_zy, t, coef_nm, device, avail_mem, use_stoch_trace,
                      needs_input_grad):
        """ Splitting along the first dimension of X """
        # Decide block size (this is super random for now: if OOM increase `coef_nm`).
        blk_n = select_dim_over_n(max_n=X.shape[0], m=M.shape[0], d=X.shape[1], max_mem=avail_mem,
                                  coef_nm=coef_nm, coef_nd=1, coef_md=1, coef_n=0,
                                  coef_m=0, coef_d=0, rest=0)
        # Initialize forward pass elements.
        dfit_fwd = Y.square().sum().to(device)
        deff_fwd = torch.tensor(0, dtype=X.dtype, device=device)
        _trace_fwd = torch.tensor(X.shape[0], dtype=X.dtype, device=device)
        grads = None
        it = 0
        with ExitStack() as stack:
            if device.type == 'cuda':
                s1 = torch.cuda.current_stream(device)
                stack.enter_context(torch.cuda.device(device))
                stack.enter_context(torch.cuda.stream(s1))
            for i in range(0, X.shape[0], blk_n):
                it += 1
                leni = min(blk_n, X.shape[0] - i)
                c_X = X[i: i + leni, :].to(device=device, non_blocking=True)
                c_zy = zy[i: i + leni, :].to(device=device, non_blocking=True)
                with Timer(RegLossAndDeffv2.iter_prep_times), torch.autograd.enable_grad():
                    k_mn = full_rbf_kernel(c_X, M,
                                           kernel_args).T  # Done to get F-contig k_mn (faster trsm)
                    k_mn_zy = k_mn @ c_zy  # MxN * Nx(T+1) = Mx(T+1)
                    zy_knm_solve_zy = (k_mn_zy * solve_zy).sum(0)  # (T+1)

                # Forward
                with Timer(RegLossAndDeffv2.fwd_times), torch.autograd.no_grad():
                    # Nystrom kernel trace forward
                    _trace_fwd, solve2 = RegLossAndDeffv2.trace_fwd(
                        _trace_fwd, k_mn=k_mn, k_mn_zy=k_mn_zy, kmm_chol=kmm_chol,
                        use_stoch_trace=use_stoch_trace, t=t)
                    # Nystrom effective dimension forward
                    deff_fwd += zy_knm_solve_zy[:t].mean()
                    # Data-fit forward
                    dfit_fwd -= zy_knm_solve_zy[t:].mean()
                # Backward
                with Timer(RegLossAndDeffv2.bwd_times), torch.autograd.enable_grad():
                    zy_solve_knm_knm_solve_zy = (k_mn.T @ solve_zy).square().sum(0)  # (T+1)
                    pen_n = penalty * X.shape[0]
                    # Nystrom kernel trace backward
                    trace_bwd = RegLossAndDeffv2.trace_bwd(
                        k_mn=k_mn, k_mn_zy=k_mn_zy, solve2=solve2, kmm=kmm,
                        use_stoch_trace=use_stoch_trace, t=t)
                    # Nystrom effective dimension backward
                    deff_bwd = RegLossAndDeffv2.deff_bwd(
                        zy_knm_solve_zy, zy_solve_knm_knm_solve_zy, zy_solve_kmm_solve_zy, pen_n, t,
                        include_kmm_term=i == 0)
                    # Data-fit backward
                    dfit_bwd = RegLossAndDeffv2.dfit_bwd(
                        zy_knm_solve_zy, zy_solve_knm_knm_solve_zy, zy_solve_kmm_solve_zy, pen_n, t,
                        include_kmm_term=i == 0)
                    bwd = deff_bwd + dfit_bwd + trace_bwd

                # Calc grads
                with Timer(RegLossAndDeffv2.grad_times):
                    new_grads = calc_grads_tensors(inputs=(kernel_args, penalty, M),
                                                   inputs_need_grad=needs_input_grad, backward=bwd,
                                                   retain_graph=True, allow_unused=True)
                    if grads is None:
                        grads = []
                        for g in new_grads:
                            if g is not None:
                                grads.append(g.to(device=X.device))
                            else:
                                grads.append(None)
                    else:
                        for gi in range(len(grads)):
                            if (grads[gi] is None) != (new_grads[gi] is None):
                                continue  # This can happen since bwd at iter-0 is different from following iters.
                            if grads[gi] is not None:
                                grads[gi] += new_grads[gi].to(X.device)
        return (deff_fwd, dfit_fwd, _trace_fwd), grads

    @staticmethod
    def forward(
            ctx,
            kernel_args: torch.Tensor,
            penalty: torch.Tensor,
            M: torch.Tensor,
            X: torch.Tensor,
            Y: torch.Tensor,
            t: int,
            deterministic: bool,
            solve_options: FalkonOptions,
            solve_maxiter: int,
            gaussian_random: bool,
            use_stoch_trace: bool,
            warm_start: bool
    ):
        use_direct_for_stoch = RegLossAndDeffv2.use_direct_for_stoch
        if RegLossAndDeffv2._last_t is not None and RegLossAndDeffv2._last_t != t:
            RegLossAndDeffv2._last_solve_y = None
            RegLossAndDeffv2._last_solve_z = None
            RegLossAndDeffv2.last_alpha = None
        RegLossAndDeffv2._last_t = t
        if deterministic:
            torch.manual_seed(12)

        if use_stoch_trace and use_direct_for_stoch:
            device, avail_mem = X.device, None
        else:
            device, avail_mem = RegLossAndDeffv2.choose_device_mem(X.device, X.dtype, solve_options)
        coef_nm = 20

        with Timer(RegLossAndDeffv2.iter_times):
            # Initialize hutch trace estimation vectors (t of them)
            Z = init_random_vecs(X.shape[0], t, dtype=X.dtype, device=X.device,
                                 gaussian_random=gaussian_random)
            ZY = torch.cat((Z, Y), dim=1)
            M_dev = M.to(device, copy=False).requires_grad_(M.requires_grad)
            kernel_args_dev = kernel_args.to(device, copy=False).requires_grad_(kernel_args.requires_grad)
            penalty_dev = penalty.to(device, copy=False).requires_grad_(penalty.requires_grad)

            with Timer(RegLossAndDeffv2.solve_times):
                solve_zy, num_flk_iters = RegLossAndDeffv2.solve_flk(
                    X, M_dev, Y, Z, ZY, penalty_dev, kernel_args_dev, solve_options, solve_maxiter, warm_start)
                RegLossAndDeffv2.num_flk_iters.append(num_flk_iters)

            with Timer(RegLossAndDeffv2.kmm_times):  # Move small matrices to the computation device
                solve_zy_dev = solve_zy.to(device, copy=False)

                with torch.autograd.enable_grad():
                    kmm = full_rbf_kernel(M_dev, M_dev, kernel_args_dev)
                    zy_solve_kmm_solve_zy = (kmm @ solve_zy_dev * solve_zy_dev).sum(0)  # (T+1)
                    # The following should be identical but seems to introduce errors in the bwd pass.
                    # zy_solve_kmm_solve_zy = (kmm_chol.T @ solve_zy_dev).square().sum(0)  # (T+1)
                with torch.autograd.no_grad():
                    mm_eye = torch.eye(M_dev.shape[0], device=device, dtype=M_dev.dtype) * EPS
                    kmm_chol, info = torch.linalg.cholesky_ex(kmm + mm_eye, check_errors=False)

            if use_stoch_trace and use_direct_for_stoch:
                kernel = DiffGaussianKernel(kernel_args_dev, solve_options)
                fwd, bwd = RegLossAndDeffv2.direct_nosplit(X, M_dev, Y, penalty, kmm, kmm_chol, ZY,
                                                           solve_zy, zy_solve_kmm_solve_zy, kernel,
                                                           t)
                with Timer(RegLossAndDeffv2.grad_times):
                    grads = calc_grads_tensors(inputs=(kernel_args_dev, penalty_dev, M_dev),
                                               inputs_need_grad=ctx.needs_input_grad, backward=bwd,
                                               retain_graph=False, allow_unused=True)
            else:
                fwd, grads = RegLossAndDeffv2.direct_wsplit(X, M_dev, Y, penalty_dev, kernel_args_dev, kmm,
                                                            kmm_chol, ZY, solve_zy_dev,
                                                            zy_solve_kmm_solve_zy, t, coef_nm,
                                                            device, avail_mem, use_stoch_trace,
                                                            ctx.needs_input_grad)

        deff_fwd, dfit_fwd, trace_fwd = fwd
        ctx.grads = grads
        print(f"Stochastic: D-eff {deff_fwd:.3e} Data-Fit {dfit_fwd:.3e} Trace {trace_fwd:.3e}")
        return (deff_fwd + dfit_fwd + trace_fwd).to(X.device)

    @staticmethod
    def backward(ctx, out):
        grads_out = []
        for g in ctx.grads:
            if g is not None:
                g = g * out
            grads_out.append(g)
        return tuple(grads_out)

    @staticmethod
    def grad_check():
        torch.manual_seed(3)
        X = torch.randn(50, 6, dtype=torch.float64)
        w = torch.randn(X.shape[1], 1, dtype=torch.float64)
        Y = X @ w
        M = X[:10].clone().detach().requires_grad_()
        s = torch.tensor([10.0], dtype=X.dtype).requires_grad_()
        p = torch.tensor(1e-2, dtype=X.dtype).requires_grad_()

        torch.autograd.gradcheck(
            lambda sigma, pen, centers:
            RegLossAndDeffv2.apply(sigma, pen, centers, X, Y, 20, True, FalkonOptions(), 30, False,
                                   True, False),
            (s, p, M))
        torch.autograd.gradcheck(
            lambda sigma, pen, centers:
            RegLossAndDeffv2.apply(sigma, pen, centers, X, Y, 20, True, FalkonOptions(), 30, False,
                                   False, False),
            (s, p, M))


def validation_loss(kernel_args, penalty, centers, Xtr, Ytr, Xval, Yval, solve_options,
                    solve_maxiter, warm_start=True):
    return ValidationLoss.apply(kernel_args, penalty, centers, Xtr, Ytr, Xval, Yval,
                                solve_options, solve_maxiter, warm_start)


# noinspection PyMethodOverriding
class ValidationLoss(torch.autograd.Function):
    last_alpha = None
    _last_alpha_prec = None
    _last_alpha2_prec = None

    @staticmethod
    def forward(
            ctx,
            kernel_args: torch.Tensor,
            penalty: torch.Tensor,
            M: torch.Tensor,
            Xtr: torch.Tensor,
            Ytr: torch.Tensor,
            Xval: torch.Tensor,
            Yval: torch.Tensor,
            solve_options: FalkonOptions,
            solve_maxiter: int,
            warm_start: bool,
    ):
        with torch.autograd.no_grad():
            kernel = GaussianKernel(kernel_args, opt=solve_options)
            pc = FalkonPreconditioner(penalty.item(), kernel, opt=solve_options)
            pc.init(M.detach())
            optim = FalkonConjugateGradient(kernel, pc, opt=solve_options)
            beta1 = optim.solve(Xtr, M.detach(), Ytr, penalty.item(),
                                ValidationLoss._last_alpha_prec, solve_maxiter)
            alpha1 = pc.apply(beta1)

        with torch.autograd.enable_grad():
            kvm_alpha = kernel.mmv(Xval, M, alpha1)

        if warm_start:
            ValidationLoss._last_alpha_prec = beta1
        ValidationLoss.last_alpha = alpha1

        ctx.save_for_backward(kernel_args, penalty, M)
        ctx.alpha1, ctx.kvm_alpha = alpha1, kvm_alpha
        ctx.pc, ctx.solve_maxiter, ctx.kernel, ctx.optim = pc, solve_maxiter, kernel, optim
        ctx.Xtr, ctx.Xval, ctx.Ytr, ctx.Yval = Xtr, Xval, Ytr, Yval
        ctx.warm_start = warm_start

        with torch.autograd.no_grad():
            val_loss = torch.sum(torch.square(kvm_alpha - Yval))
        return val_loss

    @staticmethod
    def backward(ctx, out):
        kernel_args, penalty, M = ctx.saved_tensors
        alpha1, kvm_alpha = ctx.alpha1, ctx.kvm_alpha
        pc, solve_maxiter, kernel, optim = ctx.pc, ctx.solve_maxiter, ctx.kernel, ctx.optim
        Xtr, Xval, Ytr, Yval = ctx.Xtr, ctx.Xval, ctx.Ytr, ctx.Yval

        with torch.autograd.no_grad():
            # 2 right-hand-sides: kvm_alpha and y_val
            slv_shape = kvm_alpha.shape[1]
            solve2rhs = torch.cat((kvm_alpha.detach(), Yval), dim=1)

            beta_slv2 = optim.solve_val_rhs(Xtr, Xval, M.detach(), solve2rhs, penalty.item(),
                                            ValidationLoss._last_alpha2_prec, solve_maxiter)
            alpha_slv2 = pc.apply(beta_slv2)
            if ctx.warm_start:
                ValidationLoss._last_alpha2_prec = beta_slv2

            all_alphas = torch.cat((alpha1, alpha_slv2), dim=1)
            a2, a3 = torch.split(alpha_slv2, slv_shape, dim=1)

        with torch.autograd.enable_grad():
            kmn_y = kernel.mmv(M, Xtr, Ytr)
            kmv_yv = kernel.mmv(M, Xval, Yval)
            knm_slv_all = kernel.mmv(Xtr, M, all_alphas)
            knm_a1, knm_a2, knm_a3 = torch.split(knm_slv_all, slv_shape, dim=1)
            kmm_a1 = kernel.mmv(M, M, alpha1)

            pen_n = penalty * Xtr.shape[0]
            bg = out * (
                    + 2 * (kmn_y * a2).sum()
                    + 2 * (kvm_alpha.detach() * kvm_alpha).sum()
                    - 2 * ((knm_a1 * knm_a2.detach()).sum() + (
                    knm_a1.detach() * knm_a2).sum() + pen_n * (kmm_a1 * a2).sum())
                    - 2 * (kmn_y * a3).sum()
                    - 2 * (alpha1 * kmv_yv).sum()
                    + 2 * ((knm_a1 * knm_a3.detach()).sum() + (
                    knm_a1.detach() * knm_a3).sum() + pen_n * (kmm_a1 * a3).sum())
            )
        return calc_grads(ctx, bg, 3)

    @staticmethod
    def grad_check():
        torch.manual_seed(3)
        X = torch.randn(100, 6, dtype=torch.float64)
        Xtr = X[:50].clone()
        Xval = X[50:].clone()
        w = torch.randn(X.shape[1], 1, dtype=torch.float64)
        Ytr = Xtr @ w
        Yval = Xval @ w
        M = Xtr[:10].clone().detach().requires_grad_()
        s = torch.tensor([10.0], dtype=X.dtype).requires_grad_()
        p = torch.tensor(1e-2, dtype=X.dtype).requires_grad_()

        torch.autograd.gradcheck(
            lambda sigma, pen, centers:
            ValidationLoss.apply(sigma, pen, centers, Xtr, Ytr, Xval, Yval, FalkonOptions(), 30,
                                 False),
            (s, p, M))
