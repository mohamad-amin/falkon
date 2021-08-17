import time
from typing import Dict, Optional

import scipy.linalg
import torch

from falkon import FalkonOptions
from falkon.hypergrad.common import full_rbf_kernel
from falkon.kernels import GaussianKernel
from falkon.kernels.diff_rbf_kernel import DiffGaussianKernel
from falkon.optim import FalkonConjugateGradient
from falkon.preconditioner import FalkonPreconditioner

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

EPS = 1e-6


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
        Tinv = scipy.linalg.lapack.strtri(T.cpu().detach().numpy(), lower=lower, unitdiag=0, overwrite_c=0)
    elif T.dtype == torch.float64:
        Tinv = scipy.linalg.lapack.dtrtri(T.cpu().detach().numpy(), lower=lower, unitdiag=0, overwrite_c=0)
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


def solve_falkon(X, centers, penalty, rhs, kernel_args, solve_options, solve_maxiter, init_sol=None):
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
    sol_full = precond.apply(beta)  # eta, alpha
    return sol_full, beta


def calc_grads(ctx, backward, num_diff_args):
    needs_grad = []
    for i in range(num_diff_args):
        if ctx.needs_input_grad[i]:
            needs_grad.append(ctx.saved_tensors[i])
    grads = torch.autograd.grad(backward, needs_grad, retain_graph=True, allow_unused=True)
    result = []
    j = 0
    for i in range(len(ctx.needs_input_grad)):
        if ctx.needs_input_grad[i]:
            result.append(grads[j])
            j += 1
        else:
            result.append(None)
    return tuple(result)


""" Nystrom Kernel Trace (2 methods) """


def nystrom_trace_fwd(kernel_args, M, X, kmn_z=None, use_ste=False):
    if use_ste:
        if kmn_z is None:
            raise RuntimeError("Cannot calculate nystrom-kernel trace with STE if kmn_z is not specified.")
        return nystrom_trace_hutch_fwd(kernel_args, M, kmn_z=kmn_z)
    else:
        return nystrom_trace_trinv_fwd(kernel_args, M, X)


def nystrom_trace_bwd(ctx: Dict[str, torch.Tensor], use_ste=False):
    if use_ste:
        return nystrom_trace_hutch_bwd(**ctx)
    return nystrom_trace_trinv_bwd(**ctx)


def nystrom_trace_hutch_fwd(kernel_args, M, kmn_z):
    mm_eye = torch.eye(M.shape[0], device=M.device, dtype=M.dtype) * EPS
    with torch.autograd.enable_grad():
        kmm = full_rbf_kernel(M, M, kernel_args)
        kmm_chol = torch.cholesky(kmm + mm_eye)

    with torch.autograd.no_grad():
        l_solve_1 = torch.triangular_solve(kmn_z, kmm_chol, upper=False, transpose=False).solution  # m * t
        l_solve_2 = torch.triangular_solve(l_solve_1, kmm_chol, upper=False, transpose=True).solution.contiguous()  # m * t
        output = torch.square(l_solve_1).sum(0).mean()

    ctx = dict(kmn_z=kmn_z, kmm_chol=kmm_chol, l_solve_1=l_solve_1, l_solve_2=l_solve_2)
    return output, ctx


def nystrom_trace_hutch_bwd(kmn_z, kmm_chol, l_solve_1, l_solve_2):
    with torch.autograd.enable_grad():
        bg = (
            2 * (kmn_z * l_solve_2).sum(0).mean()
            - 2 * (l_solve_2 * (kmm_chol @ l_solve_1)).sum(0).mean()
        )
    return bg


def nystrom_trace_trinv_fwd(kernel_args, M, X):
    diff_kernel = DiffGaussianKernel(kernel_args, opt=FalkonOptions(keops_active="no"))
    with torch.autograd.enable_grad():
        kmm = full_rbf_kernel(M, M, kernel_args)
        kmm_chol = torch.cholesky(kmm)
    with torch.autograd.no_grad():
        linv = tri_inverse(kmm_chol, lower=1)
    with torch.autograd.enable_grad():
        k_linv = diff_kernel.mmv(X, M, linv.T)  # n * m (problematic)

    return k_linv.square().sum(), dict(k_linv=k_linv, kmm_chol=kmm_chol, linv=linv)


def nystrom_trace_trinv_bwd(linv, k_linv, kmm_chol):
    with torch.autograd.no_grad():
        k_linv_linv = k_linv @ linv  # n * m
    with torch.autograd.enable_grad():
        bg = (
            2 * (k_linv * k_linv.detach()).sum() -
            2 * (k_linv.detach() @ kmm_chol.T * k_linv_linv).sum()
        )
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
            data.solve_zy, data.solve_zy_prec = solve_falkon(
                X, M, penalty, ZY, kernel_args, solve_opt, solve_maxiter, init_sol=last_solve_zy)

    with torch.autograd.no_grad():
        d_eff = (data.kmn_z * data.solve_z).sum(0).mean()

    return d_eff, data


def nystrom_deff_bwd(kernel_args, penalty, M, X, data):
    diff_kernel = DiffGaussianKernel(kernel_args)
    with torch.autograd.enable_grad():
        if data.knm_solve_zy is None:
            data.knm_solve_zy = diff_kernel.mmv(X, M, data.solve_zy)  # k_nm @ alpha  and  k_nm @ eta
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
    diff_kernel = DiffGaussianKernel(kernel_args, opt=solve_opt)

    ZY = torch.cat((data.z, Y), dim=1)
    if data.kmn_zy is None:
        with torch.autograd.enable_grad():
            data.kmn_zy = diff_kernel.mmv(M, X, ZY)
    if data.solve_zy is None:
        with torch.autograd.no_grad():
            # Solve Falkon part 1
            data.solve_zy, data.solve_zy_prec = solve_falkon(
                X, M, penalty, ZY, kernel_args, solve_opt, solve_maxiter, init_sol=last_solve_zy)
    if data.knm_solve_zy is None:
        with torch.autograd.enable_grad():
            # Note that knm @ alpha = y_tilde. This is handled by the data-class, hence we need to run this in fwd.
            data.knm_solve_zy = diff_kernel.mmv(X, M, data.solve_zy)
    if data.solve_ytilde is None:
        with torch.autograd.no_grad():
            # Solve Falkon part 2 (alpha_tilde = H^{-1} @ k_nm.T @ y_tilde
            data.solve_ytilde, data.solve_ytilde_prec = solve_falkon(
                X, M, penalty, data.y_tilde, kernel_args, solve_opt, solve_maxiter,
                init_sol=last_solve_ytilde)

    with torch.autograd.no_grad():
        # Loss = Y.T @ Y - 2 Y.T @ KNM @ alpha + alpha.T @ KNM.T @ KNM @ alpha
        loss = Y.square().sum()
        loss -= 2 * (data.kmn_y * data.solve_y).sum(0).mean()
        loss += data.y_tilde.square().sum(0).mean()

    return loss, data


def datafit_bwd(kernel_args, penalty, M, X, data):
    diff_kernel = DiffGaussianKernel(kernel_args)
    with torch.autograd.enable_grad():
        pen_n = penalty * X.shape[0]
        if data.kmm_solve_y is None:
            data.kmm_solve_zy = diff_kernel.mmv(M, M, data.solve_zy)

        # Loss without regularization
        loss_bg = (
            -4 * (data.kmn_y * data.solve_y).sum(0).mean()  # -4 * Y.T @ g(k_nm) @ alpha
            + 2 * (data.knm_solve_y.square().sum(0).mean() +
                   pen_n * (data.kmm_solve_y * data.solve_y).sum(0).mean())  # 2 * alpha.T @ g(H) @ alpha
            + 2 * (data.kmn_y * data.solve_ytilde).sum(0).mean()  # 2 * Y.T @ g(k_nm) @ alpha_tilde
            + 2 * (data.y_tilde * data.y_tilde.detach()).sum(0).mean()  # 2 * alpha.T @ g(k_nm.T) @ y_tilde
            - 2 * ((data.knm_solve_y * diff_kernel.mmv(X, M, data.solve_ytilde)).sum(0) +
                   pen_n * (data.kmm_solve_y * data.solve_ytilde).sum(0)).mean()  # -2 alpha @ g(H) @ alpha
        )
    return loss_bg, data


""" Nystrom Datafit (with penalty) """


def penalized_datafit_fwd(kernel_args, penalty, M, X, Y, solve_opt, solve_maxiter,
                          data: NoRegLossAndDeffCtx):
    diff_kernel = DiffGaussianKernel(kernel_args, opt=solve_opt)
    if data.solve_y is None:
        with torch.autograd.no_grad():
            data.solve_y, _ = solve_falkon(X, M, penalty, Y, kernel_args, solve_opt, solve_maxiter)
    if data.kmn_y is None:
        with torch.autograd.enable_grad():
            data.kmn_y = diff_kernel.mmv(M, X, Y)

    with torch.autograd.no_grad():
        loss = Y.square().sum()
        loss -= (data.kmn_y * data.solve_y).sum(0).mean()

    return loss, data


def penalized_datafit_bwd(kernel_args, penalty, M, X, data: NoRegLossAndDeffCtx):
    diff_kernel = DiffGaussianKernel(kernel_args)
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


def creg_plainfit(kernel_args, penalty, centers, X, Y, num_estimators, deterministic, solve_options, solve_maxiter, gaussian_random, use_stoch_trace, warm_start: bool = True):
    return NoRegLossAndDeff.apply(
        kernel_args, penalty, centers, X, Y, num_estimators, deterministic, solve_options, solve_maxiter, gaussian_random, use_stoch_trace, warm_start
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
        data.z = init_random_vecs(X.shape[0], t, dtype=X.dtype, device=X.device, gaussian_random=gaussian_random)

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
        with torch.autograd.enable_grad():
            kmn_z = data.kmn_z  # Need to diff through the slicing
        trace, tr_ctx = nystrom_trace_fwd(
            kernel_args=kernel_args, M=M, X=X, kmn_z=kmn_z, use_ste=use_stoch_trace)
        NoRegLossAndDeff.t_tr_fwd.append(time.time() - t_s)

        if warm_start:
            NoRegLossAndDeff._last_solve_zy = data.solve_zy_prec.detach()
            NoRegLossAndDeff._last_solve_ytilde = data.solve_ytilde_prec.detach()
        NoRegLossAndDeff.last_alpha = data.solve_y.detach()
        ctx.save_for_backward(kernel_args, penalty, M)
        ctx.data, ctx.tr_ctx, ctx.X, ctx.use_stoch_trace = data, tr_ctx, X, use_stoch_trace
        print(f"Stochastic: D-eff {d_eff:.3e} Data-Fit {datafit:.3e} Trace {X.shape[0] - trace:.3e}")
        return d_eff + datafit + (X.shape[0] - trace)

    @staticmethod
    def backward(ctx, out):
        kernel_args, penalty, M = ctx.saved_tensors
        data = ctx.data

        t_s = time.time()
        deff_bwd, data = nystrom_deff_bwd(kernel_args=kernel_args, penalty=penalty, M=M, X=ctx.X, data=data)
        NoRegLossAndDeff.t_deff_bwd.append(time.time() - t_s)
        t_s = time.time()
        dfit_bwd, data = datafit_bwd(kernel_args=kernel_args, penalty=penalty, M=M, X=ctx.X, data=data)
        NoRegLossAndDeff.t_fit_bwd.append(time.time() - t_s)
        t_s = time.time()
        tr_bwd = nystrom_trace_bwd(ctx.tr_ctx, use_ste=ctx.use_stoch_trace)
        NoRegLossAndDeff.t_tr_bwd.append(time.time() - t_s)

        t_s = time.time()
        with torch.autograd.enable_grad():
            bg = out * (deff_bwd + dfit_bwd - tr_bwd)
        out = calc_grads(ctx, bg, NoRegLossAndDeff.NUM_DIFF_ARGS)
        NoRegLossAndDeff.t_grad.append(time.time() - t_s)
        return out

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
                NoRegLossAndDeff.apply(sigma, pen, centers, X, Y, 20, True, FalkonOptions(), 30, False, True, False),
            (s, p, M))

        torch.autograd.gradcheck(
            lambda sigma, pen, centers:
                NoRegLossAndDeff.apply(sigma, pen, centers, X, Y, 20, True, FalkonOptions(), 30, False, False, False),
            (s, p, M))


def gcv(kernel_args, penalty, centers, X, Y, num_estimators, deterministic, solve_options, solve_maxiter, gaussian_random, warm_start: bool = True):
    return GCV.apply(
        kernel_args, penalty, centers, X, Y, num_estimators, deterministic, solve_options, solve_maxiter, gaussian_random, warm_start
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
                warm_start: bool,):
        if GCV._last_t is not None and GCV._last_t != t:
            GCV._last_solve_zy = None
            GCV._last_solve_ytilde = None
            GCV.last_alpha = None
        GCV._last_t = t
        data = NoRegLossAndDeffCtx(t)
        if deterministic:
            torch.manual_seed(12)
        data.z = init_random_vecs(X.shape[0], t, dtype=X.dtype, device=X.device, gaussian_random=gaussian_random)

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

        deff_bwd, data = nystrom_deff_bwd(kernel_args=kernel_args, penalty=penalty, M=M, X=ctx.X, data=data)
        dfit_bwd, data = datafit_bwd(kernel_args=kernel_args, penalty=penalty, M=M, X=ctx.X, data=data)
        with torch.autograd.enable_grad():
            bg = out * (dfit_bwd * denominator - ctx.datafit * (- 2 / n + 2 * ctx.d_eff / n**2) * deff_bwd) / torch.square(denominator)
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


def creg_penfit(kernel_args, penalty, centers, X, Y, num_estimators, deterministic, solve_options, solve_maxiter, gaussian_random, use_stoch_trace, warm_start=True):
    return RegLossAndDeffv2.apply(
        kernel_args, penalty, centers, X, Y, num_estimators, deterministic, solve_options, solve_maxiter, gaussian_random, use_stoch_trace, warm_start
    )


# noinspection PyMethodOverriding
class RegLossAndDeffv2(torch.autograd.Function):
    NUM_DIFF_ARGS = 3
    _last_solve_zy = None
    last_alpha = None
    _last_t = None

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
        if RegLossAndDeffv2._last_t is not None and RegLossAndDeffv2._last_t != t:
            RegLossAndDeffv2._last_solve_zy = None
            RegLossAndDeffv2.last_alpha = None
        RegLossAndDeffv2._last_t = t
        data = NoRegLossAndDeffCtx(t)
        if deterministic:
            torch.manual_seed(12)
        data.z = init_random_vecs(X.shape[0], t, dtype=X.dtype, device=X.device, gaussian_random=gaussian_random)

        d_eff, data = nystrom_deff_fwd(
            kernel_args=kernel_args, penalty=penalty, M=M, X=X, Y=Y,
            solve_opt=solve_options, solve_maxiter=solve_maxiter,
            last_solve_zy=RegLossAndDeffv2._last_solve_zy, data=data)
        datafit, data = penalized_datafit_fwd(
            kernel_args=kernel_args, penalty=penalty, M=M, X=X, Y=Y,
            solve_maxiter=solve_maxiter, solve_opt=solve_options, data=data)
        with torch.autograd.enable_grad():
            kmn_z = data.kmn_z  # Need to diff through the slicing
        trace, tr_ctx = nystrom_trace_fwd(
            kernel_args=kernel_args, M=M, X=X, kmn_z=kmn_z, use_ste=use_stoch_trace)

        if warm_start:
            RegLossAndDeffv2._last_solve_zy = data.solve_zy_prec.detach().clone()
        RegLossAndDeffv2.last_alpha = data.solve_y.detach()
        ctx.save_for_backward(kernel_args, penalty, M)
        ctx.data, ctx.tr_ctx, ctx.X, ctx.use_stoch_trace = data, tr_ctx, X, use_stoch_trace
        print(f"Stochastic: D-eff {d_eff:.3e} Data-Fit {datafit:.3e} Trace {X.shape[0] - trace:.3e}")
        return d_eff + datafit + (X.shape[0] - trace)

    @staticmethod
    def backward(ctx, out):
        kernel_args, penalty, M = ctx.saved_tensors
        data = ctx.data

        deff_bwd, data = nystrom_deff_bwd(kernel_args=kernel_args, penalty=penalty, M=M, X=ctx.X, data=data)
        dfit_bwd, data = penalized_datafit_bwd(kernel_args=kernel_args, penalty=penalty, M=M, X=ctx.X, data=data)
        tr_bwd = nystrom_trace_bwd(ctx.tr_ctx, use_ste=ctx.use_stoch_trace)

        with torch.autograd.enable_grad():
            bg = out * (deff_bwd + dfit_bwd - tr_bwd)
        return calc_grads(ctx, bg, RegLossAndDeffv2.NUM_DIFF_ARGS)

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
                RegLossAndDeffv2.apply(sigma, pen, centers, X, Y, 20, True, FalkonOptions(), 30, False, True, False),
            (s, p, M))
        torch.autograd.gradcheck(
            lambda sigma, pen, centers:
                RegLossAndDeffv2.apply(sigma, pen, centers, X, Y, 20, True, FalkonOptions(), 30, False, False, False),
            (s, p, M))


def validation_loss(kernel_args, penalty, centers, Xtr, Ytr, Xval, Yval, solve_options, solve_maxiter, warm_start=True):
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
            diff_kernel = DiffGaussianKernel(kernel_args, opt=solve_options)
            nondiff_kernel = GaussianKernel(kernel_args, opt=solve_options)
            pc = FalkonPreconditioner(penalty.item(), nondiff_kernel, opt=solve_options)
            pc.init(M.detach())
            optim = FalkonConjugateGradient(nondiff_kernel, pc, opt=solve_options)
            beta1 = optim.solve(Xtr, M.detach(), Ytr, penalty.item(), ValidationLoss._last_alpha_prec, solve_maxiter)
            alpha1 = pc.apply(beta1)

        with torch.autograd.enable_grad():
            kvm_alpha = diff_kernel.mmv(Xval, M, alpha1)

        if warm_start:
            ValidationLoss._last_alpha_prec = beta1
        ValidationLoss.last_alpha = alpha1

        ctx.save_for_backward(kernel_args, penalty, M)
        ctx.alpha1, ctx.kvm_alpha = alpha1, kvm_alpha
        ctx.pc, ctx.solve_maxiter, ctx.diff_kernel, ctx.optim = pc, solve_maxiter, diff_kernel, optim
        ctx.Xtr, ctx.Xval, ctx.Ytr, ctx.Yval = Xtr, Xval, Ytr, Yval
        ctx.warm_start = warm_start

        with torch.autograd.no_grad():
            val_loss = torch.sum(torch.square(kvm_alpha - Yval))
        return val_loss

    @staticmethod
    def backward(ctx, out):
        kernel_args, penalty, M = ctx.saved_tensors
        alpha1, kvm_alpha = ctx.alpha1, ctx.kvm_alpha
        pc, solve_maxiter, diff_kernel, optim = ctx.pc, ctx.solve_maxiter, ctx.diff_kernel, ctx.optim
        Xtr, Xval, Ytr, Yval = ctx.Xtr, ctx.Xval, ctx.Ytr, ctx.Yval

        with torch.autograd.no_grad():
            # 2 right-hand-sides: kvm_alpha and y_val
            slv_shape = kvm_alpha.shape[1]
            solve2rhs = torch.cat((kvm_alpha.detach(), Yval), dim=1)

            beta_slv2 = optim.solve_val_rhs(Xtr, Xval, M.detach(), solve2rhs, penalty.item(), ValidationLoss._last_alpha2_prec, solve_maxiter)
            alpha_slv2 = pc.apply(beta_slv2)
            if ctx.warm_start:
                ValidationLoss._last_alpha2_prec = beta_slv2

            all_alphas = torch.cat((alpha1, alpha_slv2), dim=1)
            a2, a3 = torch.split(alpha_slv2, slv_shape, dim=1)

        with torch.autograd.enable_grad():
            kmn_y = diff_kernel.mmv(M, Xtr, Ytr)
            kmv_yv = diff_kernel.mmv(M, Xval, Yval)
            knm_slv_all = diff_kernel.mmv(Xtr, M, all_alphas)
            knm_a1, knm_a2, knm_a3 = torch.split(knm_slv_all, slv_shape, dim=1)
            kmm_a1 = diff_kernel.mmv(M, M, alpha1)

            pen_n = penalty * Xtr.shape[0]
            bg = out * (
                + 2 * (kmn_y * a2).sum()
                + 2 * (kvm_alpha.detach() * kvm_alpha).sum()
                - 2 * ((knm_a1 * knm_a2.detach()).sum() + (knm_a1.detach() * knm_a2).sum() + pen_n * (kmm_a1 * a2).sum())
                - 2 * (kmn_y * a3).sum()
                - 2 * (alpha1 * kmv_yv).sum()
                + 2 * ((knm_a1 * knm_a3.detach()).sum() + (knm_a1.detach() * knm_a3).sum() + pen_n * (kmm_a1 * a3).sum())
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
                ValidationLoss.apply(sigma, pen, centers, Xtr, Ytr, Xval, Yval, FalkonOptions(), 30, False),
            (s, p, M))
