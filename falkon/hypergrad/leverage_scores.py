from typing import Optional

import scipy.linalg
import torch

from falkon import FalkonOptions
from falkon.hypergrad.common import full_rbf_kernel
from falkon.kernels import GaussianKernel
from falkon.kernels.diff_rbf_kernel import DiffGaussianKernel
from falkon.optim import FalkonConjugateGradient
from falkon.preconditioner import FalkonPreconditioner

__all__ = (
    "creg_plainfit",
    "NoRegLossAndDeff",
    "GCV",
    "gcv",
    "creg_penfit",
    "RegLossAndDeffv2",
)


class NoRegLossAndDeffCtx():
    __slots__ = ("_flk_solve_zy", "_kmn_zy", "_flk_solve_ytilde", "_len_z",
                 "_knm_solve_zy", "_kmm_solve_zy", "_nys_trace", "_nys_deff",
                 "_nys_d_eff",
                 "_nys_data_fit",
                 "_flk_solve_y",
                 "_kmn_y",
                 )

    def __init__(self,
                 len_z: int,
                 ):
        self._len_z = len_z

        self._flk_solve_zy = None
        self._flk_solve_y = None
        self._kmn_zy = None
        self._kmn_y = None
        self._flk_solve_ytilde = None
        self._knm_solve_zy = None
        self._kmm_solve_zy = None
        self._nys_trace = None
        self._nys_deff = None
        self._nys_d_eff = None
        self._nys_data_fit = None

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


def init_random_vecs(n, t, dtype, device, gaussian_random: bool):
    if gaussian_random:
        Z = torch.randn(n, t, dtype=dtype, device=device)
    else:
        Z = torch.empty(n, t, dtype=dtype, device=device).bernoulli_().mul_(2).sub_(1)
    return Z


def solve_falkon(X, centers, penalty, rhs, kernel_args, solve_options, solve_maxiter):
    penalty = penalty.item() #* X.shape[0]
    M_ = centers.detach()
    kernel_args_ = kernel_args.detach()
    K = GaussianKernel(kernel_args_, opt=solve_options)  # here which opt doesnt matter
    precond = FalkonPreconditioner(penalty , K, solve_options)  # here which opt doesnt matter
    precond.init(M_)
    optim = FalkonConjugateGradient(K, precond, solve_options)
    beta = optim.solve(
        X, M_, rhs, penalty,
        initial_solution=None,
        max_iter=solve_maxiter,
    )
    sol_full = precond.apply(beta)  # eta, alpha
    return sol_full


def calc_grads(ctx, backward, num_diff_args):
    needs_grad = []
    for i in range(num_diff_args):
        if ctx.needs_input_grad[i]:
            needs_grad.append(ctx.saved_tensors[i])
    grads = torch.autograd.grad(backward, needs_grad, retain_graph=True)
    result = []
    j = 0
    for i in range(len(ctx.needs_input_grad)):
        if ctx.needs_input_grad[i]:
            result.append(grads[j])
            j += 1
        else:
            result.append(None)
    return tuple(result)


# noinspection PyMethodOverriding
class NystromKernelTraceHutch(torch.autograd.Function):
    EPS = 1e-6
    """
    STE with bernoulli Z (m x t)
    note Tr(knm @ Kmm^{-1} @ knm.T) = Tr( knm L^{-T} L^{-1} knm.T) so
    solve g1 = L^{-1} @ knm.T @ Z
    solve g2 = L^{-T} @ g1  - only used for backward

    fwd: g1.T @ g1
    """
    @staticmethod
    def forward(ctx,
                kernel_args,
                M: torch.Tensor,
                X: torch.Tensor,
                t: int,
                deterministic: bool,
                gaussian_random: bool,
                ):
        diff_kernel = DiffGaussianKernel(
            kernel_args,
            opt=FalkonOptions(keops_active="no"))
        if deterministic:
            torch.manual_seed(22)

        Z = init_random_vecs(X.shape[0], t, dtype=X.dtype, device=X.device, gaussian_random=gaussian_random)
        mm_eye = torch.eye(M.shape[0], device=M.device, dtype=M.dtype) * NystromKernelTraceHutch.EPS
        with torch.autograd.enable_grad():
            kmn_z = diff_kernel.mmv(M, X, Z)  # m * t
            kmm = full_rbf_kernel(M, M, kernel_args)
            L = torch.cholesky(kmm + mm_eye)

        with torch.autograd.no_grad():
            g1 = torch.triangular_solve(kmn_z, L, upper=False, transpose=False).solution  # m * t
            g2 = torch.triangular_solve(g1, L, upper=False, transpose=True).solution.contiguous()  # m * t

        ctx.save_for_backward(kernel_args, M)
        ctx.kmn_z, ctx.L, ctx.g1, ctx.g2 = kmn_z, L, g1, g2
        return torch.square(g1).sum(0).mean()

    @staticmethod
    def backward(ctx, out):
        with torch.autograd.enable_grad():
            bg = out * (
                2 * (ctx.kmn_z * ctx.g2).sum(0).mean()
                - 2 * (ctx.g2 * (ctx.L @ ctx.g1)).sum(0).mean()
            )
        return calc_grads(ctx, bg, 2)

    @staticmethod
    def grad_check():
        torch.manual_seed(3)
        X = torch.randn(150, 10, dtype=torch.float64)
        M = X[:20].clone().detach().requires_grad_()
        s = torch.tensor([10.0] * X.shape[1], dtype=X.dtype).requires_grad_()

        torch.autograd.gradcheck(
            lambda sigma, centers:
                NystromKernelTraceHutch.apply(sigma, centers, X, 20, True, False), (s, M))


def nystrom_kernel_trace(kernel_args, M, X, t=None, deterministic=None, gaussian_random=None, use_ste=False):
    if use_ste:
        if t is None or deterministic is None or gaussian_random is None:
            raise RuntimeError("Cannot calculate nystrom-kernel trace with STE if t, deterministic, gaussian_random are not specified.")
        return NystromKernelTraceHutch.apply(kernel_args, M, X, t, deterministic, gaussian_random)
    else:
        return NystromKernelTrace.apply(kernel_args, M, X)


# noinspection PyMethodOverriding
class NystromKernelTrace(torch.autograd.Function):
    @staticmethod
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

    @staticmethod
    def forward(ctx, kernel_args, M, X):
        diff_kernel = DiffGaussianKernel(
            kernel_args,
            opt=FalkonOptions(keops_active="no"))
        with torch.autograd.enable_grad():
            kmm = full_rbf_kernel(M, M, kernel_args)
            L = torch.cholesky(kmm)
        with torch.autograd.no_grad():
            Linv = NystromKernelTrace.tri_inverse(L, lower=1)
        with torch.autograd.enable_grad():
            k_linv = diff_kernel.mmv(X, M, Linv.T)

        ctx.save_for_backward(kernel_args, M)
        ctx.k_linv = k_linv
        ctx.L = L
        ctx.linv = Linv

        return k_linv.square().sum()

    @staticmethod
    def backward(ctx, out):
        with torch.autograd.no_grad():
            k_linv_linv = ctx.k_linv @ ctx.linv
        with torch.autograd.enable_grad():
            bg = out * (
                2 * (ctx.k_linv * ctx.k_linv.detach()).sum() -
                2 * (ctx.k_linv.detach() @ ctx.L.T * k_linv_linv).sum()
            )

        return calc_grads(ctx, bg, 2)

    @staticmethod
    def grad_check():
        torch.manual_seed(3)
        X = torch.randn(50, 10, dtype=torch.float64)
        M = X[:20].clone().detach().requires_grad_()
        s = torch.tensor([10.0] * X.shape[1], dtype=X.dtype).requires_grad_()

        torch.autograd.gradcheck(
            lambda sigma, centers:
                NystromKernelTrace.apply(sigma, centers, X),
            (s, M))


def nystrom_effective_dim(kernel_args, penalty, M, X, Y, t, gaussian_random, solve_opt, solve_maxiter, data):
    return NystromEffectiveDimension.apply(
        kernel_args, penalty, M, X, Y, t, gaussian_random, solve_opt, solve_maxiter,
        data, False)


def nystrom_effective_dim_wdata(kernel_args, penalty, M, X, data):
    return NystromEffectiveDimension.apply(
        kernel_args, penalty, M, X, None, None, None, None, None, data, False)


# noinspection PyMethodOverriding
class NystromEffectiveDimension(torch.autograd.Function):
    r"""
    The effective dimension is $trace((\tilde{K} + n \lambda)^{-1} \tilde{K})$
    or alternatively $trace(k_nm (k_nm.T @ k_nm + n \lambda k_mm)^{-1} k_nm.T)$.
    The two formulations are equal!
    """
    NUM_DIFF_ARGS = 3

    @staticmethod
    def forward(ctx,
                kernel_args: torch.Tensor,
                penalty: torch.Tensor,
                M: torch.Tensor,
                X: torch.Tensor,
                Y: torch.Tensor,
                t: int,
                gaussian_random: bool,
                solve_options: FalkonOptions,
                solve_maxiter: int,
                data: NoRegLossAndDeffCtx,
                is_gradcheck: Optional[bool]):
        diff_kernel = DiffGaussianKernel(kernel_args, opt=solve_options)

        if is_gradcheck:
            torch.manual_seed(22)
        if data.kmn_zy is None or data.solve_zy is None:
            Z = init_random_vecs(X.shape[0], t, dtype=X.dtype, device=X.device, gaussian_random=gaussian_random)
            ZY = torch.cat((Z, Y), dim=1)
            with torch.autograd.enable_grad():
                data.kmn_zy = diff_kernel.mmv(M, X, ZY)
            with torch.autograd.no_grad():
                data.solve_zy = solve_falkon(X, M, penalty, ZY, kernel_args, solve_options, solve_maxiter)

        d_eff = (data.kmn_z * data.solve_z).sum(0).mean()

        ctx.save_for_backward(kernel_args, penalty, M)
        ctx.data, ctx.diff_kernel, ctx.X = data, diff_kernel, X
        return d_eff

    @staticmethod
    def backward(ctx, out_deff):
        data, diff_kernel, X = ctx.data, ctx.diff_kernel, ctx.X
        kernel_args, penalty, M = ctx.saved_tensors

        with torch.autograd.enable_grad():
            if data.knm_solve_zy is None:
                data.knm_solve_zy = diff_kernel.mmv(X, M, data.solve_zy)  # k_nm @ alpha  and  k_nm @ eta
            if data.kmm_solve_zy is None:
                data.kmm_solve_zy = diff_kernel.mmv(M, M, data.solve_zy)

            # Effective dimension
            deff_bg = out_deff * (
                2 * (data.kmn_z * data.solve_z).sum(0).mean()
                - (data.knm_solve_z.square().sum(0).mean() +
                   penalty * X.shape[0] * (data.solve_z * data.kmm_solve_z).sum(0).mean())
            )

        return calc_grads(ctx, deff_bg, NystromEffectiveDimension.NUM_DIFF_ARGS)

    @staticmethod
    def grad_check():
        torch.manual_seed(3)
        X = torch.randn(50, 10, dtype=torch.float64)
        w = torch.randn(X.shape[1], 1, dtype=torch.float64)
        Y = X @ w
        M = torch.randn(20, 10, dtype=torch.float64).requires_grad_()
        s = torch.tensor([10.0] * X.shape[1], dtype=X.dtype).requires_grad_()
        p = torch.tensor(1e-3, dtype=X.dtype).requires_grad_()

        torch.autograd.gradcheck(
            lambda sigma, pen, centers:
                NystromEffectiveDimension.apply(sigma, pen, centers, X, Y, 20, True, FalkonOptions(cg_tolerance=1e-8, cg_epsilon_64=1e-10), 30, NoRegLossAndDeffCtx(20), True),
            (s, p, M))


def nystrom_data_fit(kernel_args, penalty, M, X, Y, t, deterministic, solve_options, solve_maxiter, gaussian_random, data):
    return NystromDataFit.apply(
        kernel_args, penalty, M, X, Y, t, deterministic, solve_options, solve_maxiter, gaussian_random, data
    )


# noinspection PyMethodOverriding
class NystromDataFit(torch.autograd.Function):
    EPS = 1e-6
    NUM_DIFF_ARGS = 3

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
            data: NoRegLossAndDeffCtx,
            ):
        diff_kernel = DiffGaussianKernel(kernel_args, opt=FalkonOptions(keops_active="no"))

        if deterministic:
            torch.manual_seed(12)
        if data.kmn_zy is None or data.solve_zy is None:
            Z = init_random_vecs(X.shape[0], t, dtype=X.dtype, device=X.device, gaussian_random=gaussian_random)
            ZY = torch.cat((Z, Y), dim=1)

            with torch.autograd.no_grad():
                # Solve Falkon part 1
                data.solve_zy = solve_falkon(X, M, penalty, ZY, kernel_args, solve_options, solve_maxiter)
            with torch.autograd.enable_grad():
                # Note that knm @ alpha = y_tilde. This is handled by the data-class
                data.knm_solve_zy = diff_kernel.mmv(X, M, data.solve_zy)
                data.kmn_zy = diff_kernel.mmv(M, X, ZY)
            with torch.autograd.no_grad():
                # Solve Falkon part 2 (alpha_tilde = H^{-1} @ k_nm.T @ y_tilde
                data.solve_ytilde = solve_falkon(X, M, penalty, data.y_tilde, kernel_args, solve_options, solve_maxiter)

        # Loss = Y.T @ Y - 2 Y.T @ KNM @ alpha + alpha.T @ KNM.T @ KNM @ alpha
        loss = Y.square().sum()
        loss -= 2 * (data.kmn_y * data.solve_y).sum(0).mean()
        loss += data.y_tilde.square().sum(0).mean()
        #loss /= X.shape[0]

        ctx.save_for_backward(kernel_args, penalty, M)
        ctx.data, ctx.diff_kernel, ctx.X = data, diff_kernel, X
        return loss

    @staticmethod
    def backward(ctx, out):
        data, diff_kernel, X = ctx.data, ctx.diff_kernel, ctx.X
        kernel_args, penalty, M = ctx.saved_tensors

        with torch.autograd.enable_grad():
            pen_n = penalty * X.shape[0]
            data.kmm_solve_zy = diff_kernel.mmv(M, M, data.solve_zy)

            # Loss without regularization
            loss_bg = out * (
                -4 * (data.kmn_y * data.solve_y).sum(0).mean()  # -4 * Y.T @ g(k_nm) @ alpha
                + 2 * (data.knm_solve_y.square().sum(0).mean() +
                       pen_n * (data.kmm_solve_y * data.solve_y).sum(0).mean())  # 2 * alpha.T @ g(H) @ alpha
                + 2 * (data.kmn_y * data.solve_ytilde).sum(0).mean()  # 2 * Y.T @ g(k_nm) @ alpha_tilde
                + 2 * (data.y_tilde * data.y_tilde.detach()).sum(0).mean()  # 2 * alpha.T @ g(k_nm.T) @ y_tilde
                - 2 * ((data.knm_solve_y * diff_kernel.mmv(X, M, data.solve_ytilde)).sum(0) +
                       pen_n * (data.kmm_solve_y * data.solve_ytilde).sum(0)).mean()  # -2 alpha @ g(H) @ alpha
            )# / X.shape[0]
        return calc_grads(ctx, loss_bg, NystromDataFit.NUM_DIFF_ARGS)

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
                NystromDataFit.apply(sigma, pen, centers, X, Y, 20, True, FalkonOptions(), 30, False, NoRegLossAndDeffCtx(20)),
            (s, p, M))


def nystrom_penalized_data_fit(kernel_args, penalty, M, X, Y, solve_options, solve_maxiter, data):
    return NystromPenalizedDataFit.apply(
        kernel_args, penalty, M, X, Y, solve_options, solve_maxiter, data
    )


# noinspection PyMethodOverriding
class NystromPenalizedDataFit(torch.autograd.Function):
    r"""
    $1/n * \norm{Y - \tilde{f}(X)}^2 + \lambda \alpha.T k_mm \alpha$
    considering
    $\alpha = (n\lambda k_mm + k_nm.T k_nm)^{-1} k_nm.T Y$
    yields
    $\lambda Y.T ( k_nm k_mm^{-1} k_nm.T + n \lambda I)^{-1} Y$
    which expands to
    $1/n * Y.T @ Y - 1/n * Y.T k_nm (n * \lambda * k_mm + k_nm.T k_nm)^{-1} k_nm.T Y$
    """
    NUM_DIFF_ARGS = 3

    @staticmethod
    def forward(
            ctx,
            kernel_args: torch.Tensor,
            penalty: torch.Tensor,
            M: torch.Tensor,
            X: torch.Tensor,
            Y: torch.Tensor,
            solve_options: FalkonOptions,
            solve_maxiter: int,
            data: NoRegLossAndDeffCtx,
            ):
        diff_kernel = DiffGaussianKernel(kernel_args, opt=FalkonOptions(keops_active="no"))
        if data.solve_y is None:
            with torch.autograd.no_grad():
                data.solve_y = solve_falkon(X, M, penalty, Y, kernel_args, solve_options, solve_maxiter)
        if data.kmn_y is None:
            with torch.autograd.enable_grad():
                data.kmn_y = diff_kernel.mmv(M, X, Y)

        with torch.autograd.enable_grad():
            loss = Y.square().sum()
            loss -= (data.kmn_y * data.solve_y).sum(0).mean()
            #loss /= X.shape[0]

        ctx.save_for_backward(kernel_args, penalty, M)
        ctx.data, ctx.diff_kernel, ctx.X = data, diff_kernel, X
        return loss

    @staticmethod
    def backward(ctx, out):
        data, diff_kernel, X = ctx.data, ctx.diff_kernel, ctx.X
        kernel_args, penalty, M = ctx.saved_tensors

        with torch.autograd.enable_grad():
            pen_n = penalty * X.shape[0]
            kmm_solve_y = diff_kernel.mmv(M, M, data.solve_y)
            knm_solve_y = diff_kernel.mmv(X, M, data.solve_y)

            loss_bg = out * (
                -2 * (data.kmn_y * data.solve_y).sum(0).mean()
                + (knm_solve_y.square().sum(0).mean() +
                   pen_n * (kmm_solve_y * data.solve_y).sum(0).mean())
            )# / X.shape[0]
        return calc_grads(ctx, loss_bg, NystromPenalizedDataFit.NUM_DIFF_ARGS)

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
                NystromPenalizedDataFit.apply(sigma, pen, centers, X, Y, FalkonOptions(), 30, NoRegLossAndDeffCtx(20)),
            (s, p, M))


def creg_plainfit(kernel_args, penalty, centers, X, Y, num_estimators, deterministic, solve_options, solve_maxiter, gaussian_random, use_stoch_trace):
    return NoRegLossAndDeff.apply(
        kernel_args, penalty, centers, X, Y, num_estimators, deterministic, solve_options, solve_maxiter, gaussian_random, use_stoch_trace
    )


# noinspection PyMethodOverriding
class NoRegLossAndDeff(torch.autograd.Function):
    EPS = 1e-6
    NUM_DIFF_ARGS = 3
    last_alpha = None

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
            ):
        data = NoRegLossAndDeffCtx(t)

        with torch.autograd.enable_grad():
            data.data_fit = nystrom_data_fit(
                kernel_args, penalty, M, X, Y, t, deterministic, solve_options, solve_maxiter,
                gaussian_random, data)
            # Trace of the Nystrom kernel: Tr(KNM @ KMM^{-1} @ KNM.T) Prolem here: formation of N*M matrix!
            data.nys_trace = nystrom_kernel_trace(kernel_args, M, X, use_ste=use_stoch_trace, t=t, deterministic=deterministic, gaussian_random=gaussian_random)
            trace = X.shape[0] - data.nys_trace
            # D-Eff = Tr((KNM @ KMM^{-1} @ KNM.T + lambda I)^{-1} KNM @ KMM^{-1} @ KNM.T)
            data.nys_d_eff = nystrom_effective_dim_wdata(kernel_args, penalty, M, X, data)

        NoRegLossAndDeff.last_alpha = data.solve_y
        ctx.save_for_backward(kernel_args, penalty, M)
        ctx.data = data
        return data.nys_d_eff + data.data_fit + trace

    @staticmethod
    def backward(ctx, out):
        data = ctx.data
        with torch.autograd.enable_grad():
            bg = out * (data.nys_d_eff + data.data_fit - data.nys_trace)
        return calc_grads(ctx, bg, NoRegLossAndDeff.NUM_DIFF_ARGS)

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
                NoRegLossAndDeff.apply(sigma, pen, centers, X, Y, 20, True, FalkonOptions(), 30, False),
            (s, p, M))


def gcv(kernel_args, penalty, centers, X, Y, num_estimators, deterministic, solve_options, solve_maxiter, gaussian_random):
    return GCV.apply(
        kernel_args, penalty, centers, X, Y, num_estimators, deterministic, solve_options, solve_maxiter, gaussian_random
    )


# noinspection PyMethodOverriding
class GCV(torch.autograd.Function):
    last_alpha = None
    NUM_DIFF_ARGS = 3
    """
    Numerator: Exactly the data-fit term of NoRegLossAndDeff
    Denominator: Similar to Nystrom Effective Dim.
    """
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
                gaussian_random: bool,):
        data = NoRegLossAndDeffCtx(t)

        with torch.autograd.enable_grad():
            data.data_fit = nystrom_data_fit(
                kernel_args, penalty, M, X, Y, t, deterministic, solve_options, solve_maxiter,
                gaussian_random, data)
            data.nys_d_eff = nystrom_effective_dim_wdata(kernel_args, penalty, M, X, data)
            data.nys_d_eff = torch.square((1.0 - data.nys_d_eff/X.shape[0]))

        GCV.last_alpha = data.solve_y
        ctx.save_for_backward(kernel_args, penalty, M)
        ctx.data = data
        return data.data_fit / data.nys_d_eff

    @staticmethod
    def backward(ctx, out):
        data = ctx.data
        with torch.autograd.enable_grad():
            bg = out * (data.data_fit / data.nys_d_eff)
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
                GCV.apply(sigma, pen, centers, X, Y, 20, True, FalkonOptions(), 30, False),
            (s, p, M))


def creg_penfit(kernel_args, penalty, centers, X, Y, num_estimators, deterministic, solve_options, solve_maxiter, gaussian_random, use_stoch_trace):
    return RegLossAndDeffv2.apply(
        kernel_args, penalty, centers, X, Y, num_estimators, deterministic, solve_options, solve_maxiter, gaussian_random, use_stoch_trace
    )


# noinspection PyMethodOverriding
class RegLossAndDeffv2(torch.autograd.Function):
    EPS = 1e-6
    NUM_DIFF_ARGS = 3
    last_alpha = None

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
            ):
        data = NoRegLossAndDeffCtx(t)

        with torch.autograd.enable_grad():
            # D-eff calculation initializes the data object.
            # D-Eff = Tr((KNM @ KMM^{-1} @ KNM.T + lambda I)^{-1} KNM @ KMM^{-1} @ KNM.T)
            data.nys_d_eff = nystrom_effective_dim(kernel_args, penalty, M, X, Y, t, gaussian_random, solve_options, solve_maxiter, data)
            # Penalized data-fit
            data.data_fit = nystrom_penalized_data_fit(
                kernel_args, penalty, M, X, Y, solve_options, solve_maxiter, data)
            # Trace of the Nystrom kernel: Tr(KNM @ KMM^{-1} @ KNM.T) Prolem here: formation of N*M matrix!
            data.nys_trace = nystrom_kernel_trace(kernel_args, M, X, use_ste=use_stoch_trace, t=t, deterministic=deterministic, gaussian_random=gaussian_random)
            trace = X.shape[0] - data.nys_trace

        RegLossAndDeffv2.last_alpha = data.solve_y
        ctx.save_for_backward(kernel_args, penalty, M)
        ctx.data = data
        print(f"Stochastic: D-eff {data.nys_d_eff:.3e} Data-Fit {data.data_fit:.3e} Trace {trace:.3e}")
        return data.nys_d_eff + data.data_fit + trace

    @staticmethod
    def backward(ctx, out):
        data = ctx.data
        with torch.autograd.enable_grad():
            bg = out * (data.nys_d_eff + data.data_fit - data.nys_trace)
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
                RegLossAndDeffv2.apply(sigma, pen, centers, X, Y, 20, True, FalkonOptions(), 30, False),
            (s, p, M))
