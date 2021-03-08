import time
import math

import torch
from falkon.hypergrad.common import full_rbf_kernel

import falkon
from falkon import Falkon, InCoreFalkon, FalkonOptions
from falkon.center_selection import FixedSelector
from falkon.optim import FalkonConjugateGradient
from falkon.preconditioner import FalkonPreconditioner
from falkon.models.model_utils import FalkonBase
from falkon.kernels import GaussianKernel
from falkon.kernels.diff_rbf_kernel import DiffGaussianKernel
from falkon.la_helpers import potrf, trsm


__all__ = ("gauss_effective_dimension", "gauss_nys_effective_dimension")

def subs_deff_simple(kernel: falkon.kernels.Kernel,
                     penalty: torch.Tensor,
                     X: torch.Tensor,
                     J: torch.Tensor) -> torch.Tensor:
    n = X.shape[0]
    K_JJ = kernel(J, J)
    K_XJ = kernel(X, J)

    # Inversion
    U, S, _ = torch.svd(K_JJ + penalty * n * torch.eye(K_JJ.shape[0], dtype=K_JJ.dtype, device=K_JJ.device))
    # Exclude eigen-values close to 0
    thresh = (S[0] * S.shape[0] * torch.finfo(S.dtype).eps).item()
    stable_eig = (S > thresh)
    print("%d stable eigenvalues" % (stable_eig.sum()))
    U_thin = U[:, stable_eig]  # n x m
    S_thin = S[stable_eig]     # m
    S_thin_root_inv = torch.sqrt(S_thin).reciprocal().reshape(-1, 1)  # m x 1
    K_JJ_inv = S_thin_root_inv * U_thin.T   # square root inverse (m x n)

    # Multiply by the large kernel
    E = K_JJ_inv @ K_XJ.T  # x * j

    # Calculate the trace, which is just the squared norm of the columns
    dim_eff = (1 / penalty) - (1 / (penalty * n)) * torch.sum(torch.norm(E, dim=1) ** 2)
    return dim_eff


def full_deff_simple(kernel, penalty, X):
    K = kernel(X, X)
    K_inv = torch.pinverse(K + penalty * X.shape[0] * torch.eye(X.shape[0]))
    tau = torch.diagonal(K @ K_inv)
    return tau.sum()


def full_deff(kernel, penalty, X: torch.Tensor):
    K = kernel(X, X)  # n * n
    n = X.shape[0]

    K_invertible = K + penalty * n * torch.eye(X.shape[0])

    U_DD, S_DD, _ = torch.svd(K_invertible)
    # Exclude eigen-values close to 0
    thresh = (S_DD.max() * S_DD.shape[0] * torch.finfo(S_DD.dtype).eps).item()
    stable_eig = (S_DD > thresh)
    U_thin = U_DD[:, stable_eig]  # n x m
    S_thin = S_DD[stable_eig]     # m
    S_thin_inv = torch.sqrt(S_thin).reciprocal().reshape(-1, 1)  # m x 1

    # diag(S_root_inv) @ U.T
    E = S_thin_inv * U_thin.T   # square root inverse (m x n)
    E = E @ K
    # the diagonal entries of XX'(X'X + lam*S^(-2))^(-1)XX' are just the squared
    # ell-2 norm of the columns of (X'X + lam*S^(-2))^(-1/2)XX'
    tau = (torch.diagonal(K) - torch.square(E).sum(0)) / (penalty * n)
    assert torch.all(tau >= 0)
    d_eff = tau.sum()
    return d_eff


@torch.jit.script
def my_cdist(x1, x2):
    x1_norm = torch.norm(x1, p=2, dim=-1, keepdim=True).pow(2)
    x2_norm = torch.norm(x2, p=2, dim=-1, keepdim=True).pow(2)
    res = torch.addmm(x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
    res = res.clamp_min_(1e-30)
    return res


def gauss_effective_dimension(kernel_args, penalty, X, t, deterministic=False):
    return GaussEffectiveDimension.apply(kernel_args, penalty, X, t, deterministic)


# noinspection PyMethodOverriding
class GaussEffectiveDimension(torch.autograd.Function):
    NUM_DIFF_ARGS = 3

    @staticmethod
    @torch.jit.script
    def _naive_torch_gaussian_kernel(X1, X2, sigma):
        pairwise_dists = my_cdist(X1 / sigma, X2 / sigma)
        return torch.exp(-0.5 * pairwise_dists)

    @staticmethod
    def _cholesky_solve(L, b):
        y = trsm(b, L, alpha=1.0, lower=True, transpose=0)
        return trsm(y, L, alpha=1.0, lower=True, transpose=1)

    @staticmethod
    def forward(ctx, kernel_args, penalty, X, t, deterministic):
        n = X.shape[0]
        if deterministic:
            torch.manual_seed(12)
        Z = torch.randn(t, n, dtype=X.dtype, device=X.device).T  # n x t

        with torch.autograd.enable_grad():
            # Using naive or ours is equivalent, as long as splitting doesn't occur?
            K = GaussEffectiveDimension._naive_torch_gaussian_kernel(X, X, kernel_args).T
            KZ = K @ Z   # n x t
        with torch.autograd.no_grad():
            _penalty = penalty * n                # Rescaled penalty
            Keye = K + torch.diag_embed(_penalty.expand(n))
            if K.is_cuda:
                from falkon.ooc_ops.ooc_potrf import gpu_cholesky
                K_chol = gpu_cholesky(Keye, upper=False, clean=False, overwrite=True, opt=FalkonOptions())
            else:
                K_chol = potrf(Keye, upper=False, clean=False, overwrite=True, cuda=Keye.is_cuda)
            KinvZ = GaussEffectiveDimension._cholesky_solve(K_chol, Z)
            ctx.save_for_backward(kernel_args, penalty, X)
            ctx.K_chol = K_chol
            ctx.KinvZ = KinvZ
            ctx.KZ = KZ
            ctx.K = K
            return (KZ * KinvZ).sum(0).mean()

    @staticmethod
    def backward(ctx, grad_output):
        """
        Z.T @ K' @ Kinv @ Z - Z.T @ K @ Kinv @ Kl' @ Kinv @ Z

        where Kl' @ Kinv @ Z = K' @ Kinv @ Z + l' * Kinv @ Z
        """
        kernel_args, penalty, X = ctx.saved_tensors
        with torch.autograd.no_grad():
            KinvKZ = GaussEffectiveDimension._cholesky_solve(ctx.K_chol, ctx.KZ)

        with torch.autograd.enable_grad():
            _penalty = X.shape[0] * penalty
            loss_p1 = (ctx.KZ * ctx.KinvZ).sum(0).mean()
            loss_p2 = (KinvKZ * (ctx.K @ ctx.KinvZ + _penalty * ctx.KinvZ)).sum(0).mean()
            loss = grad_output * (loss_p1 - loss_p2)

        needs_grad = []
        for i in range(GaussEffectiveDimension.NUM_DIFF_ARGS):
            if ctx.needs_input_grad[i]:
                needs_grad.append(ctx.saved_tensors[i])
        grads = torch.autograd.grad(loss, needs_grad, retain_graph=True)
        result = []
        j = 0
        for i in range(len(ctx.needs_input_grad)):
            if ctx.needs_input_grad[i]:
                result.append(grads[j])
                j += 1
            else:
                result.append(None)
        return tuple(result)


def gauss_nys_effective_dimension(kernel_args, penalty, M, X, t, deterministic=False, preconditioner=None):
    opt = FalkonOptions(keops_active="no")
    K = GaussianKernel(kernel_args.detach(), opt=opt)
    if preconditioner is None:
        preconditioner = FalkonPreconditioner(penalty.detach(), K, opt)
        preconditioner.init(M.detach())
    optim = FalkonConjugateGradient(K, preconditioner, opt)

    return GaussNysEffectiveDimension.apply(kernel_args, penalty, M, X, t, deterministic, optim, preconditioner)


# noinspection PyMethodOverriding
class GaussNysEffectiveDimension(torch.autograd.Function):
    NUM_DIFF_ARGS = 3
    MAX_ITER = 10

    @staticmethod
    @torch.jit.script
    def _naive_torch_gaussian_kernel(X1, X2, sigma):
        pairwise_dists = my_cdist(X1 / sigma, X2 / sigma)
        return torch.exp(-0.5 * pairwise_dists)

    @staticmethod
    def forward(ctx, kernel_args, penalty, M, X, t, deterministic, optim: FalkonConjugateGradient, precond):
        n = X.shape[0]
        if deterministic:
            torch.manual_seed(12)
        Z = torch.randn(t, n, dtype=X.dtype, device=X.device).T  # n x t

        # TODO: flk already has args, penalty, M
        with torch.autograd.no_grad():
            beta = optim.solve(X, M, Z, penalty, initial_solution=None,
                               max_iter=GaussNysEffectiveDimension.MAX_ITER,
                               callback=None)
            f1 = precond.apply(beta)

        with torch.autograd.enable_grad():
            f2 = GaussNysEffectiveDimension._naive_torch_gaussian_kernel(M, X, kernel_args) @ Z

            ctx.save_for_backward(kernel_args, penalty, M)
            ctx.f1 = f1
            ctx.f2 = f2
            ctx.X = X
            return (f1 * f2).sum(0).mean()

    @staticmethod
    def backward(ctx, grad_output):
        X, f1, f2 = ctx.X, ctx.f1, ctx.f2
        n = X.shape[0]
        kernel_args, penalty, M = ctx.saved_tensors

        with torch.autograd.enable_grad():
            l1 = (f2 * f1).sum(0).mean()
            K_nm = GaussNysEffectiveDimension._naive_torch_gaussian_kernel(X, M, kernel_args)
            K_mm = GaussNysEffectiveDimension._naive_torch_gaussian_kernel(M, M, kernel_args)
            l2_p1 = K_nm @ f1
            l2_p2 = K_mm @ f1
            l2 = (torch.square(l2_p1)).sum(0).mean() + penalty * n * (f1 * l2_p2).sum(0).mean()

            loss = grad_output * (2 * l1 - l2)

        needs_grad = []
        for i in range(GaussNysEffectiveDimension.NUM_DIFF_ARGS):
            if ctx.needs_input_grad[i]:
                needs_grad.append(ctx.saved_tensors[i])
        grads = torch.autograd.grad(loss, needs_grad, retain_graph=True, allow_unused=True)
        result = []
        j = 0
        for i in range(len(ctx.needs_input_grad)):
            if ctx.needs_input_grad[i]:
                result.append(grads[j])
                j += 1
            else:
                result.append(None)
        return tuple(result)


def loss_and_deff(kernel_args, penalty, M, X, Y, t, deterministic=False, preconditioner=None):
    opt = FalkonOptions(keops_active="no")
    K = GaussianKernel(kernel_args.detach(), opt=opt)
    if preconditioner is None:
        preconditioner = FalkonPreconditioner(penalty.detach(), K, opt)
        preconditioner.init(M.detach())
    optim = FalkonConjugateGradient(K, preconditioner, opt)

    return LossAndDeff.apply(kernel_args, penalty, M, X, Y, t, deterministic, optim, preconditioner)


# noinspection PyMethodOverriding
class LossAndDeff(torch.autograd.Function):
    NUM_DIFF_ARGS = 3
    MAX_ITER = 10

    @staticmethod
    @torch.jit.script
    def _naive_torch_gaussian_kernel(X1, X2, sigma):
        pairwise_dists = my_cdist(X1 / sigma, X2 / sigma)
        return torch.exp(-0.5 * pairwise_dists)

    @staticmethod
    def forward(ctx, kernel_args, penalty, M, X, Y, t, deterministic, optim: FalkonConjugateGradient, precond):
        n = X.shape[0]
        if deterministic:
            torch.manual_seed(12)
        Z = torch.randn(t, n, dtype=X.dtype, device=X.device)  # t x n
        ZY = torch.cat((Z, Y.T), dim=0).T  # n x t+p

        with torch.autograd.no_grad():
            beta = optim.solve(X.detach(), M.detach(), ZY.detach(), penalty.detach(), initial_solution=None,
                               max_iter=LossAndDeff.MAX_ITER,
                               callback=None)
            sol_full = precond.apply(beta)  # alpha, eta

        with torch.autograd.enable_grad():
            # K_MN @ ZY => m x t+p
            K_MN = LossAndDeff._naive_torch_gaussian_kernel(M, X, kernel_args)
            k_vec_full = K_MN @ ZY  # d, e

        # Trace terms
        with torch.autograd.enable_grad():
            K_MM = LossAndDeff._naive_torch_gaussian_kernel(M, M, kernel_args)

        with torch.autograd.no_grad():
            T = torch.cholesky(K_MM, upper=True)  # T.T @ T = K_MM
            g = torch.triangular_solve(k_vec_full[:, :t], T, transpose=True, upper=True).solution


        # TODO: Useful?
        with torch.autograd.no_grad():
            Y_tilde = K_MN.T @ f1#[:, t:]       # KNM @ alpha  (n x p)
            beta = optim.solve(X.detach(), M.detach(), Y_tilde, penalty.detach(), initial_solution=None,
                               max_iter=LossAndDeff.MAX_ITER,
                               callback=None)  # precond(H^-1 @ KMN @ KNM @ alpha)
            alpha_tilde = precond.apply(beta)  # => m x t+p  (needed in backward)

        ctx.save_for_backward(kernel_args, penalty, M)
        ctx.f1 = f1
        ctx.f2 = f2
        ctx.X = X
        ctx.t = t
        ctx.Y_tilde = Y_tilde
        ctx.alpha_tilde = alpha_tilde

        d_eff = (k_vec_full[:,:t] * sol_full[:, :t]).sum(0).mean()

        d_eff = 2 * (f1[:, :t] * f2[:, :t]).sum(0).mean()
        #d_eff = (
        #    2 * (f1[:, :t] * f2[:, :t]).sum(0).mean() -            # normal_deff
        #        (Y_tilde[:, :t] * Y_tilde[:, :t]).sum(0).mean()  # full_deff
        #)
        l_t1 = (Y_tilde[:, t:] * Y_tilde[:, t:]).sum(0).mean()            # Y_tilde^T @ Y_tilde
        l_t2 = - 2 * (f2[:, t:] * f1[:, t:]).sum(0).mean()  # -2 * Y^T @ KNM @ alpha
        l_t3 = (Y * Y).sum(0).mean()                        # Y^T @ Y
        loss = l_t1 + l_t2 + l_t3
        print(f"loss {loss / n:.2f} - denom {(n - d_eff)**2 / n**2:.2f} - deff {d_eff:.2f}")
        out = (loss / n) / ((n - d_eff)**2 / n**2)
        return out

    @staticmethod
    def backward(ctx, out):#_deff, out_loss):
        X, f1, f2, t, Y_tilde, alpha_tilde = ctx.X, ctx.f1, ctx.f2, ctx.t, ctx.Y_tilde, ctx.alpha_tilde
        n = X.shape[0]
        kernel_args, penalty, M = ctx.saved_tensors

        with torch.autograd.enable_grad():
            K_nm = LossAndDeff._naive_torch_gaussian_kernel(X, M, kernel_args)
            K_mm = penalty * n * LossAndDeff._naive_torch_gaussian_kernel(M, M, kernel_args)
            l1_dot = (f1 * f2).sum(0)                              # alpha^T @ KNM^T @ Y
            l1_deff = l1_dot[:t].mean()

            l2_p1 = K_nm @ f1
            l2_p2 = K_mm @ f1
            l2_p1_dot = (torch.square(l2_p1)).sum(0)               # alpha^T @ KNM^T @ KNM @ alpha
            l2_p2_dot = (f1 * l2_p2).sum(0)                        # alpha^T @ (n*l*KMM) @ alpha
            l2_deff = l2_p1_dot[:t].mean() + l2_p2_dot[:t].mean()  # alpha^T @ H @ alpha

            #deff_bg = out_deff * (
            #    2 * (2 * l1_deff - l2_deff) -
            #    2 * (
            #        (f2[:, :t] * alpha_tilde[:, :t]).sum(0) +
            #        (l2_p1[:, :t] * Y_tilde[:, :t]).sum(0) -
            #        (l2_p2[:, :t] * alpha_tilde[:, :t]).sum(0) -
            #        (l2_p1[:, :t] * (K_nm @ alpha_tilde[:, :t])).sum(0)
            #    ).mean()
            #)
            deff_bg = 2 * l1_deff - l2_deff

            loss_bg = (
                2 * (  # c-term: grad(alpha^T @ KNM^T @ KNM @ alpha)
                    (f2[:, t:] * alpha_tilde[:, t:]).sum(0) +             # Y^T @ KNM @ alpha_tilde
                    (l2_p1[:, t:] * Y_tilde[:, t:]).sum(0) -              # alpha^T @ KNM^T @ Y_tilde
                    (l2_p2[:, t:] * alpha_tilde[:, t:]).sum(0) -          # alpha^T @ (n*l*KMM) @ alpha_tilde
                    (l2_p1[:, t:] * (K_nm @ alpha_tilde[:, t:])).sum(0)   # alpha^T @ KNM.T @ KNM @ alpha_tilde
                ).mean() -
                2 * (  # b-term: grad(-2 * Y^T @ KNM @ alpha)
                    2 * l1_dot[t:].mean() -                        # 2 * Y^T @ KNM @ alpha
                    (l2_p1_dot[t:].mean() + l2_p2_dot[t:].mean())  # alpha^T @ H @ alpha
                )
            )
            bg = (loss_bg / n) / ((n - deff_bg)**2 / n**2)
            #bg = deff_bg + loss_bg

        needs_grad = []
        for i in range(LossAndDeff.NUM_DIFF_ARGS):
            if ctx.needs_input_grad[i]:
                needs_grad.append(ctx.saved_tensors[i])
        grads = torch.autograd.grad(bg, needs_grad, retain_graph=True, allow_unused=True)
        result = []
        j = 0
        for i in range(len(ctx.needs_input_grad)):
            if ctx.needs_input_grad[i]:
                result.append(grads[j])
                j += 1
            else:
                result.append(None)
        return tuple(result)


def sgpr_trace(kernel_args, penalty, M, X, t, deterministic=False):
    return SGPRTrace.apply(kernel_args, penalty, M, X, t, deterministic)


# noinspection PyMethodOverriding
class SGPRTrace(torch.autograd.Function):
    NUM_DIFF_ARGS = 3

    @staticmethod
    def _cholesky_solve(L, b):
        y = trsm(b, L, alpha=1.0, lower=True, transpose=0)
        return trsm(y, L, alpha=1.0, lower=True, transpose=1)

    @staticmethod
    def forward(ctx, kernel_args, penalty, M, X, t, deterministic):
        n = X.shape[0]
        if n <= 30_000:
            keops = "no"
        else:
            keops = "force"
        kernel = DiffGaussianKernel(kernel_args, opt=FalkonOptions(keops_active=keops))
        if deterministic:
            torch.manual_seed(12)

        Z = torch.randn(n, t, dtype=X.dtype, device=X.device)

        with torch.autograd.enable_grad():
            KMN_Z = kernel.mmv(M, X, Z)
            # Note that KNN_Z needs to be differentiable wrt kernel_args!
            KNN_Z = kernel.mmv(X, X, Z)  # TODO: This seems super expensive, can probably specialize taking the diagonal of the kernel easily.

            mul_const = 1 / (torch.exp(-penalty))

        # Solve beta = K_MM^{-1} @ K_NM.T @ Z
        with torch.autograd.no_grad():
            K_MM = kernel(M, M)
            if K_MM.is_cuda:
                from falkon.ooc_ops.ooc_potrf import gpu_cholesky
                K_chol = gpu_cholesky(K_MM, upper=True, clean=False, overwrite=True, opt=FalkonOptions()).T
            else:
                K_chol = potrf(K_MM, upper=False, clean=False, overwrite=True, cuda=Keye.is_cuda)
            #KMN_Z_F = torch.empty(KMN_Z.shape[1], KMN_Z.shape[0], device=KMN_Z.device, dtype=KMN_Z.dtype)
            #KMN_Z_F.copy_(KMN_Z.T)
            beta = SGPRTrace._cholesky_solve(K_chol, KMN_Z)

        with torch.autograd.enable_grad():
            tr_1 = (Z * KNN_Z).sum(0).mean()
            tr_2 = (KMN_Z * beta).sum(0).mean()

        ctx.save_for_backward(kernel_args, penalty, M)
        ctx.beta = beta
        ctx.fwd_1 = tr_1
        ctx.fwd_2 = tr_2
        ctx.kernel = kernel
        ctx.mul_const = mul_const

        return mul_const * (tr_1 - tr_2)

    @staticmethod
    def backward(ctx, grad_out):
        beta, fwd_1, fwd_2, mul_const, kernel = ctx.beta, ctx.fwd_1, ctx.fwd_2, ctx.mul_const, ctx.kernel
        kernel_args, penalty, M = ctx.saved_tensors

        with torch.autograd.enable_grad():
            loss_1 = fwd_1
            loss_2 = - 2 * fwd_2
            loss_3 = (beta * kernel.mmv(M, M, beta)).sum(0).mean()
            loss = grad_out * mul_const * (loss_1 + loss_2 + loss_3)

        needs_grad = []
        for i in range(SGPRTrace.NUM_DIFF_ARGS):
            if ctx.needs_input_grad[i]:
                needs_grad.append(ctx.saved_tensors[i])
        grads = torch.autograd.grad(loss, needs_grad)

        result = []
        j = 0
        for i in range(len(ctx.needs_input_grad)):
            if ctx.needs_input_grad[i]:
                result.append(grads[j])
                j += 1
            else:
                result.append(None)
        return tuple(result)


def regloss_and_deff(
            kernel_args: torch.Tensor,
            penalty: torch.Tensor,
            M: torch.Tensor,
            X: torch.Tensor,
            Y: torch.Tensor,
            t: int,
            deterministic: bool = False,
            use_precise_trace: bool = True,
            solve_options: FalkonOptions = None,
            solve_maxiter: int = 10,
            gaussian_random: bool = True,
            ):
    """

    Parameters
    ----------
    gaussian_random
    kernel_args
    penalty
    M
    X
    Y
    t
    deterministic
    use_precise_trace
    solve_options
    solve_maxiter

    Returns
    -------
    (d_eff, loss, trace) : Tuple[float]
        The three terms making up complexity-regularized loss
    """
    if solve_options is None:
        solve_options = FalkonOptions(
                cg_tolerance=1e-4,  # default is 1e-7
                cg_full_gradient_every=10,  # default is 10
                pc_epsilon_32=1e-6,  # default is 1e-5
                cg_epsilon_32=1e-7,  # default is 1e-7
            )

    return RegLossAndDeff.apply(
        kernel_args,
        penalty,
        M, X, Y, t,
        deterministic,
        use_precise_trace,
        solve_options,
        solve_maxiter,
        gaussian_random,
    )


# noinspection PyMethodOverriding
class RegLossAndDeff(torch.autograd.Function):
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
            use_precise_trace: bool,
            solve_options: FalkonOptions,
            solve_maxiter: int,
            gaussian_random: bool,
            ):
        if X.shape[0] <= 30_000:
            keops = "no"
        else:
            keops = "force"
        diff_kernel = DiffGaussianKernel(
            kernel_args,
            opt=FalkonOptions(keops_active=keops))

        n = X.shape[0]
        if deterministic:
            torch.manual_seed(12)
        if gaussian_random:
            Z = torch.randn(n, t, dtype=X.dtype, device=X.device)
        else:
            Z = torch.empty(n, t, dtype=X.dtype, device=X.device).bernoulli_().mul_(2).sub_(1)
        ZY = torch.cat((Z, Y), dim=1)
        _penalty = torch.exp(-penalty)

        ### Solve Falkon
        with torch.autograd.no_grad():
            M_ = M.detach()
            kernel_args_ = kernel_args.detach()
            K = GaussianKernel(kernel_args_, opt=solve_options)  # here which opt doesnt matter
            precond = FalkonPreconditioner(_penalty.item(), K, solve_options)  # here which opt doesnt matter
            precond.init(M_)
            optim = FalkonConjugateGradient(K, precond, solve_options)
            beta = optim.solve(
                X, M_, ZY, _penalty.item(),
                initial_solution=None,
                max_iter=solve_maxiter,
            )
            sol_full = precond.apply(beta)  # eta, alpha
            RegLossAndDeff.last_alpha = sol_full[:, t:].clone()
        ### End Falkon Solve

        ### K_MN @ vector
        with torch.autograd.enable_grad():
            if use_precise_trace:
                K_MN = full_rbf_kernel(M, X, kernel_args)
                k_vec_full = K_MN @ ZY  # d, e
            else:
                k_vec_full = diff_kernel.mmv(M, X, ZY)
                K_MN = None
        ### End large K-vec product

        ### Trace terms
        with torch.autograd.enable_grad():
            K_MM = full_rbf_kernel(M, M, kernel_args)

        mm_eye = torch.eye(K_MM.shape[0], device=K_MM.device, dtype=K_MM.dtype)
        if use_precise_trace:
            # This is old-version for when trace is calculated directly.
            # This trace calculation is very precise.
            with torch.autograd.enable_grad():
                T = torch.cholesky(K_MM + mm_eye * RegLossAndDeff.EPS, upper=True)  # T.T @ T = K_MM
                g = torch.triangular_solve(K_MN, T, transpose=True, upper=True).solution  # g == A  M*N
        else:
            # Trace is calculated using hutchinson. Not very accurate!
            with torch.autograd.no_grad():
                T = torch.cholesky(K_MM + mm_eye * RegLossAndDeff.EPS, upper=True)  # T.T @ T = K_MM
                g1 = torch.triangular_solve(k_vec_full[:, :t], T, upper=True, transpose=True).solution
                g = torch.triangular_solve(g1, T, upper=True, transpose=False).solution
        ### End Trace

        d_eff = - (k_vec_full[:, :t] * sol_full[:, :t]).sum(0).mean()
        loss = - torch.square(Y).sum()
        loss += (k_vec_full[:, t:] * sol_full[:, t:]).sum(0).mean()
        if use_precise_trace:
            trace = - (1 / (_penalty) - torch.square(g).sum() / (_penalty * n))
        else:
            trace = - (1 / (_penalty) - torch.square(g).sum(0).mean() / (_penalty * n))

        ctx.save_for_backward(kernel_args, penalty, M)
        ctx.sol_full = sol_full
        ctx.K_MN = K_MN
        ctx.k_vec_full = k_vec_full
        ctx.K_MM = K_MM
        ctx.T = T
        ctx.g = g
        ctx.t = t
        ctx.diff_kernel = diff_kernel
        ctx.X = X
        ctx.use_precise_trace = use_precise_trace

        return d_eff, loss, trace

    @staticmethod
    def backward(ctx, out_deff, out_loss, out_trace):
        sol_full, K_MN, k_vec_full, K_MM, T, g, t, diff_kernel, X, use_precise_trace = (
            ctx.sol_full,
            ctx.K_MN,
            ctx.k_vec_full,
            ctx.K_MM,
            ctx.T,
            ctx.g,
            ctx.t,
            ctx.diff_kernel,
            ctx.X,
            ctx.use_precise_trace,
        )
        n = X.shape[0]
        kernel_args, penalty, M = ctx.saved_tensors

        with torch.autograd.enable_grad():
            _penalty = torch.exp(-penalty)

            if use_precise_trace:
                KNM_sol_full = K_MN.T @ sol_full
            else:
                KNM_sol_full = diff_kernel.mmv(X, M, sol_full)
            KMM_sol_full = (_penalty * n * K_MM) @ sol_full

            deff_bg = -out_deff * (
                2 * (k_vec_full[:, :t] * sol_full[:, :t]).sum(0).mean()
                - (torch.square(KNM_sol_full[:, :t]).sum(0) + (sol_full[:, :t] * KMM_sol_full[:, :t]).sum(0)).mean()
            )
            loss_bg = out_loss * (
                2 * (k_vec_full[:, t:] * sol_full[:, t:]).sum(0).mean()
                - (torch.square(KNM_sol_full[:, t:]).sum(0) + (sol_full[:, t:] * KMM_sol_full[:, t:]).sum(0)).mean()
            )
            if use_precise_trace:
                # Normal trace with A @ A.T
                trace_bg = -out_trace * (
                    1 / (_penalty) -
                    torch.square(g).sum() / (_penalty * n)
                )
            else:
                # Trace with stoch trace est
                trace_bg = -out_trace * (
                    1 / _penalty -
                    (
                        2 * (k_vec_full[:, :t] * g).sum(0).mean()
                        - torch.square(T @ g).sum(0).mean()
                    ) / (_penalty * n)
                )

            bg = deff_bg + loss_bg + trace_bg

        needs_grad = []
        for i in range(LossAndDeff.NUM_DIFF_ARGS):
            if ctx.needs_input_grad[i]:
                needs_grad.append(ctx.saved_tensors[i])
        grads = torch.autograd.grad(bg, needs_grad, retain_graph=True)

        result = []
        j = 0
        for i in range(len(ctx.needs_input_grad)):
            if ctx.needs_input_grad[i]:
                result.append(grads[j])
                j += 1
            else:
                result.append(None)
        return tuple(result)

