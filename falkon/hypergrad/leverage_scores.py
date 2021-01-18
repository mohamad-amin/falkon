import time

import torch
from falkon.center_selection import FixedSelector

from falkon import Falkon
from falkon.models.model_utils import FalkonBase

from falkon.kernels import GaussianKernel

import falkon
from falkon.options import FalkonOptions
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
        fwd_start = time.time()
        n = X.shape[0]
        if deterministic:
            torch.manual_seed(12)
        Z = torch.randn(t, n, dtype=X.dtype, device=X.device).T  # n x t

        with torch.autograd.enable_grad():
            # Using naive or ours is equivalent, as long as splitting doesn't occur?
            K = GaussEffectiveDimension._naive_torch_gaussian_kernel(X, X, kernel_args).T
            # K = GaussianKernel(kernel_args)(X, X).T
            KZ = K @ Z   # n x t
            kz_done = time.time()
        with torch.autograd.no_grad():
            _penalty = penalty * n                # Rescaled penalty
            Keye = K + torch.diag_embed(_penalty.expand(n))
            chol_start = time.time()
            if K.is_cuda:
                from falkon.ooc_ops.ooc_potrf import gpu_cholesky
                from falkon.options import FalkonOptions
                K_chol = gpu_cholesky(Keye, upper=False, clean=False, overwrite=True, opt=FalkonOptions())
            else:
                K_chol = potrf(Keye, upper=False, clean=False, overwrite=True, cuda=Keye.is_cuda)
            chol_end = time.time()
            KinvZ = GaussEffectiveDimension._cholesky_solve(K_chol, Z)
            trsm_end = time.time()
            ctx.save_for_backward(kernel_args, penalty, X)
            ctx.K_chol = K_chol
            ctx.KinvZ = KinvZ
            ctx.KZ = KZ
            ctx.K = K
            # print("HutchTrEst forward took %.2fs - kernel %.2fs - chol %.2fs - trsm %.2fs" %
            #       (time.time() - fwd_start, kz_done - fwd_start, chol_end - chol_start, trsm_end - chol_end))
            return (KZ * KinvZ).sum(0).mean()

    @staticmethod
    def backward(ctx, grad_output):
        """
        Z.T @ K' @ Kinv @ Z - Z.T @ K @ Kinv @ Kl' @ Kinv @ Z

        where Kl' @ Kinv @ Z = K' @ Kinv @ Z + l' * Kinv @ Z
        """
        kernel_args, penalty, X = ctx.saved_tensors
        bwd_start = time.time()
        with torch.autograd.no_grad():
            KinvKZ = GaussEffectiveDimension._cholesky_solve(ctx.K_chol, ctx.KZ)

        with torch.autograd.enable_grad():
            _penalty = X.shape[0] * penalty
            loss_p1 = (ctx.KZ * ctx.KinvZ).sum(0).mean()
            loss_p2 = KinvKZ * (ctx.K @ ctx.KinvZ + _penalty * ctx.KinvZ)

            loss_p2 = loss_p2.sum(0).mean()
            loss = grad_output * (loss_p1 - loss_p2)
        bwd_mid = time.time()

        needs_grad = []
        for i in range(GaussEffectiveDimension.NUM_DIFF_ARGS):
            if ctx.needs_input_grad[i]:
                needs_grad.append(ctx.saved_tensors[i])
        grads = torch.autograd.grad(loss, needs_grad, retain_graph=True)
        # print("HutchTrEst backward took %.2fs + %.2fs" % (bwd_mid - bwd_start, time.time() - bwd_mid))
        result = []
        j = 0
        for i in range(len(ctx.needs_input_grad)):
            if ctx.needs_input_grad[i]:
                result.append(grads[j])
                j += 1
            else:
                result.append(None)
        return tuple(result)


def gauss_nys_effective_dimension(kernel_args, penalty, M, X, t, deterministic=False):
    opt = FalkonOptions(keops_active="no")
    K = GaussianKernel(kernel_args, opt=opt)
    flk = Falkon(kernel=K, penalty=penalty, M=M.shape[0],
                 center_selection=FixedSelector(M), maxiter=10, options=opt)

    return GaussNysEffectiveDimension.apply(kernel_args, penalty, M, X, t, deterministic, flk)


# noinspection PyMethodOverriding
class GaussNysEffectiveDimension(torch.autograd.Function):
    NUM_DIFF_ARGS = 3

    @staticmethod
    @torch.jit.script
    def _naive_torch_gaussian_kernel(X1, X2, sigma):
        pairwise_dists = my_cdist(X1 / sigma, X2 / sigma)
        return torch.exp(-0.5 * pairwise_dists)

    @staticmethod
    def forward(ctx, kernel_args, penalty, M, X, t, deterministic, flk: 'FalkonBase'):
        n = X.shape[0]
        if deterministic:
            torch.manual_seed(12)
        Z = torch.randn(t, n, dtype=X.dtype, device=X.device).T  # n x t

        # TODO: flk already has args, penalty, M
        with torch.autograd.no_grad():
            f1 = flk.fit(X, Z).alpha_

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
