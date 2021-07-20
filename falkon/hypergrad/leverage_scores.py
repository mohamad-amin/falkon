import dataclasses
from typing import Optional

import scipy.linalg
import torch

import falkon
from falkon import FalkonOptions
from falkon.hypergrad.common import full_rbf_kernel
from falkon.kernels import GaussianKernel
from falkon.kernels.diff_rbf_kernel import DiffGaussianKernel
from falkon.la_helpers import potrf, trsm
from falkon.optim import FalkonConjugateGradient
from falkon.preconditioner import FalkonPreconditioner

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
        return self._flk_solve_zy[:, self._len_z]

    @property
    def solve_y(self):
        if self._flk_solve_y is None:
            if self._flk_solve_zy is None:
                return None
            return self._flk_solve_zy[:, self._len_z:]
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
        return self._kmn_zy[:, self._len_z]

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
        return self._knm_solve_zy[:, self._len_z]

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
        return self._kmm_solve_zy[:, self._len_z]

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
    M_ = centers.detach()
    kernel_args_ = kernel_args.detach()
    K = GaussianKernel(kernel_args_, opt=solve_options)  # here which opt doesnt matter
    precond = FalkonPreconditioner(penalty.item(), K, solve_options)  # here which opt doesnt matter
    precond.init(M_)
    optim = FalkonConjugateGradient(K, precond, solve_options)
    beta = optim.solve(
        X, M_, rhs, penalty.item(),
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
    """
    STE with bernoulli Z (m x t)
    note Tr(knm @ Kmm^{-1} @ knm.T) = Tr(knm.T @ knm @ kmm^{-1}) = Tr(L^{-1} knm.T knm L^{-T}) so
    solve knm @ (L^{-T} @ Z)

    Z = torch.empty(M.shape[0], T, dtype=X.dtype, device=X.device).bernoulli_().mul_(2).sub_(1)
    part1 = torch.triangular_solve(Z, L, upper=False, transpose=True).solution.contiguous()
    part2 = kernel.mmv(X, M, part1)
    return part2.square().sum(0).mean()
    """
    @staticmethod
    def forward(ctx,
                kernel_args,
                M,
                X,
                t: int,
                deterministic: bool,
                gaussian_random: bool,
                ):
        diff_kernel = DiffGaussianKernel(
            kernel_args,
            opt=FalkonOptions(keops_active="no"))
        if deterministic:
            torch.manual_seed(22)
        Z = init_random_vecs(M.shape[0], t, dtype=X.dtype, device=X.device, gaussian_random=gaussian_random)
        with torch.autograd.enable_grad():
            kmm = full_rbf_kernel(M, M, kernel_args)
            L = torch.cholesky(kmm)
            part1 = torch.triangular_solve(Z, L, upper=False, transpose=True).solution.contiguous()
            part2 = diff_kernel.mmv(X, M, part1)

        ctx.save_for_backward(kernel_args, M)
        return part2.square().sum(0).mean()

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


def nystrom_kernel_trace(kernel_args, M, X):
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


def nystrom_effective_dim(kernel_args, penalty, M, X, Y, t, gaussian_random, solve_opt, solve_maxiter):
    return NystromEffectiveDimension.apply(
        kernel_args, penalty, M, X, Y, t, gaussian_random, solve_opt, solve_maxiter,
        NoRegLossAndDeffCtx(t), False)


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
        loss /= X.shape[0]

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
            ) / X.shape[0]
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

        loss = Y.square().sum()
        loss -= (data.kmn_y * data.solve_y).sum(0).mean()
        loss /= X.shape[0]

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
            ) / X.shape[0]
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
            ):
        data = NoRegLossAndDeffCtx(t)

        with torch.autograd.enable_grad():
            data.data_fit = nystrom_data_fit(
                kernel_args, penalty, M, X, Y, t, deterministic, solve_options, solve_maxiter,
                gaussian_random, data)
            # Trace of the Nystrom kernel: Tr(KNM @ KMM^{-1} @ KNM.T) Prolem here: formation of N*M matrix!
            data.nys_trace = nystrom_kernel_trace(kernel_args, M, X)
            trace = X.shape[0] - data.nys_trace
            # D-Eff = Tr((KNM @ KMM^{-1} @ KNM.T + lambda I)^{-1} KNM @ KMM^{-1} @ KNM.T)
            data.nys_d_eff = nystrom_effective_dim_wdata(kernel_args, penalty, M, X, data)

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


# noinspection PyMethodOverriding
class GCV(torch.autograd.Function):
    """
    Numerator: Exactly the data-fit term of NoRegLossAndDeff (TODO: Extract into its own function)
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

        ctx.save_for_backward(kernel_args, penalty, M)
        ctx.data = data
        return data.data_fit / data.nys_d_eff

    @staticmethod
    def backward(ctx, out):
        data = ctx.data
        with torch.autograd.enable_grad():
            bg = out * (data.data_fit / data.nys_d_eff)
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
                GCV.apply(sigma, pen, centers, X, Y, 20, True, FalkonOptions(), 30, False),
            (s, p, M))


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
            ):
        data = NoRegLossAndDeffCtx(t)

        with torch.autograd.enable_grad():
            # D-eff calculation initializes the data object.
            # D-Eff = Tr((KNM @ KMM^{-1} @ KNM.T + lambda I)^{-1} KNM @ KMM^{-1} @ KNM.T)
            data.nys_d_eff = nystrom_effective_dim(kernel_args, penalty, M, X, Y, t, gaussian_random, solve_options, solve_maxiter)
            # Penalized data-fit
            data.data_fit = nystrom_penalized_data_fit(
                kernel_args, penalty, M, X, Y, solve_options, solve_maxiter, data)
            # Trace of the Nystrom kernel: Tr(KNM @ KMM^{-1} @ KNM.T) Prolem here: formation of N*M matrix!
            data.nys_trace = nystrom_kernel_trace(kernel_args, M, X)
            trace = X.shape[0] - data.nys_trace

        ctx.save_for_backward(kernel_args, penalty, M)
        ctx.data = data
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


# noinspection PyMethodOverriding
class RegLossAndDeff(torch.autograd.Function):
    # TODO: Rewrite this function using re-usable blocks. Use Tri-Inverse for Trace.
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
                # NOTE: K_MN can be a very large matrix.
                # We need to store all of it to enable the triangular-solve operation.
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
        for i in range(RegLossAndDeff.NUM_DIFF_ARGS):
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

