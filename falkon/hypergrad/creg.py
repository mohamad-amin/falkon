import torch

from falkon import FalkonOptions
from falkon.hypergrad.common import full_rbf_kernel, get_scalar, cholesky
from falkon.hypergrad.complexity_reg import NystromKRRModelMixinN, HyperOptimModel
from falkon.hypergrad.leverage_scores import (
    creg_penfit, RegLossAndDeffv2, creg_plainfit,
    NoRegLossAndDeff
)
from falkon.kernels import GaussianKernel


class StochasticDeffPenFitTr(NystromKRRModelMixinN, HyperOptimModel):
    def __init__(
            self,
            sigma_init,
            penalty_init,
            centers_init,
            opt_centers,
            opt_sigma,
            opt_penalty,
            cuda: bool,
            flk_opt: FalkonOptions,
            num_trace_est: int = 20,
            flk_maxiter: int = 10,
            nystrace_ste: bool = False,
    ):
        super().__init__(
            penalty=penalty_init,
            sigma=sigma_init,
            centers=centers_init,
            cuda=cuda,
            verbose=True,
        )
        self.opt_sigma, self.opt_centers, self.opt_penalty = opt_sigma, opt_centers, opt_penalty
        if opt_sigma:
            self.register_parameter("sigma", self.sigma_.requires_grad_(True))
        if opt_penalty:
            self.register_parameter("penalty", self.penalty_.requires_grad_(True))
        if opt_centers:
            self.register_parameter("centers", self.centers_.requires_grad_(True))

        self.flk_opt = flk_opt
        self.num_trace_est = num_trace_est
        self.flk_maxiter = flk_maxiter
        self.nystrace_ste = nystrace_ste

    def hp_loss(self, X, Y):
        loss = creg_penfit(kernel_args=self.sigma, penalty=self.penalty, centers=self.centers,
                           X=X, Y=Y, num_estimators=self.num_trace_est, deterministic=False,
                           solve_options=self.flk_opt, solve_maxiter=self.flk_maxiter,
                           gaussian_random=False, use_stoch_trace=self.nystrace_ste)
        return [loss]

    def predict(self, X):
        if RegLossAndDeffv2.last_alpha is None:
            raise RuntimeError("Call hp_loss before calling predict.")
        alpha = RegLossAndDeffv2.last_alpha
        kernel = GaussianKernel(self.sigma, opt=self.flk_opt)
        return kernel.mmv(X, self.centers, alpha)

    @property
    def loss_names(self):
        return "stoch-creg-penfit"

    def __repr__(self):
        return f"StochasticDeffPenFitTr(sigma={get_scalar(self.sigma)}, penalty={get_scalar(self.penalty)}, " \
               f"num_centers={self.centers.shape[0]}, opt_centers={self.opt_centers}, " \
               f"opt_sigma={self.opt_sigma}, opt_penalty={self.opt_penalty}, t={self.num_trace_est}, " \
               f"flk_iter={self.flk_maxiter})"


class DeffPenFitTr(NystromKRRModelMixinN, HyperOptimModel):
    def __init__(
            self,
            sigma_init,
            penalty_init,
            centers_init,
            opt_centers,
            opt_sigma,
            opt_penalty,
            cuda: bool,
    ):
        super().__init__(
            penalty=penalty_init,
            sigma=sigma_init,
            centers=centers_init,
            cuda=cuda,
            verbose=True,
        )
        self.opt_sigma, self.opt_centers, self.opt_penalty = opt_sigma, opt_centers, opt_penalty
        if opt_sigma:
            self.register_parameter("sigma", self.sigma_.requires_grad_(True))
        if opt_penalty:
            self.register_parameter("penalty", self.penalty_.requires_grad_(True))
        if opt_centers:
            self.register_parameter("centers", self.centers_.requires_grad_(True))

        self.L, self.LB, self.c = None, None, None

    def hp_loss(self, X, Y):
        variance = self.penalty * X.shape[0]
        sqrt_var = torch.sqrt(variance)
        Kdiag = X.shape[0]

        m = self.centers.shape[0]
        kmn = full_rbf_kernel(self.centers, X, self.sigma)
        kmm = (full_rbf_kernel(self.centers, self.centers, self.sigma) +
               torch.eye(m, device=X.device, dtype=X.dtype) * 1e-6)
        self.L = cholesky(kmm)  # L @ L.T = kmm
        # A = L^{-1} K_mn / (sqrt(n*pen))
        A = torch.triangular_solve(kmn, self.L, upper=False).solution / sqrt_var
        AAT = A @ A.T  # m*n @ n*m = m*m in O(n * m^2), equivalent to kmn @ knm.
        # B = A @ A.T + I
        B = AAT + torch.eye(AAT.shape[0], device=X.device, dtype=X.dtype)
        self.LB = cholesky(B)  # LB @ LB.T = B
        AY = A @ Y  # m*1
        self.c = torch.triangular_solve(AY, self.LB, upper=False).solution / sqrt_var  # m*1

        C = torch.triangular_solve(A, self.LB, upper=False).solution  # m*n

        # Complexity (nystrom-deff)
        ndeff = C.square().sum()  # = torch.trace(C.T @ C)
        datafit = torch.square(Y).sum() - torch.square(self.c * sqrt_var).sum()
        trace = Kdiag - torch.trace(AAT) * variance

        return ndeff, datafit, trace

    def predict(self, X):
        if self.L is None or self.LB is None or self.c is None:
            raise RuntimeError("Call hp_loss before calling predict.")

        tmp1 = torch.triangular_solve(self.c, self.LB, upper=False, transpose=True).solution
        tmp2 = torch.triangular_solve(tmp1, self.L, upper=False, transpose=True).solution
        kms = full_rbf_kernel(self.centers, X, self.sigma)
        return kms.T @ tmp2

    @property
    def loss_names(self):
        return "nys-deff", "data-fit", "trace"

    def __repr__(self):
        return f"DeffPenFitTr(sigma={get_scalar(self.sigma)}, penalty={get_scalar(self.penalty)}, num_centers={self.centers.shape[0]}, " \
               f"opt_centers={self.opt_centers}, opt_sigma={self.opt_sigma}, opt_penalty={self.opt_penalty})"


class StochasticDeffNoPenFitTr(NystromKRRModelMixinN, HyperOptimModel):
    def __init__(
            self,
            sigma_init,
            penalty_init,
            centers_init,
            opt_centers,
            opt_sigma,
            opt_penalty,
            cuda: bool,
            flk_opt: FalkonOptions,
            num_trace_est: int = 20,
            flk_maxiter: int = 10,
            nystrace_ste: bool = False,
    ):
        super().__init__(
            penalty=penalty_init,
            sigma=sigma_init,
            centers=centers_init,
            cuda=cuda,
            verbose=True,
        )
        self.opt_sigma, self.opt_centers, self.opt_penalty = opt_sigma, opt_centers, opt_penalty
        if opt_sigma:
            self.register_parameter("sigma", self.sigma_.requires_grad_(True))
        if opt_penalty:
            self.register_parameter("penalty", self.penalty_.requires_grad_(True))
        if opt_centers:
            self.register_parameter("centers", self.centers_.requires_grad_(True))

        self.flk_opt = flk_opt
        self.num_trace_est = num_trace_est
        self.flk_maxiter = flk_maxiter
        self.nystrace_ste = nystrace_ste

    def hp_loss(self, X, Y):
        loss = creg_plainfit(kernel_args=self.sigma, penalty=self.penalty, centers=self.centers,
                             X=X, Y=Y, num_estimators=self.num_trace_est, deterministic=False,
                             solve_options=self.flk_opt, solve_maxiter=self.flk_maxiter,
                             gaussian_random=False, use_stoch_trace=self.nystrace_ste)
        return [loss]

    def predict(self, X):
        if NoRegLossAndDeff.last_alpha is None:
            raise RuntimeError("Call hp_loss before calling predict.")
        alpha = NoRegLossAndDeff.last_alpha
        kernel = GaussianKernel(self.sigma, opt=self.flk_opt)
        return kernel.mmv(X, self.centers, alpha)

    @property
    def loss_names(self):
        return "stoch-creg-plainfit"

    def __repr__(self):
        return f"StochasticDeffNoPenFitTr(sigma={get_scalar(self.sigma)}, penalty={get_scalar(self.penalty)}, " \
               f"num_centers={self.centers.shape[0]}, opt_centers={self.opt_centers}, " \
               f"opt_sigma={self.opt_sigma}, opt_penalty={self.opt_penalty}, t={self.num_trace_est}, " \
               f"flk_iter={self.flk_maxiter})"


class DeffNoPenFitTr(NystromKRRModelMixinN, HyperOptimModel):
    def __init__(
            self,
            sigma_init,
            penalty_init,
            centers_init,
            opt_centers,
            opt_sigma,
            opt_penalty,
            cuda: bool,
    ):
        super().__init__(
            penalty=penalty_init,
            sigma=sigma_init,
            centers=centers_init,
            cuda=cuda,
            verbose=True,
        )
        self.opt_sigma, self.opt_centers, self.opt_penalty = opt_sigma, opt_centers, opt_penalty
        if opt_sigma:
            self.register_parameter("sigma", self.sigma_.requires_grad_(True))
        if opt_penalty:
            self.register_parameter("penalty", self.penalty_.requires_grad_(True))
        if opt_centers:
            self.register_parameter("centers", self.centers_.requires_grad_(True))

        self.alpha = None

    def hp_loss(self, X, Y):
        variance = self.penalty * X.shape[0]
        sqrt_var = torch.sqrt(variance)
        Kdiag = X.shape[0]

        m = self.centers.shape[0]
        kmn = full_rbf_kernel(self.centers, X, self.sigma)
        kmm = (full_rbf_kernel(self.centers, self.centers, self.sigma) +
               torch.eye(m, device=X.device, dtype=X.dtype) * 1e-6)
        L = cholesky(kmm)  # L @ L.T = kmm
        # A = L^{-1} K_mn / (sqrt(n*pen))
        A = torch.triangular_solve(kmn, L, upper=False).solution / sqrt_var
        AAT = A @ A.T
        # B = A @ A.T + I
        B = AAT + torch.eye(AAT.shape[0], device=X.device, dtype=X.dtype)
        LB = cholesky(B)  # LB @ LB.T = B
        AY = A @ Y
        c = torch.triangular_solve(AY, LB, upper=False).solution / sqrt_var

        tmp1 = torch.triangular_solve(c, LB, upper=False, transpose=True).solution
        self.alpha = torch.triangular_solve(tmp1, L, upper=False, transpose=True).solution
        d = A.T @ tmp1

        C = torch.triangular_solve(A, LB, upper=False).solution

        # Complexity (nystrom-deff)
        ndeff = C.square().sum()  # = torch.trace(C.T @ C)
        datafit = torch.square(Y).sum() - 2 * torch.square(
            c * sqrt_var).sum() + variance * torch.square(d).sum()
        trace = Kdiag - torch.trace(AAT) * variance
        # trace = trace / variance  # TODO: This is a temporary addition!

        return ndeff, datafit, trace

    def predict(self, X):
        if self.alpha is None:
            raise RuntimeError("Call hp_loss before calling predict.")

        kms = full_rbf_kernel(self.centers, X, self.sigma)
        return kms.T @ self.alpha

    @property
    def loss_names(self):
        return "nys-deff", "data-fit", "trace"

    def __repr__(self):
        return f"DeffNoPenFitTr(sigma={get_scalar(self.sigma)}, penalty={get_scalar(self.penalty)}, num_centers={self.centers.shape[0]}, " \
               f"opt_centers={self.opt_centers}, opt_sigma={self.opt_sigma}, opt_penalty={self.opt_penalty})"
