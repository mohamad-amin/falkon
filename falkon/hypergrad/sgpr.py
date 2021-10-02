import numpy as np
import torch

from falkon.hypergrad.common import full_rbf_kernel, get_scalar, cholesky
from falkon.hypergrad.complexity_reg import NystromKRRModelMixinN, KRRModelMixinN, HyperOptimModel


class GPR(KRRModelMixinN, HyperOptimModel):
    def __init__(
            self,
            sigma_init,
            penalty_init,
            opt_sigma,
            opt_penalty,
            cuda: bool
    ):
        super().__init__(
            penalty=penalty_init,
            sigma=sigma_init,
            cuda=cuda,
            verbose=True,
        )
        self.opt_sigma, self.opt_penalty = opt_sigma, opt_penalty
        if opt_sigma:
            self.register_parameter("sigma", self.sigma_.requires_grad_(True))
        if opt_penalty:
            self.register_parameter("penalty", self.penalty_.requires_grad_(True))
        self.L, self.data_X, self.data_Y = None, None, None

    def hp_loss(self, X, Y):
        knn = (full_rbf_kernel(X, X, self.sigma) +
               self.penalty * torch.eye(X.shape[0], dtype=X.dtype, device=X.device))
        self.L = cholesky(knn, upper=False)
        self.data_X, self.data_Y = X, Y
        alpha = torch.triangular_solve(Y, self.L, upper=False).solution
        datafit = -0.5 * torch.square(alpha).sum(0)
        complexity = - torch.log(torch.diag(self.L)).sum()
        const = -0.5 * X.shape[0] * torch.log(2 * torch.tensor(np.pi, dtype=X.dtype))

        return complexity, datafit, const

    def predict(self, X):
        if self.data_X is None or self.data_Y is None or self.L is None:
            raise RuntimeError("Call hp_loss before calling predict.")
        Kmn = full_rbf_kernel(self.data_X, X)
        A = torch.triangular_solve(Kmn, self.L, upper=False).solution
        A = torch.triangular_solve(A, self.L, upper=False).solution
        return A.T @ self.data_Y

    @property
    def loss_names(self):
        return "log-det", "data-fit", "const"

    def __repr__(self):
        return f"GPR(sigma={get_scalar(self.sigma)}, penalty={get_scalar(self.penalty)}, " \
               f" opt_sigma={self.opt_sigma}, opt_penalty={self.opt_penalty})"


class SGPR(NystromKRRModelMixinN, HyperOptimModel):
    def __init__(
            self,
            sigma_init,
            penalty_init,
            centers_init,
            opt_centers,
            opt_sigma,
            opt_penalty,
            cuda: bool,
            no_log_det: bool = False,
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
        self.no_log_det = no_log_det

    def hp_loss(self, X, Y):
        variance = self.penalty * X.shape[0]
        sqrt_var = torch.sqrt(variance)
        Kdiag = X.shape[0]

        m = self.centers.shape[0]
        kmn = full_rbf_kernel(self.centers, X, self.sigma)
        kmm = full_rbf_kernel(self.centers, self.centers, self.sigma)
        try:
            # L @ L.T = kmm
            eps = 1e-5
            self.L = cholesky(kmm + torch.eye(m, device=X.device, dtype=X.dtype) * eps)
        except RuntimeError as e:
            try:
                eps = 1e-4
                self.L = cholesky(kmm + torch.eye(m, device=X.device, dtype=X.dtype) * eps)
            except RuntimeError as e:
                eps = 1e-3
                self.L = cholesky(kmm + torch.eye(m, device=X.device, dtype=X.dtype) * eps)

        # A = L^{-1} K_mn / (sqrt(n*pen))
        A = torch.triangular_solve(kmn, self.L, upper=False).solution / sqrt_var
        AAT = A @ A.T
        # B = A @ A.T + I
        B = AAT + torch.eye(AAT.shape[0], device=X.device, dtype=X.dtype)
        self.LB = cholesky(B)  # LB @ LB.T = B
        AY = A @ Y
        self.c = torch.triangular_solve(AY, self.LB, upper=False).solution / sqrt_var

        # Complexity
        if not self.no_log_det:
            logdet = torch.log(torch.diag(self.LB)).sum()
            logdet += 0.5 * X.shape[0] * torch.log(variance)
        # Data-fit
        datafit = 0.5 * torch.square(Y).sum() / variance
        datafit -= 0.5 * torch.square(self.c).sum()
        # Traces (minimize)
        trace = 0.5 * Kdiag / variance
        trace -= 0.5 * torch.diag(AAT).sum()

        const = 0.5 * X.shape[0] * torch.log(2 * torch.tensor(np.pi, dtype=X.dtype))

        if self.no_log_det:
            return datafit, trace
        return logdet, datafit, trace

    def predict(self, X):
        if self.L is None or self.LB is None or self.c is None:
            raise RuntimeError("Call hp_loss before calling predict.")
        kms = full_rbf_kernel(self.centers, X, self.sigma)
        tmp1 = torch.triangular_solve(kms, self.L, upper=False).solution
        tmp2 = torch.triangular_solve(tmp1, self.LB, upper=False).solution
        return tmp2.T @ self.c

    @property
    def loss_names(self):
        if self.no_log_det:
            return "data-fit", "trace"
        return "log-det", "data-fit", "trace"

    def __repr__(self):
        return f"SGPR(sigma={get_scalar(self.sigma)}, penalty={get_scalar(self.penalty)}, num_centers={self.centers.shape[0]}, " \
               f"opt_centers={self.opt_centers}, opt_sigma={self.opt_sigma}, opt_penalty={self.opt_penalty}, " \
               f"log_det={not self.no_log_det})"
