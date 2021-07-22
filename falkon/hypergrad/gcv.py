import torch
from falkon.kernels import GaussianKernel

from falkon.hypergrad.leverage_scores import gcv, GCV

from falkon import FalkonOptions
from falkon.hypergrad.common import full_rbf_kernel, get_scalar, cholesky
from falkon.hypergrad.complexity_reg import NystromKRRModelMixinN, HyperOptimModel


class StochasticNystromGCV(NystromKRRModelMixinN, HyperOptimModel):
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

    def hp_loss(self, X, Y):
        loss = gcv(kernel_args=self.sigma, penalty=self.penalty, centers=self.centers,
                   X=X, Y=Y, num_estimators=self.num_trace_est, deterministic=False,
                   solve_options=self.flk_opt, solve_maxiter=self.flk_maxiter,
                   gaussian_random=False)
        return [loss]

    def predict(self, X):
        if GCV.last_alpha is None:
            raise RuntimeError("Call hp_loss before calling predict.")
        alpha = GCV.last_alpha
        kernel = GaussianKernel(self.sigma, opt=self.flk_opt)
        return kernel.mmv(X, self.centers, alpha)

    @property
    def loss_names(self):
        return ("gcv",)

    def __repr__(self):
        return f"StochasticNystromGCV(sigma={get_scalar(self.sigma)}, penalty={get_scalar(self.penalty)}, " \
               f"num_centers={self.centers.shape[0]}, opt_centers={self.opt_centers}, " \
               f"opt_sigma={self.opt_sigma}, opt_penalty={self.opt_penalty}, t={self.num_trace_est}, " \
               f"flk_iter={self.flk_maxiter})"


class NystromGCV(NystromKRRModelMixinN, HyperOptimModel):
    r"""
    GCV objective is

    ..math:

        \dfrac{\dfrac{1}{n} \lVert (I - \widetilde{K}_\lambda \widetilde{K}) Y \rVert^2}
              {\Big(\frac{1}{n} \mathrm{Tr}(I - \widetilde{K}_\lambda \widetilde{K}) \Big)}

    We must compute the two terms denoted as the numerator and the denominator.
    Using the usual names for matrix variable substitutions (taken from gpflow code), we have that
    the numerator can be computed as

    ..math:

        \dfrac{1}{n} \lVert (I - A^\top \mathrm{LB}^{-\top} \mathrm{LB}^{-1} A) Y \rVert^2

    We compute the terms inside the norm first, from right to left using matrix-vector multiplications
    and triangular solves. Finally we compute the norm.

    The denominator is far less efficient to compute, since it requires working with m*n matrices.
    It can be expressed in terms of the same matrices as above:

    ..math:

        \Big( \frac{1}{n} (\mathrm{Tr}(I} - \lVert \mathrm{LB}^{-1}A \rVert_F^2 \Big)^2

    """

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

        self.L, self.LB, self.d = None, None, None

    def hp_loss(self, X, Y):
        # Like with LOOCV we are virtually using an estimator trained with n - 1 points.
        variance = self.penalty * (X.shape[0] - 1)
        sqrt_var = torch.sqrt(variance)

        m = self.centers.shape[0]
        kmn = full_rbf_kernel(self.centers, X, self.sigma)
        kmm = (full_rbf_kernel(self.centers, self.centers, self.sigma) +
               torch.eye(m, device=X.device, dtype=X.dtype) * 1e-6)
        self.L = cholesky(kmm)  # L @ L.T = kmm
        # A = L^{-1} K_mn / (sqrt(n*pen))
        A = torch.triangular_solve(kmn, self.L, upper=False).solution / sqrt_var
        AAT = A @ A.T
        # B = A @ A.T + I
        B = AAT + torch.eye(AAT.shape[0], device=X.device, dtype=X.dtype)
        self.LB = cholesky(B)  # LB @ LB.T = B

        AY = A @ Y
        # numerator is (1/n)*||(I - A.T @ LB^{-T} @ LB^{-1} @ A) @ Y||^2
        # compute A.T @ LB^{-T} @ LB^{-1} @ A @ Y
        tmp1 = torch.triangular_solve(AY, self.LB, upper=False).solution
        tmp2 = torch.triangular_solve(tmp1, self.LB, upper=False, transpose=True).solution
        self.d = tmp2 / sqrt_var  # only for predictions
        tmp3 = Y - A.T @ tmp2
        numerator = torch.square(tmp3).sum(0).mean()

        # Denominator
        C = torch.triangular_solve(A, self.LB, upper=False).solution
        denominator = (1 - torch.square(C).sum() / X.shape[0]) ** 2
        return ((1 / X.shape[0]) * (numerator / denominator),)

    def predict(self, X):
        if self.L is None or self.LB is None or self.d is None:
            raise RuntimeError("Call hp_loss before calling predict.")
        kms = full_rbf_kernel(self.centers, X, self.sigma)
        tmp1 = torch.triangular_solve(self.d, self.L, upper=False, transpose=True).solution
        return kms.T @ tmp1

    @property
    def loss_names(self):
        return ("gcv",)

    def __repr__(self):
        return f"NystromGCV(sigma={get_scalar(self.sigma)}, penalty={get_scalar(self.penalty)}, num_centers={self.centers.shape[0]}, " \
               f"opt_centers={self.opt_centers}, opt_sigma={self.opt_sigma}, opt_penalty={self.opt_penalty})"
