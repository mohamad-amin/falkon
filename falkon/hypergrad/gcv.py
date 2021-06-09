import torch

from falkon.hypergrad.common import full_rbf_kernel, get_scalar
from falkon.hypergrad.complexity_reg import NystromKRRModelMixinN, HyperOptimModel


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

        self.L, self.LB, self.c = None, None, None

    def hp_loss(self, X, Y):
        variance = self.penalty
        sqrt_var = torch.sqrt(variance)

        m = self.centers.shape[0]
        kmn = full_rbf_kernel(self.centers, X, self.sigma)
        kmm = (full_rbf_kernel(self.centers, self.centers, self.sigma) +
               torch.eye(m, device=X.device, dtype=X.dtype) * 1e-6)
        self.L = torch.cholesky(kmm)   # L @ L.T = kmm
        # A = L^{-1} K_mn / (sqrt(n*pen))
        A = torch.triangular_solve(kmn, self.L, upper=False).solution / sqrt_var
        AAT = A @ A.T
        # B = A @ A.T + I
        B = AAT + torch.eye(AAT.shape[0], device=X.device, dtype=X.dtype)
        self.LB = torch.cholesky(B)  # LB @ LB.T = B

        AY = A @ Y
        # Now we need to compute A.T @ LB^{-T} @ LB^{-1} @ A @ Y
        d = torch.triangular_solve(AY, self.LB, upper=False).solution
        self.c = d / sqrt_var  # only for predictions
        d = torch.triangular_solve(d, self.LB, upper=False, transpose=True).solution
        d = A.T @ d
        numerator = torch.square(d).sum(0) / X.shape[0]

        # Denomoinator
        C = torch.triangular_solve(A, self.LB, upper=False).solution
        denominator = (1 - torch.square(C).sum() / X.shape[0])**2
        loss = (numerator / denominator)[0]
        return (loss, )

    def predict(self, X):
        if self.L is None or self.LB is None or self.c is None:
            raise RuntimeError("Call hp_loss before calling predict.")
        # Predictions are handled directly.
        kms = full_rbf_kernel(self.centers, X, self.sigma)
        tmp1 = torch.triangular_solve(kms, self.L, upper=False).solution
        tmp2 = torch.triangular_solve(tmp1, self.LB, upper=False).solution
        return tmp2.T @ self.c

    @property
    def loss_names(self):
        return ("gcv", )

    def __repr__(self):
        return f"NystromGCV(sigma={get_scalar(self.sigma)}, penalty={get_scalar(self.penalty)}, num_centers={self.centers.shape[0]}, " \
               f"opt_centers={self.opt_centers}, opt_sigma={self.opt_sigma}, opt_penalty={self.opt_penalty})"
