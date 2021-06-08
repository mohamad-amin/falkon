from functools import partial
from typing import Iterable

import torch

from falkon import FalkonOptions
from falkon.kernels import GaussianKernel
from falkon.hypergrad.common import full_rbf_kernel, get_scalar, cg
from falkon.hypergrad.complexity_reg import NystromKRRModelMixinN, HyperOptimModel


class NystromClosedFormHgrad(NystromKRRModelMixinN, HyperOptimModel):
    r"""

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
            tr_indices,
            ts_indices,
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

        self.tr_indices = tr_indices
        self.ts_indices = ts_indices
        self.L, self.LB, self.c = None, None, None

    def hp_loss(self, X, Y):
        Xtr = X[self.tr_indices]
        Xval = X[self.ts_indices]
        Ytr = Y[self.tr_indices]
        Yval = Y[self.ts_indices]

        variance = self.penalty
        sqrt_var = torch.sqrt(variance)

        m = self.centers.shape[0]
        kmn = full_rbf_kernel(self.centers, Xtr, self.sigma)
        kmm = (full_rbf_kernel(self.centers, self.centers, self.sigma) +
               torch.eye(m, device=Xtr.device, dtype=Xtr.dtype) * 1e-6)
        self.L = torch.cholesky(kmm)   # L @ L.T = kmm
        # A = L^{-1} K_mn / (sqrt(n*pen))
        A = torch.triangular_solve(kmn, self.L, upper=False).solution / sqrt_var
        AAT = A @ A.T
        # B = A @ A.T + I
        B = AAT + torch.eye(AAT.shape[0], device=Xtr.device, dtype=Xtr.dtype)
        self.LB = torch.cholesky(B)  # LB @ LB.T = B
        AYtr = A @ Ytr
        self.c = torch.triangular_solve(AYtr, self.LB, upper=False).solution / sqrt_var

        kmval = full_rbf_kernel(self.centers, Xval, self.sigma)
        tmp1 = torch.triangular_solve(kmval, self.L, upper=False).solution
        tmp2 = torch.triangular_solve(tmp1, self.LB, upper=False).solution
        val_preds = tmp2 @ self.c

        return torch.sum(torch.square(Yval - val_preds))

    def predict(self, X):
        if self.L is None or self.LB is None or self.c is None:
            raise RuntimeError("Call hp_loss before calling predict.")
        kms = full_rbf_kernel(self.centers, X, self.sigma)
        tmp1 = torch.triangular_solve(kms, self.L, upper=False).solution
        tmp2 = torch.triangular_solve(tmp1, self.LB, upper=False).solution
        return tmp2.T @ self.c

    @property
    def loss_names(self):
        return ("val-mse", )

    @property
    def train_pct(self):
        tot = len(self.tr_indices) + len(self.ts_indices)
        return len(self.tr_indices) / tot * 100

    def __repr__(self):
        return f"NystromClosedFormHgrad(sigma={get_scalar(self.sigma)}, penalty={get_scalar(self.penalty)}, num_centers={self.centers.shape[0]}, " \
               f"opt_centers={self.opt_centers}, opt_sigma={self.opt_sigma}, opt_penalty={self.opt_penalty}, train_pct={self.train_pct})"


class NystromIFTHgrad(NystromKRRModelMixinN, HyperOptimModel):
    r"""

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
            tr_indices,
            ts_indices,
            cg_tol: float,
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

        self.tr_indices = tr_indices
        self.ts_indices = ts_indices
        self.cg_steps = 100
        self.cg_tol = cg_tol
        self.falkon_opt = FalkonOptions()
        self.alpha = None

    def compute_alpha(self, Xtr, Ytr) -> torch.Tensor:
        sqrt_var = torch.sqrt(self.penalty)
        kmn = full_rbf_kernel(self.centers, Xtr, self.sigma)
        kmm = (full_rbf_kernel(self.centers, self.centers, self.sigma) +
               torch.eye(self.centers.shape[0], device=Xtr.device, dtype=Xtr.dtype) * 1e-6)
        L = torch.cholesky(kmm)  # L @ L.T = kmm
        # A = L^{-1} K_mn / (sqrt(n*pen))
        A = torch.triangular_solve(kmn, L, upper=False).solution / sqrt_var
        AAT = A @ A.T
        # B = A @ A.T + I
        B = AAT + torch.eye(AAT.shape[0], device=Xtr.device, dtype=Xtr.dtype)
        LB = torch.cholesky(B)  # LB @ LB.T = B

        # Now we need to compute la^{-1/2} * L^{-T} @ LB^{-T} @ LB^{-1} @ A @ Y
        AY = A @ Ytr
        d = torch.triangular_solve(AY, LB, upper=False).solution / sqrt_var
        d = torch.triangular_solve(d, LB, upper=False, transpose=True).solution
        d = torch.triangular_solve(d, L, upper=False, transpose=True).solution
        return d

    def compute_val_loss(self, alpha, Xval, Yval) -> torch.Tensor:
        kernel = GaussianKernel(self.sigma.detach(), self.falkon_opt)
        preds = kernel.mmv(Xval, self.centers, alpha)
        return torch.mean((preds - Yval) ** 2)

    def compute_tr_loss(self, alpha, Xtr, Ytr) -> torch.Tensor:
        kernel = GaussianKernel(self.sigma.detach(), self.falkon_opt)
        preds = kernel.mmv(Xtr, self.centers, alpha)
        return torch.mean((preds - Ytr) ** 2) + self.penalty * alpha.T @ kernel.mmv(self.centers, self.centers, alpha)

    def hessian_vector_product(self, Xtr, vector) -> torch.Tensor:
        kernel = GaussianKernel(self.sigma.detach(), self.falkon_opt)
        return (kernel.mmv(self.centers, Xtr, kernel.mmv(Xtr, self.centers, vector)) +
                self.penalty * kernel.mmv(self.centers, self.centers, vector))

    def mixed_vector_product(self, first_derivative, vector) -> Iterable[torch.Tensor]:
        return torch.autograd.grad(first_derivative, self.parameters(), grad_outputs=vector, allow_unused=True)

    def hp_loss(self, X, Y):
        Xtr = X[self.tr_indices]
        Xval = X[self.ts_indices]
        Ytr = Y[self.tr_indices]
        Yval = Y[self.ts_indices]

        hparams = self.parameters()

        with torch.autograd.no_grad():
            self.alpha = self.compute_alpha(Xtr, Ytr)
        with torch.autograd.enable_grad():
            val_loss = self.compute_val_loss(self.alpha, Xval, Yval)
        val_grad_alpha = torch.autograd.grad(val_loss, self.alpha, allow_unused=True, create_graph=False, retain_graph=True)
        val_grad_hp = torch.autograd.grad(val_loss, hparams, allow_unused=True, create_graph=False, retain_graph=False)

        with torch.autograd.enable_grad():
            tr_loss = self.compute_tr_loss(self.alpha, Xtr, Ytr)
            tr_grad_alpha = torch.autograd.grad(tr_loss, self.alpha, retain_graph=True)

        with torch.autograd.no_grad():
            hvp = partial(self.hessian_vector_product, Xtr)
            vs, cg_iter_completed, hvp_time = cg(hvp, val_grad_alpha, max_iter=self.cg_steps, epsilon=self.cg_tol)

        # Multiply the mixed inner gradient by `vs`
        grads = self.mixed_vector_product(tr_grad_alpha, vs)

        final_grads = []
        for ohp, g in zip(val_grad_hp, grads):
            if ohp is not None:
                final_grads.append(ohp - g)
            else:
                final_grads.append(-g)
        return final_grads

    def predict(self, X):
        if self.alpha is None:
            raise RuntimeError("Call hp_loss before calling predict.")
        # Predictions are handled directly.
        kms = full_rbf_kernel(self.centers, X, self.sigma)
        return kms @ self.alpha.T

    @property
    def losses_are_grads(self):
        return True

    @property
    def loss_names(self):
        return ("hypergrad", )

    @property
    def train_pct(self):
        tot = len(self.tr_indices) + len(self.ts_indices)
        return len(self.tr_indices) / tot * 100

    def __repr__(self):
        return f"NystromIFTHgrad(sigma={get_scalar(self.sigma)}, penalty={get_scalar(self.penalty)}, num_centers={self.centers.shape[0]}, " \
               f"opt_centers={self.opt_centers}, opt_sigma={self.opt_sigma}, opt_penalty={self.opt_penalty}, train_pct={self.train_pct}," \
               f"cg_tol={self.cg_tol}, cg_steps={self.cg_steps})"
