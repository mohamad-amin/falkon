import numpy as np
import torch

from falkon import FalkonOptions
from falkon.hypergrad.leverage_scores import (
    NoRegLossAndDeff, GCV, RegLossAndDeffv2, ValidationLoss, creg_plainfit,
)
from falkon.tests.gen_random import gen_random


def test_gpytorch_quad_derivative():
    X = torch.from_numpy(gen_random(1000, 3, np.float32, F=False))
    Y = (X ** 2).mean(1) + torch.randn(X.shape[0]) * 0.3

    import gpytorch
    # We will use the simplest form of GP model, exact inference
    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(X, Y, likelihood)
    model.train()
    likelihood.train()
    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=0.1)  # Includes GaussianLikelihood parameters
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    for i in range(10):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(X)
        # Calc loss and backprop gradients
        loss = -mll(output, Y)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            i + 1, 40, loss.item(),
            model.covar_module.base_kernel.lengthscale.item(),
            model.likelihood.noise.item()
        ))
        optimizer.step()


def test_complexity_reg_impl():
    # RegLossAndDeffv2.grad_check()
    # GCV.grad_check()
    # NoRegLossAndDeff.grad_check()
    ValidationLoss.grad_check()


def test_creg_timing():
    n, m, d = 10_000, 100, 10
    torch.manual_seed(3)
    X = torch.randn(n, d, dtype=torch.float64)
    w = torch.randn(d, 1, dtype=torch.float64)
    Y = X @ w
    M = X[:m].clone().detach().requires_grad_()
    s = torch.tensor([10.0] * d, dtype=X.dtype).requires_grad_()
    p = torch.tensor(1e-2, dtype=X.dtype).requires_grad_()

    for i in range(10):
        loss = creg_plainfit(s, p, M, X, Y, num_estimators=200, deterministic=False,
                             solve_options=FalkonOptions(cg_tolerance=1e-2), solve_maxiter=100,
                             gaussian_random=False, use_stoch_trace=True, warm_start=True)
        loss.backward()

    print("Timings:")
    print(f"D-Eff: F {np.mean(NoRegLossAndDeff.t_deff_fwd):.2f}s  B {np.mean(NoRegLossAndDeff.t_deff_bwd):.2f}s")
    print(f"Fit: F {np.mean(NoRegLossAndDeff.t_fit_fwd):.2f}s  B {np.mean(NoRegLossAndDeff.t_fit_bwd):.2f}s")
    print(f"Trace: F {np.mean(NoRegLossAndDeff.t_tr_fwd):.2f}s  B {np.mean(NoRegLossAndDeff.t_tr_bwd):.2f}s")
    print(f"Grad: {np.mean(NoRegLossAndDeff.t_grad):.2f}s")
