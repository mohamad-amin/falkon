import time

import torch
import numpy as np
from falkon.kernels.diff_rbf_kernel import DiffGaussianKernel

from falkon.hypergrad.leverage_scores import (
    full_deff, full_deff_simple, subs_deff_simple,
    GaussEffectiveDimension
)

from falkon.kernels import GaussianKernel
from falkon.tests.gen_random import gen_random


def test_full_impl():
    torch.manual_seed(129)
    s = torch.tensor([5.0]).requires_grad_()
    kernel = DiffGaussianKernel(sigma=s)
    print()
    for p in [1e-10, 1e-7, 1e-5, 1e-3, 1e-1, 1.0, 2.0]:
        X = torch.from_numpy(gen_random(2000, 30, np.float32, F=False))#.requires_grad_()
        p = torch.tensor(p).to(X.dtype).requires_grad_()

        # Efficient
        t_eff = time.time()
        d_eff = full_deff(kernel, p, X)
        t_eff = time.time() - t_eff

        # Simple (naive estimation)
        t_smp = time.time()
        f_d_eff = full_deff_simple(kernel, p, X)
        t_smp = time.time() - t_smp

        t_hutch2 = time.time()
        f_d_hutch2 = GaussEffectiveDimension.apply(100, s, p, X)
        t_hutch2 = time.time() - t_hutch2

        # Subsampled
        J = X[:1000]
        t_3 = time.time()
        f_d_eff2 = subs_deff_simple(kernel, p, X, J)
        t_3 = time.time() - t_3

        # print("Estimated effective dimension for penalty %e: \n\t"
        #       "1/l=%.5f, efficient=%.5f, simple=%.5f, subsampled=%.5f, hutch=%.5f" % (p, 1/p, d_eff, f_d_eff, f_d_eff2, f_d_hutch))
        print("Efficient time %.2fs - Simple time %.2fs - Subsampled time %.2fs - Hutch time %.2fs" %
              (t_eff, t_smp, t_3, t_hutch2))
        print("Sigma gradients:")
        print(torch.autograd.grad(d_eff, s, retain_graph=True),
              torch.autograd.grad(f_d_eff, s, retain_graph=True),
              torch.autograd.grad(f_d_eff2, s, retain_graph=True),
              torch.autograd.grad(f_d_hutch2, s, retain_graph=True)
            )
        # print(torch.autograd.grad(d_eff, s), torch.autograd.grad(f_d_eff, s), torch.autograd.grad(f_d_eff2, s))


def test_hutch_grad():
    X = torch.from_numpy(gen_random(5000, 30, np.float32, F=False)).requires_grad_()
    s = torch.tensor([5.0] * X.shape[1], dtype=X.dtype).requires_grad_()
    p = torch.tensor(0.01).to(X.dtype).requires_grad_()
    print()
    # loss = GaussEffectiveDimension.apply(100, s, p, X)
    # loss.backward()
    torch.autograd.gradcheck(
        lambda si, Xi, pi: GaussEffectiveDimension.apply(100, si, pi, Xi),
        (s, X, p), eps=1e-6, atol=1e-4)


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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
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

