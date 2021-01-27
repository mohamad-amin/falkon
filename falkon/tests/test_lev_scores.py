import time

import pytest
import torch
import numpy as np
from falkon.center_selection import FixedSelector

from falkon import Falkon, FalkonOptions
from falkon.utils import decide_cuda

from falkon.kernels.diff_rbf_kernel import DiffGaussianKernel
from falkon.hypergrad.leverage_scores import (
    full_deff, full_deff_simple, subs_deff_simple,
    GaussEffectiveDimension, gauss_effective_dimension, gauss_nys_effective_dimension,
    loss_and_deff,
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


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda:0", marks=[pytest.mark.skipif(not decide_cuda(), reason="No GPU found.")])
])
@pytest.mark.parametrize("dtype", [np.float64])
def test_hutch_grad(device, dtype):
    torch.manual_seed(3)
    f_order = False

    X = torch.from_numpy(gen_random(100, 3, dtype, F=f_order)).to(device=device).requires_grad_()
    s = torch.tensor([5.0] * X.shape[1], dtype=X.dtype, device=device).requires_grad_()
    p = torch.tensor(0.01, dtype=X.dtype, device=device).requires_grad_()
    print()
    #loss = GaussEffectiveDimension.apply(5, s, p, X)
    #loss.backward()
    torch.autograd.gradcheck(
        lambda si, Xi, pi: gauss_effective_dimension(si, pi, Xi, t=50, deterministic=True),
        (s, X, p), eps=1e-4, atol=1e-4)


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda:0", marks=[pytest.mark.skipif(not decide_cuda(), reason="No GPU found.")])
])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_hutch_grad_nys(device, dtype):
    torch.manual_seed(3)
    f_order = False

    X = torch.from_numpy(gen_random(100, 3, dtype, F=f_order)).to(device=device)
    M = X[:20].clone().detach().requires_grad_()
    s = torch.tensor([5.0] * X.shape[1], dtype=X.dtype, device=device).requires_grad_()
    p = torch.tensor(0.01, dtype=X.dtype, device=device).requires_grad_()

    print()
    torch.autograd.gradcheck(
        lambda si, Mi, Xi, pi: gauss_nys_effective_dimension(si, pi, Mi, Xi, t=10, deterministic=True),
        (s, M, X, p), eps=1e-4, atol=1e-4)


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda:0", marks=[pytest.mark.skipif(not decide_cuda(), reason="No GPU found.")])
])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_lossdeff_grad_nys(device, dtype):
    torch.manual_seed(3)
    f_order = False

    X = torch.from_numpy(gen_random(100, 3, dtype, F=f_order)).to(device=device)
    Y = X.sum(1).reshape(-1, 1)
    M = X[:20].clone().detach().requires_grad_()
    s = torch.tensor([5.0] * X.shape[1], dtype=X.dtype, device=device).requires_grad_()
    p = torch.tensor(0.01, dtype=X.dtype, device=device).requires_grad_()

    print()
    torch.autograd.gradcheck(
        lambda si, Mi, Xi, Yi, pi: loss_and_deff(si, pi, Mi, Xi, Yi, t=10, deterministic=True),
        (s, M, X, Y, p), eps=1e-4, atol=1e-4)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_nystrom_deff_timings(dtype):
    torch.manual_seed(3)
    f_order = False
    device = "cpu"

    X = torch.from_numpy(gen_random(2000, 3, dtype, F=f_order)).to(device=device)
    M = X[:100].clone().detach().requires_grad_()
    s = torch.tensor([5.0] * X.shape[1], dtype=X.dtype, device=device).requires_grad_()
    p = torch.tensor(0.001, dtype=X.dtype, device=device).requires_grad_()

    t_ful = []
    t_nys = []

    for i in range(10):
        t_s = time.time()
        d_eff = gauss_effective_dimension(s, p, X, t=30, deterministic=True)
        t_fwd = time.time()
        d_eff.backward()
        t_bwd = time.time()
        t_ful.append((t_fwd - t_s, t_bwd - t_fwd))

        t_s = time.time()
        d_eff_nys = gauss_nys_effective_dimension(s, p, M, X, t=30, deterministic=True)
        t_fwd = time.time()
        d_eff_nys.backward()
        t_bwd = time.time()
        t_nys.append((t_fwd - t_s, t_bwd - t_fwd))
        np.testing.assert_allclose(d_eff.cpu().detach().numpy(), d_eff_nys.cpu().detach().numpy(), rtol=1e-2)

    print("Full timings: %.4fs - %.4fs" % (np.min([t[0] for t in t_ful]), np.min([t[1] for t in t_ful])))
    print("Nystrom timings: %.4fs - %.4fs" % (np.min([t[0] for t in t_nys]), np.min([t[1] for t in t_nys])))

@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_compare_deff_outputs(dtype):
    torch.manual_seed(3)
    f_order = False
    device = "cpu"

    X = torch.from_numpy(gen_random(1000, 100, dtype, F=f_order)).to(device=device)
    M = X[:40].clone().detach().requires_grad_()
    s = torch.tensor([10.0] * X.shape[1], dtype=X.dtype, device=device).requires_grad_()
    p = torch.tensor(1e-2, dtype=X.dtype, device=device).requires_grad_()
    K = GaussianKernel(s)

    print()
    print("True formula d_eff", torch.diag(K(X, X) @ torch.pinverse(
        K(X, X) + torch.diag_embed((p*X.shape[0]).expand(X.shape[0])).to(X))).sum().item())

    d_eff = gauss_effective_dimension(s, p, X, t=30, deterministic=False)
    print("d_eff", d_eff.item())

    d_eff_nys = gauss_nys_effective_dimension(s, p, M, X, t=30, deterministic=False)
    print("d_eff_nys", d_eff_nys.item())

    d_eff_nys_real = torch.diag(
        K(X, M) @ torch.pinverse(K(M, X) @ K(X, M) + p * X.shape[0] * K(M, M)) @ K(M, X)).sum()
    print("d_eff_nys_real", d_eff_nys_real.item())

    d_eff_double_nys = torch.diag(
        K(X, M) @ torch.pinverse(K(M, M) @ K(M, M) + p * X.shape[0] * K(M, M)) @ K(M, X)).sum()
    print("d_eff_double_nys", d_eff_double_nys.item())

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

