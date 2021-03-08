import dataclasses
import time
from typing import Union, List, Optional
from enum import Enum

import pandas as pd
import numpy as np
import torch
from torch.distributions import transforms
from torch.utils.data import DataLoader, TensorDataset

import falkon
from falkon.center_selection import UniformSelector, FixedSelector, CenterSelector
from falkon.kernels.diff_rbf_kernel import DiffGaussianKernel
from falkon.hypergrad.common import AbsHypergradModel
from falkon.hypergrad.hypergrad import compute_hypergrad, direct_valgrad
from falkon.optim import ConjugateGradient


class ValidationLoss(Enum):
    PenalizedMSE = "penalized-mse"
    MSE = "mse"

    def __str__(self):
        return self.value

    def __repr__(self):
        return str(self)


class HyperParam():
    def __init__(self, starting_value: torch.Tensor, transform: Optional[transforms.Transform] = None):
        self.val = starting_value
        self.trans = transforms

    def __get__(self):
        return self.trans(self.val)


class FalkonHO(AbsHypergradModel):
    def __init__(self,
                 M,
                 center_selection,
                 maxiter,
                 Xtr,
                 Ytr,
                 Xts,
                 Yts,
                 opt,
                 loss: ValidationLoss,
                 tot_N: int,
                 warm_start: bool = True):
        super().__init__()
        self.Xtr = Xtr
        self.Ytr = Ytr
        self.Xts = Xts
        self.Yts = Yts

        self.M = M
        self.tot_N = tot_N
        self.maxiter = maxiter
        self.opt = opt

        self.center_selection = center_selection
        self.flk = None
        self.loss = loss
        self.warm_start = warm_start

    def inner_opt(self, params, hparams):
        """This is NOT DIFFERENTIABLE"""
        alpha, beta = params['alpha'], params['alpha_pc']
        penalty, sigma = hparams['penalty'].detach(), hparams['sigma'].detach()
        if 'centers' in hparams:
            center_selection = FixedSelector(hparams['centers'].detach().clone())
        else:
            center_selection = self.center_selection

        kernel = DiffGaussianKernel(sigma, self.opt)
        self.flk = self.flk_class(
            kernel,
            torch.exp(-penalty).item(),
            self.M,
            center_selection=center_selection,
            maxiter=self.maxiter,
            seed=129,
            options=self.opt,
            N=self.tot_N)
        if self.warm_start:
            self.flk.fit(self.Xtr, self.Ytr, warm_start=beta.detach().clone())
        else:
            self.flk.fit(self.Xtr, self.Ytr)

        return {'alpha': self.flk.alpha_, 'alpha_pc': self.flk.beta_}

    def _val_loss_mse(self, params, hparams):
        alpha = params['alpha']
        sigma = hparams['sigma']
        kernel = DiffGaussianKernel(sigma, self.opt)
        if 'centers' in hparams:
            ny_points = hparams['centers']
        else:
            ny_points = self.flk.ny_points_
        kernel = DiffGaussianKernel(sigma, self.opt)
        preds = kernel.mmv(self.Xts, ny_points, alpha)
        return torch.mean((preds - self.Yts) ** 2)

    def _val_loss_penalized_mse(self, params, hparams):
        alpha = params['alpha']
        penalty, sigma = hparams['penalty'], hparams['sigma']
        kernel = DiffGaussianKernel(sigma, self.opt)
        if 'centers' in hparams:
            ny_points = hparams['centers']
        else:
            ny_points = self.flk.ny_points_
        preds = kernel.mmv(self.Xts, ny_points, alpha)
        return torch.mean((preds - self.Yts) ** 2) + torch.exp(-penalty) * alpha.T @ kernel.mmv(ny_points, ny_points, alpha)

    def val_loss(self, params, hparams):
        if self.loss == ValidationLoss.PenalizedMSE:
            return self._val_loss_penalized_mse(params, hparams)
        elif self.loss == ValidationLoss.MSE:
            return self._val_loss_mse(params, hparams)
        else:
            raise RuntimeError("Loss %s unrecognized" % (self.loss))

    def val_loss_grads(self, params, hparams):
        o_loss = self.val_loss(params, hparams)
        return (
            torch.autograd.grad(o_loss, list(params.values()), allow_unused=True, create_graph=False,
                                retain_graph=True),
            torch.autograd.grad(o_loss, list(hparams.values()), allow_unused=True, create_graph=False,
                                retain_graph=False)
        )

    def param_derivative(self, params, hparams):
        """Derivative of the training loss, with respect to the parameters"""
        alpha = params['alpha']
        penalty, sigma = hparams['penalty'], hparams['sigma']
        N = self.Xtr.shape[0]
        if 'centers' in hparams:
            ny_points = hparams['centers']
        else:
            ny_points = self.flk.ny_points_

        kernel = DiffGaussianKernel(sigma, self.opt)

        # 2/N * (K_MN(K_NM @ alpha - Y)) + 2*lambda*(K_MM @ alpha)
        out = (kernel.mmv(ny_points, self.Xtr, kernel.mmv(self.Xtr, ny_points, alpha, opt=self.opt) - self.Ytr, opt=self.opt) +
                N * torch.exp(-penalty) * kernel.mmv(ny_points, ny_points, alpha, opt=self.opt))
        return [out]

    def hparam_derivative(self, params, hparams):
        """Derivative of the training loss with respect to the parameters
        Unused in hypergradient computation.
        """
        pass

    def mixed_vector_product(self, hparams, first_derivative, vector):
        return torch.autograd.grad(first_derivative, list(hparams.values()), grad_outputs=vector,
                                   allow_unused=True)

    def hessian_vector_product(self, params, first_derivative, vector):
        N = self.Xtr.shape[0]
        ny_points = self.flk.ny_points_  # Use saved value instead of from hparams to simplify code
        kernel = self.flk.kernel
        penalty = torch.tensor(self.flk.penalty)
        vector = vector[0]

        out = (kernel.mmv(ny_points, self.Xtr, kernel.mmv(self.Xtr, ny_points, vector, opt=self.opt), opt=self.opt) +
                N * torch.exp(-penalty) * kernel.mmv(ny_points, ny_points, vector, opt=self.opt))
        return [out]

    def solve_hessian(self, params, hparams, vector, max_iter, cg_tol):
        penalty = torch.exp(-torch.tensor(self.flk.penalty))

        opt = dataclasses.replace(self.opt, cg_tolerance=cg_tol)
        vector = vector[0].detach()
        N = self.Xtr.shape[0]

        cg = ConjugateGradient(opt)
        kernel = self.flk.kernel
        precond = self.flk.precond

        start_time = time.time()
        B = precond.apply_t(vector / N)
        def mmv(sol):
            v = precond.invA(sol)
            cc = kernel.dmmv(self.Xtr, self.flk.ny_points_, precond.invT(v), None)
            return precond.invAt(precond.invTt(cc / N) + penalty * v)
        d = cg.solve(X0=None, B=B, mmv=mmv, max_iter=max_iter)
        c = precond.apply(d)
        elapsed = time.time() - start_time

        return [c], max_iter, elapsed / max_iter

    def to(self, device):
        self.Xtr = self.Xtr.to(device)
        self.Ytr = self.Ytr.to(device)
        self.Xts = self.Xts.to(device)
        self.Yts = self.Yts.to(device)
        return self

    @property
    def flk_class(self):
        if self.Xtr.device.type == 'cuda':
            return falkon.InCoreFalkon
        return falkon.Falkon

    @property
    def model(self):
        return self.flk


def map_gradient(data,
                 falkon_centers: CenterSelector,
                 falkon_M: int,
                 falkon_maxiter,
                 falkon_opt,
                 sigma_type: str,
                 hessian_cg_steps: int,
                 hessian_cg_tol: float,
                 loss: ValidationLoss,
                 err_fns: list):
    Xtr, Ytr, Xts, Yts = data['Xtr'], data['Ytr'], data['Xts'], data['Yts']
    t = Ytr.size(1)
    dt, dev = Xtr.dtype, Xtr.device
    which_grads = "hyper_grads"  # Can also be 'val_grads'

    # Choose start value for sigma
    if sigma_type != 'single':
        raise ValueError("sigma_type %s invalid for mapping gradient" % (sigma_type))

    params = {
        'alpha': torch.zeros(falkon_M, t, requires_grad=True, dtype=dt, device=dev),
        'alpha_pc': torch.zeros(falkon_M, t, requires_grad=True, dtype=dt, device=dev),
    }
    penalty_range = np.linspace(0.1, 20, num=60)
    sigma_range = np.linspace(0.1, 20, num=60)
    hps = []
    for p in penalty_range:
        for s in sigma_range:
            hps.append({
                'penalty': torch.tensor(p, requires_grad=True, dtype=dt, device=dev),  # e^{-penalty}
                'sigma': torch.tensor([s], requires_grad=True, dtype=dt, device=dev),
            })

    flk_helper = FalkonHO(falkon_M, falkon_centers, falkon_maxiter, Xtr, Ytr, Xts, Yts, falkon_opt, loss)

    df = pd.DataFrame(columns=["sigma", "sigma_g", "penalty", "penalty_g", "loss"])
    for hp in hps:
        params = flk_helper.inner_opt(params, hp)
        if which_grads == 'hyper_grads':
            loss, hp_grad = compute_hypergrad(
                params, hp, model=flk_helper, cg_steps=hessian_cg_steps, cg_tol=hessian_cg_tol,
                set_grad=False, timings=False)
        elif which_grads == 'val_grads':
            params['alpha'].requires_grad_()
            params['alpha_pc'].requires_grad_()
            _, hp_grad = flk_helper.val_loss_grads(params, hp)
            loss = flk_helper.val_loss(params, hp)
        else:
            raise ValueError(which_grads)

        new_row = {
            'penalty': hp['penalty'].cpu().item(),
            'penalty_g': hp_grad[0].cpu().item(),
            'sigma': hp['sigma'][0].cpu().item(),
            'sigma_g': hp_grad[1][0].cpu().item(),
            'loss': loss.cpu().item(),
        }
        ts_preds = flk_helper.flk.predict(Xts)
        for efn in err_fns:
            err, err_name = efn(Yts.cpu(), ts_preds.cpu())
            new_row["test_%s" % err_name] = err

        df = df.append(new_row, ignore_index=True)
        print(hp)
    return df


def run_falkon_hypergrad(data,
                         falkon_centers: CenterSelector,
                         falkon_M: int,
                         optimize_centers: bool,
                         falkon_maxiter,
                         falkon_opt,
                         sigma_type: str,
                         outer_lr,
                         outer_steps,
                         hessian_cg_steps,
                         hessian_cg_tol,
                         callback,
                         debug,
                         loss: ValidationLoss,
                         warm_start: bool,
                         sigma_init: float = 2,
                         penalty_init: float = 1):
    Xtr, Ytr, Xts, Yts = data['Xtr'], data['Ytr'], data['Xts'], data['Yts']

    n, d = Xtr.size()
    t = Ytr.size(1)
    dt, dev = Xtr.dtype, Xtr.device

    # Choose start value for sigma
    if sigma_type == 'single':
        start_sigma = [sigma_init]
    elif sigma_type == 'diag':
        start_sigma = [sigma_init] * d
    else:
        raise ValueError("sigma_type %s unrecognized" % (sigma_type))

    hparams = {
        'penalty': torch.tensor(penalty_init, requires_grad=True, dtype=dt, device=dev),  # e^{-penalty}
        'sigma': torch.tensor(start_sigma, requires_grad=True, dtype=dt, device=dev),
    }
    params = {
        'alpha': torch.zeros(falkon_M, t, requires_grad=True, dtype=dt, device=dev),
        'alpha_pc': torch.zeros(falkon_M, t, requires_grad=True, dtype=dt, device=dev),
    }
    # Nystrom centers: Need to decide whether to optimize them as well
    if optimize_centers:
        hparams['centers'] = falkon_centers.select(Xtr, Y=None, M=falkon_M).requires_grad_()

    outer_opt = torch.optim.Adam(lr=outer_lr, params=hparams.values())
    flk_helper = FalkonHO(falkon_M, falkon_centers, falkon_maxiter, Xtr, Ytr, Xts, Yts, falkon_opt, loss,
                          tot_N=None, warm_start=warm_start)

    hparam_history = [[h.detach().clone() for h in hparams.values()]]
    val_loss_history = []
    hgrad_history = []
    time_history = []
    for o_step in range(outer_steps):
        # Run inner loop to get alpha*
        i_start = time.time()
        params = flk_helper.inner_opt(params, hparams)
        if callback is not None:
            callback(o_step, flk_helper.model)
        inner_opt_t = time.time()

        outer_opt.zero_grad()
        hgrad_out = compute_hypergrad(params, hparams, model=flk_helper,
                                      cg_steps=hessian_cg_steps, cg_tol=hessian_cg_tol, set_grad=True,
                                      timings=debug)
        outer_opt.step()
        hparams['penalty'].data.clamp_(min=1e-10)
        hparams['sigma'].data.clamp_(min=1e-10)
        i_end = time.time()
        if True:
            #print("[%4d/%4d] - time %.2fs (inner-opt %.2fs, outer-opt %.2fs)" % (o_step, outer_steps, i_end - i_start, inner_opt_t - i_start, i_end - inner_opt_t))
            #print("GRADIENT", hgrad_out[1])
            print(f"Epoch {o_step} - sigma: {hparams['sigma'].detach().cpu()[0].item():.2f} - "
                  f"penalty: {torch.exp(-hparams['penalty'].detach().cpu()).item():.3e} - "
                  f"Validation loss: {hgrad_out[0].item():.4f}")
        hparam_history.append([h.detach().clone() for h in hparams.values()])
        val_loss_history.append(hgrad_out[0])
        hgrad_history.append([g.detach() for g in hgrad_out[1]])
        time_history.append(i_end - i_start)

    return hparam_history, val_loss_history, hgrad_history, flk_helper.model, time_history


def stochastic_flk_hypergrad(
                         data,
                         falkon_centers: CenterSelector,
                         falkon_M: int,
                         falkon_maxiter: int,
                         falkon_opt,
                         sigma_type: str,
                         outer_lr: float,
                         batch_size: int,
                         outer_steps: int,
                         hessian_cg_steps: int,
                         hessian_cg_tol: float,
                         callback,
                         debug: bool,
                         loss: ValidationLoss,
                         sigma_init: float = 2,
                         penalty_init: float = 1,
                         warm_start: bool = True,
                         cuda: bool = False
                         ):
    Xtr, Ytr, Xts, Yts = data['Xtr'], data['Ytr'], data['Xts'], data['Yts']

    if batch_size < falkon_M:
        raise ValueError("Batch-size must be at least as large as the number of Falkon centers.")

    n, d = Xtr.size()
    t = Ytr.size(1)
    dt, dev = Xtr.dtype, Xtr.device
    if cuda and not Xtr.is_cuda:
        dev = torch.device("cuda:0")

    # Choose start value for sigma
    if sigma_type == 'single':
        start_sigma = [sigma_init]
    elif sigma_type == 'diag':
        start_sigma = [sigma_init] * d
    else:
        raise ValueError("sigma_type %s unrecognized" % (sigma_type))

    hparams = {
        'penalty': torch.tensor(penalty_init, requires_grad=True, dtype=dt, device=dev),  # e^{-penalty}
        'sigma': torch.tensor(start_sigma, requires_grad=True, dtype=dt, device=dev),
        'centers': falkon_centers.select(Xtr, Y=None, M=falkon_M).to(dev).requires_grad_(),
    }
    params = {
        'alpha': torch.zeros(falkon_M, t, requires_grad=True, dtype=dt, device=dev),
        'alpha_pc': torch.zeros(falkon_M, t, requires_grad=True, dtype=dt, device=dev),
    }

    outer_opt = torch.optim.Adam([
            {'params': hparams['penalty']},
            {'params': hparams['sigma']},
            {'params': hparams['centers']}
        ], lr=outer_lr)
    hparam_history = [[h.detach().cpu().clone() for h in hparams.values()]]
    val_loss_history = []
    hgrad_history = []
    time_history = []

    train_loader = FastTensorDataLoader(Xtr, Ytr, batch_size=batch_size, shuffle=True, drop_last=False, cuda=cuda)
    train_loader = iter(train_loader)
    test_loader = FastTensorDataLoader(Xts, Yts, batch_size=batch_size, shuffle=True, drop_last=False, cuda=cuda)
    test_loader = iter(test_loader)

    for o_step in range(outer_steps):
        i_start = time.time()
        b_tr_x, b_tr_y = next(train_loader)
        b_ts_x, b_ts_y = next(test_loader)
        flk_helper = FalkonHO(
            falkon_M, falkon_centers, falkon_maxiter, b_tr_x, b_tr_y, b_ts_x, b_ts_y, falkon_opt, loss,
            warm_start=warm_start, tot_N=None)

        i_data_load = time.time()
        params = flk_helper.inner_opt(params, hparams)
        if callback is not None:
            callback(o_step, flk_helper.model)
        inner_opt_t = time.time()

        # Start 'outer_opt'
        outer_opt.zero_grad()
        hgrad_out = direct_valgrad(params, hparams, flk_helper, set_grad=True)
        #hgrad_out = compute_hypergrad(params, hparams, model=flk_helper, cg_steps=hessian_cg_steps,
        #                              cg_tol=hessian_cg_tol, set_grad=True, timings=False)

        #hparams['penalty'].grad = None
        #hparams['sigma'].grad = None
        #hparams['centers'].grad = None
        outer_opt.step()
        hparams['penalty'].data.clamp_(min=1e-10)
        hparams['sigma'].data.clamp_(min=1e-10)
        i_end = time.time()

        if debug and o_step % 1 == 0:
            #print("[%4d/%4d] - time %.2fs (data-load %.2fs, inner-opt %.2fs, outer-opt %.2fs)" % (o_step, outer_steps, i_end - i_start, i_data_load - i_start, inner_opt_t - i_data_load, i_end - inner_opt_t))
            #print("centers", hparams['centers'].mean(), hparams['centers'].min(), hparams['centers'].max(), hparams['centers'].sum())
            #print("centers\n", hparams['centers'])
            #print("alpha\n", params['alpha'])
            #print("GRADIENT", hgrad_out[1])
            #print("NEW HPARAMS", hparams)
            print("%5.3f, %5.3f" % (hparams['penalty'].item(), hparams['sigma'][0].item()))
            print(hgrad_out[0].item())

        hparam_history.append([h.detach().clone().cpu() for h in hparams.values()])
        val_loss_history.append(hgrad_out[0])
        hgrad_history.append([g.detach().cpu() for g in hgrad_out[1]])
        time_history.append(i_end - i_start)

    return hparam_history, val_loss_history, hgrad_history, flk_helper.model, time_history
