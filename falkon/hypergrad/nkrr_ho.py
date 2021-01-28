import itertools
import time
import math

from reprint import output
import numpy as np
import torch
import torch.nn as nn
from falkon.optim import ConjugateGradient

import falkon
from falkon.center_selection import FixedSelector, CenterSelector
from falkon.hypergrad.leverage_scores import (
    subs_deff_simple, gauss_effective_dimension,
    gauss_nys_effective_dimension, loss_and_deff
)
from falkon.kernels.diff_rbf_kernel import DiffGaussianKernel


class FastTensorDataLoader:
    def __init__(self, *tensors, batch_size, shuffle=False, drop_last=False, cuda=False):
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors
        self.num_points = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.cuda = cuda

        n_batches, remainder = divmod(self.num_points, self.batch_size)
        if remainder > 0 and not drop_last:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            self.indices = torch.randperm(self.num_points)
        else:
            self.indices = None
        self.i = 0
        return self

    def __next__(self):
        try:
            if self.i >= self.n_batches:  # This should handle drop_last correctly
                raise StopIteration()
        except AttributeError:
            raise RuntimeError(
                "Make sure you make the tensor data-loader an iterator before iterating over it!")

        if self.indices is not None:
            indices = self.indices[self.i * self.batch_size: (self.i + 1) * self.batch_size]
            batch = tuple(t[indices] for t in self.tensors)
        else:
            batch = tuple(
                t[self.i * self.batch_size: (self.i + 1) * self.batch_size] for t in self.tensors)
        if self.cuda:
            batch = tuple(t.cuda() for t in batch)
        self.i += 1
        return batch

    def __len__(self):
        return self.n_batches


def test_predict(model,
                 test_loader: FastTensorDataLoader,
                 err_fn: callable,
                 epoch: int,
                 time_start: float,
                 cum_time: float,
                 train_error: float,
                 ):
    t_elapsed = time.time() - time_start  # Stop the time
    cum_time += t_elapsed
    model.eval()
    test_loader = iter(test_loader)
    test_preds, test_labels = [], []
    try:
        while True:
            b_ts_x, b_ts_y = next(test_loader)
            test_preds.append(model.predict(b_ts_x))
            test_labels.append(b_ts_y)
    except StopIteration:
        test_preds = torch.cat(test_preds)
        test_labels = torch.cat(test_labels)
        test_err, err_name = err_fn(test_labels.detach().cpu(), test_preds.detach().cpu())
    print(f"Epoch {epoch} ({cum_time:5.2f}s) - "
          f"Tr {err_name} = {train_error:6.4f} , "
          f"Ts {err_name} = {test_err:6.4f} -- "
          f"Sigma {model.sigma[0].item():.3f} - Penalty {np.exp(-model.penalty.item()):.2e}")
    return cum_time


def test_train_predict(model,
                       test_loader: FastTensorDataLoader,
                       train_loader: FastTensorDataLoader,
                       err_fn: callable,
                       epoch: int,
                       time_start: float,
                       cum_time: float,
                       ):
    t_elapsed = time.time() - time_start  # Stop the time
    cum_time += t_elapsed
    model.eval()
    test_loader = iter(test_loader)
    train_loader = iter(train_loader)
    test_preds, test_labels = [], []
    train_preds, train_labels = [], []
    try:
        while True:
            b_ts_x, b_ts_y = next(test_loader)
            test_preds.append(model.predict(b_ts_x))
            test_labels.append(b_ts_y)
    except StopIteration:
        test_preds = torch.cat(test_preds)
        test_labels = torch.cat(test_labels)
        test_err, err_name = err_fn(test_labels.detach().cpu(), test_preds.detach().cpu())
    try:
        while True:
            b_tr_x, b_tr_y = next(train_loader)
            train_preds.append(model.predict(b_tr_x))
            train_labels.append(b_tr_y)
    except StopIteration:
        train_preds = torch.cat(train_preds)
        train_labels = torch.cat(train_labels)
        train_err, err_name = err_fn(train_labels.detach().cpu(), train_preds.detach().cpu())
    print(f"Epoch {epoch} ({cum_time:5.2f}s) - "
          f"Tr {err_name} = {train_err:6.4f} , "
          f"Ts {err_name} = {test_err:6.4f} -- "
          f"Sigma {model.sigma[0].item():.3f} - Penalty {np.exp(-model.penalty.item()):.2e}")
    return cum_time


def report_out(idx, total, l):
    MAX_CHAR = 100

    ending = f"{idx:d}/{total:d}"
    num_char = int(math.ceil(idx / total * MAX_CHAR))
    num_char = min(num_char, MAX_CHAR - len(ending))
    num_empty = MAX_CHAR - num_char - len(ending)
    l[0] = "{prog}{pad}{end}".format(
        prog="-" * num_char,
        pad=" " * num_empty if num_empty > 0 else "",
        end=ending
    )


class NKRR(nn.Module):
    def __init__(self, sigma_init, penalty_init, centers_init, opt):
        super().__init__()
        penalty = nn.Parameter(torch.tensor(penalty_init, requires_grad=True))
        self.register_parameter('penalty', penalty)
        sigma = nn.Parameter(torch.tensor(sigma_init, requires_grad=True))
        self.register_parameter('sigma', sigma)
        centers = nn.Parameter(centers_init.requires_grad_())
        self.register_parameter('centers', centers)
        alpha = nn.Parameter(torch.zeros(centers_init.shape[0], 1, requires_grad=True))
        self.register_parameter('alpha', alpha)

        self.opt = opt

    def forward(self, X, Y):
        """
        l = 1/N ||K_{NM} @ a - Y|| + lambda * alpha.T @ K_{MM} @ alpha
        """
        k = DiffGaussianKernel(self.sigma, self.opt)

        preds = self.predict(X)
        loss = torch.mean((preds - Y) ** 2)
        reg = torch.exp(-self.penalty) * (
                self.alpha.T @ (k.mmv(self.centers, self.centers, self.alpha)))

        return (loss + reg), preds

    def predict(self, X):
        k = DiffGaussianKernel(self.sigma, self.opt)
        return k.mmv(X, self.centers, self.alpha)


class FLK_NKRR(nn.Module):
    def __init__(self, sigma_init, penalty_init, centers_init, opt, regularizer, opt_centers,
                 tot_n=None):
        super().__init__()
        falkon.cuda.initialization.init(opt)
        penalty = nn.Parameter(torch.tensor(penalty_init, requires_grad=True))
        self.register_parameter('penalty', penalty)
        sigma = nn.Parameter(torch.tensor(sigma_init, requires_grad=True))
        self.register_parameter('sigma', sigma)

        centers = nn.Parameter(centers_init.requires_grad_(opt_centers))
        if opt_centers:
            self.register_parameter('centers', centers)
        else:
            self.register_buffer('centers', centers)

        self.f_alpha = torch.zeros(centers_init.shape[0], 1, requires_grad=False)
        self.register_buffer('alpha', self.f_alpha)
        self.f_alpha_pc = torch.zeros(centers_init.shape[0], 1, requires_grad=False)
        self.register_buffer('alpha_pc', self.f_alpha_pc)

        self.opt = opt
        self.flk_maxiter = 10
        self.regularizer = regularizer
        self.tot_n = tot_n

    def forward(self, X, Y):
        """
        l = 1/N ||K_{NM} @ a - Y|| + lambda * alpha.T @ K_{MM} @ alpha
        """
        k = DiffGaussianKernel(self.sigma, self.opt)

        preds = self.predict(X)
        loss = torch.mean((preds - Y) ** 2)
        pen = torch.exp(-self.penalty)
        if self.regularizer == "deff":
            d_eff = gauss_effective_dimension(self.sigma, pen, self.centers, t=100)
            reg = d_eff / X.shape[0] ** 4
            # print("d_eff: %.3e - Loss: %.3e" % (d_eff, loss + reg))
        elif self.regularizer == "tikhonov":
            # This is the normal RKHS norm of the function
            reg = pen * (self.alpha.T @ (k.mmv(self.centers, self.centers, self.alpha)))
        else:
            raise ValueError("Regularizer %s not implemented" % (self.regularizer))

        return (loss), preds

    def adapt_alpha(self, X, Y, n_tot=None):
        k = DiffGaussianKernel(self.sigma.detach(), self.opt)
        if X.is_cuda:
            fcls = falkon.InCoreFalkon
        else:
            fcls = falkon.Falkon

        model = fcls(k,
                     torch.exp(-self.penalty).item(),
                     M=self.centers.shape[0],
                     center_selection=FixedSelector(self.centers.detach()),
                     maxiter=self.flk_maxiter,
                     options=self.opt,
                     N=self.tot_n)
        model.fit(X, Y, warm_start=self.alpha_pc)

        self.alpha = model.alpha_.detach()
        self.alpha_pc = model.beta_.detach()

    def predict(self, X):
        k = DiffGaussianKernel(self.sigma, self.opt)
        preds = k.mmv(X, self.centers, self.alpha)
        return preds

    def get_model(self):
        k = DiffGaussianKernel(self.sigma.detach(), self.opt)
        # TODO: make this return the correct class
        model = falkon.InCoreFalkon(k,
                                    torch.exp(-self.penalty).item(),
                                    M=self.centers.shape[0],
                                    center_selection=FixedSelector(self.centers.detach()),
                                    maxiter=self.flk_maxiter,
                                    options=self.opt,
                                    )
        return model


class FLK_HYP_NKRR(nn.Module):
    def __init__(self, sigma_init, penalty_init, centers_init, opt, regularizer, opt_centers,
                 tot_n=None):
        super().__init__()
        falkon.cuda.initialization.init(opt)
        penalty = nn.Parameter(torch.tensor(penalty_init, requires_grad=True))
        self.register_parameter('penalty', penalty)
        sigma = nn.Parameter(torch.tensor(sigma_init, requires_grad=True))
        self.register_parameter('sigma', sigma)

        centers = nn.Parameter(centers_init.requires_grad_(opt_centers))
        if opt_centers:
            self.register_parameter('centers', centers)
        else:
            self.register_buffer('centers', centers)

        self.f_alpha = torch.zeros(centers_init.shape[0], 1, requires_grad=False)
        self.register_buffer('alpha', self.f_alpha)
        self.f_alpha_pc = torch.zeros(centers_init.shape[0], 1, requires_grad=False)
        self.register_buffer('alpha_pc', self.f_alpha_pc)

        self.opt = opt
        self.flk_maxiter = 10
        self.regularizer = regularizer
        self.tot_n = tot_n
        self.model = None

    def forward(self):
        pass

    def adapt_hps(self, X, Y):
        use_hyper = False
        use_deff = True
        nys_d_eff_c = 2
        k = DiffGaussianKernel(self.sigma, self.opt)
        hparams = [w for k, w in self.named_parameters()]

        # 1: Derivative of validation loss wrt hps and wrt params. Validation loss here is simply the unregularized MSE on training data
        preds = self.predict(X)
        loss = torch.mean((preds - Y) ** 2) + torch.exp(-self.penalty) * (
                    self.alpha.T @ (k.mmv(self.centers, self.centers, self.alpha)))

        mse_grad_hp = torch.autograd.grad(loss, hparams, allow_unused=True, retain_graph=True)
        if use_hyper:
            mse_grad_alpha = k.mmv(self.centers, X, preds - Y) / X.shape[0]

            # 2: Derivative of training loss wrt params
            # 2/N * (K_MN(K_NM @ alpha - Y)) + 2*lambda*(K_MM @ alpha)
            func_norm = k.mmv(self.centers, self.centers, self.alpha)
            first_diff = mse_grad_alpha + torch.exp(-self.penalty) * func_norm

            # 3: inverse-hessian of the training loss wrt params @ val-gradient wrt params
            vs = self.solve_hessian(X, mse_grad_alpha)

            # 4: Multiply the result by the derivative of `first_diff` wrt hparams.
            mixed = torch.autograd.grad(first_diff, hparams, grad_outputs=vs, allow_unused=True)
        else:
            mixed = [None] * len(hparams)

        if use_deff:
            # 5: d_eff gradient
            nys_d_eff = gauss_nys_effective_dimension(self.sigma, torch.exp(-self.penalty),
                                                      M=self.centers,
                                                      X=X, t=20,
                                                      preconditioner=None)  # self.model.precond)
            nys_reg = (nys_d_eff_c * nys_d_eff) / X.shape[0]
            # d_eff = gauss_effective_dimension(self.sigma, torch.exp(-self.penalty), X, t=20)
            # print("Full d_eff=%.4e - Nystrom d_eff=%.4e" % (d_eff, nys_d_eff))
            # reg = d_eff / X.shape[0]
            # d_eff_grad = torch.autograd.grad(reg, hparams, allow_unused=True)
            nys_d_eff_grad = torch.autograd.grad(nys_reg, hparams, allow_unused=False)
            # print("lambda grad (full d_eff = %.4e) (nystrom d_eff = %.4e)" % (d_eff_grad[0].item(), nys_d_eff_grad[0].item()))
        else:
            d_eff_grad = [None] * len(hparams)

        final_grads = []
        for ghp, mix, deg in zip(mse_grad_hp, mixed, nys_d_eff_grad):
            grad = 0
            if ghp is not None:
                grad += ghp
            if mix is not None:
                grad -= mix
            if deg is not None:
                grad += deg
            final_grads.append(grad)
        # print("center grads: %6.4e - %6.4e + %6.4e = %6.4e" % (mse_grad_hp[2].sum(), mixed[2].sum(), d_eff_grad[2].sum(), final_grads[2].sum()))
        # print("d_eff: %6.4e" % (reg))
        # print("penalty grads: %6.4e + %6.4e = %6.4e" % (mse_grad_hp[0], d_eff_grad[0], final_grads[0]))
        # print("sigma grads: %6.4e - %6.4e + %6.4e = %6.4e" % (mse_grad_hp[1], mixed[1], d_eff_grad[1], final_grads[1]))
        # print("sigma = %.3f - grad %.2e" % (self.sigma[0].item(), final_grads[1][0].item()))
        # print("penalty = %.3f - grad %.2e" % (self.penalty.item(), final_grads[0].item()))

        for l, g in zip(hparams, final_grads):
            if not l.requires_grad:
                continue
            if l.grad is None:
                l.grad = torch.zeros_like(l)
            if g is not None:
                l.grad += g

    def solve_hessian(self, X, vector):
        if self.model is None:
            raise RuntimeError("Model must be present when solving for the inverse hessian.")
        with torch.no_grad():
            penalty = torch.exp(-self.penalty)

            vector = vector.detach()
            N = X.shape[0]

            cg = ConjugateGradient(self.opt)
            kernel = self.model.kernel
            precond = self.model.precond

            B = precond.apply_t(vector)

            def mmv(sol):
                v = precond.invA(sol)
                cc = kernel.dmmv(X, self.model.ny_points_, precond.invT(v), None)
                return precond.invAt(precond.invTt(cc) + penalty * v)

            d = cg.solve(X0=None, B=B, mmv=mmv, max_iter=self.flk_maxiter)
            c = precond.apply(d)

            return c

    def adapt_alpha(self, X, Y, n_tot=None):
        k = DiffGaussianKernel(self.sigma.detach(), self.opt)
        if X.is_cuda:
            fcls = falkon.InCoreFalkon
        else:
            fcls = falkon.Falkon

        model = fcls(k,
                     torch.exp(-self.penalty).item(),
                     M=self.centers.shape[0],
                     center_selection=FixedSelector(self.centers.detach()),
                     maxiter=self.flk_maxiter,
                     options=self.opt,
                     N=self.tot_n)
        model.fit(X, Y, warm_start=self.alpha_pc)

        self.alpha = model.alpha_.detach()
        self.alpha_pc = model.beta_.detach()
        self.model = model

    def predict(self, X):
        k = DiffGaussianKernel(self.sigma, self.opt)
        preds = k.mmv(X, self.centers, self.alpha)
        return preds

    def get_model(self):
        k = DiffGaussianKernel(self.sigma.detach(), self.opt)
        # TODO: make this return the correct class
        model = falkon.InCoreFalkon(k,
                                    torch.exp(-self.penalty).item(),
                                    M=self.centers.shape[0],
                                    center_selection=FixedSelector(self.centers.detach()),
                                    maxiter=self.flk_maxiter,
                                    options=self.opt,
                                    )
        return model


class FLK_HYP_NKRR_FIX(nn.Module):
    def __init__(self, sigma_init, penalty_init, centers_init, opt, regularizer, opt_centers,
                 nys_d_eff_c, tot_n=None):
        super().__init__()
        falkon.cuda.initialization.init(opt)
        penalty = nn.Parameter(torch.tensor(penalty_init, requires_grad=True))
        self.register_parameter('penalty', penalty)
        sigma = nn.Parameter(torch.tensor(sigma_init, requires_grad=True))
        self.register_parameter('sigma', sigma)

        centers = nn.Parameter(centers_init.requires_grad_(opt_centers))
        if opt_centers:
            self.register_parameter('centers', centers)
        else:
            self.register_buffer('centers', centers)

        self.f_alpha = torch.zeros(centers_init.shape[0], 1, requires_grad=False)
        self.register_buffer('alpha', self.f_alpha)
        self.f_alpha_pc = torch.zeros(centers_init.shape[0], 1, requires_grad=False)
        self.register_buffer('alpha_pc', self.f_alpha_pc)

        self.opt = opt
        self.flk_maxiter = 10
        self.regularizer = regularizer
        self.tot_n = tot_n
        self.model = None
        self.nys_d_eff_c = nys_d_eff_c

    def forward(self):
        pass

    def adapt_hps(self, X, Y):
        hparams = [w for k, w in self.named_parameters()]

        deff, loss = loss_and_deff(kernel_args=self.sigma,
                                   penalty=torch.exp(-self.penalty),
                                   M=self.centers,
                                   X=X, Y=Y, t=20)
        deff_reg = (self.nys_d_eff_c * deff) / X.shape[0]
        final_grads = torch.autograd.grad([deff_reg, loss], hparams, allow_unused=False)

        # print("center grads: %6.4e - %6.4e + %6.4e = %6.4e" % (mse_grad_hp[2].sum(), mixed[2].sum(), d_eff_grad[2].sum(), final_grads[2].sum()))
        # print("d_eff: %6.4e" % (reg))
        # print("penalty grads: %6.4e + %6.4e = %6.4e" % (mse_grad_hp[0], d_eff_grad[0], final_grads[0]))
        # print("sigma grads: %6.4e - %6.4e + %6.4e = %6.4e" % (mse_grad_hp[1], mixed[1], d_eff_grad[1], final_grads[1]))
        # print("sigma = %.3f - grad %.2e" % (self.sigma[0].item(), final_grads[1][0].item()))
        # print("penalty = %.3f - grad %.2e" % (self.penalty.item(), final_grads[0].item()))

        for l, g in zip(hparams, final_grads):
            if not l.requires_grad:
                continue
            if l.grad is None:
                l.grad = torch.zeros_like(l)
            if g is not None:
                l.grad += g

    def adapt_alpha(self, X, Y, n_tot=None):
        k = DiffGaussianKernel(self.sigma.detach(), self.opt)
        if X.is_cuda:
            fcls = falkon.InCoreFalkon
        else:
            fcls = falkon.Falkon

        model = fcls(k,
                     torch.exp(-self.penalty).item(),
                     M=self.centers.shape[0],
                     center_selection=FixedSelector(self.centers.detach()),
                     maxiter=self.flk_maxiter,
                     options=self.opt,
                     N=self.tot_n)
        model.fit(X, Y, warm_start=self.alpha_pc)

        self.alpha = model.alpha_.detach()
        self.alpha_pc = model.beta_.detach()
        self.model = model

    def predict(self, X):
        k = DiffGaussianKernel(self.sigma, self.opt)
        preds = k.mmv(X, self.centers, self.alpha)
        return preds

    def get_model(self):
        k = DiffGaussianKernel(self.sigma.detach(), self.opt)
        # TODO: make this return the correct class
        model = falkon.InCoreFalkon(k,
                                    torch.exp(-self.penalty).item(),
                                    M=self.centers.shape[0],
                                    center_selection=FixedSelector(self.centers.detach()),
                                    maxiter=self.flk_maxiter,
                                    options=self.opt,
                                    )
        return model


def nkrr_ho(Xtr, Ytr,
            Xts, Yts,
            num_epochs: int,
            sigma_type: str,
            sigma_init: float,
            penalty_init: float,
            falkon_centers: CenterSelector,
            falkon_M: int,
            hp_lr: float,
            p_lr: float,
            batch_size: int,
            cuda: bool,
            loss_every: int,
            err_fn,
            opt,
            ):
    # Choose start value for sigma
    if sigma_type == 'single':
        start_sigma = [sigma_init]
    elif sigma_type == 'diag':
        start_sigma = [sigma_init] * Xtr.shape[1]
    else:
        raise ValueError("sigma_type %s unrecognized" % (sigma_type))

    model = NKRR(
        start_sigma,
        penalty_init,
        falkon_centers.select(Xtr, Y=None, M=falkon_M),
        opt,
    )
    if cuda:
        model = model.cuda()

    opt_p = torch.optim.Adam([
        {"params": [model.alpha], "lr": p_lr},
    ])
    opt_hp = torch.optim.Adam([
        {"params": [model.sigma, model.penalty, model.centers], "lr": hp_lr},
    ])

    train_loader = FastTensorDataLoader(Xtr, Ytr, batch_size=batch_size, shuffle=True,
                                        drop_last=False, cuda=cuda)
    test_loader = FastTensorDataLoader(Xts, Yts, batch_size=batch_size, shuffle=False,
                                       drop_last=False, cuda=cuda)

    for epoch in range(num_epochs):
        train_loader = iter(train_loader)
        model.train()
        e_start = time.time()

        running_error = 0
        samples_processed = 0
        try:
            for i in itertools.count(0):
                b_tr_x, b_tr_y = next(train_loader)
                samples_processed += b_tr_x.shape[0]

                opt_p.zero_grad()
                opt_hp.zero_grad()
                loss, preds = model(b_tr_x, b_tr_y)
                loss.backward()
                opt_p.step()
                opt_hp.step()

                err, err_name = err_fn(b_tr_y.detach().cpu(), preds.detach().cpu())
                running_error += err * b_tr_x.shape[0]
                if i % loss_every == (loss_every - 1):
                    print(f"step {i} - {err_name} {running_error / samples_processed}")
        except StopIteration:
            test_predict(model=model, test_loader=test_loader, err_fn=err_fn,
                         epoch=epoch, time_start=e_start,
                         train_error=running_error / samples_processed)


def flk_nkrr_ho_fix(Xtr, Ytr,
                    Xts, Yts,
                    num_epochs: int,
                    sigma_type: str,
                    sigma_init: float,
                    penalty_init: float,
                    falkon_centers: CenterSelector,
                    falkon_M: int,
                    hp_lr: float,
                    p_lr: float,  # Only for signature compatibility
                    batch_size: int,
                    cuda: bool,
                    loss_every: int,
                    err_fn,
                    opt,
                    regularizer: str,
                    opt_centers: bool,
                    deff_factor: int,
                    ):
    """
    Algorithm description:
        Only use the training-data to minimize the objective with respect to params and hyperparams.
        At each iteration, a mini-batch of training data is picked and both params and hps are
        optimized simultaneously: the former using Falkon, the latter using Adam.
        The optimization objective is the squared loss plus a regularizer which depends on the
        'regularizer' parameter (it can be either 'tikhonov' which means the squared norm of the
        predictor will be used for regularization, or 'deff' so the effective-dimension will be
        used instead. Note that the effective dimension calculation is likely to be unstable.)
    """
    print("Starting Falkon-NKRR-HO - FIXED VERSION - optimization.")
    print(f"{num_epochs} epochs - {sigma_type} sigma ({sigma_init}) - penalty ({penalty_init}) - "
          f"{falkon_M} centers. HP-LR={hp_lr} - batch {batch_size} - D-Eff factor {deff_factor}")
    # Choose start value for sigma
    if sigma_type == 'single':
        start_sigma = [sigma_init]
    elif sigma_type == 'diag':
        start_sigma = [sigma_init] * Xtr.shape[1]
    else:
        raise ValueError("sigma_type %s unrecognized" % (sigma_type))

    model = FLK_HYP_NKRR_FIX(
        start_sigma,
        penalty_init,
        falkon_centers.select(Xtr, Y=None, M=falkon_M),
        opt,
        regularizer,
        opt_centers,
        nys_d_eff_c=deff_factor,
    )
    if cuda:
        model = model.cuda()

    opt_hp = torch.optim.Adam([
        {"params": model.parameters(), "lr": hp_lr},
    ])

    train_loader = FastTensorDataLoader(Xtr, Ytr, batch_size=batch_size, shuffle=True,
                                        drop_last=False, cuda=cuda)
    test_loader = FastTensorDataLoader(Xts, Yts, batch_size=batch_size, shuffle=False,
                                       drop_last=False, cuda=cuda)

    cum_time = 0
    for epoch in range(num_epochs):
        train_loader_it = iter(train_loader)
        model.train()
        e_start = time.time()

        with output(output_type="list", initial_len=1, interval=0) as output_list:
            try:
                for i in itertools.count(0):
                    b_tr_x, b_tr_y = next(train_loader_it)

                    # Optimize the parameters alpha using Falkon (on training-batch)
                    # model.adapt_alpha(b_tr_x, b_tr_y)
                    # Calculate gradient for the hyper-parameters (on training-batch)
                    opt_hp.zero_grad()
                    model.adapt_hps(b_tr_x, b_tr_y)
                    opt_hp.step()
                    report_out(i, len(train_loader), output_list)
            except StopIteration:
                model.adapt_alpha(Xtr, Ytr)
                cum_time = test_train_predict(
                    model=model, test_loader=test_loader, train_loader=train_loader,
                    err_fn=err_fn, epoch=epoch, time_start=e_start, cum_time=cum_time)
    return model.get_model()


def flk_nkrr_ho(Xtr, Ytr,
                Xts, Yts,
                num_epochs: int,
                sigma_type: str,
                sigma_init: float,
                penalty_init: float,
                falkon_centers: CenterSelector,
                falkon_M: int,
                hp_lr: float,
                p_lr: float,  # Only for signature compatibility
                batch_size: int,
                cuda: bool,
                loss_every: int,
                err_fn,
                opt,
                regularizer: str,
                opt_centers: bool,
                ):
    """
    Algorithm description:
        Only use the training-data to minimize the objective with respect to params and hyperparams.
        At each iteration, a mini-batch of training data is picked and both params and hps are
        optimized simultaneously: the former using Falkon, the latter using Adam.
        The optimization objective is the squared loss plus a regularizer which depends on the
        'regularizer' parameter (it can be either 'tikhonov' which means the squared norm of the
        predictor will be used for regularization, or 'deff' so the effective-dimension will be
        used instead. Note that the effective dimension calculation is likely to be unstable.)
    """
    print("Starting Falkon-NKRR-HO optimization.")
    print(f"{num_epochs} epochs - {sigma_type} sigma ({sigma_init}) - penalty ({penalty_init}) - "
          f"{falkon_M} centers. HP-LR={hp_lr} - batch {batch_size} - {regularizer} regularizer")
    # Choose start value for sigma
    if sigma_type == 'single':
        start_sigma = [sigma_init]
    elif sigma_type == 'diag':
        start_sigma = [sigma_init] * Xtr.shape[1]
    else:
        raise ValueError("sigma_type %s unrecognized" % (sigma_type))

    model = FLK_HYP_NKRR(
        start_sigma,
        penalty_init,
        falkon_centers.select(Xtr, Y=None, M=falkon_M),
        opt,
        regularizer,
        opt_centers,
    )
    if cuda:
        model = model.cuda()

    opt_hp = torch.optim.Adam([
        {"params": model.parameters(), "lr": hp_lr},
    ])

    train_loader = FastTensorDataLoader(Xtr, Ytr, batch_size=batch_size, shuffle=True,
                                        drop_last=False, cuda=cuda)
    test_loader = FastTensorDataLoader(Xts, Yts, batch_size=batch_size, shuffle=False,
                                       drop_last=False, cuda=cuda)

    cum_time = 0
    for epoch in range(num_epochs):
        train_loader = iter(train_loader)
        model.train()
        e_start = time.time()

        running_error = 0
        samples_processed = 0
        try:
            # model.adapt_alpha(Xtr.cuda(), Ytr.cuda())
            for i in itertools.count(0):
                b_tr_x, b_tr_y = next(train_loader)
                samples_processed += b_tr_x.shape[0]

                # Optimize the parameters alpha using Falkon (on training-batch)
                model.adapt_alpha(b_tr_x, b_tr_y)
                # Calculate gradient for the hyper-parameters (on training-batch)
                opt_hp.zero_grad()
                # loss, preds = model(b_tr_x, b_tr_y)
                # loss.backward()
                model.adapt_hps(b_tr_x, b_tr_y)
                # Change theta
                opt_hp.step()

                preds = model.predict(b_tr_x)  # Redo predictions to check adapted model
                err, err_name = err_fn(b_tr_y.detach().cpu(), preds.detach().cpu())
                running_error += err * preds.shape[0]
                if i % loss_every == (loss_every - 1):
                    print(f"step {i} - {err_name} {running_error / samples_processed}")
        except StopIteration:
            train_error = 1
            if samples_processed > 0:
                train_error = running_error / samples_processed
            cum_time = test_predict(model=model, test_loader=test_loader, err_fn=err_fn,
                                    epoch=epoch, time_start=e_start, cum_time=cum_time,
                                    train_error=train_error)
    return model.get_model()


def flk_nkrr_ho_val(Xtr, Ytr,
                    Xts, Yts,
                    num_epochs: int,
                    sigma_type: str,
                    sigma_init: float,
                    penalty_init: float,
                    falkon_centers: CenterSelector,
                    falkon_M: int,
                    hp_lr: float,
                    p_lr: float,  # Only for signature compatibility
                    batch_size: int,
                    cuda: bool,
                    loss_every: int,
                    err_fn,
                    opt,
                    regularizer: str,
                    opt_centers: bool,
                    ):
    """
    Algorithm description:
        Use a training-set (mini-batched) to minimize the objective wrt parameters (using Falkon)
        and a validation-set (mini-batched) to minimize wrt the hyper-parameters (using Adam).

        At each iteration, a mini-batch of training data and one of validation data are picked.
        First the hyper-parameters are moved in the direction of the validation gradient, and then
        the parameters are moved in the direction of the training gradient (using Falkon).

        The hyper-parameter (validation-data) objective is the squared loss plus a regularizer which
        depends on the 'regularizer' parameter.

        Since each iteration involves one mini-batch of two differently sized sets, behaviour
        around mini-batch selection is a bit strange: check the code!
    """
    print("Starting Falkon-NKRR-HO-VAL optimization.")
    print(f"{num_epochs} epochs - {sigma_type} sigma ({sigma_init}) - penalty ({penalty_init}) - "
          f"{falkon_M} centers. HP-LR={hp_lr} - batch {batch_size} - {regularizer} regularizer")
    # Choose start value for sigma
    if sigma_type == 'single':
        start_sigma = [sigma_init]
    elif sigma_type == 'diag':
        start_sigma = [sigma_init] * Xtr.shape[1]
    else:
        raise ValueError("sigma_type %s unrecognized" % (sigma_type))

    n_tr_samples = int(Xtr.shape[0] * 0.8)
    model = FLK_NKRR(
        start_sigma,
        penalty_init,
        falkon_centers.select(Xtr, Y=None, M=falkon_M),
        opt,
        regularizer,
        opt_centers,
    )
    if cuda:
        model = model.cuda()

    opt_hp = torch.optim.Adam([
        {"params": model.parameters(), "lr": hp_lr},
    ])

    print("Using %d training samples - %d validation samples." %
          (n_tr_samples, Xtr.shape[0] - n_tr_samples))
    train_loader = FastTensorDataLoader(Xtr[:n_tr_samples], Ytr[:n_tr_samples],
                                        batch_size=batch_size, shuffle=True, drop_last=False,
                                        cuda=cuda)
    val_loader = FastTensorDataLoader(Xtr[n_tr_samples:], Ytr[n_tr_samples:], batch_size=batch_size,
                                      shuffle=True, drop_last=False, cuda=cuda)
    test_loader = FastTensorDataLoader(Xts, Yts, batch_size=batch_size, shuffle=False,
                                       drop_last=False, cuda=cuda)
    cum_time = 0

    for epoch in range(num_epochs):
        train_loader = iter(train_loader)
        val_loader = iter(val_loader)
        model.train()
        e_start = time.time()

        running_error = 0
        samples_processed = 0
        try:
            # model.adapt_alpha(Xtr[:n_tr_samples].cuda(), Ytr[:n_tr_samples].cuda())
            for i in itertools.count(0):
                b_tr_x, b_tr_y = next(train_loader)
                try:
                    b_vl_x, b_vl_y = next(val_loader)
                except StopIteration:  # We assume that the validation loader is smaller, so we always restart it.
                    val_loader = iter(val_loader)
                    b_vl_x, b_vl_y = next(val_loader)
                samples_processed += b_vl_x.shape[0]

                # Outer (hp) opt with validation
                opt_hp.zero_grad()
                loss, preds = model(b_vl_x, b_vl_y)
                loss.backward()
                opt_hp.step()
                # Inner opt with train
                model.adapt_alpha(b_tr_x, b_tr_y)
                # Redo predictions to check adapted model
                preds = model.predict(b_vl_x)
                err, err_name = err_fn(b_vl_y.detach().cpu(), preds.detach().cpu())
                running_error += err * preds.shape[0]
                if i % loss_every == (loss_every - 1):
                    print(f"step {i} - {err_name} {running_error / samples_processed}")
        except StopIteration:
            cum_time = test_predict(model=model, test_loader=test_loader, err_fn=err_fn,
                                    epoch=epoch, time_start=e_start, cum_time=cum_time,
                                    train_error=running_error / samples_processed)
