import itertools
import time

import numpy as np
import torch
import torch.nn as nn

import falkon
from falkon.center_selection import FixedSelector, CenterSelector
from falkon.hypergrad.leverage_scores import subs_deff_simple, GaussEffectiveDimension
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
    def __init__(self, sigma_init, penalty_init, centers_init, opt, regularizer, opt_centers, tot_n=None):
        super().__init__()
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
            d_eff = GaussEffectiveDimension.apply(
                t=100, kernel_args=self.sigma, penalty=pen, X=self.centers)
            reg = d_eff / X.shape[0]
        elif self.regularizer == "tikhonov":
            # This is the normal RKHS norm of the function
            reg = pen * (self.alpha.T @ (k.mmv(self.centers, self.centers, self.alpha)))
        else:
            raise ValueError("Regularizer %s not implemented" % (self.regularizer))

        return (loss + reg), preds

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
            model.adapt_alpha(Xtr.cuda(), Ytr.cuda())
            for i in itertools.count(0):
                b_tr_x, b_tr_y = next(train_loader)
                samples_processed += b_tr_x.shape[0]

                # Calculate gradient for the hyper-parameters (on training-batch)
                opt_hp.zero_grad()
                loss, preds = model(b_tr_x, b_tr_y)
                loss.backward()
                # Change theta
                opt_hp.step()
                # Optimize the parameters alpha using Falkon (on training-batch)
                #model.adapt_alpha(b_tr_x, b_tr_y)

                preds = model.predict(b_tr_x)  # Redo predictions to check adapted model
                err, err_name = err_fn(b_tr_y.detach().cpu(), preds.detach().cpu())
                running_error += err * preds.shape[0]
                if i % loss_every == (loss_every - 1):
                    print(f"step {i} - {err_name} {running_error / samples_processed}")
                    running_error = 0
                    samples_processed = 0
        except StopIteration:
            cum_time = test_predict(model=model, test_loader=test_loader, err_fn=err_fn,
                                    epoch=epoch, time_start=e_start, cum_time=cum_time,
                                    train_error=running_error / samples_processed)
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
            #model.adapt_alpha(Xtr[:n_tr_samples].cuda(), Ytr[:n_tr_samples].cuda())
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

