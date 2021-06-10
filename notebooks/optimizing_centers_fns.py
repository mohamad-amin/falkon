import math
import time

import numpy as np
import torch
from libsvmdata import fetch_libsvm

import falkon
from falkon.hypergrad.common import test_train_predict
from falkon.hypergrad.complexity_reg import (
    GPComplexityReg, SimpleFalkonComplexityReg, reporting,
    NoRegFalkonComplexityReg
)

dset_sigmas_15la = {
    "cpusmall": 8.,
    "abalone": 20.,
    "space_ga": 3.,
    "svmguide1": 3.,
    "cadata": 5.,
}
default_falkon_opt = falkon.FalkonOptions(keops_active="no", use_cpu=True)


def compare_ker_fro(model, X):
    centers = model.centers.detach()
    M = centers.shape[0]
    kernel = falkon.kernels.GaussianKernel(model.sigma.detach(), default_falkon_opt)

    # Nystrom B, G
    Bnm_nys = kernel(X, centers)
    Gmm_nys = kernel(centers, centers)
    # SVD B, G
    full_kernel = kernel(X, X)
    u, s, v = torch.svd(full_kernel)
    Bnm_svd = full_kernel @ u[:, :M]
    Gmm_svd = u[:,:M].T @ full_kernel @ u[:, :M]

    Ktilde_nys = Bnm_nys @ torch.pinverse(Gmm_nys) @ Bnm_nys.T
    Ktilde_svd = Bnm_svd @ torch.pinverse(Gmm_svd) @ Bnm_svd.T
    Ktilde_svd_2 = u[:, :M] @ torch.diag(s[:M]) @ u[:, :M].T
    # torch.testing.assert_allclose(Ktilde_svd, Ktilde_svd_2)
    return math.sqrt(torch.sum((Ktilde_nys - Ktilde_svd)**2).item())


def compare_ker_tr(model, X):
    centers = model.centers.detach()
    M = centers.shape[0]
    kernel = falkon.kernels.GaussianKernel(model.sigma.detach(), default_falkon_opt)

    # Nystrom B, G
    Bnm_nys = kernel(X, centers)
    Gmm_nys = kernel(centers, centers)
    # SVD B, G
    full_kernel = kernel(X, X)
    u, s, v = torch.svd(full_kernel)
    Bnm_svd = full_kernel @ u[:, :M]
    Gmm_svd = u[:,:M].T @ full_kernel @ u[:, :M]

    Ktilde_nys = Bnm_nys @ torch.pinverse(Gmm_nys) @ Bnm_nys.T
    Ktilde_svd = Bnm_svd @ torch.pinverse(Gmm_svd) @ Bnm_svd.T
    Ktilde_svd_2 = u[:, :M] @ torch.diag(s[:M]) @ u[:, :M].T
    # torch.testing.assert_allclose(Ktilde_svd, Ktilde_svd_2)

    return torch.trace(Ktilde_nys - Ktilde_svd)


def simple_solve_nkrr(Bnm, Gmm, Y, la):
    h = Bnm.T @ Bnm + Bnm.shape[0] * la * Gmm
    alpha = torch.pinverse(h) @ Bnm.T @ Y
    return alpha


def simple_solve_krr(kernel, Xtr, Ytr, la):
    h = kernel(Xtr, Xtr) + Xtr.shape[0] * la * torch.eye(Xtr.shape[0])
    alpha = torch.pinverse(h) @ Ytr
    return alpha


def simple_pred_nkrr(Bnm, alpha):
    return Bnm @ alpha


def simple_pred_krr(kernel, Xtr, Xts, alpha):
    return kernel(Xts, Xtr) @ alpha


def mse(y_true, y_pred):
    if not isinstance(y_true, np.ndarray):
        y_true = y_true.numpy()
    if not isinstance(y_pred, np.ndarray):
        y_pred = y_pred.numpy()

    y_mean = np.abs(np.mean(y_true))

    return math.sqrt(np.mean((y_true.reshape(-1) - y_pred.reshape(-1))**2)) / y_mean, "nrmse"


def preprocess_dataset(X, Y, n_train):
    shuffle = np.random.permutation(X.shape[0])
    X = X[shuffle]
    Y = Y[shuffle]
    Xtr = X[:n_train].clone()#.to(dtype=torch.float64)
    Ytr = Y[:n_train].clone()#.to(dtype=torch.float64)
    Xts = X[n_train:].clone()#.to(dtype=torch.float64)
    Yts = Y[n_train:].clone()#.to(dtype=torch.float64)
    # Preprocess X
    mean = Xtr.mean(axis=0, keepdims=True)
    std = Xtr.std(axis=0, keepdims=True)
    Xtr -= mean
    Xts -= mean
    Xtr /= std
    Xts /= std
    # Preprocess Y
    #norm = torch.linalg.norm(Ytr)
    #Ytr /= norm
    #Yts /= norm
    return Xtr, Ytr, Xts, Yts


def load_dset(dset_name, n_train=1000, penalty=8., sigma=None, dtype=torch.float32):
    X, y = fetch_libsvm(dset_name)
    X = torch.from_numpy(np.asarray(X.todense())).to(dtype=dtype)
    Y = torch.from_numpy(y.reshape(-1, 1)).to(dtype=dtype)
    print(X.shape, Y.shape)
    Xtr, Ytr, Xts, Yts = preprocess_dataset(X, Y, n_train=n_train)

    penalty_init = torch.tensor(penalty, dtype=Xtr.dtype)
    if sigma is not None:
        sigma_init = torch.tensor([sigma], dtype=Xtr.dtype)
    else:
        sigma_init = torch.tensor([dset_sigmas_15la[dset_name]], dtype=Xtr.dtype)

    kernel = falkon.kernels.GaussianKernel(sigma_init, opt=default_falkon_opt)

    return Xtr, Ytr, Xts, Yts, penalty_init, sigma_init, kernel


def get_random_bg(Xtr, M, kernel, Xts=None):
    centers_init = torch.randn(M, Xtr.shape[1], dtype=Xtr.dtype)
    Bnm = kernel(Xtr, centers_init)
    Gmm = kernel(centers_init, centers_init)
    Bnm_test = None
    if Xts is not None:
        Bnm_test = kernel(Xts, centers_init)
    return Bnm, Gmm, Bnm_test


def get_nystrom_bg(Xtr, M, kernel, Xts=None):
    centers_init = Xtr[np.random.choice(Xtr.shape[0], size=M, replace=False), :]
    Bnm = kernel(Xtr, centers_init)
    Gmm = kernel(centers_init, centers_init)
    Bnm_test = None
    if Xts is not None:
        Bnm_test = kernel(Xts, centers_init)
    return Bnm, Gmm, Bnm_test


def get_svd_bg(Xtr, M, kernel, Xts=None):
    full_kernel = kernel(Xtr, Xtr)
    u, s, v = torch.svd(full_kernel)
    Bnm = full_kernel @ u[:, :M]
    Gmm = u[:,:M].T @ full_kernel @ u[:, :M]
    Bnm_test = None
    if Xts is not None:
        Bnm_test = kernel(Xts, Xtr) @ u[:, :M]
    return Bnm, Gmm, Bnm_test


def get_train_error(Bnm, Bnm_test, Gmm, Ytr, Yts, penalty):
    alpha = simple_solve_nkrr(Bnm, Gmm, Ytr, penalty)
    train_preds = simple_pred_nkrr(Bnm, alpha)
    err, name = mse(Ytr, train_preds)
    return err


def get_test_error(Bnm, Bnm_test, Gmm, Ytr, Yts, penalty):
    alpha = simple_solve_nkrr(Bnm, Gmm, Ytr, penalty)
    test_preds = simple_pred_nkrr(Bnm_test, alpha)
    err, name = mse(Yts, test_preds)
    return err


def krr_train_error(Xtr, Xts, Ytr, Yts, kernel, la):
    alpha = simple_solve_krr(kernel, Xtr, Ytr, la)
    train_preds = simple_pred_krr(kernel, Xtr, Xtr, alpha)
    err, name = mse(Ytr, train_preds)
    return err


def krr_test_error(Xtr, Xts, Ytr, Yts, kernel, la):
    alpha = simple_solve_krr(kernel, Xtr, Ytr, la)
    test_preds = simple_pred_krr(kernel, Xtr, Xts, alpha)
    err, name = mse(Yts, test_preds)
    return err


def choose_centers(Xtr, M, random=False):
    if random:
        return torch.randn(M, Xtr.shape[1], dtype=Xtr.dtype, device=Xtr.device)
    else:
        idx = torch.randperm(Xtr.shape[0])[:M]
        return Xtr[idx].clone().detach()


# MAIN FUNCTIONS
def train_sgpr_like(opt_model,
                    penalty_init,
                    sigma_init,
                    centers_init,
                    only_trace,
                    lr,
                    epochs,
                    Xtr,
                    Ytr,
                    Xts,
                    Yts,
                    kernel,
                    opt_centers=True,
                    opt_sigma=True,
                    opt_penalty=True):
    if opt_model == "SGPR":
        opt_model_cls = GPComplexityReg
    elif opt_model == "Falkon":
        opt_model_cls = SimpleFalkonComplexityReg
    elif opt_model == "NoRegFalkon":
        opt_model_cls = NoRegFalkonComplexityReg
    else:
        raise ValueError(opt_model)
    model = opt_model_cls(
        penalty_init=penalty_init,
        sigma_init=sigma_init,
        centers_init=centers_init,
        opt_centers=opt_centers,
        opt_sigma=opt_sigma,
        opt_penalty=opt_penalty,
        flk_opt=default_falkon_opt,
        flk_maxiter=10,
        verbose_tboard=False,
        cuda=False,
        T=1,
        only_trace=only_trace,
    )
    ## Closed-form baselines
    penalty = model.penalty_val.detach()
    tr_err_krr = krr_train_error(Xtr, Xts, Ytr, Yts, kernel, penalty)
    ts_err_krr = krr_test_error(Xtr, Xts, Ytr, Yts, kernel, penalty)
    Bnm, Gmm, Bnm_test = get_svd_bg(Xtr, centers_init.shape[0], kernel, Xts)
    tr_err_svd = get_train_error(Bnm, Bnm_test, Gmm, Ytr, Yts, penalty)
    ts_err_svd = get_test_error(Bnm, Bnm_test, Gmm, Ytr, Yts, penalty)
    ## Run SGD training
    train_errors, test_errors, fro_errors = [], [], []
    opt_hp = torch.optim.Adam(model.parameters(), lr=lr)
    cum_time, cum_step = 0, 0
    for epoch in range(epochs):
        e_start = time.time()

        opt_hp.zero_grad()
        losses = model.hp_loss(Xtr, Ytr)
        grads = model.hp_grad(*losses, accumulate_grads=True)
        reporting(model.named_parameters(), grads, losses, model.loss_names, verbose=False, step=cum_step)
        opt_hp.step()

        model.adapt_alpha(Xtr, Ytr)
        cum_time, train_err, test_err = test_train_predict(
            model=model, Xtr=Xtr, Ytr=Ytr, Xts=Xts, Yts=Yts,
            err_fn=mse, epoch=epoch, time_start=e_start, cum_time=cum_time)
        # Frobenius error: the squared-frobenius norm of the difference between the
        # kernel matrix with the optimized centers (Nystrom), and the kernel
        # matrix obtained by truncated-SVD of the kernel (with the same number of sing-vecs)
        fro_errors.append(compare_ker_fro(model, Xtr))
        train_errors.append(train_err)
        test_errors.append(test_err)
        print("Fro err: %.3f" % (fro_errors[-1]))
        cum_step += 1

    return {
        "tr_errs": np.asarray(train_errors),
        "ts_errs": np.asarray(test_errors),
        "fro_errs": np.asarray(fro_errors),
        "tr_err_krr": tr_err_krr,
        "ts_err_krr": ts_err_krr,
        "tr_err_svd": tr_err_svd,
        "ts_err_svd": ts_err_svd,
    }
