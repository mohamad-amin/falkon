import argparse
import time
import datetime
import functools

import numpy as np
import pandas as pd
from falkon.hypergrad.nkrr_ho import flk_nkrr_ho_fix

from datasets import get_load_fn, equal_split
from benchmark_utils import *
from error_metrics import get_err_fns, mse


def run_gmap_exp(dataset: Dataset,
                 sigma_type: str,
                 inner_maxiter: int,
                 hessian_cg_steps: int,
                 hessian_cg_tol: float,
                 loss: str,
                 seed: int):
    import torch
    torch.manual_seed(seed)
    from falkon import FalkonOptions
    from falkon.center_selection import FixedSelector
    from falkon.hypergrad.falkon_ho import run_falkon_hypergrad, map_gradient, ValidationLoss

    loss = ValidationLoss(loss)
    err_fns = get_err_fns(dataset)
    Xtr, Ytr, Xts, Yts, metadata = get_load_fn(dataset)(np.float32, as_torch=True)
    centers = torch.from_numpy(metadata['centers']).cuda()
    val_type = "full"

    if val_type == "split":
        # We use a validation split (redefinition of Xtr, Ytr).
        train_frac = 0.8
        idx_tr, idx_val = equal_split(Xtr.shape[0], train_frac=train_frac)
        Xval, Yval = Xtr[idx_val], Ytr[idx_val]
        Xtr, Ytr = Xtr[idx_tr], Ytr[idx_tr]
    elif val_type == "full":
        Xval, Yval = Xts, Yts
    else:
        raise ValueError("Validation type %s" % (val_type))
    print("Will use %d train - %d validation samples for gradient map evaluation" %
          (Xtr.shape[0], Xval.shape[0]))
    data = {'Xtr': Xtr.cuda(), 'Ytr': Ytr.cuda(), 'Xts': Xval.cuda(), 'Yts': Yval.cuda()}

    falkon_opt = FalkonOptions(use_cpu=False)

    df: pd.DataFrame = map_gradient(data,
                                    falkon_centers=FixedSelector(centers),
                                    falkon_M=centers.shape[0],
                                    falkon_maxiter=inner_maxiter,
                                    falkon_opt=falkon_opt,
                                    sigma_type=sigma_type,
                                    hessian_cg_steps=hessian_cg_steps,
                                    hessian_cg_tol=hessian_cg_tol,
                                    loss=loss,
                                    err_fns=err_fns,
                                   )
    out_fn = f"./logs/gd_map_{dataset}_{int(datetime.datetime.timestamp(datetime.datetime.now()) * 1000)}.csv"
    print("Saving gradient map to %s" % (out_fn))
    df.to_csv(out_fn)


def run_gpflow(dataset: Dataset,
               num_iter: int,
               lr: float,
               sigma_type: str,
               sigma_init: float,
               opt_centers: bool,
               seed: int,
               gradient_map: bool,
               ):
    batch_size = 1000
    dt = np.float64
    model_type = "sgpr"
    import gpflow
    import tensorflow_probability as tfp
    gpflow.config.set_default_float(dt)
    from gpflow_model import TrainableSVGP, TrainableSGPR, TrainableGPR

    # Load data
    Xtr, Ytr, Xts, Yts, metadata = get_load_fn(dataset)(dt, as_torch=False, as_tf=True)
    err_fns = get_err_fns(dataset)
    err_fns = [functools.partial(fn, **metadata) for fn in err_fns]
    centers = metadata['centers'].astype(dt)

    # We use a validation split (redefinition of Xtr, Ytr).
    if not gradient_map:
        train_frac = 0.8
        idx_tr, idx_val = equal_split(Xtr.shape[0], train_frac=train_frac)
        Xval, Yval = Xtr[idx_val], Ytr[idx_val]
        Xtr, Ytr = Xtr[idx_tr], Ytr[idx_tr]
        print("Splitting data for validation and testing: Have %d train - %d validation samples" %
              (Xtr.shape[0], Xval.shape[0]))

    # Data are divided by `lengthscales`
    # variance is multiplied outside of the exponential
    if sigma_type == "single":
        initial_sigma = np.array([sigma_init], dtype=dt)
    elif sigma_type == "diag":
        initial_sigma = np.array([sigma_init] * Xtr.shape[1], dtype=dt)
    else:
        raise ValueError("Sigma type %s not recognized" % (sigma_type))
    kernel_variance = 3
    kernel = gpflow.kernels.SquaredExponential(lengthscales=initial_sigma, variance=kernel_variance)
    kernel.lengthscales = gpflow.Parameter(initial_sigma, transform=tfp.bijectors.Identity())
    gpflow.set_trainable(kernel.variance, False)
    gpflow.set_trainable(kernel.lengthscales, True)


    if model_type == "sgpr":
        gpflow_model = TrainableSGPR(kernel,
                                       centers,
                                       num_iter=num_iter,
                                       err_fn=err_fns[0],
                                       train_hyperparams=True,
                                       lr=lr,
                                       )
    elif model_type == "svgp":
        gpflow_model = TrainableSVGP(kernel,
                                       centers,
                                       batch_size=batch_size,
                                       num_iter=num_iter,
                                       err_fn=err_fns[0],
                                       var_dist="full",
                                       classif=None,
                                       error_every=10,
                                       train_hyperparams=False,
                                       optimize_centers=False,
                                       lr=lr,
                                       natgrad_lr=0.1)
    elif model_type == "gpr":
        gpflow_model = TrainableGPR(kernel,
                                    num_iter=num_iter,
                                    err_fn=err_fns[0],
                                    lr=lr,
                                    )
    else:
        raise ValueError("Model type %s" % (model_type))

    if gradient_map:
        if model_type not in ["sgpr", "svgp"]:
            raise ValueError("Gradient-map only doable with SGPR or SVGP models")
        df = gpflow_model.gradient_map(Xtr, Ytr, Xts, Yts, variance_list=np.linspace(0.1, 2.0, 20), lengthscale_list=np.linspace(1, 20, 20))
        out_fn = f"./logs/gd_map_{model_type}_{dataset}_{int(datetime.datetime.timestamp(datetime.datetime.now()) * 1000)}.csv"
        print("Saving gradient map to %s" % (out_fn))
        df.to_csv(out_fn)
    else:
        gpflow_model.fit(Xtr, Ytr, Xval, Yval)
        train_pred = gpflow_model.predict(Xtr)
        test_pred = gpflow_model.predict(Xts)
        print("Test (unseen) errors (no retrain)")
        for efn in err_fns:
            train_err, err = efn(Ytr, train_pred)
            test_err, err = efn(Yts, test_pred)
            print(f"Train {err}: {train_err:.5f} -- Test {err}: {test_err:.5f}")


def run_nkrr(dataset: Dataset,
             num_epochs: int,
             hp_lr: float,
             p_lr: float,
             sigma_type: str,
             sigma_init: float,
             penalty_init: float,
             M: int,
             seed: int,
             regularizer: str,
             opt_centers: bool,
             deff_factor: int,
             loss_type: str,
             ):
    cuda = True
    batch_size = 20_000_000
    loss_every = 1
    mode = "flk_fix"  # flk, flk_val

    print("Running Hyperparameter Tuning Experiment.")
    print(f"CUDA: {cuda} -- Batch {batch_size} -- Loss report {loss_every} -- Mode {mode} -- "
          f"Optimize centers {opt_centers}.")

    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    from falkon import FalkonOptions
    from falkon.center_selection import FixedSelector, UniformSelector
    from falkon.hypergrad.nkrr_ho import nkrr_ho, flk_nkrr_ho, flk_nkrr_ho_val

    Xtr, Ytr, Xts, Yts, metadata = get_load_fn(dataset)(np.float32, as_torch=True)
    err_fns = get_err_fns(dataset)

    # Center selection
    if 'centers' in metadata and False:
        centers = torch.from_numpy(metadata['centers'])
    else:
        selector = UniformSelector(np.random.default_rng(seed))
        centers = selector.select(Xtr, None, M)

    # Initialize Falkon model
    falkon_opt = FalkonOptions(use_cpu=False, debug=False, cg_tolerance=1e-8, pc_epsilon_32=1e-6)

    if mode == "nkrr":
        nkrr_ho(
            Xtr, Ytr, Xts, Yts,
            num_epochs=num_epochs,
            sigma_type=sigma_type,
            sigma_init=sigma_init,
            penalty_init=penalty_init,
            falkon_centers=FixedSelector(centers),
            falkon_M=M,
            hp_lr=hp_lr,
            p_lr=p_lr,
            batch_size=batch_size,
            cuda=cuda,
            err_fn=err_fns[0],
            opt=falkon_opt,
            loss_every=loss_every,
            opt_centers=opt_centers,
        )
    elif mode == "flk":
        best_model = flk_nkrr_ho(
            Xtr, Ytr, Xts, Yts,
            num_epochs=num_epochs,
            sigma_type=sigma_type,
            sigma_init=sigma_init,
            penalty_init=penalty_init,
            falkon_centers=FixedSelector(centers),
            falkon_M=M,
            hp_lr=hp_lr,
            p_lr=0.01,  # Unused
            batch_size=batch_size,
            cuda=cuda,
            err_fn=err_fns[0],
            opt=falkon_opt,
            loss_every=loss_every,
            regularizer=regularizer,
            opt_centers=opt_centers,
        )
    elif mode == "flk_val":
        best_model = flk_nkrr_ho_val(
            Xtr, Ytr, Xts, Yts,
            num_epochs=num_epochs,
            sigma_type=sigma_type,
            sigma_init=sigma_init,
            penalty_init=penalty_init,
            falkon_centers=FixedSelector(centers),
            falkon_M=M,
            hp_lr=hp_lr,
            p_lr=0.01,  # Unused
            batch_size=batch_size,
            cuda=cuda,
            err_fn=err_fns[0],
            opt=falkon_opt,
            loss_every=loss_every,
            regularizer=regularizer,
            opt_centers=opt_centers,
        )
    elif mode == "flk_fix":
        best_model = flk_nkrr_ho_fix(
            Xtr, Ytr, Xts, Yts,
            num_epochs=num_epochs,
            sigma_type=sigma_type,
            sigma_init=sigma_init,
            penalty_init=penalty_init,
            falkon_centers=FixedSelector(centers),
            falkon_M=M,
            hp_lr=hp_lr,
            p_lr=0.01,  # Unused
            batch_size=batch_size,
            cuda=cuda,
            err_fn=err_fns[0],
            opt=falkon_opt,
            loss_every=loss_every,
            opt_centers=opt_centers,
            deff_factor=deff_factor,
            loss_type=loss_type,
        )

    if cuda:
        Xtr, Ytr, Xts, Yts = Xtr.cuda(), Ytr.cuda(), Xts.cuda(), Yts.cuda()
    best_model.fit(Xtr, Ytr)
    train_pred = best_model.predict(Xtr).cpu()
    test_pred = best_model.predict(Xts).cpu()

    print("Test (unseen) errors after retraining on the full train dataset")
    if sigma_type == "diag":
        print("Penalty: %.5e - Sigma: %s" % (best_model.penalty, best_model.kernel.sigma))
    else:
        print("Penalty: %.5e - Sigma: %.5f" % (best_model.penalty, best_model.kernel.sigma))
    for efn in err_fns:
        train_err, err = efn(Ytr.cpu(), train_pred, **metadata)
        test_err, err = efn(Yts.cpu(), test_pred, **metadata)
        print(f"Train {err}: {train_err:.5f} -- Test {err}: {test_err:.5f}")


def run_exp(dataset: Dataset,
            inner_maxiter: int,
            outer_lr: float,
            outer_steps: int,
            hessian_cg_steps: int,
            hessian_cg_tol: float,
            sigma_type: str,
            sigma_init: float,
            penalty_init: float,
            opt_centers: bool,
            loss: str,
            M: int,
            seed: int):
    cuda = True
    train_frac = 0.8
    sgd = False
    batch_size = 25_000
    cg_tol = 1e-4
    warm_start = True
    error_every = 10

    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    from falkon import FalkonOptions
    from falkon.center_selection import FixedSelector, UniformSelector
    from falkon.hypergrad.falkon_ho import run_falkon_hypergrad, ValidationLoss, stochastic_flk_hypergrad
    torch.autograd.set_detect_anomaly(True)
    loss = ValidationLoss(loss)

    Xtr, Ytr, Xts, Yts, metadata = get_load_fn(dataset)(np.float32, as_torch=True)
    err_fns = get_err_fns(dataset)

    # We use a validation split (redefinition of Xtr, Ytr).
    if train_frac < 1.0:
        #idx_tr, idx_val = equal_split(Xtr.shape[0], train_frac=train_frac)
        n_train = int(Xtr.shape[0] * train_frac)
        idx_tr = torch.arange(n_train)
        idx_val = torch.arange(n_train, Xtr.shape[0])
        Xval, Yval = Xtr[idx_val], Ytr[idx_val]
        Xtr, Ytr = Xtr[idx_tr], Ytr[idx_tr]
        data = {'Xtr': Xtr, 'Ytr': Ytr, 'Xts': Xval, 'Yts': Yval}
    else:
        data = {'Xtr': Xtr, 'Ytr': Ytr, 'Xts': Xtr, 'Yts': Ytr}

    fmt_str = f"Dataset:{dataset}, cuda:{cuda}, sgd:{sgd}, warm_start:{warm_start}, cg_tol:{cg_tol}, opt-centers:{opt_centers}, loss:{loss}\n"
    fmt_str += f"hp-lr:{outer_lr}, {sigma_type} sigma, training-fraction:{train_frac}"
    if sgd: fmt_str += f", batch size:{batch_size}"
    if 'centers' in metadata: fmt_str += f", stored centers"
    else: fmt_str += ", fixed centers"

    print("Starting FalkonHO training.")
    print(fmt_str)

    # Center selection
    if 'centers' in metadata:
        centers = torch.from_numpy(metadata['centers'])
        print("Loading centers")
    else:
        selector = UniformSelector(np.random.default_rng(seed))
        centers = selector.select(Xtr, None, M)

    # Move to GPU if needed
    if cuda:
        data = {k: v.cuda() for k, v in data.items()}
        centers = centers.cuda()

    # Initialize Falkon model
    falkon_opt = FalkonOptions(use_cpu=False, debug=False, cg_tolerance=cg_tol)

    t_s = time.time()
    def cback(i, model):
        if i % error_every != 0:
            return
        train_pred = model.predict(data['Xtr'].cuda())
        #val_pred = model.predict(data['Xts'])
        val_pred = model.predict(Xts.cuda())
        train_err, err = err_fns[0](data['Ytr'].cpu(), train_pred.cpu(), **metadata)
        #val_err, err = err_fns[0](data['Yts'].cpu(), val_pred.cpu(), **metadata)
        val_err, err = err_fns[0](Yts.cpu(), val_pred.cpu(), **metadata)
        print(f"Iteration {i} ({time.time() - t_s:.2f}s) - Train {err}: {train_err:.5f} -- Test {err}: {val_err:.5f}")

    if sgd:
        hps, val_loss, hgrads, best_model, times = stochastic_flk_hypergrad(
            data,
            sigma_type=sigma_type,
            sigma_init=sigma_init,
            penalty_init=penalty_init,
            falkon_M=centers.shape[0],
            falkon_centers=FixedSelector(centers),
            falkon_maxiter=inner_maxiter,
            falkon_opt=falkon_opt,
            outer_lr=outer_lr,
            outer_steps=outer_steps,
            hessian_cg_steps=hessian_cg_steps,
            hessian_cg_tol=hessian_cg_tol,
            callback=cback,
            debug=True,
            loss=loss,
            batch_size=batch_size,
            warm_start=warm_start,
            cuda=True,
        )
    else:
        hps, val_loss, hgrads, best_model, times = run_falkon_hypergrad(
            data,
            sigma_type=sigma_type,
            sigma_init=sigma_init,
            penalty_init=penalty_init,
            falkon_M=centers.shape[0],
            falkon_centers=FixedSelector(centers),
            optimize_centers=opt_centers,
            falkon_maxiter=inner_maxiter,
            falkon_opt=falkon_opt,
            outer_lr=outer_lr,
            outer_steps=outer_steps,
            hessian_cg_steps=hessian_cg_steps,
            hessian_cg_tol=hessian_cg_tol,
            callback=cback,
            debug=False,
            loss=loss,
            warm_start=warm_start,
        )

    # Now we have the model, retrain with the full training data and test!
    print("Retraining on the full train dataset.")
    del data  # free GPU mem
    try:
        Xtr = torch.cat([Xtr, Xval], 0)
        Ytr = torch.cat([Ytr, Yval], 0)
    except NameError:
        Xtr, Ytr = Xtr, Ytr
    if cuda:
        Xtr, Ytr, Xts, Yts = Xtr.cuda(), Ytr.cuda(), Xts.cuda(), Yts.cuda()
    best_model.maxiter = 20
    best_model.error_fn = functools.partial(err_fns[0], **metadata)
    best_model.error_every = 1
    best_model.fit(Xtr, Ytr)
    train_pred = best_model.predict(Xtr).cpu()
    test_pred = best_model.predict(Xts).cpu()

    print("Test (unseen) errors after retraining on the full train dataset")
    if sigma_type == "diag":
        print("Penalty: %.5e - Sigma: %s" % (best_model.penalty, best_model.kernel.sigma))
    else:
        print("Penalty: %.5e - Sigma: %.5f" % (best_model.penalty, best_model.kernel.sigma))
    for efn in err_fns:
        train_err, err = efn(Ytr.cpu(), train_pred, **metadata)
        test_err, err = efn(Yts.cpu(), test_pred, **metadata)
        print(f"Train {err}: {train_err:.5f} -- Test {err}: {test_err:.5f}")

    # Create a dataframe for saving the optimization trajectory.
    if sigma_type == "single":
        penalties = np.array([hp[0].cpu().item() for hp in hps[:-1]])
        sigmas = np.array([hp[1][0].cpu().item() for hp in hps[:-1]])
        penalty_g = np.array([hg[0].cpu().item() for hg in hgrads])
        sigma_g = np.array([hg[1][0].cpu().item() for hg in hgrads])
        loss = np.array([vl.cpu().item() for vl in val_loss])
        df = pd.DataFrame(columns=["sigma", "penalty", "sigma_g", "penalty_g", "loss"],
                          data=np.stack((sigmas, penalties, sigma_g, penalty_g, loss), axis=1))
        print(df.head())
        out_fn = f"./logs/hotraj_{dataset}_{int(datetime.datetime.timestamp(datetime.datetime.now()) * 1000)}.csv"
        print("Saving HyperOpt trajectory to %s" % (out_fn))
        df.to_csv(out_fn)
    else:
        print("Cannot save trajectory with multiple lengthscales!")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="FALKON Benchmark Runner")

    p.add_argument('-n', '--name', type=str, required=True)
    p.add_argument('-d', '--dataset', type=Dataset, choices=list(Dataset), required=True,
                   help='Dataset')
    p.add_argument('-s', '--seed', type=int, required=True, help="Random seed")
    p.add_argument('--flk-steps', type=int, help="Maximum number of Falkon steps",
                   default=10)
    p.add_argument('--lr', type=float, help="Learning rate for the outer-problem solver",
                   default=0.01)
    p.add_argument('--steps', type=int, help="Number of outer-problem steps",
                   default=100)
    p.add_argument('--hessian-cg-steps', type=int,
                   help="Maximum steps for finding the Hessian via CG",
                   default=10)
    p.add_argument('--hessian-cg-tol', type=float, help="Tolerance for Hessian CG problem",
                   default=1e-4)
    p.add_argument('--sigma-type', type=str,
                   help="Use diagonal or single lengthscale for the kernel",
                   default='single')
    p.add_argument('--sigma-init', type=float, default=2.0, help="Starting value for sigma")
    p.add_argument('--penalty-init', type=float, default=1.0, help="Starting value for penalty")
    p.add_argument('--optimize-centers', action='store_true',
                   help="Whether to optimize Nystrom centers")
    p.add_argument('--loss', type=str, default="penalized-mse")
    p.add_argument('--map-gradient', action='store_true', help="Creates a gradient map")
    p.add_argument('--gpflow', action='store_true', help="Run GPflow model")
    p.add_argument('--nkrr', action='store_true', help="Run NKRR model")
    p.add_argument('--M', type=int, default=1000, required=False,
                   help="Number of Nystrom centers for Falkon")
    p.add_argument('--regularizer', type=str, default='tikhonov',
                   help="How to regularize the loss in FLK-NKRR")
    p.add_argument('--deff-factor', default=1,
                   help="Constant factor multiplying the effective-dimension regularizer")
    p.add_argument('--loss-type', type=str, default='reg',
                   help="Loss to use in FLK_FIX NKRR setting. can either be regularized loss or just the squared loss")


    args = p.parse_args()
    print("-------------------------------------------")
    print(datetime.datetime.now())
    print("############### SEED: %d ################" % (args.seed))
    print("-------------------------------------------")

    np.random.seed(args.seed)

    if args.gpflow:
        run_gpflow(dataset=args.dataset,
                   num_iter=args.steps,
                   lr=args.lr,
                   sigma_type=args.sigma_type,
                   sigma_init=args.sigma_init,
                   opt_centers=args.optimize_centers,
                   seed=args.seed,
                   gradient_map=args.map_gradient,
                   )
    elif args.map_gradient:
        run_gmap_exp(dataset=args.dataset,
                     sigma_type=args.sigma_type,
                     inner_maxiter=args.flk_steps,
                     hessian_cg_steps=args.hessian_cg_steps,
                     hessian_cg_tol=args.hessian_cg_tol,
                     loss=args.loss,
                     seed=args.seed,
                     )
    elif args.nkrr:
        from summary import get_writer
        get_writer(args.name)
        run_nkrr(dataset=args.dataset,
                 num_epochs=args.steps,
                 hp_lr=args.lr,
                 p_lr=args.lr,
                 sigma_type=args.sigma_type,
                 sigma_init=args.sigma_init,
                 penalty_init=args.penalty_init,
                 M=args.M,
                 seed=args.seed,
                 regularizer=args.regularizer,
                 opt_centers=args.optimize_centers,
                 deff_factor=args.deff_factor,
                 loss_type=args.loss_type,
                 )
    else:
        run_exp(dataset=args.dataset,
                inner_maxiter=args.flk_steps,
                outer_lr=args.lr,
                outer_steps=args.steps,
                hessian_cg_steps=args.hessian_cg_steps,
                hessian_cg_tol=args.hessian_cg_tol,
                sigma_type=args.sigma_type,
                sigma_init=args.sigma_init,
                penalty_init=args.penalty_init,
                opt_centers=args.optimize_centers,
                loss=args.loss,
                seed=args.seed,
                M=args.M,
                )
