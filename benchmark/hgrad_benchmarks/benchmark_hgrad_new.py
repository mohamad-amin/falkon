import argparse
import datetime
import functools

import numpy as np
import torch

from common.benchmark_utils import *
from common.datasets import get_load_fn, equal_split
from common.error_metrics import get_err_fns
from falkon import FalkonOptions
from falkon.center_selection import UniformSelector, FixedSelector
from falkon.hypergrad import validation_hp, complexity_reg
from common.summary import get_writer


def retrain_and_test(model, Xtr, Ytr, Xts, Yts, err_fns, metadata, cuda, maxiter):
    # Retrain with the full training data and test!
    print("Retraining on the full train dataset.")
    if cuda:
        Xtr, Ytr, Xts, Yts = Xtr.cuda(), Ytr.cuda(), Xts.cuda(), Yts.cuda()
    model.maxiter = maxiter
    model.error_fn = functools.partial(err_fns[0], **metadata)
    model.error_every = 1
    model.fit(Xtr, Ytr)
    train_pred = model.predict(Xtr).cpu()
    test_pred = model.predict(Xts).cpu()
    writer = get_writer()
    print("Best model settings:")
    print("Penalty: %.5e - Sigma: %s" % (model.penalty, model.kernel.sigma))
    print("Test (unseen) errors after retraining on the full train dataset")
    for efn in err_fns:
        train_err, err = efn(Ytr.cpu(), train_pred, **metadata)
        test_err, err = efn(Yts.cpu(), test_pred, **metadata)
        print(f"Train {err}: {train_err:.5f} -- Test {err}: {test_err:.5f}")
        writer.add_scalar(f"RetrainError/test-{err}", test_err, 5)


def run_validation_hp_opt_hgrad(
        dataset: Dataset,
        penalty_init: float,
        sigma_type: str,
        sigma_init: float,
        opt_centers: bool,
        num_centers: int,
        num_epochs: int,
        learning_rate: float,
        val_loss_type: str,
        train_frac: float,
        falkon_maxiter: int,
        cuda: bool,
        seed: int,
):
    loss_every = 1
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr, Ytr, Xts, Yts, metadata = get_load_fn(dataset)(np.float32, as_torch=True)
    err_fns = get_err_fns(dataset)

    # Split training set into train, validation
    if train_frac <= 0 or train_frac >= 1:
        raise ValueError("train_frac must be between 0 and 1.")
    idx_tr, idx_val = equal_split(Xtr.shape[0], train_frac=train_frac)
    Xval, Yval = Xtr[idx_val], Ytr[idx_val]
    Xtr_, Ytr_ = Xtr[idx_tr], Ytr[idx_tr]

    # Center selection
    if False and 'centers' in metadata:
        centers = torch.from_numpy(metadata['centers'])
        print("Ignoring `num_centers` argument since dataset metadata "
              "contains centers. Picked %d centers" % (centers.shape[0]))
    else:
        selector = UniformSelector(np.random.default_rng(seed))
        centers = selector.select(Xtr, None, num_centers)

    # Falkon Options
    falkon_opt = FalkonOptions(
        use_cpu=False,
        debug=False,
        cg_tolerance=1e-4,  # default is 1e-7
        cg_full_gradient_every=10,  # default is 10
        pc_epsilon_32=1e-6,  # default is 1e-5
        cg_epsilon_32=1e-7,  # default is 1e-7
    )

    # Run the training
    best_model = validation_hp.train_hypergrad(
        Xtr_, Ytr_, Xval, Yval, Xts, Yts,
        penalty_init=penalty_init,
        sigma_type=sigma_type,
        sigma_init=sigma_init,
        num_centers=centers.shape[0],
        opt_centers=opt_centers,
        falkon_centers=FixedSelector(centers),
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        val_loss_type=val_loss_type,
        cuda=cuda,
        loss_every=loss_every,
        err_fn=functools.partial(err_fns[0], **metadata),
        falkon_opt=falkon_opt,
        falkon_maxiter=falkon_maxiter,
    )

    # Retrain with the full training data and test!
    retrain_and_test(best_model, Xtr, Ytr, Xts, Yts, err_fns, metadata, cuda, max(20, falkon_maxiter))


def run_complexity_reg_hp_opt(
        dataset: Dataset,
        penalty_init: float,
        sigma_type: str,
        sigma_init: float,
        opt_centers: bool,
        num_centers: int,
        num_epochs: int,
        learning_rate: float,
        model_type: str,
        falkon_maxiter: int,
        cuda: bool,
        seed: int,
):
    loss_every = 1
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr, Ytr, Xts, Yts, metadata = get_load_fn(dataset)(np.float32, as_torch=True)
    err_fns = get_err_fns(dataset)

    # Center selection
    if False and 'centers' in metadata:
        centers = torch.from_numpy(metadata['centers'])
        print("Ignoring `num_centers` argument since dataset metadata "
              "contains centers. Picked %d centers" % (centers.shape[0]))
    else:
        selector = UniformSelector(np.random.default_rng(seed))
        centers = selector.select(Xtr, None, num_centers)

    # Falkon Options
    falkon_opt = FalkonOptions(
        use_cpu=False,
        debug=False,
        cg_tolerance=1e-4,  # default is 1e-7
        cg_full_gradient_every=10,  # default is 10
        pc_epsilon_32=1e-6,  # default is 1e-5
        cg_epsilon_32=1e-7,  # default is 1e-7
    )

    # Run the training
    best_model = complexity_reg.train_complexity_reg(
        Xtr, Ytr, Xts, Yts,
        penalty_init=penalty_init,
        sigma_type=sigma_type,
        sigma_init=sigma_init,
        num_centers=centers.shape[0],
        opt_centers=opt_centers,
        falkon_centers=FixedSelector(centers),
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        model_type=model_type,
        cuda=cuda,
        loss_every=loss_every,
        err_fn=functools.partial(err_fns[0], **metadata),
        falkon_opt=falkon_opt,
        falkon_maxiter=falkon_maxiter,
    )

    # Retrain with the full training data and test!
    retrain_and_test(best_model, Xtr, Ytr, Xts, Yts, err_fns, metadata, cuda, max(20, falkon_maxiter))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="FALKON Benchmark Runner")

    p.add_argument('-n', '--name', type=str, required=True)
    p.add_argument('-d', '--dataset', type=Dataset, choices=list(Dataset), required=True,
                   help='Dataset')
    p.add_argument('-s', '--seed', type=int, required=True, help="Random seed")
    p.add_argument('--lr', type=float, help="Learning rate for the outer-problem solver",
                   default=0.01)
    p.add_argument('--epochs', type=int, help="Number of outer-problem steps",
                   default=100)
    p.add_argument('--sigma-type', type=str,
                   help="Use diagonal or single lengthscale for the kernel",
                   default='single')
    p.add_argument('--sigma-init', type=float, default=2.0, help="Starting value for sigma")
    p.add_argument('--penalty-init', type=float, default=1.0, help="Starting value for penalty")
    p.add_argument('--optimize-centers', action='store_true',
                   help="Whether to optimize Nystrom centers")
    p.add_argument('--num-centers', type=int, default=1000, required=False,
                   help="Number of Nystrom centers for Falkon")
    p.add_argument('--flk-maxiter', type=int, required=True,
                   help="Number of falkon iterations")
    p.add_argument('--val-loss', type=str, default="")
    p.add_argument('--train-frac', type=float, default=0,
                   help="Fraction of training data in the training-validation split. Only necessary "
                        "for validation-data experiments")
    p.add_argument('--creg-type', type=str, default="")
    p.add_argument('--exp', type=str, required=True)
    p.add_argument('--cuda', action='store_true')
    args = p.parse_args()
    print("-------------------------------------------")
    print(datetime.datetime.now())
    print("############### SEED: %d ################" % (args.seed))
    print("-------------------------------------------")
    np.random.seed(args.seed)

    get_writer(args.name)

    if args.exp == "creg":
        if args.creg_type == "":
            raise ValueError("creg-type must be specified (either 'gp', 'deff-simple', 'deff-fast', 'deff-precise')")
        run_complexity_reg_hp_opt(
            dataset=args.dataset, penalty_init=args.penalty_init, sigma_type=args.sigma_type,
            sigma_init=args.sigma_init, opt_centers=args.optimize_centers, num_centers=args.num_centers,
            num_epochs=args.epochs, learning_rate=args.lr, model_type=args.creg_type, cuda=args.cuda,
            seed=args.seed, falkon_maxiter=args.flk_maxiter,
        )
    elif args.exp == "hgrad":
        if args.val_loss == "":
            raise ValueError("val-loss type must be specified (either 'mse' or 'penalized-mse').")
        if args.train_frac == 0:
            raise ValueError("train-frac must be specified (a number between 0 and 1).")
        run_validation_hp_opt_hgrad(
            dataset=args.dataset, penalty_init=args.penalty_init, sigma_type=args.sigma_type,
            sigma_init=args.sigma_init, opt_centers=args.optimize_centers, num_centers=args.num_centers,
            num_epochs=args.epochs, learning_rate=args.lr, val_loss_type=args.val_loss,
            train_frac=args.train_frac, cuda=args.cuda, seed=args.seed, falkon_maxiter=args.flk_maxiter,
        )
