from collections import defaultdict
import argparse
import functools
import datetime
from typing import Optional

import torch
import numpy as np
import matplotlib.pyplot as plt

from common.benchmark_utils import *
from common.datasets import get_load_fn, equal_split
from common.error_metrics import get_err_fns

from falkon.models import falkon
from falkon.utils import TicToc
from falkon.hypergrad.falkon_ho import run_falkon_hypergrad


RANDOM_SEED = 1219


def run_on_dataset(dset: Dataset,
                   dtype: Optional[DataType],
                   seed: int,
                   inner_iter: int,
                   outer_iter: int,
                   outer_lr: float,
                   num_centers: int,
                   exp_name: str):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Data types
    if dtype is None:
        dtype = DataType.float64

    # Data load
    load_fn = get_load_fn(dset)
    err_fns = get_err_fns(dset)
    Xtr, Ytr, Xts, Yts, kwargs = load_fn(dtype=dtype.to_numpy_dtype(), as_torch=True)
    Xtr = Xtr[:400_000]
    Ytr = Ytr[:400_000]
    print("Subsampled to %d points" % (Xtr.shape[0]))
    Xtr = Xtr.pin_memory()
    Ytr = Ytr.pin_memory()
    # Further split the data into train/validation
    idx_tr, idx_val = equal_split(Xtr.shape[0], 0.8)
    data = {'Xtr': Xtr[idx_tr, :], 'Ytr': Ytr[idx_tr, :], 'Xts': Xtr[idx_val, :], 'Yts': Ytr[idx_val, :]}

    # Options and parameters
    opt = falkon.FalkonOptions(
        compute_arch_speed=False,
        no_single_kernel=True,
        pc_epsilon_32=1e-6,
        pc_epsilon_64=1e-13,
        cg_tolerance=1e-7,
        debug=False
    )
    hessian_cg_steps = 20
    hessian_cg_tol = 1e-7
    debug = True

    partial_test_errs = defaultdict(list)
    def cback(model):
        test_preds = model.predict(Xts)
        for err_fn in err_fns:
            err, err_name = err_fn(Yts, test_preds, **kwargs)
            print(f"Trained on train, test {err_name} = {err:.5f}")
            partial_test_errs[err_name].append(err)


    with TicToc("Hyperparameter optimization"):
        out_dict = run_falkon_hypergrad(data=data,
                                        falkon_M=num_centers,
                                        falkon_maxiter=inner_iter,
                                        falkon_opt=opt,
                                        outer_lr=outer_lr,
                                        outer_steps=outer_iter,
                                        hessian_cg_steps=hessian_cg_steps,
                                        hessian_cg_tol=hessian_cg_tol,
                                        callback=cback,
                                        debug=True)
    hparam_history, val_loss_history, hgrad_history, model, time_history = out_dict

    # Refit model
    model.fit(Xtr, Ytr)
    test_preds = model.predict(Xts)
    for err_fn in err_fns:
        test_error, err_name = err_fn(Yts, test_preds, **kwargs)
        print(f"Test {err_name} = {test_error:.5f}")

    print(f"\nFull refit with same parameters, more centers")
    Xtr, Ytr, Xts, Yts, kwargs = load_fn(dtype=dtype.to_numpy_dtype(), as_torch=True)
    flk = falkon.Falkon(model.kernel, penalty=model.penalty, M=50_000,
                        center_selection="uniform",
                        maxiter=10,
                        seed=120,
                        error_fn=functools.partial(err_fns[0], **kwargs),
                        options=opt)
    flk.fit(Xtr, Ytr)
    test_preds = flk.predict(Xts)
    final_test_errs = {}
    for err_fn in err_fns:
        test_error, err_name = err_fn(Yts, test_preds, **kwargs)
        print(f"Test {err_name} = {test_error:.5f}")
        final_test_errs[err_name] = test_error

    ## Run some plots ##
    # Sigma evolution
    fig, ax = plt.subplots()
    num_sigmas = len(hparam_history[0][1])
    num_iter = len(hparam_history)
    for i in range(num_sigmas):
        ax.plot(np.arange(num_iter), [hparam_history[j][1][i] for j in range(num_iter)])
    ax.set_xlabel("HO Iterations")
    ax.set_ylabel("$\sigma_i$")
    ax.set_title("Lengthscale evolution")
    fig.savefig("logs/sigma_evolution_%s.png" % (exp_name), bbox_inches='tight')

    # Test error
    fig, ax = plt.subplots()
    err_name = "MSE"
    err_list = partial_test_errs[err_name]
    ax.plot(np.cumsum(time_history), err_list, lw=2)
    ax.scatter(np.cumsum(time_history)[-1], final_test_errs[err_name], marker='o', s=80, c='b', label="Full-refit error")
    ax.set_xlabel("HO Time (s)")
    ax.set_ylabel(err_name)
    ax.grid()
    fig.savefig("logs/err_evolution_%s.png" % (exp_name), bbox_inches='tight')

    # Val-loss evolution
    fig, ax = plt.subplots()
    num_iter = len(val_loss_history)
    ax.plot(np.arange(num_iter), val_loss_history, lw=2)
    ax.set_xlabel("HO Iterations")
    ax.set_ylabel("validation-loss")
    fig.savefig("logs/val_evolution_%s.png" % (exp_name), bbox_inches='tight')
    
    print("Saved plots in 'logs/' folder with name '%s'" % (exp_name))



if __name__ == "__main__":

    p = argparse.ArgumentParser(description="Falkon HyperOpt Runner")

    p.add_argument('-d', '--dataset', type=Dataset, choices=list(Dataset), required=True,
                   help='Dataset')
    p.add_argument('-t', '--dtype', type=DataType.argparse, choices=list(DataType),
                   required=False, default=None,
                   help='Floating point precision to work with. Lower precision will be '
                        'faster but less accurate. Certain algorithms require a specific precision. '
                        'If this argument is not specified we will use the highest precision '
                        'supported by the chosen algorithm.')
    p.add_argument('-i', '--inner-iter', type=int, required=True,
                   help='Number of inner CG iterations to run Falkon')
    p.add_argument('-o', '--outer-iter', type=int, required=True,
                   help='Number of hyperparameter gradient descent iterations')
    p.add_argument('--seed', type=int, default=RANDOM_SEED,
                    help='Random number generator seed')
    # Algorithm-specific arguments
    p.add_argument('-M', '--num-centers', type=int, required=True,
                        help='Number of Nystroem centers. Used for algorithms '
                              'falkon, gpytorch and gpflow.')
    p.add_argument('--outer-lr', type=float, default=0.1, help="Outer gradient descent learning rate")
    p.add_argument('-n', '--exp-name', type=str, required=False,
                   default='default-exp')

    args = p.parse_args()
    print("-------------------------------------------")
    print(f"STARTING AT {datetime.datetime.now()} -- SEED={args.seed}")

    run_on_dataset(dset=args.dataset, dtype=args.dtype, seed=args.seed,
                   inner_iter=args.inner_iter, outer_iter=args.outer_iter,
                   outer_lr=args.outer_lr, num_centers=args.num_centers,
                   exp_name=args.exp_name)
