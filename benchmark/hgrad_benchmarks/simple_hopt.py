import argparse
import datetime
import os
import pickle
import warnings
from functools import partial
from typing import Any, List

import numpy as np
import pandas as pd
import torch

from common.datasets import get_load_fn
from common.error_metrics import get_err_fns
from common.summary import get_writer
from common.benchmark_utils import Dataset
from falkon.center_selection import UniformSelector
from falkon.hypergrad.training import init_model, train_complexity_reg, HPGridPoint, run_on_grid


def save_logs(logs: Any, exp_name: str, log_folder: str = "./logs"):
    new_exp_name = exp_name
    log_path = os.path.join(log_folder, f"{new_exp_name}.pkl")
    name_counter = 1
    while os.path.isfile(log_path):
        name_counter += 1
        new_exp_name = f"{exp_name}_{name_counter}"
        log_path = os.path.join(log_folder, f"{new_exp_name}.pkl")
    if new_exp_name != exp_name:
        warnings.warn(f"Logs will be saved to '{log_path}' because original name was taken.")

    with open(log_path, "wb") as file:
        pickle.dump(logs, file, protocol=4)
    print(f"Log saved to {log_path}", flush=True)


def read_gs_file(file_name: str) -> List[HPGridPoint]:
    df = pd.read_csv(file_name, header=0, index_col=False)
    points = []
    for row in df.itertuples():
        point = HPGridPoint(attributes=row._asdict())
        points.append(point)
    return points


def run_grid_search(
        exp_name: str,
        dataset: Dataset,
        model_type: str,
        penalty_init: float,
        sigma_type: str,
        sigma_init: float,
        gs_file: str,
        num_centers: int,
        val_pct: float,
        cg_tol: float,
        cuda: bool,
        seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr, Ytr, Xts, Yts, metadata = get_load_fn(dataset)(np.float64, as_torch=True)
    err_fns = get_err_fns(dataset)

    # Center selection (not on GPR!)
    if model_type == "gpr":
        centers = None
    else:
        if 'centers' in metadata:
            print(f"Ignoring default centers and picking new {num_centers} centers.")
        selector = UniformSelector(np.random.default_rng(seed))
        centers = selector.select(Xtr, None, num_centers)

    grid_spec = read_gs_file(gs_file)
    model = init_model(model_type=model_type,
                       data={'X': Xtr, 'Y': Ytr},
                       penalty_init=penalty_init,
                       centers_init=centers,
                       sigma_init=sigma_init,
                       opt_penalty=False,
                       opt_sigma=False,
                       opt_centers=False,
                       sigma_type=sigma_type,
                       cuda=cuda,
                       val_pct=val_pct,
                       cg_tol=cg_tol)
    logs = run_on_grid(Xtr=Xtr, Ytr=Ytr,
                       Xts=Xts, Yts=Yts,
                       model=model, err_fn=partial(err_fns[0], **metadata),
                       grid_spec=grid_spec, cuda=cuda)
    save_logs(logs, exp_name=exp_name)


def run_optimization(
        exp_name: str,
        dataset: Dataset,
        model_type: str,
        penalty_init: float,
        sigma_type: str,
        sigma_init: float,
        opt_centers: bool,
        opt_sigma: bool,
        opt_penalty: bool,
        num_centers: int,
        num_epochs: int,
        learning_rate: float,
        val_pct: float,
        cg_tol: float,
        cuda: bool,
        seed: int
):
    loss_every = 1
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr, Ytr, Xts, Yts, metadata = get_load_fn(dataset)(np.float32, as_torch=True)
    err_fns = get_err_fns(dataset)

    # Center selection (not on GPR!)
    if model_type == "gpr":
        centers = None
    else:
        if 'centers' in metadata:
            print(f"Ignoring default centers and picking new {num_centers} centers.")
        selector = UniformSelector(np.random.default_rng(seed))
        centers = selector.select(Xtr, None, num_centers)

    model = init_model(model_type=model_type,
                       data={'X': Xtr, 'Y': Ytr},
                       penalty_init=penalty_init,
                       centers_init=centers,
                       sigma_init=sigma_init,
                       opt_penalty=opt_penalty,
                       opt_sigma=opt_sigma,
                       opt_centers=opt_centers,
                       sigma_type=sigma_type,
                       cuda=cuda,
                       val_pct=val_pct,
                       cg_tol=cg_tol)
    logs = train_complexity_reg(Xtr=Xtr, Ytr=Ytr,
                                Xts=Xts, Yts=Yts,
                                model=model, err_fn=partial(err_fns[0], **metadata),
                                learning_rate=learning_rate, num_epochs=num_epochs,
                                cuda=cuda, verbose=True, loss_every=loss_every)
    save_logs(logs, exp_name=exp_name)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="FALKON Benchmark Runner")

    p.add_argument('-n', '--name', type=str, required=True)
    p.add_argument('-d', '--dataset', type=Dataset, choices=list(Dataset), required=True,
                   help='Dataset')
    p.add_argument('-s', '--seed', type=int, required=True, help="Random seed")
    p.add_argument('--model', type=str, required=True, help="Model type")
    p.add_argument('--cg-tol', type=float, required=False, default=1e-5,
                   help="CG algorithm tolerance for hyper-gradient models.")
    p.add_argument('--lr', type=float, help="Learning rate for the outer-problem solver",
                   default=0.01)
    p.add_argument('--epochs', type=int, help="Number of outer-problem steps",
                   default=100)
    p.add_argument('--sigma-type', type=str,
                   help="Use diagonal or single lengthscale for the kernel",
                   default='single')
    p.add_argument('--sigma-init', type=float, default=2.0, help="Starting value for sigma")
    p.add_argument('--penalty-init', type=float, default=1.0, help="Starting value for penalty")
    p.add_argument('--oc', action='store_true',
                   help="Whether to optimize Nystrom centers")
    p.add_argument('--os', action='store_true',
                   help="Whether to optimize kernel parameters (sigma)")
    p.add_argument('--op', action='store_true',
                   help="Whether to optimize penalty")
    p.add_argument('--num-centers', type=int, default=1000, required=False,
                   help="Number of Nystrom centers for Falkon")
    p.add_argument('--val-pct', type=float, default=0,
                   help="Fraction of validation data (hgrad experiments)")
    p.add_argument('--grid-spec', type=str, default=None,
                   help="Grid-spec file. Triggers a grid-search run instead of optimization.")
    p.add_argument('--cuda', action='store_true')
    args = p.parse_args()
    print("-------------------------------------------")
    print(datetime.datetime.now())
    print("############### SEED: %d ################" % (args.seed))
    print("-------------------------------------------")
    np.random.seed(args.seed)
    get_writer(args.name)
    args.model = args.model.lower()

    if args.grid_spec is not None:
        run_grid_search(exp_name=args.name, dataset=args.dataset, model_type=args.model,
                        penalty_init=args.penalty_init, sigma_type=args.sigma_type, gs_file=args.grid_spec,
                        sigma_init=args.sigma_init, num_centers=args.num_centers, val_pct=args.val_pct,
                        cg_tol=args.cg_tol, cuda=args.cuda, seed=args.seed)
    else:
        run_optimization(exp_name=args.name, dataset=args.dataset, model_type=args.model,
                         penalty_init=args.penalty_init, sigma_type=args.sigma_type, sigma_init=args.sigma_init,
                         opt_centers=args.oc, opt_sigma=args.os, opt_penalty=args.op, num_centers=args.num_centers,
                         num_epochs=args.num_epochs, learning_rate=args.lr, val_pct=args.val_pct,
                         cg_tol=args.cg_tol, cuda=args.cuda, seed=args.seed)
