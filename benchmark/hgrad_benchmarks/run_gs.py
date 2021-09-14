import io
import itertools
import pathlib
import subprocess
import os

import numpy as np

os.environ['PYTHONPATH'] = '..'
SIMPLE_HOPT_PATH = pathlib.Path(__file__).parent.joinpath("simple_hopt.py").resolve()
DEFAULT_SEED = 12319


def write_gridspec_file(out_file, sigmas, penalties):
    with open(out_file, "w") as fh:
        fh.write("sigma,penalty\n")
        for ex in itertools.product(sigmas, penalties):
            fh.write("%.8e,%.8e\n" % (ex[0], ex[1]))


def run_gs(
        val_pct: float,
        num_centers: int,
        dataset: str,
        model: str,
        gs_file: str,
        exp_name: str,
        seed: int = DEFAULT_SEED,):
    proc_args = [
        f"python {SIMPLE_HOPT_PATH}",
        f"--seed {seed}",
        f"--cg-tol 1e-1",  # ignored
        f"--val-pct {val_pct}",
        f"--sigma-type single",
        f"--sigma-init 1.0",  # ignored
        f"--penalty-init 1.0",  # ignored
        f"--num-centers {num_centers}",
        f"--dataset {dataset}",
        f"--model {model}",
        f"--grid-spec {gs_file}",
        f"--name {dataset}_gs_{model}_{exp_name}"
    ]
    if model == "svgp":
        proc_args.append("--mb 16000")
    proc = subprocess.Popen([" ".join(proc_args)], shell=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    for line in io.TextIOWrapper(proc.stdout, encoding='utf-8'):
        print(line, end='')
    ret_code = proc.wait()
    if ret_code != 0:
        raise RuntimeError("Process returned error", ret_code)


def run():
    gs_file = "tmp_gs_file"
    exp_name = "test_exp_m100"
    datasets = {
        "boston": {
           "num_centers": 100,
           "sigmas": np.logspace(0, 2, 10),
           "penalties": np.logspace(-8, 2, 15),
        },
        "energy": {
           "num_centers": 100,
           "sigmas": np.logspace(0, 2, 10),
           "penalties": np.logspace(-8, 2, 15),
        },
        "ho-higgs": {
           "num_centers": 100,
           "sigmas": np.logspace(0, 1.5, 10),
           "penalties": np.logspace(-6, 2, 15),
        },
        "protein": {
            "num_centers": 100,
            "sigmas": np.logspace(0, 1.5, 10),
            "penalties": np.logspace(-6, 2, 15),
        },
    }
    models = {
        "svgp": {},
        "gcv": {},
        "loocv": {},
        "sgpr": {},
        "hgrad-closed": {'val_pct': 0.5},
        "creg-penfit": {},
        "creg-nopenfit": {},
        "creg-nopenfit-divtr": {},
        "creg-nopenfit-divtrdeff": {},
    }
    for dset, dset_params in datasets.items():
        for model, model_params in models.items():
            write_gridspec_file(gs_file, dset_params['sigmas'], dset_params['penalties'])
            run_gs(val_pct=model_params.get('val_pct', 0.2),
                   num_centers=dset_params['num_centers'],
                   dataset=dset,
                   model=model,
                   gs_file=gs_file,
                   exp_name=exp_name,
                   seed=DEFAULT_SEED)
