import pathlib
import io
import subprocess
from typing import List

import numpy as np


SIMPLE_HOPT_PATH = pathlib.Path(__file__).parent.joinpath("simple_hopt.py").resolve()
DEFAULT_SEED = 123199


def gen_exp_name(
        num_centers,
        penalty,
        sigma,
        sigma_type,
        model,
        dataset,
        extra_name):
    return f"{dataset}_fetchloss_{model}_m{num_centers}_pinit{penalty}_{sigma_type}sinit{sigma}" \
           f"_{extra_name}"


def run_fetch_loss(sigma_init: float,
                   pen_init: float,
                   M: int,
                   dataset: str,
                   model: str,
                   sigma: str,
                   exp_name: str,
                   seed: int = DEFAULT_SEED, ):
    proc_args = [
        f"python {SIMPLE_HOPT_PATH}",
        f"--fetch-loss",
        f"--seed {seed}",
        f"--sigma-type {sigma}",
        f"--sigma-init {sigma_init}",
        f"--penalty-init {pen_init}",
        f"--num-centers {M}",
        f"--dataset {dataset}",
        f"--model {model}",
        f"--name {gen_exp_name(M, pen_init, sigma_init, sigma, model, dataset, exp_name)}",
    ]
    proc = subprocess.Popen([" ".join(proc_args)], shell=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    for line in io.TextIOWrapper(proc.stdout, encoding='utf-8'):
        print(line, end='')
    ret_code = proc.wait()
    if ret_code != 0:
        raise RuntimeError("Process returned error", ret_code)


def run_for_models(sigma_init: float,
                   pen_init: float,
                   M: int,
                   dataset: str,
                   models: List[str],
                   sigma: str,
                   exp_name: str,
                   num_rep: int = 1,
                   ):
    for model in models:
        for i in range(num_rep):
            run_fetch_loss(sigma_init, pen_init, M, dataset, model, sigma, exp_name,
                           seed=DEFAULT_SEED + i,)


def run():
    datasets = ["protein", "chiet", "ictus", "codrna", "svmguide1", "phishing",
                "spacega", "cadata", "mg", "cpusmall", "abalone", "blogfeedback",
                "energy", "covtype", "ho-higgs",]
    datasets = ["mnist-small", "svhn", "cifar10", "fashionmnist"]
    M = 100
    sigma = "diag"
    extra_exp_name = "for_realzz2"
    sigma_init = 5.0
    penalty_init = 1e-5
    models = ["sgpr", "creg-penfit", "creg-nopenfit"]

    for dset in datasets:
        run_for_models(sigma_init, penalty_init, M, dset, models,
                       sigma, extra_exp_name, num_rep=1)


if __name__ == "__main__":
    run()
