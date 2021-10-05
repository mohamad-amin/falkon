import pathlib
import io
import subprocess
from typing import List

import numpy as np


SIMPLE_HOPT_PATH = pathlib.Path(__file__).parent.joinpath("simple_hopt.py").resolve()
DEFAULT_SEED = 123199


def gen_exp_name(optimizer, num_centers, learning_rate, penalty, sigma, val_percentage, opt_m,
                 sigma_type, model, dataset, num_trace_vecs, approx_trace, flk_maxiter, cg_tol,
                 extra_name):
    val_pct_str = f"_val{val_percentage}_" if True else "" # model in VAL_MODELS else ""
    trace_vec_str = f"_ste{num_trace_vecs}_" if model in STOCH_MODELS else ""
    flk_miter_str = f"_{flk_maxiter}fits_" if model in STOCH_MODELS else ""
    approx_trace_str = f"_trappr_" if model in STOCH_MODELS and approx_trace else ""
    cg_tol_str = f"_cg{cg_tol:.1e}" if model in STOCH_MODELS else ""
    opt_m_str = "_optM_" if opt_m else ""

    return f"{dataset}_hopt_{model}_test_hopt_{optimizer}_m{num_centers}_lr{learning_rate}_" \
           f"pinit{penalty}_{sigma_type}sinit{sigma}{val_pct_str}{trace_vec_str}{approx_trace_str}" \
           f"{flk_miter_str}{cg_tol_str}{opt_m_str}_{extra_name}"


def run_simple_hopt(sigma_init: float,
                    pen_init: float,
                    lr: float,
                    num_epochs: int,
                    M: int,
                    dataset: str,
                    val_pct: float,
                    model: str,
                    optim: str,
                    sigma: str,
                    opt_centers: bool,
                    num_trace_vecs: int,
                    flk_maxiter: int,
                    exp_name: str,
                    cg_tol: float,
                    approx_trace: bool,
                    seed: int = DEFAULT_SEED, ):
    exp_name_final = gen_exp_name(
        optim, M, lr, pen_init, sigma_init, val_pct, opt_centers, sigma,
        model, dataset, num_trace_vecs, approx_trace, flk_maxiter, cg_tol, exp_name)
    proc_args = [
        f"python {SIMPLE_HOPT_PATH}",
        f"--seed {seed}",
        f"--cg-tol {cg_tol}",
        f"--val-pct {val_pct}",
        f"--sigma-type {sigma}",
        f"--sigma-init {sigma_init}",
        f"--penalty-init {pen_init}",
        f"--lr {lr}",
        f"--epochs {num_epochs}",
        f"--optimizer {optim}",
        f"--op",
        f"--os",
        f"--num-centers {M}",
        f"--dataset {dataset}",
        f"--model {model}",
        f"--num-t {num_trace_vecs}",
        f"--flk-maxiter {flk_maxiter}",
        #f"--cuda",
        f"--name {exp_name_final}",
    ]
    if model == "svgp":
        proc_args.append("--mb 16000")
    if opt_centers:
        proc_args.append("--oc")
    if approx_trace:
        proc_args.append("--approx-trace")
    proc = subprocess.Popen([" ".join(proc_args)], shell=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    with open(f"logs/tee_{exp_name_final}.txt", "a+") as out_f:
        for line in io.TextIOWrapper(proc.stdout, encoding='utf-8'):
            print(line, end='')
            out_f.write(line)
    ret_code = proc.wait()
    if ret_code != 0:
        raise RuntimeError("Process returned error", ret_code)


def run_for_models(sigma_init: float,
                   pen_init: float,
                   lr: float,
                   num_epochs: int,
                   M: int,
                   dataset: str,
                   optim: str,
                   val_pct: float,
                   models: List[str],
                   sigma: str,
                   opt_centers: bool,
                   num_trace_vecs: int,
                   flk_maxiter: int,
                   cg_tol: float,
                   approx_trace: bool,
                   exp_name: str,
                   num_rep: int = 1,
                   ):
    for model in models:
        for i in range(num_rep):
            if i == 0:
                continue
            run_simple_hopt(sigma_init, pen_init, lr, num_epochs, M, dataset, val_pct, model,
                            optim, seed=DEFAULT_SEED + i, sigma=sigma, opt_centers=opt_centers,
                            num_trace_vecs=num_trace_vecs, flk_maxiter=flk_maxiter,
                            exp_name=exp_name, cg_tol=cg_tol, approx_trace=approx_trace)


def run_for_valpct(sigma_init: float,
                   pen_init: float,
                   lr: float,
                   num_epochs: int,
                   M: int,
                   dataset: str,
                   optim: str,
                   val_pcts: List[float],
                   sigma: str,
                   opt_centers: bool,
                   num_trace_vecs: int,
                   flk_maxiter: int,
                   cg_tol: float,
                   approx_trace: bool,
                   exp_name: str,
                   model: str = "hgrad-closed",
                   num_rep: int = 1,
                   ):
    for val_pct in val_pcts:
        for i in range(num_rep):
            run_simple_hopt(sigma_init, pen_init, lr, num_epochs, M, dataset, val_pct, model,
                            optim, seed=DEFAULT_SEED + i, sigma=sigma, opt_centers=opt_centers,
                            num_trace_vecs=num_trace_vecs, flk_maxiter=flk_maxiter, cg_tol=cg_tol,
                            exp_name=exp_name, approx_trace=approx_trace)


SIGMA_PEN_PAIRS = [
    (2.0, 1.5e-5),
    (1.0, 1.0),
    (15.0, 1e-4),
    (15.0, 1.0),
]
VAL_PCTS = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
STOCH_MODELS = [
    "stoch-gcv",
    "stoch-creg-nopenfit",
    "stoch-creg-penfit",
    "comp-creg-penfit",
    "comp-creg-nopenfit",
]
VAL_MODELS = [
    "hgrad-ift",
    "hgrad-closed",
    "flk-hgrad-closed",
]
MODELS = [
    "loocv",
    "sgpr",
    "gcv",
    "creg-nopenfit",
    "creg-penfit",
    "hgrad-closed",
]



def run_multistart_2params():
    datasets = ["cpusmall", "energy", "protein", "svmguide1",
                "cadata", "ho-higgs", "road3d", "houseleectric"]
    datasets = ["houseelectric"]
    models = ["svgp", "sgpr", "hgrad-closed", "gcv", "creg-penfit", "creg-nopenfit",
              "creg-nopenfit-divtr", "creg-nopenfit-divtrdeff",]
    num_epochs = 200
    opt_m = False
    M = 100
    val_pct = 0.6
    optim = "adam"
    learning_rate = 0.1
    sigma = "single"
    extra_exp_name = "test_v5"

    sigma_pen_pairs = [(1.0, 1e-2), (15.0, 1e-2), (1.0, 1e-7), (15.0, 1e-7)]

    for dataset in datasets:
        for si, pi in sigma_pen_pairs:
            run_for_models(
                sigma_init=si, pen_init=pi,
                lr=learning_rate, num_epochs=num_epochs, M=M, dataset=dataset,
                val_pct=val_pct, models=models, num_rep=1, optim=optim, sigma=sigma,
                opt_centers=opt_m, exp_name=extra_exp_name, flk_maxiter=30,
                num_trace_vecs=30, cg_tol=1e-3, approx_trace=True)


def run():
    datasets = ["svhn", "mnist-small", "fashionmnist", "svhn", "cifar10"]
    datasets = ["protein", "chiet", "ictus", "codrna", "svmguide1", "phishing",
                "spacega", "cadata", "mg", "cpusmall", "abalone", "blogfeedback",
                "energy", "covtype", "ho-higgs", "ijcnn1",
                "road3d", "buzz", "houseelectric",]
    datasets = ["flights"]
    num_epochs = 200
    learning_rate = 0.05
    M = 2000
    opt_m = True
    val_pct = 0.6
    optim = "adam"
    sigma = "diag"
    extra_exp_name = "test"
    sigma_init = 1#"auto"
    penalty_init = "auto"
    # Stochastic stuff
    flk_maxiter = 150
    num_trace_vecs = 20
    cg_tol = 5e-4
    approx_trace = True
    # Models to use for training
    #models = ["gcv", "sgpr", "hgrad-closed", "creg-penfit", "creg-nopenfit", "creg-nopenfit-divtr", "creg-nopenfit-divtrdeff", "creg-nopenfit-divdeff"]
    models = ["creg-notrace", "hgrad-closed", "creg-penfit", "sgpr", "gcv", "loocv"]
    models = ["sgpr", "creg-penfit", "creg-nopenfit", "hgrad-closed", "creg-nopenfit-divtr", "creg-nopenfit-divtrdeff", "gcv"]
    models = ["stoch-creg-penfit"]

    if False:  # Experiment with increasing M
        ms = [100, 200, 400, 800, 1600]
        dataset = "codrna"
        models = ["creg-penfit-divtr", "sgpr", "creg-penfit", "creg-notrace", "creg-nopenfit", "creg-nopenfit-divtr"]
        for M in ms:
            run_for_models(
                sigma_init=sigma_init, pen_init=penalty_init,
                lr=learning_rate, num_epochs=num_epochs, M=M, dataset=dataset,
                val_pct=val_pct, models=models, num_rep=3, optim=optim, sigma=sigma,
                opt_centers=opt_m, exp_name=extra_exp_name, flk_maxiter=flk_maxiter,
                num_trace_vecs=num_trace_vecs, cg_tol=cg_tol, approx_trace=approx_trace)
    elif False:
        for t in [10, 20, 40, 70, 100]:
            run_for_models(
                sigma_init=sigma_init, pen_init=penalty_init,
                lr=learning_rate, num_epochs=num_epochs, M=M, dataset=dataset,
                val_pct=val_pct, models=models, num_rep=5, optim=optim, sigma=sigma,
                opt_centers=opt_m, exp_name=extra_exp_name, flk_maxiter=flk_maxiter,
                num_trace_vecs=t, cg_tol=cg_tol, approx_trace=approx_trace)
    elif False:
        for cg in np.logspace(-1, -6, 10):
            run_for_models(
                sigma_init=sigma_init, pen_init=penalty_init,
                lr=learning_rate, num_epochs=num_epochs, M=M, dataset=dataset,
                val_pct=val_pct, models=models, num_rep=5, optim=optim, sigma=sigma,
                opt_centers=opt_m, exp_name=extra_exp_name, flk_maxiter=flk_maxiter,
                num_trace_vecs=num_trace_vecs, cg_tol=cg, approx_trace=approx_trace)
    elif False:
        for si, pi in SIGMA_PEN_PAIRS:
            run_for_models(
                sigma_init=si, pen_init=pi,
                lr=learning_rate, num_epochs=num_epochs, M=M, dataset=dataset,
                val_pct=val_pct, models=models, num_rep=5, optim=optim, sigma=sigma,
                opt_centers=opt_m, exp_name=extra_exp_name, flk_maxiter=flk_maxiter,
                num_trace_vecs=num_trace_vecs, cg_tol=cg_tol, approx_trace=approx_trace)
    elif False:  # Changing validation percentages (hgrad-closed)
        for val_pct in VAL_PCTS:
            run_for_models(
                sigma_init=sigma_init, pen_init=penalty_init,
                lr=learning_rate, num_epochs=num_epochs, M=M, dataset=dataset,
                val_pct=val_pct, models=["hgrad-closed"], num_rep=5, optim=optim, sigma=sigma,
                opt_centers=opt_m, exp_name=extra_exp_name, flk_maxiter=flk_maxiter,
                num_trace_vecs=num_trace_vecs, cg_tol=cg_tol, approx_trace=approx_trace)
    elif False:
        for si, pi in SIGMA_PEN_PAIRS:
            run_for_models(
                sigma_init=si, pen_init=pi,
                lr=learning_rate, num_epochs=num_epochs, M=M, dataset=dataset,
                val_pct=val_pct, models=MODELS, num_rep=1, optim=optim, sigma=sigma,
                opt_centers=opt_m, exp_name=extra_exp_name, flk_maxiter=flk_maxiter,
                num_trace_vecs=num_trace_vecs, cg_tol=cg_tol, approx_trace=approx_trace)
    else:
        for dset in datasets:
            run_for_models(
                sigma_init=sigma_init, pen_init=penalty_init,
                lr=learning_rate, num_epochs=num_epochs, M=M, dataset=dset,
                val_pct=val_pct, models=models, num_rep=3, optim=optim, sigma=sigma,
                opt_centers=opt_m, exp_name=extra_exp_name, flk_maxiter=flk_maxiter,
                num_trace_vecs=num_trace_vecs, cg_tol=cg_tol, approx_trace=approx_trace)


if __name__ == "__main__":
    run()
    #run_multistart_2params()
