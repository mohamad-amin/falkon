import pathlib
import io
import subprocess
from typing import List

SIMPLE_HOPT_PATH = pathlib.Path(__file__).parent.joinpath("simple_hopt.py").resolve()
DEFAULT_SEED = 12319


def gen_exp_name(optimizer, num_centers, learning_rate, penalty, sigma, val_percentage, opt_m,
                 sigma_type, model, dataset, num_trace_vecs, flk_maxiter, extra_name):
    val_pct_str = f"_val{val_percentage}_" if model in VAL_MODELS else ""
    trace_vec_str = f"_ste{num_trace_vecs}_" if model in STOCH_MODELS else ""
    flk_miter_str = f"_{flk_maxiter}fits_" if model in STOCH_MODELS else ""
    opt_m_str = "_optM_" if opt_m else ""

    return f"{dataset}_hopt_{model}_test_hopt_{optimizer}_m{num_centers}_lr{learning_rate}_" \
           f"pinit{penalty}_{sigma_type}sinit{sigma}{val_pct_str}{trace_vec_str}{flk_miter_str}{opt_m_str}_" \
           f"{extra_name}"


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
                    seed: int = DEFAULT_SEED, ):
    proc_args = [
        f"python {SIMPLE_HOPT_PATH}",
        f"--seed {seed}",
        f"--cg-tol {1e-3}",
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
        f"--cuda",
        f"--name {gen_exp_name(optim, M, lr, pen_init, sigma_init, val_pct, opt_centers, sigma, model, dataset, num_trace_vecs, flk_maxiter, exp_name)}",
    ]
    if opt_centers:
        proc_args.append("--oc")
    proc = subprocess.Popen([" ".join(proc_args)], shell=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    for line in io.TextIOWrapper(proc.stdout, encoding='utf-8'):
        print(line, end='')
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
                   exp_name: str,
                   num_rep: int = 1,
                   ):
    for model in models:
        for i in range(num_rep):
            run_simple_hopt(sigma_init, pen_init, lr, num_epochs, M, dataset, val_pct, model,
                            optim, seed=DEFAULT_SEED + i, sigma=sigma, opt_centers=opt_centers,
                            num_trace_vecs=num_trace_vecs, flk_maxiter=flk_maxiter,
                            exp_name=exp_name)


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
                   exp_name: str,
                   model: str = "hgrad-closed",
                   num_rep: int = 1,
                   ):
    for val_pct in val_pcts:
        for i in range(num_rep):
            run_simple_hopt(sigma_init, pen_init, lr, num_epochs, M, dataset, val_pct, model,
                            optim, seed=DEFAULT_SEED + i, sigma=sigma, opt_centers=opt_centers,
                            num_trace_vecs=num_trace_vecs, flk_maxiter=flk_maxiter,
                            exp_name=exp_name)


SIGMA_PEN_PAIRS = [
    (1.0, 1.0),
    (15.0, 1.0),
    (1.0, 1e-4),
    (15.0, 1e-4),
]
VAL_PCTS = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
STOCH_MODELS = [
    "stoch-gcv",
    "stoch-creg-nopenfit",
    "stoch-creg-penfit",
]
VAL_MODELS = [
    "hgrad-ift",
    "hgrad-closed",
]
MODELS = [
    # "loocv",
    # "sgpr",
    "gcv",
    # "hgrad-ift",
    # "hgrad-closed",
    "creg-nopenfit",
    "creg-penfit",
]


def run():
    num_epochs = 200
    learning_rate = 0.1
    M = 20
    dataset = "protein"
    val_pct = 0.4
    optim = "adam"
    sigma = "diag"
    opt_m = True
    extra_exp_name = "meanrem"
    flk_maxiter = 20
    num_trace_vecs = 20

    if True:
        for si, pi in SIGMA_PEN_PAIRS:
            run_for_models(
                sigma_init=si, pen_init=pi,
                lr=learning_rate, num_epochs=num_epochs, M=M, dataset=dataset,
                val_pct=val_pct, models=MODELS, num_rep=5, optim=optim, sigma=sigma,
                opt_centers=opt_m, exp_name=extra_exp_name, flk_maxiter=flk_maxiter,
                num_trace_vecs=num_trace_vecs)
    if False:
        for val_pct in VAL_PCTS:
            run_for_models(
                sigma_init=si, pen_init=pi,
                lr=learning_rate, num_epochs=num_epochs, M=M, dataset=dataset,
                val_pct=val_pct, models=["hgrad-closed"], num_rep=10, optim=optim, sigma=sigma,
                opt_centers=opt_m, exp_name=extra_exp_name, flk_maxiter=flk_maxiter,
                num_trace_vecs=num_trace_vecs)


if __name__ == "__main__":
    run()
