import pathlib
import io
import subprocess
from typing import List

SIMPLE_HOPT_PATH = pathlib.Path(__file__).parent.joinpath("simple_hopt.py").resolve()
DEFAULT_SEED = 12319


def run_simple_hopt(sigma_init: float,
                    pen_init: float,
                    lr: float,
                    num_epochs: int,
                    M: int,
                    dataset: str,
                    ename: str,
                    val_pct: float,
                    model: str,
                    optim: str,
                    sigma: str,
                    seed: int = DEFAULT_SEED,):
    proc = subprocess.Popen([
        #"PYTHONPATH=..",
        f"python {SIMPLE_HOPT_PATH} "
        f"--seed {seed} "
        f"--cg-tol {1e-3} "
        f"--val-pct {val_pct} "
        f"--sigma-type {sigma} "
        f"--sigma-init {sigma_init} "
        f"--penalty-init {pen_init} "
        f"--lr {lr} "
        f"--epochs {num_epochs} "
        f"--optimizer {optim} "
        f"--op "
        f"--os "
        f"--oc "
        f"--num-centers {M} "
        f"--dataset {dataset} "
        f"--model {model} "
        f"--cuda "
        f"--name {dataset}_hopt_{model}_{ename} "
    ], shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
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
                   ename: str,
                   val_pct: float,
                   models: List[str],
                   sigma: str,
                   num_rep: int = 1,
                   ):
    for model in models:
        for i in range(num_rep):
            run_simple_hopt(sigma_init, pen_init, lr, num_epochs, M, dataset, ename, val_pct, model,
                            optim, seed=DEFAULT_SEED+i, sigma=sigma)


def run_for_valpct(sigma_init: float,
                   pen_init: float,
                   lr: float,
                   num_epochs: int,
                   M: int,
                   dataset: str,
                   optim: str,
                   ename: str,
                   val_pcts: List[float],
                   sigma: str,
                   model: str = "hgrad-closed",
                   num_rep: int = 1,
                   ):
    for val_pct in val_pcts:
        for i in range(num_rep):
            run_simple_hopt(sigma_init, pen_init, lr, num_epochs, M, dataset, ename, val_pct, model,
                            optim, seed=DEFAULT_SEED+i, sigma=sigma)


SIGMA_PEN_PAIRS = [
    (1.0, 1.0),
    (15.0, 1.0),
    (1.0, 1e-4),
    (15.0, 1e-4),
]
VAL_PCTS = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
MODELS = [
    #"loocv",
    "sgpr",
    "gcv",
    #"hgrad-ift",
    "hgrad-closed",
    "creg-nopenfit",
    "creg-penfit",
]
if __name__ == "__main__":
    si = 1.0
    pi = 1.0
    num_epochs = 500
    learning_rate = 0.01
    M = 100
    dataset = "protein"
    val_pct = 0.4
    optim = "adam"
    sigma = "diag"

    if True:
        for si, pi in SIGMA_PEN_PAIRS:
            ename = f"test_hopt_{optim}_m{M}_lr{learning_rate}_pinit{pi}sinit{si}_meanrem_val{val_pct}_{sigma}sig_optM_sec"
            run_for_models(
                sigma_init=si, pen_init=pi,
                lr=learning_rate, num_epochs=num_epochs, M=M, dataset=dataset, ename=ename,
                val_pct=val_pct, models=MODELS, num_rep=5, optim=optim, sigma=sigma)
    if False:
        for val_pct in VAL_PCTS:
            ename = f"test_hopt_{optim}_m{M}_lr{learning_rate}_pinit{pi}sinit{si}_meanrem_val{val_pct}_{sigma}sig"
            run_for_models(
                sigma_init=si, pen_init=pi,
                lr=learning_rate, num_epochs=num_epochs, M=M, dataset=dataset, ename=ename,
                val_pct=val_pct, models=["hgrad-closed"], num_rep=10, optim=optim, sigma=sigma)
