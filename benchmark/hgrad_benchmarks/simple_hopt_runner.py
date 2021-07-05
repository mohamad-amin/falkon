import subprocess
from typing import List

SIMPLE_HOPT_PATH = "./simple_hopt.py"
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
                    seed: int = DEFAULT_SEED,):
    subprocess.run([
        "PYTHONPATH=..",
        "python",
        SIMPLE_HOPT_PATH,
        "--seed", str(seed),
        "--cg-tol", str(1e-3),
        "--val-pct", str(val_pct),
        "--sigma-type", "single",
        "--sigma-init", str(sigma_init),
        "--penalty-init", str(pen_init),
        "--lr", str(lr),
        "--epochs", str(num_epochs),
        "--op",
        "--os",
        "--num-centers", str(M),
        "--dataset", str(dataset),
        "--model", str(model),
        "--cuda",
        "--name", f"{dataset}_hopt_{model}_{ename}",
        "--optimizer", optim,
    ])


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
                   num_rep: int = 1):
    for model in models:
        for i in range(num_rep):
            run_simple_hopt(sigma_init, pen_init, lr, num_epochs, M, dataset, ename, val_pct, model,
                            optim, seed=DEFAULT_SEED+i)


def run_for_valpct(sigma_init: float,
                   pen_init: float,
                   lr: float,
                   num_epochs: int,
                   M: int,
                   dataset: str,
                   optim: str,
                   ename: str,
                   val_pcts: List[float],
                   model: str = "hgrad-closed",
                   num_rep: int = 1):
    for val_pct in val_pcts:
        for i in range(num_rep):
            run_simple_hopt(sigma_init, pen_init, lr, num_epochs, M, dataset, ename, val_pct, model,
                            optim, seed=DEFAULT_SEED+i)


SIGMA_PEN_PAIRS = [
    (1.0, 1.0),
    (15.0, 1.0),
    (1.0, 1e-4),
    (15.0, 1e-4),
]
VAL_PCTS = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
MODELS = [
    "loocv",
    "sgpr",
    "gcv",
    "hgrad-ift",
    "hgrad-closed",
    "creg-nopenfit",
    "creg-penfit",
]
if __name__ == "__main__":
    num_epochs = 50
    learning_rate = 0.005
    M = 20
    dataset = "protein"
    val_pct = 0.2
    optim = "adam"

    if False:
        for si, pi in SIGMA_PEN_PAIRS:
            ename = f"test_hopt_{optim}_m{M}_lr{learning_rate}_pinit{pi}sinit{si}_meanrem_val{val_pct}"
            run_for_models(
                sigma_init=si, pen_init=pi,
                lr=learning_rate, num_epochs=num_epochs, M=M, dataset=dataset, ename=ename,
                val_pct=val_pct, models=MODELS, num_rep=1, optim=optim)
    if True:
        for val_pct in VAL_PCTS:
            ename = f"test_hopt_{optim}_m{M}_lr{learning_rate}_pinit{pi}sinit{si}_meanrem_val{val_pct}"
            run_for_models(
                sigma_init=si, pen_init=pi,
                lr=learning_rate, num_epochs=num_epochs, M=M, dataset=dataset, ename=ename,
                val_pct=val_pct, models=["hgrad-closed"], num_rep=10, optim=optim)
