import subprocess

import numpy as np

from benchmark.hgrad_benchmarks import gen_grid_spec

DEFAULT_VAL_PCT = 0.2
gs_file = "tmp_gs_file"
exp_name = "test_exp_1"

datasets = {
    "boston": {
        "num_centers": 20,
        "sigmas": np.logspace(0, 2, 10),
        "penalties": np.logspace(-8, 2, 15),
    },
}
models = {
    "sgpr": {},
    "creg-nopenfit": {},
    "creg-penfit": {},
    "loocv": {},
    "gcv": {},
    "hgrad-closed": {'val_pct': 0.2}
}

run_str = """
PYTHONPATH=.. python hgrad_benchmarks/simple_hopt.py \
    --seed 12319 \
    --cg-tol 1e-3 \
    --val-pct {val_pct} \
    --sigma-type single \
    --sigma-init 1.0 \
    --penalty-init 1e-4 \
    --num-centers {num_centers} \
    --dataset {dataset} \
    --model {model} \
    --grid-spec "{gs_file}" \
    --name "{dataset}_gs_{model}_{exp_name}"
"""

for dset, dset_params in datasets.items():
    for model, model_params in models.items():
        gen_grid_spec.sigmas = dset_params['sigmas']
        gen_grid_spec.penalties = dset_params['penalties']
        gen_grid_spec.gen_grid_spec(gs_file)
        conc_run_str = run_str.format(
            dataset=dset,
            model=model,
            gs_file=gs_file,
            exp_name=exp_name,
            num_centers=dset_params['num_centers'],
            val_pct=model_params.get('val_pct', DEFAULT_VAL_PCT),
        )
        subprocess.call(["/bin/bash", conc_run_str])
