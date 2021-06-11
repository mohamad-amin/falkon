import subprocess
import os

import numpy as np

from benchmark.hgrad_benchmarks import gen_grid_spec

os.environ['PYTHONPATH'] = '..'
file_dir = os.path.abspath(os.path.dirname(__file__))
runner = os.path.join(file_dir, "simple_hopt.py")

DEFAULT_VAL_PCT = 0.2
gs_file = "tmp_gs_file"
exp_name = "test_exp_m100"

datasets = {
    #"boston": {
    #    "num_centers": 100,
    #    "sigmas": np.logspace(0, 2, 10),
    #    "penalties": np.logspace(-8, 2, 15),
    #},
    #"energy": {
    #    "num_centers": 100,
    #    "sigmas": np.logspace(0, 2, 10),
    #    "penalties": np.logspace(-8, 2, 15),
    #},
    #"ho-higgs": {
    #    "num_centers": 100,
    #    "sigmas": np.logspace(0, 1.5, 10),
    #    "penalties": np.logspace(-6, 2, 15),
    #},
    "protein": {
        "num_centers": 100,
        "sigmas": np.logspace(0, 1.5, 10),
        "penalties": np.logspace(-6, 2, 15),
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

run_str = """python %s \
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
    --name "{dataset}_gs_{model}_{exp_name}" """ % (runner)

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
        print("Running command: %s" % (conc_run_str), flush=True)
        subprocess.call(conc_run_str, shell=True)
