import argparse

import numpy as np
import pandas as pd
import itertools


sigmas = np.logspace(0, 2, 10)
penalties = np.logspace(-8, 2, 15)


def gen_grid_spec(out_file):
    with open(out_file, "w") as fh:
        fh.write("sigma,penalty\n")
        for ex in itertools.product(sigmas, penalties):
            fh.write("%.8e,%.8e\n" % (ex[0], ex[1]))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Grid-Spec generator")
    p.add_argument('-f', '--out-file', type=str, required=True,
                   help='Output file name')
    args = p.parse_args()
    print("Generating grid-spec to file %s" % (args.out_file))
    gen_grid_spec(args.out_file)

