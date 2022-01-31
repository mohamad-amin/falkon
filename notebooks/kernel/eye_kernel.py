from typing import Union, Optional, Dict

import torch

from falkon import sparse
from falkon.kernels.diff_kernel import DiffKernel
from falkon.la_helpers.square_norm_fn import square_norm_diff
from falkon.options import FalkonOptions
from falkon.sparse import SparseTensor


def eye_core(mat1, mat2, out: Optional[torch.Tensor]):
    """
    Note 1: if out is None, then this function will be differentiable wrt all three remaining inputs.
    Note 2: this function can deal with batched inputs

    Parameters
    ----------
    sigma
    mat1
    mat2
    out

    Returns
    -------

    """
    n1, n2 = mat1.shape[0], mat2.shape[0]
    if n1 == n2:
        out = torch.eye(n1)
    else:
        out = torch.rand(n1, n2)
    print('Called with shapes', n1, n2)
    return out


class EyeKernel(DiffKernel):
    kernel_name = "eye"
    core_fn = eye_core

    def __init__(self, opt: Optional[FalkonOptions] = None):
        super().__init__(self.kernel_name, opt, core_fn=EyeKernel.core_fn)

