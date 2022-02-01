import warnings
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

    def _keops_mmv_impl(self, X1, X2, v, kernel, out, opt: FalkonOptions):
        return RuntimeError("SigmoidKernel is not implemented in KeOps")

    def _decide_mmv_impl(self, X1, X2, v, opt):
        if self.keops_can_handle_mmv(X1, X2, v, opt):
            warnings.warn("KeOps MMV implementation for %s kernel is not available. "
                          "Falling back to matrix-multiplication based implementation."
                          % (self.name))
        return super()._decide_mmv_impl(X1, X2, v, opt)

    def _decide_dmmv_impl(self, X1, X2, v, w, opt):
        if self.keops_can_handle_dmmv(X1, X2, v, w, opt):
            warnings.warn("KeOps dMMV implementation for %s kernel is not available. "
                          "Falling back to matrix-multiplication based implementation."
                          % (self.name))
        return super()._decide_dmmv_impl(X1, X2, v, w, opt)

    def compute_sparse(self, X1: SparseTensor, X2: SparseTensor, out: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError('This kernel doesn\'t support sparse computations')

    def detach(self):
        return self
