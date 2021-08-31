from contextlib import ExitStack
from typing import Optional, Union

import numpy as np
import torch
import torch.cuda as tcd

from falkon.mmv_ops.fmmv_cuda import ArgsFmmv, ArgsFdmmv
from falkon.mmv_ops.utils import *
from falkon.options import BaseOptions
from falkon.sparse import SparseTensor
from falkon.utils.device_copy import copy
from falkon.utils.helpers import (
    calc_gpu_block_sizes,
    sizeof_dtype,
    select_dim_over_nm_v2,
    select_dim_over_n,
)
from falkon.utils.tensor_helpers import (
    create_same_stride,
    extract_fortran, create_fortran,
)


def _sparse_mmv_blk_sizes(n, d, m, t, avail_mem, extra_mem, incore: bool, m1_density: float, m2_density: float):
    # Memory needs:
    # chunk of x1: n + 2*d*n*density
    # chunk of x2: d + 2*d*m*density (because is transposed)
    # chunk of kernel: n + 2*n*m*density (assume density=1)
    # dense matrices: kernel (n*m) + output (n*t) + vector (m*t)
    coef_nm = 3  # Both dense and sparse spspmm output must be allocated.
    coef_n, coef_m, coef_rest = 0, 0, 0
    if not incore:
        coef_n += 2 + 2 * d * m1_density + t
        coef_m += 2 * d * m2_density + t
        coef_rest = d
    blk_n, blk_m = select_dim_over_nm_v2(
        max_n=n, max_m=m, max_mem=avail_mem,
        coef_nm=coef_nm + extra_mem.get('nm', 0),
        coef_n=coef_n + extra_mem.get('n', 0) + extra_mem.get('nd', 0) * d,
        coef_m=coef_m + extra_mem.get('m', 0) + extra_mem.get('md', 0) * d,
        rest=coef_rest + extra_mem.get('d', 0))
    # here mem_needed is only the dense blocks!
    mem_needed = blk_m * blk_n
    if not incore:
        mem_needed += (blk_n + blk_m) * t
    return blk_n, blk_m, mem_needed


def _dense_mmv_blk_sizes(n, d, m, t, avail_mem, extra_mem, incore: bool):
    coef_n, coef_m = 0, 0
    if not incore:
        coef_n += d + t
        coef_m += d + t
    blk_n, blk_m = select_dim_over_nm_v2(
        max_n=n, max_m=m, max_mem=avail_mem,
        coef_nm=1 + extra_mem.get('nm', 0),
        coef_n=coef_n + extra_mem.get('n', 0) + extra_mem.get('nd', 0) * d,
        coef_m=coef_m + extra_mem.get('m', 0) + extra_mem.get('md', 0) * d,
        rest=extra_mem.get('d', 0))
    mem_needed = blk_m * blk_n
    if not incore:
        mem_needed += blk_n * (d + t) + blk_m * (d + t)
    return blk_n, blk_m, mem_needed


def mmv_run_starter(proc_idx, queue, device_id):
    a: ArgsFmmv = queue.get()
    X1, X2, v, out = a.X1, a.X2, a.v, a.out
    kernel = a.kernel
    max_mem = a.max_mem
    dev = _dev_from_id(device_id)
    incore = _is_incore(dev, X1.device)
    is_sparse = isinstance(X1, SparseTensor) and isinstance(X2, SparseTensor)

    # Choose batch sizes
    avail_mem = max_mem / sizeof_dtype(X1.dtype)
    extra_mem = kernel.extra_mem()
    n, d = X1.shape
    m, t = v.shape
    if is_sparse:
        blk_n, blk_m, mem_needed = _sparse_mmv_blk_sizes(
            n=n, m=m, d=d, t=t, avail_mem=avail_mem, extra_mem=extra_mem, incore=incore,
            m1_density=X1.density, m2_density=X2.density)
        sparse_mmv_run_thread(X1, X2, v, out, kernel, blk_n, blk_m, mem_needed, dev)
    else:
        blk_n, blk_m, mem_needed = _dense_mmv_blk_sizes(
            n=n, m=m, d=d, t=t, avail_mem=avail_mem, extra_mem=extra_mem, incore=incore)
        mmv_run_thread(X1, X2, v, out, kernel, blk_n, blk_m, mem_needed, dev)


def sparse_mmv_run_thread(m1: SparseTensor, m2: SparseTensor, v: torch.Tensor,
                          out: torch.Tensor, kernel: 'Kernel', blk_n: int, blk_m: int,
                          mem_needed: int, dev: torch.device):
    incore = _is_incore(dev, m1.device)
    N, D = m1.shape
    M, T = v.shape

    """ Initialize extra buffers """
    flat_gpu = torch.empty(size=(mem_needed,), dtype=m1.dtype, device=dev)
    flat_offset = 0
    # ker_gpu must be fortran-ordered due to cusparse csr2dense function (TODO: only on CUDA)
    ker_gpu = extract_fortran(flat_gpu, size=(blk_n, blk_m), offset=flat_offset)
    flat_offset += np.prod(ker_gpu.shape)
    dev_v, dev_out = None, None
    if not incore:
        dev_out, flat_offset = _extract_flat(flat_gpu, size=(blk_n, T), other=out, offset=flat_offset)
        dev_v, flat_offset = _extract_flat(flat_gpu, size=(blk_m, T), other=v, offset=flat_offset)

    with ExitStack() as stack:
        if dev.type == 'cuda':
            s1 = tcd.current_stream(dev)
            s2 = tcd.Stream(dev)
            stack.enter_context(tcd.device(dev))
            stack.enter_context(tcd.stream(s1))

        for i in range(0, N, blk_n):
            leni = min(blk_n, N - i)

            c_m1 = m1.narrow_rows(i, leni)
            if incore:  # Note that CUDA-incore is not allowed to happen (so this is CPU->CPU)
                c_dev_out = out[i: i + leni]
                c_dev_m1 = c_m1
            else:  # CPU -> CUDA
                c_dev_out = dev_out[:leni]
                c_dev_m1 = m1.index_to_int().to(device=dev, non_blocking=True)
            c_dev_out.fill_(0.0)

            for j in range(0, M, blk_m):
                lenj = min(blk_m, M - j)

                c_m2 = m2.narrow_rows(j, lenj)
                if incore:  # CPU -> CPU
                    c_dev_m2 = c_m2.transpose_csc()
                    c_dev_v = v[j: j + lenj]
                else:  # CPU -> CUDA
                    c_dev_m2 = SparseTensor.from_scipy(
                        c_m2.transpose_csc().to_scipy().tocsr(copy=False)) \
                        .index_to_int() \
                        .to(device=dev, non_blocking=True)
                    c_dev_v = copy(v[j: j + lenj], dev_v[:lenj], s=s2)
                c_dev_ker = ker_gpu[:leni, :lenj].fill_(0.0)

                ddd = kernel._prepare_sparse(c_m1, c_m2)
                kernel._apply_sparse(c_dev_m1, c_dev_m2, c_dev_ker)
                c_dev_ker = kernel._finalize(c_dev_ker, ddd)
                if not incore:
                    s2.synchronize()
                c_dev_out.addmm_(c_dev_ker, c_dev_v)

                # Copy output to host
                if not incore:
                    copy(c_dev_out, out[i: i + leni], s=s1)


def mmv_run_thread(m1: torch.Tensor, m2: torch.Tensor, v: Optional[torch.Tensor],
                   out: torch.Tensor, kernel: 'Kernel', blk_n: int, blk_m: int,
                   mem_needed: int, dev: torch.device):
    # data(CUDA), dev(CUDA) or data(CPU), dev(CPU)
    incore = _is_incore(dev, m1.device)
    N, D = m1.shape
    M, T = v.shape

    # Initialize extra buffers
    flat_gpu = torch.empty(size=(mem_needed,), dtype=m1.dtype, device=dev)
    flat_offset = 0
    ker_gpu, flat_offset = _extract_flat(flat_gpu, size=(blk_n, blk_m), other=out, offset=flat_offset)
    dev_m1, dev_m2, dev_out, dev_v = None, None, None, None
    if not incore:
        dev_m1, flat_offset = _extract_flat(flat_gpu, size=(blk_n, D), other=m1, offset=flat_offset)
        dev_m2, flat_offset = _extract_flat(flat_gpu, size=(blk_m, D), other=m2, offset=flat_offset)
        dev_out, flat_offset = _extract_flat(flat_gpu, size=(blk_n, T), other=out, offset=flat_offset)
        dev_v, flat_offset = _extract_flat(flat_gpu, size=(blk_m, T), other=v, offset=flat_offset)

    with ExitStack() as stack:
        if dev.type == 'cuda':
            s1 = tcd.current_stream(dev)
            s2 = tcd.Stream(dev)
            stack.enter_context(tcd.device(dev))
            stack.enter_context(tcd.stream(s1))
        for i in range(0, N, blk_n):
            leni = min(blk_n, N - i)
            if incore:
                c_dev_m1 = m1[i: i + leni, :]
                c_dev_out = out[i: i + leni]
            else:
                c_dev_m1 = copy(m1[i: i + leni, :], dev_m1[:leni, :], s=s1)
                c_dev_out = dev_out[:leni]
            c_dev_out.fill_(0.0)

            for j in range(0, M, blk_m):
                lenj = min(blk_m, M - j)
                if incore:
                    c_dev_m2 = m2[j: j + lenj, :]
                    c_dev_v = v[j: j + lenj, :]
                else:
                    c_dev_m2 = copy(m2[j: j + lenj, :], dev_m2[:lenj, :], s=s1)
                    c_dev_v = copy(v[j: j + lenj, :], dev_v[:lenj, :], s=s2)
                c_dev_ker = ker_gpu[:leni, :lenj].fill_(0.0)

                c_dev_ker = kernel.compute(c_dev_m1, c_dev_m2, c_dev_ker)
                if not incore:
                    s2.synchronize()
                c_dev_out.addmm_(c_dev_ker, c_dev_v)

            if not incore:
                copy(c_dev_out, out[i: i + leni], s=s1)
            # end iter over N
        # end iter over M
    # exit context manager (device, stream)


def _sparse_dmmv_blk_sizes(n, d, m, t, avail_mem, extra_mem: dict, incore: bool,
                           dev_out_exists: bool, has_w: bool, m1_density: float, m2_density: float):
    # Memory needs:
    # chunk of X1              : n + 2*d*n*density
    # full X2                  : d + 2*d*m*density (it's transposed)
    # sparse output (internal) : n + 2*n*m*density (density assumed = 1)
    # And the dense matrices: kernel(m*n), w(n*t), v(m*t), output(m*t)
    coef_nm = 3  # Both dense and sparse spspmm output must be allocated.
    coef_nd, coef_md, coef_nt, coef_mt = 0, 0, 0, 0
    coef_nt += 1  # for dev_w allocation
    if not incore:
        coef_nd += 2 * m1_density  # for x1-chunk
        coef_md += 2 * m2_density  # for x2
        coef_mt += 1  # for v
        if not dev_out_exists:
            coef_mt += 1  # for output
    blk_n = select_dim_over_n(
        max_n=n, m=m, d=d, max_mem=avail_mem,
        coef_nm=coef_nm + extra_mem.get('nm', 0),
        coef_nd=coef_nd + extra_mem.get('nd', 0),
        coef_md=coef_md + extra_mem.get('md', 0),
        coef_n=coef_nt * t + 2 + extra_mem.get('n', 0) + t * extra_mem.get('nt', 0),
        coef_m=coef_mt * t + extra_mem.get('m', 0) + t * extra_mem.get('mt', 0),
        coef_d=1 + extra_mem.get('d', 0), rest=0)

    mem_needed = blk_n * m
    mem_needed += blk_n * t  # dev_w
    if not incore:
        mem_needed += m * t
        if not dev_out_exists:
            mem_needed += m * t

    return blk_n, mem_needed


def _dense_dmmv_blk_sizes(n, d, m, t, avail_mem: float, extra_mem: dict, incore: bool, dev_out_exists: bool, has_w: bool):
    coef_nm, coef_nd, coef_md, coef_nt, coef_mt = 1, 0, 0, 0, 0
    coef_nt += 1  # for dev_w allocation
    if not incore:
        coef_nd += 1  # x1
        coef_md += 1  # x2
        coef_mt += 1  # v
        if not dev_out_exists:
            coef_mt += 1  # output
    blk_n = select_dim_over_n(
        max_n=n, m=m, d=d, max_mem=avail_mem,
        coef_nm=1 + extra_mem.get('nm', 0),
        coef_nd=coef_nd + extra_mem.get('nd', 0),
        coef_md=coef_md + extra_mem.get('md', 0),
        coef_n=coef_nt * t + extra_mem.get('n', 0) + t * extra_mem.get('nt', 0),
        coef_m=coef_mt * t + extra_mem.get('m', 0) + t * extra_mem.get('mt', 0),
        coef_d=extra_mem.get('d', 0), rest=0)
    mem_needed = blk_n * m
    if not incore:
        mem_needed += m * t + m * d + blk_n * d
        if not dev_out_exists:
            mem_needed += m * t
    mem_needed += blk_n * t  # dev_w
    return blk_n, mem_needed


def dmmv_run_starter(proc_idx, queue, device_id):
    a: ArgsFdmmv = queue.get()
    X1, X2, v, w, out = a.X1, a.X2, a.v, a.w, a.out
    kernel = a.kernel
    max_mem = a.max_mem
    dev = _dev_from_id(device_id)
    incore = _is_incore(dev, X1.device)
    dev_out_exists = out.device == dev  # out has already been allocated on the computation device
    is_sparse = isinstance(X1, SparseTensor) and isinstance(X2, SparseTensor)

    # Choose batch sizes
    avail_mem = max_mem / sizeof_dtype(X1.dtype)
    extra_mem = kernel.extra_mem()
    n, d = X1.shape
    m, t = v.shape

    if is_sparse:
        blk_n, mem_needed = _sparse_dmmv_blk_sizes(
            n=n, d=d, m=m, t=t, avail_mem=avail_mem, extra_mem=extra_mem, incore=incore,
            dev_out_exists=dev_out_exists, has_w=w is not None, m1_density=X1.density,
            m2_density=X2.density)
        sparse_dmmv_run_thread(X1, X2, v, w, out, kernel, blk_n, mem_needed, dev)
    else:
        blk_n, mem_needed = _dense_dmmv_blk_sizes(
            n=n, d=d, m=m, t=t, avail_mem=avail_mem, extra_mem=extra_mem, incore=incore,
            dev_out_exists=dev_out_exists, has_w=w is not None)
        dmmv_run_thread(X1, X2, v, w, out, kernel, blk_n, mem_needed, dev)


def sparse_dmmv_run_thread(m1: SparseTensor, m2: SparseTensor, v: torch.Tensor,
                           w: Optional[torch.Tensor], out: torch.Tensor, kernel: 'Kernel',
                           blk_n: int, mem_needed: int, dev: torch.device):
    incore = _is_incore(dev, m1.device)
    dev_out_exists = out.device == dev  # out has already been allocated on the computation device
    N, D = m1.shape
    M, T = v.shape

    """ Initialize extra buffers """
    flat_gpu = torch.empty(size=(mem_needed,), dtype=m1.dtype, device=dev)
    flat_offset = 0
    # ker_gpu must be fortran-ordered due to cusparse csr2dense function (TODO: only on CUDA)
    ker_gpu = extract_fortran(flat_gpu, size=(blk_n, M), offset=flat_offset)
    flat_offset += np.prod(ker_gpu.shape)
    dev_w, flat_offset = _extract_flat(flat_gpu, size=(blk_n, T), other=v if w is None else w, offset=flat_offset)
    dev_out, dev_v, dev_m2 = out, v, m2
    if not incore:
        if not dev_out_exists:
            dev_out, flat_offset = _extract_flat(flat_gpu, size=(M, T), other=out, offset=flat_offset)
        dev_v, flat_offset = _extract_flat(flat_gpu, size=(M, T), other=v, offset=flat_offset)
    dev_out.fill_(0.0)

    with ExitStack() as stack:
        s1, s2 = None, None
        if dev.type == 'cuda':
            s1 = tcd.current_stream(dev)
            s2 = tcd.Stream(dev)
            stack.enter_context(tcd.device(dev))
            stack.enter_context(tcd.stream(s1))
        if not incore:  # Note that CUDA-incore is not allowed to happen (CPU->CUDA)
            copy(v, dev_v, s=s1)
            dev_m2 = SparseTensor.from_scipy(
                m2.transpose_csc().to_scipy().tocsr(copy=False)) \
                .index_to_int() \
                .to(device=dev)
        else:
            dev_m2 = m2.transpose_csc()

        for i in range(0, N, blk_n):
            leni = min(blk_n, N - i)

            c_m1 = m1.narrow_rows(i, leni)
            if incore:  # Note that CUDA-incore is not allowed to happen (so this is CPU->CPU)
                c_dev_m1 = c_m1
            else:  # CPU -> CUDA
                c_dev_m1 = c_m1.index_to_int().to(device=dev, non_blocking=True)
            if w is None:
                c_dev_w = dev_w[:leni, :].fill_(0.0)
            else:
                c_dev_w = copy(w[i: i + leni, :], dev_w[:leni, :], s=s1)

            c_dev_ker = ker_gpu[:leni].fill_(0.0)
            ddd = kernel._prepare_sparse(c_m1, m2)
            kernel._apply_sparse(c_dev_m1, dev_m2, c_dev_ker)
            c_dev_ker = kernel._finalize(c_dev_ker, ddd)

            c_dev_w.addmm_(c_dev_ker, dev_v)
            dev_out.addmm_(c_dev_ker.T, c_dev_w)
        if not incore and not dev_out_exists:
            copy(dev_out, out, s=s1)


def dmmv_run_thread(m1: torch.Tensor, m2: torch.Tensor, v: torch.Tensor,
                    w: Optional[torch.Tensor], out: torch.Tensor,
                    kernel: 'Kernel', blk_n: int, mem_needed: int, dev: torch.device):
    # k(x2, x1) @ (k(x1, x2) @ v + w)
    # data(CUDA), dev(CUDA) or data(CPU), dev(CPU)
    incore = _is_incore(dev, m1.device)
    dev_out_exists = out.device == dev  # out has already been allocated on the computation device
    N, D = m1.shape
    M, T = v.shape

    # Initialize extra buffers
    flat_gpu = torch.empty(size=(mem_needed,), dtype=m1.dtype, device=dev)
    flat_offset = 0
    dev_ker, flat_offset = _extract_flat(flat_gpu, size=(blk_n, M), other=out, offset=flat_offset)
    dev_w, flat_offset = _extract_flat(flat_gpu, size=(blk_n, T), other=v if w is None else w, offset=flat_offset)
    dev_m1, dev_m2, dev_out, dev_v = None, m2, out, v
    if not incore:
        dev_m1, flat_offset = _extract_flat(flat_gpu, size=(blk_n, D), other=m1, offset=flat_offset)
        dev_m2, flat_offset = _extract_flat(flat_gpu, size=(M, D), other=m2, offset=flat_offset)
        if not dev_out_exists:
            dev_out, flat_offset = _extract_flat(flat_gpu, size=(M, T), other=out, offset=flat_offset)
        dev_v, flat_offset = _extract_flat(flat_gpu, size=(M, T), other=v, offset=flat_offset)
    dev_out.fill_(0.0)

    with ExitStack() as stack:
        s1, s2 = None, None
        if dev.type == 'cuda':
            s1 = tcd.current_stream(dev)
            s2 = tcd.Stream(dev)
            stack.enter_context(tcd.device(dev))
            stack.enter_context(tcd.stream(s1))
        if not incore:
            copy(m2, dev_m2, s=s1)
            copy(v, dev_v, s=s1)
        for i in range(0, N, blk_n):
            leni = min(blk_n, N - i)
            if incore:
                c_dev_m1 = m1[i: i + leni, :]
            else:
                c_dev_m1 = copy(m1[i: i + leni, :], dev_m1[:leni, :], s=s1)
            if w is not None:
                c_dev_w = copy(w[i: i + leni, :], dev_w[:leni, :], s=s2)
            else:
                c_dev_w = dev_w[:leni, :].fill_(0.0)

            c_dev_ker = kernel.compute(c_dev_m1, dev_m2, dev_ker[:leni, :])
            if not incore:
                s2.synchronize()
            # noinspection PyUnboundLocalVariable
            c_dev_w.addmm_(c_dev_ker, dev_v)
            dev_out.addmm_(c_dev_ker.T, c_dev_w)
        if not incore and not dev_out_exists:
            copy(dev_out, out, s=s1)


def fmmv(X1: Union[torch.Tensor, SparseTensor],
         X2: Union[torch.Tensor, SparseTensor],
         v: torch.Tensor, kernel: 'Kernel', out: Optional[torch.Tensor] = None,
         opt: Optional[BaseOptions] = None) -> torch.Tensor:
    is_sparse = isinstance(X1, SparseTensor)
    if not is_sparse:
        _check_contiguity((X1, 'X1'), (X2, 'X2'), (v, 'v'), (out, 'out'))
    data_dev = X1.device
    comp_dev_type = 'cpu' if opt.use_cpu or not torch.cuda.is_available() else 'cuda'
    N, D = X1.shape
    T = v.shape[-1]
    # Create output matrix
    if out is None:
        if is_sparse:
            out = create_fortran((N, T), v.dtype, device=data_dev,
                                 pin_memory=data_dev.type != 'cuda' and comp_dev_type == 'cuda')
        else:
            out = create_same_stride((N, T), X1, v.dtype, device=data_dev,
                                     pin_memory=data_dev.type != 'cuda' and comp_dev_type == 'cuda')

    if comp_dev_type == 'cpu' and data_dev.type == 'cpu':
        args = ArgsFmmv(X1=X1, X2=X2, v=v, out=out, kernel=kernel, max_mem=opt.max_cpu_mem)
        _call_direct(mmv_run_starter, (args, -1))
    elif comp_dev_type == 'cuda' and data_dev.type == 'cuda':
        if is_sparse:
            raise NotImplementedError("In-core, sparse fmmv not implemented. "
                                      "Use the out-of-core version instead.")
        gpu_info = _get_gpu_info(opt, slack=0.9)
        single_gpu_info = [g for g in gpu_info if g.Id == data_dev.index][0]
        args = ArgsFmmv(X1=X1, X2=X2, v=v, out=out, kernel=kernel,
                        max_mem=single_gpu_info.usable_ram)
        _call_direct(mmv_run_starter, (args, data_dev.index))
    elif comp_dev_type == 'cuda' and data_dev.type == 'cpu':
        gpu_info = _get_gpu_info(opt, slack=0.9)
        args = []  # Arguments passed to each subprocess
        block_sizes = calc_gpu_block_sizes(gpu_info, N)
        for i, g in enumerate(gpu_info):
            bwidth = block_sizes[i + 1] - block_sizes[i]
            if bwidth <= 0:
                continue
            if is_sparse:
                X1_block = X1.narrow_rows(block_sizes[i], bwidth)
            else:
                X1_block = X1.narrow(0, block_sizes[i], bwidth)
            args.append((ArgsFmmv(
                X1=X1_block,
                X2=X2, v=v,
                out=out.narrow(0, block_sizes[i], bwidth),
                kernel=kernel, max_mem=g.usable_ram), g.Id))
        _start_wait_processes(mmv_run_starter, args)
    else:
        raise RuntimeError("Requested CPU computations with CUDA data. This should not happen.")
    return out


def fdmmv(X1: Union[torch.Tensor, SparseTensor], X2: Union[torch.Tensor, SparseTensor],
          v: torch.Tensor, w: Optional[torch.Tensor],
          kernel: 'Kernel', out: Optional[torch.Tensor] = None,
          opt: Optional[BaseOptions] = None) -> torch.Tensor:
    """
    X1 : N x D
    X2 : M x D
    v  : M x T
    w  : N x T
    performs fnc(X1*X2', X1, X2)' * ( fnc(X1*X2', X1, X2) * v  +  w )  : M x T
    """
    is_sparse = isinstance(X1, SparseTensor)
    if not is_sparse:
        _check_contiguity((X1, 'X1'), (X2, 'X2'), (v, 'v'), (out, 'out'))
    data_dev = X1.device
    comp_dev_type = 'cpu' if opt.use_cpu or not torch.cuda.is_available() else 'cuda'

    N, D = X1.shape[-2:]
    M, T = v.shape[-2:]
    # Create output matrix
    if out is None:
        if is_sparse:
            out = create_fortran((M, T), v.dtype, device=data_dev,
                                 pin_memory=data_dev.type != 'cuda' and comp_dev_type == 'cuda')
        else:
            out = create_same_stride((M, T), X1, v.dtype, device=data_dev,
                                     pin_memory=data_dev.type != 'cuda' and comp_dev_type == 'cuda')

    if comp_dev_type == 'cpu' and data_dev.type == 'cpu':
        args = ArgsFdmmv(X1=X1, X2=X2, v=v, w=w, out=out, kernel=kernel, max_mem=opt.max_cpu_mem)
        _call_direct(dmmv_run_starter, (args, -1))
    elif comp_dev_type == 'cuda' and data_dev.type == 'cuda':
        if is_sparse:
            raise NotImplementedError("In-core, sparse fdmmv not implemented. "
                                      "Use the out-of-core version instead.")
        gpu_info = _get_gpu_info(opt, slack=0.9)
        single_gpu_info = [g for g in gpu_info if g.Id == data_dev.index][0]
        args = ArgsFdmmv(X1=X1, X2=X2, v=v, w=w, out=out, kernel=kernel,
                         max_mem=single_gpu_info.usable_ram)
        _call_direct(dmmv_run_starter, (args, data_dev.index))
    elif comp_dev_type == 'cuda' and data_dev.type == 'cpu':
        gpu_info = _get_gpu_info(opt, slack=0.9)
        args = []  # Arguments passed to each subprocess
        wrlk = []  # Outputs for each subprocess
        block_sizes = calc_gpu_block_sizes(gpu_info, N)
        for i, g in enumerate(gpu_info):
            bwidth = block_sizes[i + 1] - block_sizes[i]
            if bwidth <= 0:
                continue
            cur_out_gpu = create_same_stride((M, T), out, out.dtype, f'cuda:{gpu_info[i].Id}')
            gpu_info[i].usable_ram -= M * T * sizeof_dtype(X1.dtype)
            wrlk.append(cur_out_gpu)
            if is_sparse:
                X1_block = X1.narrow_rows(block_sizes[i], bwidth)
            else:
                X1_block = X1.narrow(0, block_sizes[i], bwidth)
            args.append((ArgsFdmmv(
                X1=X1_block,
                X2=X2, v=v,
                w=w.narrow(0, block_sizes[i], bwidth) if w is not None else None,
                out=cur_out_gpu,
                kernel=kernel, max_mem=g.usable_ram), g.Id))
        _start_wait_processes(dmmv_run_starter, args)
        if len(wrlk) > 1:  # Sum up all subprocess outputs and copy to `out` on host.
            # noinspection PyTypeChecker
            fastest_device: int = np.argmax([d.speed for d in gpu_info])
            copy(torch.cuda.comm.reduce_add(wrlk, destination=gpu_info[fastest_device].Id), out)
        else:
            copy(wrlk[0], out)
    else:
        raise RuntimeError("Requested CPU computations with CUDA data. This should not happen.")
    return out
