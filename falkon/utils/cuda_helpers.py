from typing import Optional

import torch

from .helpers import sizeof_dtype
from .tensor_helpers import is_f_contig, is_contig
from falkon.cuda.cudart_gpu import cuda_memcpy2d, cuda_memcpy2d_async
from falkon.cuda.cublas_gpu import (cublasSetMatrix, cublasSetMatrixAsync,
                                    cublasGetMatrix, cublasGetMatrixAsync)


def check_copy(H, D):
    # Data-types
    if H.dtype != D.dtype:
        raise ValueError("Data types of H and D (%s, %s) do not match." % (H.dtype, D.dtype))
    # Sizes
    if H.size() != D.size():
        raise ValueError("Size of H (%s) does not match size of D (%s)" % (H.size(), D.size()))
    # Contiguity
    if is_f_contig(H, strict=False):
        if not is_f_contig(D, strict=False):
            raise ValueError("H is F-contig (strides %s), while D is not (strides %s)" % (H.stride(), D.stride()))
    elif is_contig(H):
        if not is_contig(D):
            raise ValueError("H is C-contig (strides %s), while D is not (strides %s)" % (H.stride(), D.stride()))
    else:
        raise ValueError("H is not memory-contiguous (strides %s)" % (H.stride(), ))


def flk_copy(origin, dest, s=None):
    check_copy(origin, dest)

    if origin.device.type == dest.device.type:
        dest.copy_(origin, non_blocking=s is not None)
    elif origin.device.type == "cpu":  # host -> dev
        copy_to_device(origin.shape[0], origin.shape[1], origin, 0, 0, dest, 0, 0, s)
    else:                              # dev -> host
        copy_to_host(origin.shape[0], origin.shape[1], origin, 0, 0, dest, 0, 0, s)
    return dest


def is_1d_contig(t):
    size = t.size()
    stride = t.stride()

    if size[0] == 1 and stride[1] == 1:
        return True
    if size[1] == 1 and stride[0] == 1:
        return True
    return False


def copy_to_device(rows, cols, H, Hi, Hj, D, Di, Dj, s=None):
    D_narrow = D.narrow(0, Di, rows).narrow(1, Dj, cols)
    H_narrow = H.narrow(0, Hi, rows).narrow(1, Hj, cols)

    dts = sizeof_dtype(D.dtype)

    # strict needs to be False since cublas deals with row/column matrices just fine,
    # while cuda errors-out in certain cases (width > dpitch or width > spitch...)
    if is_f_contig(H, strict=False):
        if s is not None:
            cublasSetMatrixAsync(
                rows=rows, cols=cols, elem_size=dts,
                A=H_narrow.data_ptr(), lda=H_narrow.stride(1),
                B=D_narrow.data_ptr(), ldb=D_narrow.stride(1),
                stream=s._as_parameter_)
        else:
            cublasSetMatrix(
                rows=rows, cols=cols, elem_size=dts,
                A=H_narrow.data_ptr(), lda=H_narrow.stride(1),
                B=D_narrow.data_ptr(), ldb=D_narrow.stride(1))
    elif is_contig(H):
        if s is not None:
            cuda_memcpy2d_async(
                src=H_narrow.data_ptr(), spitch=H_narrow.stride(0) * dts,
                dst=D_narrow.data_ptr(), dpitch=D_narrow.stride(0) * dts,
                width=cols * dts, height=rows, stream=s._as_parameter)
        else:
            print("spitch", H_narrow.stride(0))
            print("dpitch", D_narrow.stride(0))
            print("width", cols)
            print("height", rows)
            cuda_memcpy2d(
                src=H_narrow.data_ptr(), spitch=H_narrow.stride(0) * dts,
                dst=D_narrow.data_ptr(), dpitch=D_narrow.stride(0) * dts,
                width=cols * dts, height=rows)

    return D_narrow


def copy_to_device_noorder(rows, cols, H, Hi, Hj, D, Di, Dj, s=None, check=False):
    if check:
        if rows > H.shape[0] or cols > H.shape[1]:
            raise IndexError("rows, cols (%d, %d) into H of size %s out of range." %
                             (rows, cols, H.shape))
        if rows > D.shape[0] or cols > D.shape[1]:
            raise IndexError("rows, cols (%d, %d) into D of size %s out of range." %
                             (rows, cols, D.shape))
        if H.dtype != D.dtype:
            raise ValueError("Data types of H and D (%s, %s) do not match." % (H.dtype, D.dtype))

    D_narrow = D.narrow(0, Di, rows).narrow(1, Dj, cols)
    H_narrow = H.narrow(0, Hi, rows).narrow(1, Hj, cols)

    if is_f_contig(D, strict=True):
        return copy_to_device(rows, cols, H, Hi, Hj, D, Di, Dj, s=s)
    elif is_contig(D):
        dts = sizeof_dtype(D.dtype)
        if s is not None:
            cuda_memcpy2d_async(
                dst=H_narrow.data_ptr(), dpitch=H_narrow.stride(0) * dts,
                src=D_narrow.data_ptr(), spitch=D_narrow.stride(0) * dts,
                width=cols * dts, height=rows, stream=s._as_parameter)
        else:
            cuda_memcpy2d(
                dst=H_narrow.data_ptr(), dpitch=H_narrow.stride(0) * dts,
                src=D_narrow.data_ptr(), spitch=D_narrow.stride(0) * dts,
                width=cols * dts, height=rows)

        return D_narrow


def copy_to_host(rows, cols, D, Di, Dj, H, Hi, Hj, s=None):
    D_narrow = D.narrow(0, Di, rows).narrow(1, Dj, cols)
    H_narrow = H.narrow(0, Hi, rows).narrow(1, Hj, cols)

    dts = sizeof_dtype(D.dtype)

    # strict needs to be False since cublas deals with row/column matrices just fine,
    # while cuda errors-out in certain cases (width > dpitch or width > spitch...)
    if is_f_contig(H, strict=False):
        if s is not None:
            cublasGetMatrixAsync(
                rows=rows, cols=cols, elem_size=dts,
                A=D_narrow.data_ptr(), lda=D_narrow.stride(1),
                B=H_narrow.data_ptr(), ldb=H_narrow.stride(1),
                stream=s._as_parameter_)
        else:
            cublasGetMatrix(
                rows=rows, cols=cols, elem_size=dts,
                A=D_narrow.data_ptr(), lda=D_narrow.stride(1),
                B=H_narrow.data_ptr(), ldb=H_narrow.stride(1))
    elif is_contig(H):
        if s is not None:
            cuda_memcpy2d_async(
                src=D_narrow.data_ptr(), spitch=D_narrow.stride(0) * dts,
                dst=H_narrow.data_ptr(), dpitch=H_narrow.stride(0) * dts,
                width=cols * dts, height=rows, stream=s._as_parameter)
        else:
            cuda_memcpy2d(
                src=D_narrow.data_ptr(), spitch=D_narrow.stride(0) * dts,
                dst=H_narrow.data_ptr(), dpitch=H_narrow.stride(0) * dts,
                width=cols * dts, height=rows)

    return H_narrow


def copy_to_host_noorder(rows: int, cols: int,
                         D: torch.Tensor, Di: int, Dj: int,
                         H: torch.Tensor, Hi: int, Hj: int,
                         cpu_buf: Optional[torch.Tensor] = None,
                         s: Optional[torch.cuda.Stream] = None):
    if is_f_contig(D, strict=True):
        if cpu_buf is not None:
            if cpu_buf.shape[0] < rows or cpu_buf.shape[1] < cols:
                raise RuntimeError("Intermediate CPU Buffer is not large enough to hold data: "
                                   "expected (%d, %d); was (%d, %d)" %
                                   (rows, cols, cpu_buf.shape[0], cpu_buf.shape[1]))
            if cpu_buf.dtype != D.dtype:
                raise TypeError("Intermediate CPU buffer data type is not equal to the GPU data "
                                "type: expected %s; was %s" % (D.dtype, cpu_buf.dtype))
            restr_cpu_buf = copy_to_host(rows, cols, D, Di, Dj, cpu_buf, 0, 0, s=s, check=False)
            if s is not None:
                s.synchronize()
            H[Hi:Hi + rows, Hj:Hj + cols].copy_(restr_cpu_buf)
        else:
            copy_to_host(rows, cols, D, Di, Dj, H, Hi, Hj, s=s)
    elif is_contig(D):
        if cpu_buf is not None:
            restr_cpu_buf = copy_to_host(cols, rows, D.T, Dj, Di, cpu_buf.T, 0, 0, s=s, check=False)
            if s is not None:
                s.synchronize()
            H[Hi:Hi + rows, Hj:Hj + cols].copy_(restr_cpu_buf.T)
        else:
            copy_to_host(cols, rows, D.T, Dj, Di, H.T, Hj, Hi, s=s)
    else:
        raise RuntimeError("Cannot copy data which is not memory contiguous.")
