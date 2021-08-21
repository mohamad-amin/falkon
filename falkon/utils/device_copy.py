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


def copy(origin, dest, s=None):
    check_copy(origin, dest)

    if origin.device.type == dest.device.type:
        dest.copy_(origin)
    elif origin.device.type == "cpu":  # host -> dev
        copy_to_device(origin.shape[0], origin.shape[1], origin, 0, 0, dest, 0, 0, s)
    else:                              # dev -> host
        copy_to_host(origin.shape[0], origin.shape[1], origin, 0, 0, dest, 0, 0, s)
    return dest


def copy_to_host(rows, cols, D, Di, Dj, H, Hi, Hj, s=None):
    H_narrow = H.narrow(0, Hi, rows).narrow(1, Hj, cols)
    D_narrow = D.narrow(0, Di, rows).narrow(1, Dj, cols)

    dts = sizeof_dtype(D.dtype)

    if is_f_contig(D, strict=False):
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
    elif is_contig(D):
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
