import dataclasses

import numpy as np
import pytest
import torch
from pytest import mark

from falkon import kernels
from falkon.kernels import GaussianKernel, LinearKernel, PolynomialKernel, MaternKernel
from falkon.options import FalkonOptions
from falkon.tests.conftest import memory_checker, fix_mat, fix_sparse_mat
from falkon.tests.gen_random import gen_random, gen_sparse_matrix
from falkon.tests.naive_kernels import (
    naive_diff_gaussian_kernel, naive_diff_matern_kernel, naive_diff_polynomial_kernel,
    naive_diff_linear_kernel
)
from falkon.utils import decide_cuda

n32 = np.float32
n64 = np.float64
# Global dimensions
n = 1000
m = 850
d = 10
t = 5
# n, m, d, t = 5, 2, 2, 2
max_mem_dense = 0.5 * 2**20
max_mem_sparse = 0.5 * 2**20
cpu_params = [
    pytest.param(True),
    pytest.param(False, marks=[mark.skipif(not decide_cuda(), reason="No GPU found.")])
]


def _run_fmmv_test(fn, exp, tensors, out, rtol, opt):
    # TODO: On some systems (nest but not sperone), checking memory
    # usage for CPU functions fails miserably due to inconsistent
    # memory numbers being reported at random. We simply replace CPU
    # with a high number to avoid checking.
    extra_mem = 10 * 2**30 if opt.use_cpu else 0
    opt = dataclasses.replace(opt, max_cpu_mem=opt.max_cpu_mem + extra_mem)
    with memory_checker(opt) as new_opt:
        actual = fn(*tensors, out=out, opt=new_opt)

    # Check 1. Accuracy
    np.testing.assert_allclose(exp.detach().numpy(), actual.cpu().detach().numpy(), rtol=rtol)
    # Check 2. Output pointers
    if out is not None:
        assert out.data_ptr() == actual.data_ptr(), "Output data tensor was not used"


def _run_diff_fmmv_test(fn, exp, exp_grads, tensors, out, rtol, opt):
    diff_tensors = [tens.requires_grad_() for tens in tensors]

    # TODO: On some systems (nest but not sperone), checking memory
    # usage for CPU functions fails miserably due to inconsistent
    # memory numbers being reported at random. We simply replace CPU
    # with a high number to avoid checking.
    extra_mem = 10 * 2**30 if opt.use_cpu else 0
    opt = dataclasses.replace(opt, max_cpu_mem=opt.max_cpu_mem + extra_mem)
    with memory_checker(opt) as new_opt:
        actual = fn(*diff_tensors, out=out, opt=new_opt)
    actual_grads = torch.autograd.grad(actual.sum(), diff_tensors)

    # Check 1. Accuracy
    np.testing.assert_allclose(exp.detach().numpy(), actual.cpu().detach().numpy(), rtol=rtol)
    # Check 2. Gradients
    for i in range(len(exp_grads)):
        torch.testing.assert_allclose(exp_grads[i], actual_grads[i], msg="Gradient w.r.t. argument %d do not match" % (i))
    # Check 3. Output pointers
    if out is not None:
        assert out.data_ptr() == actual.data_ptr(), "Output data tensor was not used"


@pytest.fixture
def rtol():
    return {
        np.float64: 1e-12,
        torch.float64: 1e-12,
        np.float32: 1e-4,
        torch.float32: 1e-4
    }


@pytest.fixture(scope="module")
def A():
    return torch.from_numpy(gen_random(n, d, 'float64', False, seed=92)).requires_grad_()


@pytest.fixture(scope="module")
def B():
    return torch.from_numpy(gen_random(m, d, 'float64', False, seed=92)).requires_grad_()


@pytest.fixture(scope="module")
def v():
    return torch.from_numpy(gen_random(m, t, 'float64', False, seed=92)).requires_grad_()


@pytest.fixture(scope="module")
def w():
    return torch.from_numpy(gen_random(n, t, 'float64', False, seed=92))


@pytest.fixture(scope="module", params=[
    pytest.param(1, marks=[]),
    pytest.param(2, marks=pytest.mark.full()),
    pytest.param(3, marks=pytest.mark.full()),
    pytest.param(4, marks=pytest.mark.full()),
], ids=["Gaussian", "Linear", "Polynomial", "Matern"])
def kernel(request):
    if request.param == 1:
        return GaussianKernel(sigma=1)
    elif request.param == 2:
        return LinearKernel()
    elif request.param == 3:
        return PolynomialKernel(1.2, 3, 2.5)
    elif request.param == 4:
        return MaternKernel(sigma=1.0, nu=1.5)


@pytest.fixture(scope="module")
def gram(kernel, A, B):
    if isinstance(kernel, kernels.GaussianKernel):
        return naive_diff_gaussian_kernel(A, B, kernel.sigma)
    elif isinstance(kernel, kernels.LinearKernel):
        return naive_diff_linear_kernel(A, B, kernel.beta, kernel.sigma)
    elif isinstance(kernel, kernels.PolynomialKernel):
        return naive_diff_polynomial_kernel(A, B, kernel.alpha, kernel.beta, kernel.degree)
    elif isinstance(kernel, kernels.MaternKernel):
        return naive_diff_matern_kernel(A, B, kernel.sigma, kernel.nu)
    else:
        raise RuntimeError("Kernel %s not recognized" % (kernel))


@pytest.fixture(scope="module")
def expected_fmmv(gram, v):
    return gram @ v


@pytest.fixture(scope="module")
def expected_grad_fmmv(expected_fmmv, A, B, v):
    return torch.autograd.grad(expected_fmmv.sum(), [A, B, v])


@pytest.fixture(scope="module")
def e_dfmmv1(gram, v, w):
    return gram.T @ (gram @ v + w)


@pytest.fixture(scope="module")
def e_dfmmv2(gram, v):
    return gram.T @ (gram @ v)


@pytest.fixture(scope="module")
def e_dfmmv3(gram, w):
    return gram.T @ w


@pytest.fixture(scope="module")
def e_dfmmv(request):
    return request.getfixturevalue(request.param)


class TestDense:
    basic_options = FalkonOptions(debug=True, compute_arch_speed=False, keops_active="no",
                                  max_gpu_mem=max_mem_dense, max_cpu_mem=max_mem_dense)

    @pytest.mark.parametrize("Ao,Adt,Bo,Bdt,vo,vdt", [
        ("F", np.float32, "F", np.float32, "F", np.float32),
        ("C", np.float32, "C", np.float32, "C", np.float32),
        pytest.param("F", np.float64, "F", np.float64, "F", np.float64, marks=pytest.mark.full()),
        pytest.param("C", np.float64, "C", np.float64, "C", np.float64, marks=pytest.mark.full()),
        # A few mixed-contiguity examples
        pytest.param("F", np.float32, "C", np.float32, "F", np.float32, marks=pytest.mark.full()),
        pytest.param("F", np.float32, "C", np.float32, "C", np.float32, marks=pytest.mark.full()),
    ], ids=["AF32-BF32-vF32", "AC32-BC32-vC32", "AF64-BF64-vF64", "AC64-BC64-vC64",
            "AF32-BC32-vF32", "AF32-BC32-vC32"])
    @pytest.mark.parametrize("cpu", cpu_params, ids=["cpu", "gpu"])
    def test_fmmv(self, A, B, v, Ao, Adt, Bo, Bdt, vo, vdt, kernel, expected_fmmv, cpu, rtol):
        A = fix_mat(A, order=Ao, dtype=Adt).detach()
        B = fix_mat(B, order=Bo, dtype=Bdt).detach()
        v = fix_mat(v, order=vo, dtype=vdt).detach()

        opt = dataclasses.replace(self.basic_options, use_cpu=cpu)
        # Test normal
        _run_fmmv_test(kernel.mmv, expected_fmmv, (A, B, v), out=None, rtol=rtol[A.dtype], opt=opt)
        # Test with out
        out = torch.empty(A.shape[0], v.shape[1], dtype=A.dtype)
        _run_fmmv_test(kernel.mmv, expected_fmmv, (A, B, v), out=out, rtol=rtol[A.dtype], opt=opt)

    @pytest.mark.parametrize("Ao,Adt,Bo,Bdt,vo,vdt", [
        pytest.param("F", np.float64, "F", np.float64, "F", np.float64),
        pytest.param("C", np.float64, "C", np.float64, "C", np.float64),
        pytest.param("C", np.float64, "F", np.float64, "C", np.float64),
    ], ids=["AF64-BF64-vF64", "AC64-BC64-vC64", "AC64-BF64-vC64"])
    @pytest.mark.parametrize("cpu", cpu_params, ids=["cpu", "gpu"])
    def test_fmmv_wgrad(self, A, B, v, Ao, Adt, Bo, Bdt, vo, vdt, kernel, expected_fmmv, expected_grad_fmmv, cpu, rtol):
        A = fix_mat(A, order=Ao, dtype=Adt)
        B = fix_mat(B, order=Bo, dtype=Bdt)
        v = fix_mat(v, order=vo, dtype=vdt)

        opt = dataclasses.replace(self.basic_options, use_cpu=cpu)
        # Test normal
        _run_diff_fmmv_test(kernel.mmv, expected_fmmv, expected_grad_fmmv, (A, B, v), out=None, rtol=rtol[A.dtype], opt=opt)
        # Test with out
        out = torch.empty(A.shape[0], v.shape[1], dtype=A.dtype)
        _run_diff_fmmv_test(kernel.mmv, expected_fmmv, expected_grad_fmmv, (A, B, v), out=out, rtol=rtol[A.dtype], opt=opt)

    @pytest.mark.skipif(not decide_cuda(), reason="No GPU found.")
    @pytest.mark.parametrize("Ao,Adt,Bo,Bdt,vo,vdt", [
        ("F", np.float64, "F", np.float64, "F", np.float64),
    ], ids=["AF64-BF64-vF64"])
    def test_fmmv_input_device(
            self, A, B, v, Ao, Adt, Bo, Bdt, vo, vdt, kernel, expected_fmmv, rtol):
        input_device = "cuda:0"
        A = fix_mat(A, order=Ao, dtype=Adt, device=input_device).detach()
        B = fix_mat(B, order=Bo, dtype=Bdt, device=input_device).detach()
        v = fix_mat(v, order=vo, dtype=vdt, device=input_device).detach()

        opt = dataclasses.replace(self.basic_options, use_cpu=False)

        # Test normal
        _run_fmmv_test(kernel.mmv, expected_fmmv, (A, B, v), out=None, rtol=rtol[A.dtype], opt=opt)
        # Test with out
        out = torch.empty(A.shape[0], v.shape[1], dtype=A.dtype, device=input_device)
        _run_fmmv_test(kernel.mmv, expected_fmmv, (A, B, v), out=out, rtol=rtol[A.dtype], opt=opt)

    @pytest.mark.skipif(not decide_cuda(), reason="No GPU found.")
    @pytest.mark.parametrize("Ao,Adt,Bo,Bdt,vo,vdt", [
        ("F", np.float32, "F", np.float32, "F", np.float32),
    ], ids=["AF32-BF32-vF32"])
    def test_fmmv_input_device_wgrad(
            self, A, B, v, Ao, Adt, Bo, Bdt, vo, vdt, kernel, expected_fmmv, expected_grad_fmmv, rtol):
        input_device = "cuda:0"
        A = fix_mat(A, order=Ao, dtype=Adt, device=input_device)
        B = fix_mat(B, order=Bo, dtype=Bdt, device=input_device)
        v = fix_mat(v, order=vo, dtype=vdt, device=input_device)

        opt = dataclasses.replace(self.basic_options, use_cpu=False)

        # Test normal
        _run_diff_fmmv_test(kernel.mmv, expected_fmmv, expected_grad_fmmv, (A, B, v), out=None, rtol=rtol[A.dtype], opt=opt)
        # Test with out
        out = torch.empty(A.shape[0], v.shape[1], dtype=A.dtype, device=input_device)
        _run_diff_fmmv_test(kernel.mmv, expected_fmmv, expected_grad_fmmv, (A, B, v), out=out, rtol=rtol[A.dtype], opt=opt)

    @pytest.mark.parametrize("cpu", cpu_params, ids=["cpu", "gpu"])
    @pytest.mark.parametrize("Ao,Adt,Bo,Bdt,vo,vdt,wo,wdt,e_dfmmv", [
        pytest.param("F", n32, "F", n32, "F", n32, "F", n32, "e_dfmmv1",
                     marks=[mark.usefixtures("e_dfmmv1"), mark.full()]),
        pytest.param("C", n32, "C", n32, "C", n32, "C", n32, "e_dfmmv1",
                     marks=mark.usefixtures("e_dfmmv1")),
        pytest.param("F", n64, "F", n64, "F", n64, "F", n64, "e_dfmmv1",
                     marks=[mark.usefixtures("e_dfmmv1"), mark.full()]),
        pytest.param("C", n64, "C", n64, "C", n64, "C", n64, "e_dfmmv1",
                     marks=[mark.usefixtures("e_dfmmv1"), mark.full()]),
        pytest.param("F", n32, "F", n32, "F", n32, None, None, "e_dfmmv2",
                     marks=[mark.usefixtures("e_dfmmv2"), mark.full()]),
        pytest.param("C", n32, "C", n32, "C", n32, None, None, "e_dfmmv2",
                     marks=mark.usefixtures("e_dfmmv2")),
        pytest.param("F", n64, "F", n64, "F", n64, None, None, "e_dfmmv2",
                     marks=[mark.usefixtures("e_dfmmv2"), mark.full()]),
        pytest.param("C", n64, "C", n64, "C", n64, None, None, "e_dfmmv2",
                     marks=[mark.usefixtures("e_dfmmv2"), mark.full()]),
        # A few mixed-contiguity examples
        pytest.param("F", n32, "C", n32, "C", n32, "F", n32, "e_dfmmv1",
                     marks=mark.usefixtures("e_dfmmv1")),
    ], ids=["F32-F32-vF32-wF32", "C32-C32-vC32-wC32", "F64-F64-vF64-wF64", "C64-C64-vC64-wC64",
            "F32-F32-vF32", "C32-C32-vC32", "F64-F64-vF64", "C64-C64-vC64",
            "F32-C32-vC32-wF32"],
        indirect=["e_dfmmv"])
    def test_dfmmv(self, A, B, v, w, Ao, Adt, Bo, Bdt, vo, vdt, wo, wdt, kernel, e_dfmmv, cpu, rtol):
        A = fix_mat(A, order=Ao, dtype=Adt).detach()
        B = fix_mat(B, order=Bo, dtype=Bdt).detach()
        v = fix_mat(v, order=vo, dtype=vdt).detach()
        w = fix_mat(w, order=wo, dtype=wdt)
        if w is not None:
            w = w.detach()

        opt = dataclasses.replace(self.basic_options, use_cpu=cpu)

        # Test normal
        _run_fmmv_test(kernel.dmmv, e_dfmmv, (A, B, v, w), out=None, rtol=rtol[A.dtype], opt=opt)
        # Test with out
        out = torch.empty(m, t, dtype=A.dtype)
        _run_fmmv_test(kernel.dmmv, e_dfmmv, (A, B, v, w), out=out, rtol=rtol[A.dtype], opt=opt)

    @pytest.mark.parametrize("Ao,Adt,Bo,Bdt,vo,vdt,wo,wdt,e_dfmmv", [
        pytest.param("F", n32, "F", n32, "F", n32, "F", n32, "e_dfmmv1",
                     marks=mark.usefixtures("e_dfmmv1")),
        pytest.param("F", n32, "F", n32, "F", n32, None, None, "e_dfmmv2",
                     marks=[mark.usefixtures("e_dfmmv2"), mark.full()]),
    ], ids=["F32-F32-vF32-wF32", "F32-F32-vF32"], indirect=["e_dfmmv"])
    @pytest.mark.skipif(not decide_cuda(), reason="No GPU found.")
    def test_dfmmv_input_device(
            self, A, B, v, w, Ao, Adt, Bo, Bdt, vo, vdt, wo, wdt, kernel, e_dfmmv, rtol):
        input_device = "cuda:0"
        A = fix_mat(A, order=Ao, dtype=Adt, device=input_device).detach()
        B = fix_mat(B, order=Bo, dtype=Bdt, device=input_device).detach()
        v = fix_mat(v, order=vo, dtype=vdt, device=input_device).detach()
        w = fix_mat(w, order=wo, dtype=wdt, device=input_device)
        if w is not None:
            w = w.detach()

        opt = dataclasses.replace(self.basic_options, use_cpu=False)

        # Test normal
        _run_fmmv_test(kernel.dmmv, e_dfmmv, (A, B, v, w), out=None, rtol=rtol[A.dtype], opt=opt)
        # Test with out
        out = torch.empty(m, t, dtype=A.dtype, device=input_device)
        _run_fmmv_test(kernel.dmmv, e_dfmmv, (A, B, v, w), out=out, rtol=rtol[A.dtype], opt=opt)

    @pytest.mark.skipif(not decide_cuda(), reason="No GPU found.")
    @pytest.mark.full
    def test_incorrect_dev_setting(self, A, B, v, w, kernel, e_dfmmv1, expected_fmmv, rtol):
        # tests when use_cpu = True, but CUDA input tensors
        A = A.cuda().detach()
        B = B.cuda().detach()
        v = v.cuda().detach()
        w = w.cuda().detach()
        opt = dataclasses.replace(self.basic_options, use_cpu=True)

        with pytest.raises(RuntimeError, match='Requested CPU computations with CUDA data. This should not happen.'):
            _run_fmmv_test(kernel.dmmv, e_dfmmv1, (A, B, v, w), out=None, rtol=rtol[A.dtype], opt=opt)

        with pytest.raises(RuntimeError, match='Requested CPU computations with CUDA data. This should not happen.'):
            _run_fmmv_test(kernel.mmv, expected_fmmv, (A, B, v), out=None, rtol=rtol[A.dtype], opt=opt)


class TestKeops:
    basic_options = FalkonOptions(debug=True, compute_arch_speed=False, keops_active="force",
                                  max_cpu_mem=max_mem_dense, max_gpu_mem=max_mem_dense)

    @pytest.mark.parametrize("Ao,Adt,Bo,Bdt,vo,vdt", [
        ("C", np.float32, "C", np.float32, "C", np.float32),
        pytest.param("C", np.float64, "C", np.float64, "C", np.float64,
                     marks=[pytest.mark.full()]),
        pytest.param("F", np.float32, "F", np.float32, "F", np.float32,
                     marks=[pytest.mark.full()]),
        pytest.param("F", np.float32, "C", np.float32, "C", np.float32,
                     marks=[pytest.mark.full()]),
    ], ids=["AC32-BC32-vC32", "AC64-BC64-vC64", "AF32-BF32-vF32", "AF32-BC32-vC32"])
    @pytest.mark.parametrize("cpu", cpu_params, ids=["cpu", "gpu"])
    def test_fmmv(self, A, B, v, Ao, Adt, Bo, Bdt, vo, vdt, kernel,
                  expected_fmmv, cpu, rtol):
        A = fix_mat(A, order=Ao, dtype=Adt).detach()
        B = fix_mat(B, order=Bo, dtype=Bdt).detach()
        v = fix_mat(v, order=vo, dtype=vdt).detach()

        opt = dataclasses.replace(self.basic_options, use_cpu=cpu)

        # Test normal
        _run_fmmv_test(kernel.mmv, expected_fmmv, (A, B, v), out=None, rtol=rtol[A.dtype], opt=opt)
        # Test with out
        out = torch.empty(A.shape[0], v.shape[1], dtype=A.dtype)
        _run_fmmv_test(kernel.mmv, expected_fmmv, (A, B, v), out=out, rtol=rtol[A.dtype], opt=opt)

    @pytest.mark.parametrize("Ao,Adt,Bo,Bdt,vo,vdt", [
        pytest.param("F", np.float64, "F", np.float64, "F", np.float64),
        pytest.param("C", np.float64, "C", np.float64, "C", np.float64),
        pytest.param("C", np.float64, "F", np.float64, "C", np.float64),
    ], ids=["AF64-BF64-vF64", "AC64-BC64-vC64", "AC64-BF64-vC64"])
    @pytest.mark.parametrize("cpu", cpu_params, ids=["cpu", "gpu"])
    def test_fmmv_wgrad(self, A, B, v, Ao, Adt, Bo, Bdt, vo, vdt, kernel, expected_fmmv, expected_grad_fmmv, cpu, rtol):
        A = fix_mat(A, order=Ao, dtype=Adt)
        B = fix_mat(B, order=Bo, dtype=Bdt)
        v = fix_mat(v, order=vo, dtype=vdt)

        opt = dataclasses.replace(self.basic_options, use_cpu=cpu)
        # Test normal
        _run_diff_fmmv_test(kernel.mmv, expected_fmmv, expected_grad_fmmv, (A, B, v), out=None, rtol=rtol[A.dtype], opt=opt)
        # Test with out
        out = torch.empty(A.shape[0], v.shape[1], dtype=A.dtype)
        _run_diff_fmmv_test(kernel.mmv, expected_fmmv, expected_grad_fmmv, (A, B, v), out=out, rtol=rtol[A.dtype], opt=opt)

    @pytest.mark.skipif(not decide_cuda(), reason="No GPU found.")
    def test_gpu_inputs(self, A, B, v, kernel, expected_fmmv, rtol):
        A = fix_mat(A, order="C", dtype=n32).cuda().detach()
        B = fix_mat(B, order="C", dtype=n32, device=A.device).detach()
        v = fix_mat(v, order="C", dtype=n32, device=A.device).detach()
        opt = dataclasses.replace(self.basic_options, use_cpu=False, max_gpu_mem=np.inf)
        # Test normal
        _run_fmmv_test(kernel.mmv, expected_fmmv, (A, B, v), out=None, rtol=rtol[A.dtype], opt=opt)
        # Test with out
        out = torch.empty(A.shape[0], v.shape[1], dtype=A.dtype, device=A.device)
        _run_fmmv_test(kernel.mmv, expected_fmmv, (A, B, v), out=out, rtol=rtol[A.dtype], opt=opt)

    @pytest.mark.skipif(not decide_cuda(), reason="No GPU found.")
    def test_gpu_inputs_wgrad(self, A, B, v, kernel, expected_fmmv, expected_grad_fmmv, rtol):
        A = fix_mat(A, order="C", dtype=n32).cuda()
        B = fix_mat(B, order="C", dtype=n32, device=A.device)
        v = fix_mat(v, order="C", dtype=n32, device=A.device)
        opt = dataclasses.replace(self.basic_options, use_cpu=False, max_gpu_mem=np.inf)
        # Test normal
        _run_diff_fmmv_test(kernel.mmv, expected_fmmv, expected_grad_fmmv, (A, B, v), out=None, rtol=rtol[A.dtype], opt=opt)
        # Test with out
        out = torch.empty(A.shape[0], v.shape[1], dtype=A.dtype, device=A.device)
        _run_diff_fmmv_test(kernel.mmv, expected_fmmv, expected_grad_fmmv, (A, B, v), out=out, rtol=rtol[A.dtype], opt=opt)

    @pytest.mark.skipif(not decide_cuda(), reason="No GPU found.")
    def test_gpu_inputs_fail(self, A, B, v, kernel, expected_fmmv, rtol):
        A = fix_mat(A, order="C", dtype=n32, device="cuda:0").detach()
        B = fix_mat(B, order="C", dtype=n32, device="cuda:0").detach()
        v = fix_mat(v, order="C", dtype=n32, device="cpu").detach()
        opt = dataclasses.replace(self.basic_options, use_cpu=False, max_gpu_mem=np.inf)
        # Test normal
        with pytest.raises(RuntimeError):
            _run_fmmv_test(kernel.mmv, expected_fmmv, (A, B, v), out=None, rtol=rtol[A.dtype], opt=opt)


class TestSparse:
    # FIXME: We cannot control GPU-memory usage due to large buffers
    #        allocated inside spspmm_cuda!
    basic_options = FalkonOptions(debug=True, compute_arch_speed=False,
                                  max_cpu_mem=max_mem_sparse, max_gpu_mem=np.inf)
    # sparse_dim and sparse_density result in sparse matrices with m and n non-zero entries.
    sparse_dim = 10_000
    sparse_density = 1e-4

    @pytest.fixture(scope="class")
    def s_A(self):
        A = gen_sparse_matrix(n, self.sparse_dim, np.float64, density=self.sparse_density, seed=14)
        Ad = torch.from_numpy(A.to_scipy().todense())
        return A, Ad

    @pytest.fixture(scope="class")
    def s_B(self):
        B = gen_sparse_matrix(m, self.sparse_dim, np.float64, density=self.sparse_density, seed=14)
        Bd = torch.from_numpy(B.to_scipy().todense())
        return B, Bd

    @pytest.fixture(scope="class")
    def s_gram(self, kernel, s_A, s_B):
        opt = FalkonOptions(use_cpu=True, compute_arch_speed=False)
        return kernel(s_A[1], s_B[1], opt=opt)  # n x m kernel

    @pytest.fixture(scope="class")
    def s_expected_fmmv(self, s_gram, v):
        return s_gram @ v

    @pytest.fixture(scope="class")
    def s_e_dfmmv1(self, s_gram, v, w):
        return s_gram.T @ (s_gram @ v + w)

    @pytest.fixture(scope="class")
    def s_e_dfmmv2(self, s_gram, v):
        return s_gram.T @ (s_gram @ v)

    @pytest.fixture(scope="class")
    def s_e_dfmmv3(self, s_gram, w):
        return s_gram.T @ w

    @pytest.fixture(scope="class")
    def s_e_dfmmv(self, request):
        return request.getfixturevalue(request.param)

    @pytest.mark.parametrize("cpu", cpu_params, ids=["cpu", "gpu"])
    @pytest.mark.parametrize("Adt,Bdt,vo,vdt", [
        pytest.param(np.float32, np.float32, "F", np.float32, marks=[pytest.mark.full()]),
        pytest.param(np.float32, np.float32, "C", np.float32),
        pytest.param(np.float64, np.float64, "F", np.float64, marks=[pytest.mark.full()]),
        pytest.param(np.float64, np.float64, "C", np.float64, marks=[pytest.mark.full()]),
    ], ids=["A32-B32-vF32", "A32-B32-vC32", "A64-B64-vF64", "A64-B64-vC64"])
    def test_fmmv(self, s_A, s_B, v, Adt, Bdt, vo, vdt, kernel, s_expected_fmmv, cpu, rtol):
        A = fix_sparse_mat(s_A[0], dtype=Adt)
        B = fix_sparse_mat(s_B[0], dtype=Bdt)
        v = fix_mat(v, dtype=vdt, order=vo, copy=True)

        opt = dataclasses.replace(self.basic_options, use_cpu=cpu)

        # Test normal
        _run_fmmv_test(kernel.mmv, s_expected_fmmv, (A, B, v), out=None, rtol=rtol[A.dtype], opt=opt)
        # Test with out
        out = torch.empty(A.shape[0], v.shape[1], dtype=A.dtype)
        _run_fmmv_test(kernel.mmv, s_expected_fmmv, (A, B, v), out=out, rtol=rtol[A.dtype], opt=opt)

    @pytest.mark.skipif(not decide_cuda(), reason="No GPU found.")
    @pytest.mark.parametrize("Adt,Bdt,vo,vdt", [(np.float32, np.float32, "F", np.float32)],
                             ids=["A32-B32-vF32"])
    @pytest.mark.xfail(reason="Squared-norm not implemented for CUDA tensors", run=True)
    def test_fmmv_input_device(self, s_A, s_B, v, Adt, Bdt, vo, vdt, kernel, s_expected_fmmv, rtol):
        input_device = "cuda:0"
        A = fix_sparse_mat(s_A[0], dtype=Adt, device=input_device)
        B = fix_sparse_mat(s_B[0], dtype=Bdt, device=input_device)
        v = fix_mat(v, dtype=vdt, order=vo, copy=True, device=input_device)

        opt = dataclasses.replace(self.basic_options, use_cpu=False)

        # Test normal
        _run_fmmv_test(kernel.mmv, s_expected_fmmv, (A, B, v), out=None, rtol=rtol[A.dtype], opt=opt)
        # Test with out
        out = torch.empty(A.shape[0], v.shape[1], dtype=A.dtype, device=input_device)
        _run_fmmv_test(kernel.mmv, s_expected_fmmv, (A, B, v), out=out, rtol=rtol[A.dtype], opt=opt)

    @pytest.mark.parametrize("cpu", cpu_params, ids=["cpu", "gpu"])
    @pytest.mark.parametrize("Adt,Bdt,vo,vdt,wo,wdt,s_e_dfmmv", [
        pytest.param(n32, n32, "F", n32, "F", n32, "s_e_dfmmv1",
                     marks=[mark.usefixtures("s_e_dfmmv1"), mark.full()]),
        pytest.param(n32, n32, "C", n32, "C", n32, "s_e_dfmmv1",
                     marks=mark.usefixtures("s_e_dfmmv1")),
        pytest.param(n64, n64, "F", n64, "F", n64, "s_e_dfmmv1",
                     marks=[mark.usefixtures("s_e_dfmmv1"), mark.full()]),
        pytest.param(n64, n64, "C", n64, "C", n64, "s_e_dfmmv1",
                     marks=[mark.usefixtures("s_e_dfmmv1"), mark.full()]),
        pytest.param(n32, n32, "F", n32, None, None, "s_e_dfmmv2",
                     marks=[mark.usefixtures("s_e_dfmmv2"), mark.full()]),
        pytest.param(n32, n32, "C", n32, None, None, "s_e_dfmmv2",
                     marks=mark.usefixtures("s_e_dfmmv2")),
        pytest.param(n64, n64, "F", n64, None, None, "s_e_dfmmv2",
                     marks=[mark.usefixtures("s_e_dfmmv2"), mark.full()]),
        pytest.param(n64, n64, "C", n64, None, None, "s_e_dfmmv2",
                     marks=[mark.usefixtures("s_e_dfmmv2"), mark.full()]),
        # A few mixed-contiguity examples
        pytest.param(n32, n32, "C", n32, "F", n32, "s_e_dfmmv1",
                     marks=[mark.usefixtures("s_e_dfmmv1"), mark.full()]),
    ], ids=["32-32-vF32-wF32", "32-32-vC32-wC32", "64-64-vF64-wF64", "64-64-vC64-wC64",
            "32-32-vF32", "32-32-vC32", "64-64-vF64", "64-64-vC64",
            "32-32-vC32-wF32"
            ], indirect=["s_e_dfmmv"])
    def test_dfmmv(self, s_A, s_B, v, w, Adt, Bdt, vo, vdt, wo, wdt, kernel, s_e_dfmmv, cpu, rtol):
        A = fix_sparse_mat(s_A[0], dtype=Adt)
        B = fix_sparse_mat(s_B[0], dtype=Bdt)
        v = fix_mat(v, order=vo, dtype=vdt).detach()
        w = fix_mat(w, order=wo, dtype=wdt)
        if w is not None:
            w = w.detach()

        opt = dataclasses.replace(self.basic_options, use_cpu=cpu)

        # Test normal
        _run_fmmv_test(kernel.dmmv, s_e_dfmmv, (A, B, v, w), out=None, rtol=rtol[A.dtype], opt=opt)
        # Test with out
        out = torch.empty(m, t, dtype=A.dtype)
        _run_fmmv_test(kernel.dmmv, s_e_dfmmv, (A, B, v, w), out=out, rtol=rtol[A.dtype], opt=opt)

    @pytest.mark.skipif(not decide_cuda(), reason="No GPU found.")
    @pytest.mark.xfail(reason="Squared-norm not implemented for CUDA tensors", run=True)
    @pytest.mark.parametrize("Adt,Bdt,vo,vdt,wo,wdt,s_e_dfmmv", [
        pytest.param(n32, n32, "F", n32, "F", n32, "s_e_dfmmv1",
                     marks=mark.usefixtures("s_e_dfmmv1")),
        pytest.param(n32, n32, "F", n32, None, None, "s_e_dfmmv2",
                     marks=[mark.usefixtures("s_e_dfmmv2"), mark.full()]),
    ], ids=["32-32-vF32-wF32", "32-32-wF32"], indirect=["s_e_dfmmv"])
    def test_dfmmv_input_devices(
            self, s_A, s_B, v, w, Adt, Bdt, vo, vdt, wo, wdt, kernel, s_e_dfmmv, rtol):
        input_device = "cuda:0"
        A = fix_sparse_mat(s_A[0], dtype=Adt, device=input_device)
        B = fix_sparse_mat(s_B[0], dtype=Bdt, device=input_device)
        v = fix_mat(v, order=vo, dtype=vdt, device=input_device)
        w = fix_mat(w, order=wo, dtype=wdt, device=input_device)

        opt = dataclasses.replace(self.basic_options, use_cpu=False)

        # Test normal
        _run_fmmv_test(kernel.dmmv, s_e_dfmmv, (A, B, v, w), out=None, rtol=rtol[A.dtype], opt=opt)
        # Test with out
        out = torch.empty(m, t, dtype=A.dtype, device=input_device)
        _run_fmmv_test(kernel.dmmv, s_e_dfmmv, (A, B, v, w), out=out, rtol=rtol[A.dtype], opt=opt)


if __name__ == "__main__":
    pytest.main()
