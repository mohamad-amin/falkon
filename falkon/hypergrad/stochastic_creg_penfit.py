from contextlib import ExitStack
from typing import Tuple

import numpy as np
import torch

import falkon
import falkon.preconditioner
from falkon import FalkonOptions
from falkon.hypergrad.common import full_rbf_kernel, calc_grads_tensors, init_random_vecs
from falkon.kernels import GaussianKernel
from falkon.kernels.diff_rbf_kernel import DiffGaussianKernel
from falkon.la_helpers import trsm
from falkon.optim import FalkonConjugateGradient
from falkon.preconditioner import FalkonPreconditioner
from falkon.utils.helpers import sizeof_dtype, select_dim_over_n
from falkon.utils.tictoc import Timer


EPS = 5e-5


def calc_trace_fwd(init_val, k_mn, k_mn_zy, kmm_chol, use_stoch_trace, t):
    """ Nystrom kernel trace forward """
    if use_stoch_trace:
        solve1 = torch.triangular_solve(k_mn_zy[:, :t], kmm_chol, upper=False,
                                        transpose=False).solution  # m * t
        solve2 = torch.triangular_solve(solve1, kmm_chol, upper=False,
                                        transpose=True).solution.contiguous()  # m * t
        init_val -= solve1.square_().sum(0).mean()
    else:
        solve1 = trsm(k_mn, kmm_chol, 1.0, lower=True, transpose=False)  # (M*N)
        solve2 = trsm(solve1, kmm_chol, 1.0, lower=True, transpose=True)  # (M*N)
        init_val -= solve1.square_().sum()
    return init_val, solve2


def calc_trace_bwd(k_mn, k_mn_zy, solve2, kmm, use_stoch_trace, t):
    if use_stoch_trace:
        if k_mn_zy is None or t is None or t <= 0:
            raise ValueError("Using stochastic trace but k_mn_zy is None.")
    else:
        if k_mn is None:
            raise ValueError("Not using stochastic trace but k_mn is None.")
    if use_stoch_trace:
        return -(
                2 * (k_mn_zy[:, :t].mul(solve2)).sum(0).mean() -
                (solve2 * (kmm @ solve2)).sum(0).mean()
        )
    else:
        return -(
                2 * (k_mn.mul(solve2)).sum() -
                (solve2 * (kmm @ solve2)).sum()
        )


def calc_deff_bwd(zy_knm_solve_zy, zy_solve_knm_knm_solve_zy, zy_solve_kmm_solve_zy, pen_n, t,
                  include_kmm_term):
    out_deff_bwd = (
            2 * zy_knm_solve_zy[:t].mean() -
            zy_solve_knm_knm_solve_zy[:t].mean()
    )
    if include_kmm_term:
        out_deff_bwd -= pen_n * zy_solve_kmm_solve_zy[:t].mean()
    return out_deff_bwd


def calc_dfit_bwd(zy_knm_solve_zy, zy_solve_knm_knm_solve_zy, zy_solve_kmm_solve_zy, pen_n, t,
                  include_kmm_term):
    dfit_bwd = -(
            2 * zy_knm_solve_zy[t:].mean() -
            zy_solve_knm_knm_solve_zy[t:].mean()
    )
    if include_kmm_term:
        dfit_bwd += pen_n * zy_solve_kmm_solve_zy[t:].mean()
    return dfit_bwd


def calc_dfit_nopen_bwd(zy_knm_solve_zy, zy_solve_knm_knm_solve_zy, zy_solve_kmm_solve_zy,
                        k_mn_zy, solve_ytilde, knm_solve_zy, k_mn, y_solve_kmm_solve_ytilde,
                        pen_n, t, include_kmm_term):
    dfit_nopen_bwd = (
        -4 * zy_knm_solve_zy[t:].mean() +                                       # -4 * Y.T @ g(k_nm) @ alpha
        2 * zy_solve_knm_knm_solve_zy[t:].mean() +                              # 2 * alpha.T @ g(H) @ alpha -- part 1
        2 * (k_mn_zy[:, t:] * solve_ytilde).sum(0).mean() +                     # 2 * Y.T @ g(k_nm) @ alpha_tilde
        2 * (knm_solve_zy[:, t:] * knm_solve_zy[:, t:].detach()).sum(0).mean()  # 2 * alpha.T @ g(k_nm.T) @ y_tilde
        - 2 * (knm_solve_zy[:, t:] * k_mn.T @ solve_ytilde).sum(0).mean()       # -2 alpha @ g(H) @ alpha -- part 1
    )
    if include_kmm_term:
        dfit_nopen_bwd += 2 * pen_n * zy_solve_kmm_solve_zy[t:].mean()          # 2 * alpha.T @ g(H) @ alpha -- part 2
        dfit_nopen_bwd -= 2 * pen_n * y_solve_kmm_solve_ytilde.mean()           # -2 alpha @ g(H) @ alpha -- part 2
    return dfit_nopen_bwd


# noinspection PyMethodOverriding
class RegLossAndDeffv2(torch.autograd.Function):
    coef_nm = 40
    _last_solve_z = None
    _last_solve_y = None
    _last_solve_zy = None
    _last_t = None
    last_alpha = None
    iter_prep_times, fwd_times, bwd_times, solve_times, kmm_times, grad_times = [], [], [], [], [], []
    iter_times, num_flk_iters = [], []
    solve_together = False
    use_direct_for_stoch = True
    print(f"Initialized class RegLossAndDeffv2. solve_together={solve_together}, "
          f"use_direct_for_stoch={use_direct_for_stoch}")

    @staticmethod
    def print_times():
        num_times = len(RegLossAndDeffv2.iter_times)
        print(
            f"Timings: Preparation {np.sum(RegLossAndDeffv2.iter_prep_times) / num_times:.2f} "
            f"Falkon solve {np.sum(RegLossAndDeffv2.solve_times) / num_times:.2f} "
            f"(in {np.sum(RegLossAndDeffv2.num_flk_iters) / num_times:.1f} iters) "
            f"KMM (toCUDA) {np.sum(RegLossAndDeffv2.kmm_times) / num_times:.2f} "
            f"Forward {np.sum(RegLossAndDeffv2.fwd_times) / num_times:.2f} "
            f"Backward {np.sum(RegLossAndDeffv2.bwd_times) / num_times:.2f} "
            f"Grad {np.sum(RegLossAndDeffv2.grad_times) / num_times:.2f} "
            f"\n\tTotal {np.sum(RegLossAndDeffv2.iter_times) / num_times:.2f}"
        )
        (RegLossAndDeffv2.iter_prep_times, RegLossAndDeffv2.fwd_times, RegLossAndDeffv2.bwd_times,
         RegLossAndDeffv2.solve_times, RegLossAndDeffv2.kmm_times, RegLossAndDeffv2.grad_times,
         RegLossAndDeffv2.iter_times, RegLossAndDeffv2.num_flk_iters) = [], [], [], [], [], [], [], []

    @staticmethod
    def direct_nosplit(X, M, Y, penalty, kmm, kmm_chol, zy, solve_zy, zy_solve_kmm_solve_zy, kernel,
                       t):
        print("Start direct_nosplit")
        print("Alloced mem: %.5fMB" % (torch.cuda.memory_allocated() / 2**20))
        with Timer(RegLossAndDeffv2.iter_prep_times), torch.autograd.enable_grad():
            print("mmv to get k_mn_zy")
            k_mn_zy = kernel.mmv(M, X, zy) # M x (T+1)
            zy_knm_solve_zy = k_mn_zy.mul(solve_zy).sum(0)  # T+1
        print("end iter_prep")
        torch.cuda.empty_cache()
        print("Alloced mem: %.5fMB" % (torch.cuda.memory_allocated() / 2**20))

        # Forward
        dfit_fwd = Y.square().sum()
        deff_fwd = torch.tensor(0, dtype=X.dtype)
        trace_fwd = torch.tensor(X.shape[0], dtype=X.dtype)
        with Timer(RegLossAndDeffv2.fwd_times), torch.autograd.no_grad():
            trace_fwd, solve2 = calc_trace_fwd(
                trace_fwd, k_mn=None, k_mn_zy=k_mn_zy, kmm_chol=kmm_chol,
                use_stoch_trace=True, t=t)
            # Nystrom effective dimension forward
            deff_fwd += zy_knm_solve_zy[:t].mean()
            # Data-fit forward
            dfit_fwd -= zy_knm_solve_zy[t:].mean()
        print("end forward")
        print("Alloced mem: %.5fMB" % (torch.cuda.memory_allocated() / 2**20))

        # Backward
        with Timer(RegLossAndDeffv2.bwd_times), torch.autograd.enable_grad():
            # This OOM
            zy_solve_knm_knm_solve_zy = kernel.mmv(X, M, solve_zy).square().sum(0)  # T+1
            pen_n = penalty * X.shape[0]
            # Nystrom kernel trace backward
            trace_bwd = calc_trace_bwd(
                k_mn=None, k_mn_zy=k_mn_zy, solve2=solve2, kmm=kmm, use_stoch_trace=True, t=t)
            # Nystrom effective dimension backward
            deff_bwd = calc_deff_bwd(
                zy_knm_solve_zy, zy_solve_knm_knm_solve_zy, zy_solve_kmm_solve_zy, pen_n, t,
                include_kmm_term=True)
            # Data-fit backward
            dfit_bwd = calc_dfit_bwd(
                zy_knm_solve_zy, zy_solve_knm_knm_solve_zy, zy_solve_kmm_solve_zy, pen_n, t,
                include_kmm_term=True)
            bwd = (deff_bwd + dfit_bwd + trace_bwd)
        return (deff_fwd, dfit_fwd, trace_fwd), bwd

    @staticmethod
    def choose_device_mem(data_dev: torch.device, dtype: torch.dtype,
                          solve_options: FalkonOptions) -> Tuple[torch.device, float]:
        if data_dev.type == 'cuda':  # CUDA in-core
            from falkon.mmv_ops.utils import _get_gpu_info
            gpu_info = _get_gpu_info(solve_options, slack=0.9)
            single_gpu_info = [g for g in gpu_info if g.Id == data_dev.index][0]
            avail_mem = single_gpu_info.usable_memory / sizeof_dtype(dtype)
            device = torch.device("cuda:%d" % (single_gpu_info.Id))
        elif not solve_options.use_cpu and torch.cuda.is_available():  # CUDA out-of-core
            from falkon.mmv_ops.utils import _get_gpu_info
            gpu_info = _get_gpu_info(solve_options, slack=0.9)[0]  # TODO: Splitting across gpus
            avail_mem = gpu_info.usable_memory / sizeof_dtype(dtype)
            device = torch.device("cuda:%d" % (gpu_info.Id))
        else:  # CPU in-core
            avail_mem = solve_options.max_cpu_mem / sizeof_dtype(dtype)
            device = torch.device("cpu")

        return device, avail_mem

    @staticmethod
    def solve_flk(X, M, Y, Z, ZY, penalty, kernel_args, solve_options, solve_maxiter, warm_start):
        t = Z.shape[1]
        solve_together = RegLossAndDeffv2.solve_together
        solve_opt_precise = solve_options
        solve_maxiter_precise = solve_maxiter

        kernel_args_ = kernel_args.detach()
        penalty_ = penalty.item()
        M_ = M.detach()

        K = GaussianKernel(kernel_args_, opt=solve_opt_precise)
        precond = FalkonPreconditioner(penalty_, K, solve_opt_precise)
        precond.init(M_)

        if solve_together:
            optim = FalkonConjugateGradient(K, precond, solve_opt_precise)
            solve_zy_prec = optim.solve(
                X, M_, ZY, penalty_,
                initial_solution=RegLossAndDeffv2._last_solve_zy,
                max_iter=solve_maxiter,
            )
            solve_zy = precond.apply(solve_zy_prec)
            if warm_start:
                RegLossAndDeffv2._last_solve_zy = solve_zy_prec.detach().clone()
            RegLossAndDeffv2.last_alpha = solve_zy[:, t:].detach().clone()
            num_iters = optim.optimizer.num_iter
        else:
            optim_y = FalkonConjugateGradient(K, precond, solve_opt_precise)
            solve_y_prec = optim_y.solve(X, M_, Y, penalty_,
                                         initial_solution=RegLossAndDeffv2._last_solve_y,
                                         max_iter=solve_maxiter_precise)
            optim_z = FalkonConjugateGradient(K, precond, solve_opt_precise)
            solve_z_prec = optim_z.solve(X, M_, Z, penalty_,
                                         initial_solution=RegLossAndDeffv2._last_solve_z,
                                         max_iter=solve_maxiter_precise)
            solve_z = precond.apply(solve_z_prec)
            solve_y = precond.apply(solve_y_prec)
            solve_zy = torch.cat((solve_z, solve_y), dim=1)
            if warm_start:
                RegLossAndDeffv2._last_solve_y = solve_y_prec.detach().clone()
                RegLossAndDeffv2._last_solve_z = solve_z_prec.detach().clone()
            RegLossAndDeffv2.last_alpha = solve_y.detach().clone()
            num_iters = optim_z.optimizer.num_iter
        return solve_zy, num_iters

    @staticmethod
    def direct_wsplit(X, M, Y, penalty, kernel_args, kmm, kmm_chol, zy, solve_zy,
                      zy_solve_kmm_solve_zy, t, coef_nm, device, avail_mem, use_stoch_trace,
                      needs_input_grad):
        """ Splitting along the first dimension of X """
        # Decide block size (this is super random for now: if OOM increase `coef_nm`).
        blk_n = select_dim_over_n(max_n=X.shape[0], m=M.shape[0], d=X.shape[1], max_mem=avail_mem,
                                  coef_nm=coef_nm, coef_nd=1, coef_md=1, coef_n=0,
                                  coef_m=0, coef_d=0, rest=0)
        # Initialize forward pass elements.
        _dfit_fwd = Y.square().sum().to(device)
        _deff_fwd = torch.tensor(0, dtype=X.dtype, device=device)
        _trace_fwd = torch.tensor(X.shape[0], dtype=X.dtype, device=device)
        grads = None
        it = 0
        with ExitStack() as stack:
            if device.type == 'cuda':
                s1 = torch.cuda.current_stream(device)
                stack.enter_context(torch.cuda.device(device))
                stack.enter_context(torch.cuda.stream(s1))
            for i in range(0, X.shape[0], blk_n):
                it += 1
                leni = min(blk_n, X.shape[0] - i)
                c_X = X[i: i + leni, :].to(device=device, non_blocking=True)
                c_zy = zy[i: i + leni, :].to(device=device, non_blocking=True)
                with Timer(RegLossAndDeffv2.iter_prep_times), torch.autograd.enable_grad():
                    k_mn = full_rbf_kernel(c_X, M,
                                           kernel_args).T  # Done to get F-contig k_mn (faster trsm)
                    k_mn_zy = k_mn @ c_zy  # MxN * Nx(T+1) = Mx(T+1)
                    zy_knm_solve_zy = (k_mn_zy * solve_zy).sum(0)  # (T+1)

                # Forward
                with Timer(RegLossAndDeffv2.fwd_times), torch.autograd.no_grad():
                    pen_n = penalty * X.shape[0]
                    # Nystrom kernel trace forward
                    _trace_fwd, solve2 = calc_trace_fwd(
                        _trace_fwd, k_mn=k_mn, k_mn_zy=k_mn_zy, kmm_chol=kmm_chol,
                        use_stoch_trace=use_stoch_trace, t=t)
                    trace_fwd = _trace_fwd# / penalty
                    # Nystrom effective dimension forward
                    _deff_fwd += zy_knm_solve_zy[:t].mean()
                    deff_fwd = _deff_fwd# / pen_n
                    # Data-fit forward
                    _dfit_fwd -= zy_knm_solve_zy[t:].mean()
                    dfit_fwd = _dfit_fwd# / pen_n

                # Backward
                with Timer(RegLossAndDeffv2.bwd_times), torch.autograd.enable_grad():
                    zy_solve_knm_knm_solve_zy = (k_mn.T @ solve_zy).square().sum(0)  # (T+1)
                    pen_n = penalty * X.shape[0]
                    # Nystrom kernel trace backward
                    trace_bwd = calc_trace_bwd(
                        k_mn=k_mn, k_mn_zy=k_mn_zy, solve2=solve2, kmm=kmm,
                        use_stoch_trace=use_stoch_trace, t=t)
                    #trace_bwd = (-penalty * _trace_fwd.detach() + penalty.detach() * trace_bwd) / (penalty**2)
                    # Nystrom effective dimension backward
                    deff_bwd = calc_deff_bwd(
                        zy_knm_solve_zy, zy_solve_knm_knm_solve_zy, zy_solve_kmm_solve_zy, pen_n, t,
                        include_kmm_term=i == 0)
                    #deff_bwd = (-pen_n * _deff_fwd.detach() + pen_n.detach() * deff_bwd) / (pen_n.detach()**2)
                    # Data-fit backward
                    dfit_bwd = calc_dfit_bwd(
                        zy_knm_solve_zy, zy_solve_knm_knm_solve_zy, zy_solve_kmm_solve_zy, pen_n, t,
                        include_kmm_term=i == 0)
                    #dfit_bwd = (-pen_n * _dfit_fwd.detach() + pen_n.detach() * dfit_bwd) / (pen_n.detach()**2)
                    bwd = (deff_bwd + dfit_bwd + trace_bwd)

                # Calc grads
                with Timer(RegLossAndDeffv2.grad_times):
                    new_grads = calc_grads_tensors(inputs=(kernel_args, penalty, M),
                                                   inputs_need_grad=needs_input_grad, backward=bwd,
                                                   retain_graph=True, allow_unused=True)
                    if grads is None:
                        grads = []
                        for g in new_grads:
                            if g is not None:
                                grads.append(g.to(device=X.device))
                            else:
                                grads.append(None)
                    else:
                        for gi in range(len(grads)):
                            if (grads[gi] is None) != (new_grads[gi] is None):
                                continue  # This can happen since bwd at iter-0 is different from following iters.
                            if grads[gi] is not None:
                                grads[gi] += new_grads[gi].to(X.device)
        return (deff_fwd, dfit_fwd, trace_fwd), grads

    @staticmethod
    def forward(
            ctx,
            kernel_args: torch.Tensor,
            penalty: torch.Tensor,
            M: torch.Tensor,
            X: torch.Tensor,
            Y: torch.Tensor,
            t: int,
            deterministic: bool,
            solve_options: FalkonOptions,
            solve_maxiter: int,
            gaussian_random: bool,
            use_stoch_trace: bool,
            warm_start: bool
    ):
        use_direct_for_stoch = RegLossAndDeffv2.use_direct_for_stoch
        if RegLossAndDeffv2._last_t is not None and RegLossAndDeffv2._last_t != t:
            RegLossAndDeffv2._last_solve_y = None
            RegLossAndDeffv2._last_solve_z = None
            RegLossAndDeffv2.last_alpha = None
        RegLossAndDeffv2._last_t = t
        if deterministic:
            torch.manual_seed(12)

        if use_stoch_trace and use_direct_for_stoch:
            device, avail_mem = X.device, None
        else:
            device, avail_mem = RegLossAndDeffv2.choose_device_mem(X.device, X.dtype, solve_options)

        with Timer(RegLossAndDeffv2.iter_times):
            # Initialize hutch trace estimation vectors (t of them)
            Z = init_random_vecs(X.shape[0], t, dtype=X.dtype, device=X.device,
                                 gaussian_random=gaussian_random)
            ZY = torch.cat((Z, Y), dim=1)
            M_dev = M.to(device, copy=False).requires_grad_(M.requires_grad)
            kernel_args_dev = kernel_args.to(device, copy=False).requires_grad_(kernel_args.requires_grad)
            penalty_dev = penalty.to(device, copy=False).requires_grad_(penalty.requires_grad)

            with Timer(RegLossAndDeffv2.solve_times):
                solve_zy, num_flk_iters = RegLossAndDeffv2.solve_flk(
                    X, M_dev, Y, Z, ZY, penalty_dev, kernel_args_dev, solve_options, solve_maxiter, warm_start)
                RegLossAndDeffv2.num_flk_iters.append(num_flk_iters)
            print("Falkon solve finished.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("Alloced mem: %.5fMB" % (torch.cuda.memory_allocated() / 2**20))

            with Timer(RegLossAndDeffv2.kmm_times):  # Move small matrices to the computation device
                solve_zy_dev = solve_zy.to(device, copy=False)

                with torch.autograd.enable_grad():
                    kmm = full_rbf_kernel(M_dev, M_dev, kernel_args_dev)
                    zy_solve_kmm_solve_zy = (kmm @ solve_zy_dev * solve_zy_dev).sum(0)  # (T+1)
                    # The following should be identical but seems to introduce errors in the bwd pass.
                    # zy_solve_kmm_solve_zy = (kmm_chol.T @ solve_zy_dev).square().sum(0)  # (T+1)
                with torch.autograd.no_grad():
                    mm_eye = torch.eye(M_dev.shape[0], device=device, dtype=M_dev.dtype) * EPS
                    kmm_chol, info = torch.linalg.cholesky_ex(kmm + mm_eye, check_errors=False)
            print("Put KMM on GPU")
            print("Alloced mem: %.5fMB" % (torch.cuda.memory_allocated() / 2**20))
            print("kmm", kmm.device)

            if use_stoch_trace and use_direct_for_stoch:
                kernel = GaussianKernel(kernel_args_dev, solve_options)
                fwd, bwd = RegLossAndDeffv2.direct_nosplit(X, M_dev, Y, penalty, kmm, kmm_chol, ZY,
                                                           solve_zy, zy_solve_kmm_solve_zy, kernel,
                                                           t)
                with Timer(RegLossAndDeffv2.grad_times):
                    grads = calc_grads_tensors(inputs=(kernel_args_dev, penalty_dev, M_dev),
                                               inputs_need_grad=ctx.needs_input_grad, backward=bwd,
                                               retain_graph=False, allow_unused=True)
            else:
                fwd, grads = RegLossAndDeffv2.direct_wsplit(X, M_dev, Y, penalty_dev, kernel_args_dev, kmm,
                                                            kmm_chol, ZY, solve_zy_dev,
                                                            zy_solve_kmm_solve_zy, t, RegLossAndDeffv2.coef_nm,
                                                            device, avail_mem, use_stoch_trace,
                                                            ctx.needs_input_grad)

        deff_fwd, dfit_fwd, trace_fwd = fwd
        ctx.grads = grads
        print(f"Stochastic: D-eff {deff_fwd:.3e} Data-Fit {dfit_fwd:.3e} Trace {trace_fwd:.3e}")
        return (deff_fwd + dfit_fwd + trace_fwd).to(X.device)

    @staticmethod
    def backward(ctx, out):
        grads_out = []
        for g in ctx.grads:
            if g is not None:
                g = g * out
            grads_out.append(g)
        return tuple(grads_out)

    @staticmethod
    def grad_check():
        torch.manual_seed(3)
        X = torch.randn(50, 6, dtype=torch.float64)
        w = torch.randn(X.shape[1], 1, dtype=torch.float64)
        Y = X @ w
        M = X[:10].clone().detach().requires_grad_()
        s = torch.tensor([10.0], dtype=X.dtype).requires_grad_()
        p = torch.tensor(1e-2, dtype=X.dtype).requires_grad_()
        RegLossAndDeffv2.use_direct_for_stoch = True
        torch.autograd.gradcheck(
            lambda sigma, pen, centers:
            RegLossAndDeffv2.apply(
                sigma,              # kernel_args
                pen,                # penalty
                centers,            # M
                X,                  # X
                Y,                  # Y
                20,                 # t
                True,               # deterministic
                FalkonOptions(),    # solve_options
                30,                 # solve_maxiter
                False,              # gaussian_random
                True,               # use_stoch_trace
                False),             # warm_start
            (s, p, M))
        RegLossAndDeffv2.use_direct_for_stoch = False
        torch.autograd.gradcheck(
            lambda sigma, pen, centers:
            RegLossAndDeffv2.apply(
                sigma,              # kernel_args
                pen,                # penalty
                centers,            # M
                X,                  # X
                Y,                  # Y
                20,                 # t
                True,               # deterministic
                FalkonOptions(),    # solve_options
                30,                 # solve_maxiter
                False,              # gaussian_random
                True,               # use_stoch_trace
                False),             # warm_start
            (s, p, M))
        torch.autograd.gradcheck(
            lambda sigma, pen, centers:
            RegLossAndDeffv2.apply(
                sigma,
                pen,
                centers,
                X,
                Y,
                20,
                True,
                FalkonOptions(),
                30,
                False,
                False,
                False),
            (s, p, M))


def creg_penfit(kernel_args, penalty, centers, X, Y, num_estimators, deterministic, solve_options,
                solve_maxiter, gaussian_random, use_stoch_trace, warm_start=True):
    return RegLossAndDeffv2.apply(
        kernel_args, penalty, centers, X, Y, num_estimators, deterministic, solve_options,
        solve_maxiter, gaussian_random, use_stoch_trace, warm_start
    )


# noinspection PyMethodOverriding
class StochasticDeffNoPenFitTrFn(torch.autograd.Function):
    coef_nm = 40
    _last_solve_z = None
    _last_solve_y = None
    _last_solve_ytilde = None
    _last_t = None
    _precond = None
    last_alpha = None
    iter_prep_times, fwd_times, bwd_times, solve_times, kmm_times, grad_times = [], [], [], [], [], []
    iter_times, num_flk_iters = [], []
    print(f"Initialized class StochasticDeffNoPenFitTrFn. ")

    @classmethod
    def print_times(cls):
        num_times = len(cls.iter_times)
        print(
            f"Timings: Preparation {np.sum(cls.iter_prep_times) / num_times:.2f} "
            f"Falkon solve {np.sum(cls.solve_times) / num_times:.2f} "
            f"(in {np.sum(cls.num_flk_iters) / num_times:.1f} iters) "
            f"KMM (toCUDA) {np.sum(cls.kmm_times) / num_times:.2f} "
            f"Forward {np.sum(cls.fwd_times) / num_times:.2f} "
            f"Backward {np.sum(cls.bwd_times) / num_times:.2f} "
            f"Grad {np.sum(cls.grad_times) / num_times:.2f} "
            f"\n\tTotal {np.sum(cls.iter_times) / num_times:.2f}"
        )
        (cls.iter_prep_times, cls.fwd_times, cls.bwd_times,
         cls.solve_times, cls.kmm_times, cls.grad_times,
         cls.iter_times, cls.num_flk_iters) = [], [], [], [], [], [], [], []

    @staticmethod
    def choose_device_mem(data_dev: torch.device, dtype: torch.dtype,
                          solve_options: FalkonOptions) -> Tuple[torch.device, float]:
        if data_dev.type == 'cuda':  # CUDA in-core
            from falkon.mmv_ops.utils import _get_gpu_info
            gpu_info = _get_gpu_info(solve_options, slack=0.9)
            single_gpu_info = [g for g in gpu_info if g.Id == data_dev.index][0]
            avail_mem = single_gpu_info.usable_memory / sizeof_dtype(dtype)
            device = torch.device("cuda:%d" % (single_gpu_info.Id))
        elif not solve_options.use_cpu and torch.cuda.is_available():  # CUDA out-of-core
            from falkon.mmv_ops.utils import _get_gpu_info
            gpu_info = _get_gpu_info(solve_options, slack=0.9)[0]  # TODO: Splitting across gpus
            avail_mem = gpu_info.usable_memory / sizeof_dtype(dtype)
            device = torch.device("cuda:%d" % (gpu_info.Id))
        else:  # CPU in-core
            avail_mem = solve_options.max_cpu_mem / sizeof_dtype(dtype)
            device = torch.device("cpu")

        return device, avail_mem

    @staticmethod
    def run_flk_opt(
            kernel: falkon.kernels.Kernel,
            preconditioner: falkon.preconditioner.Preconditioner,
            solve_opt: falkon.FalkonOptions,
            m1, m2, rhs, penalty, init_solve, maxiter):
        optim = FalkonConjugateGradient(kernel, preconditioner, solve_opt)
        solve_prec = optim.solve(
            m1, m2, rhs, penalty, initial_solution=init_solve, max_iter=maxiter)
        solve = preconditioner.apply(solve_prec)
        return solve_prec, solve, optim.optimizer.num_iter

    @classmethod
    def solve_flk_zy(cls, X, M, Y, Z, penalty, kernel_args, solve_options, solve_maxiter, warm_start):
        solve_opt_precise = solve_options
        solve_maxiter_precise = solve_maxiter

        kernel_args_ = kernel_args.detach()
        penalty_ = penalty.item()
        M_ = M.detach()

        K = GaussianKernel(kernel_args_, opt=solve_opt_precise)
        precond = FalkonPreconditioner(penalty_, K, solve_opt_precise)
        precond.init(M_)

        solve_y_prec, solve_y, _ = cls.run_flk_opt(
            K, precond, solve_opt_precise, X, M_, Y, penalty_,
            cls._last_solve_y, solve_maxiter_precise)
        solve_z_prec, solve_z, num_iters = cls.run_flk_opt(
            K, precond, solve_opt_precise, X, M_, Z, penalty_,
            cls._last_solve_z, solve_maxiter_precise)
        solve_zy = torch.cat((solve_z, solve_y), dim=1)
        if warm_start:
            cls._last_solve_y = solve_y_prec.clone()
            cls._last_solve_z = solve_z_prec.clone()
        cls.last_alpha = solve_y.clone()
        cls._precond = precond
        return solve_zy, num_iters

    @classmethod
    def solve_flk_ytilde(cls, X, M, Ytilde, penalty, kernel_args, solve_options, solve_maxiter, warm_start):
        solve_opt_precise = solve_options
        solve_maxiter_precise = solve_maxiter

        kernel_args_ = kernel_args.detach()
        penalty_ = penalty.item()
        M_ = M.detach()

        K = GaussianKernel(kernel_args_, opt=solve_opt_precise)
        if cls._precond is None:
            raise RuntimeError("preconditioner empty")
        precond = cls._precond

        solve_ytilde_prec, solve_ytilde, num_iters = cls.run_flk_opt(
            K, precond, solve_opt_precise, X, M_, Ytilde, penalty_,
            cls._last_solve_ytilde, solve_maxiter_precise)
        if warm_start:
            cls._last_solve_ytilde = solve_ytilde_prec.clone()
        return solve_ytilde, num_iters

    @classmethod
    def direct_wsplit(cls, X, M, Y, penalty, kernel_args, kmm, kmm_chol, zy, solve_zy,
                      kmm_solve_zy, solve_ytilde, t, coef_nm, device,
                      avail_mem, use_stoch_trace, needs_input_grad):
        """ Splitting along the first dimension of X """
        # Decide block size (this is super random for now: if OOM increase `coef_nm`).
        blk_n = select_dim_over_n(max_n=X.shape[0], m=M.shape[0], d=X.shape[1], max_mem=avail_mem,
                                  coef_nm=coef_nm, coef_nd=1, coef_md=1, coef_n=0,
                                  coef_m=0, coef_d=0, rest=0)
        # Initialize forward pass elements.
        dfit_nopen_fwd = torch.tensor(0, dtype=X.dtype, device=device)#Y.square().sum().to(device)
        deff_fwd = torch.tensor(0, dtype=X.dtype, device=device)
        trace_fwd = torch.tensor(X.shape[0], dtype=X.dtype, device=device)
        grads = None
        it = 0
        with ExitStack() as stack:
            if device.type == 'cuda':
                s1 = torch.cuda.current_stream(device)
                stack.enter_context(torch.cuda.device(device))
                stack.enter_context(torch.cuda.stream(s1))
            with torch.autograd.enable_grad():
                pen_n = penalty * X.shape[0]
                zy_solve_kmm_solve_zy = (kmm_solve_zy * solve_zy).sum(0)  # (T+1)
                y_solve_kmm_solve_ytilde = (kmm_solve_zy[:, t:] * solve_ytilde).sum(0)  # 1

            for i in range(0, X.shape[0], blk_n):
                it += 1
                leni = min(blk_n, X.shape[0] - i)
                c_X = X[i: i + leni, :].to(device=device, non_blocking=True)
                c_zy = zy[i: i + leni, :].to(device=device, non_blocking=True)

                with Timer(cls.iter_prep_times), torch.autograd.enable_grad():
                    pen_n = penalty * X.shape[0]
                    k_mn = full_rbf_kernel(c_X, M, kernel_args).T  # Done to get F-contig k_mn (faster trsm)
                    k_mn_zy = k_mn @ c_zy  # MxN * Nx(T+1) = Mx(T+1)
                    zy_knm_solve_zy = (k_mn_zy * solve_zy).sum(0)  # (T+1)
                    knm_solve_zy = k_mn.T @ solve_zy  # cNxT

                # Forward
                with Timer(cls.fwd_times), torch.autograd.no_grad():
                    # Nystrom kernel trace forward
                    trace_fwd, solve2 = calc_trace_fwd(
                        trace_fwd, k_mn=k_mn, k_mn_zy=k_mn_zy, kmm_chol=kmm_chol,
                        use_stoch_trace=use_stoch_trace, t=t)
                    # Nystrom effective dimension forward
                    deff_fwd += zy_knm_solve_zy[:t].mean()
                    # Data-fit forward
                    dfit_nopen_fwd += (knm_solve_zy[:, t:] - c_zy[:, t:]).square_().sum(0).mean()
                # Backward
                with Timer(cls.bwd_times), torch.autograd.enable_grad():
                    zy_solve_knm_knm_solve_zy = (knm_solve_zy).square().sum(0)  # (T+1)
                    # Nystrom kernel trace backward
                    trace_bwd = calc_trace_bwd(
                        k_mn=k_mn, k_mn_zy=k_mn_zy, solve2=solve2, kmm=kmm,
                        use_stoch_trace=use_stoch_trace, t=t)
                    # When you want to divide the forward pass by pen_n, you must modify the backward
                    # pass as follows. Note that `trace_fwd_` is the trace_fwd term before division.
                    # trace_bwd = (-pen_n * trace_fwd_.detach() + pen_n.detach() * trace_bwd) / (pen_n.detach()**2)

                    # Nystrom effective dimension backward
                    deff_bwd = calc_deff_bwd(
                        zy_knm_solve_zy, zy_solve_knm_knm_solve_zy, zy_solve_kmm_solve_zy, pen_n, t,
                        include_kmm_term=i == 0)
                    # Data-fit backward
                    dfit_nopen_bwd = calc_dfit_nopen_bwd(
                        zy_knm_solve_zy, zy_solve_knm_knm_solve_zy, zy_solve_kmm_solve_zy,
                        k_mn_zy, solve_ytilde, knm_solve_zy, k_mn, y_solve_kmm_solve_ytilde,
                        pen_n, t, include_kmm_term=i == 0)
                    bwd = deff_bwd + dfit_nopen_bwd + trace_bwd

                # Calc grads
                with Timer(cls.grad_times):
                    new_grads = calc_grads_tensors(inputs=(kernel_args, penalty, M),
                                                   inputs_need_grad=needs_input_grad, backward=bwd,
                                                   retain_graph=True, allow_unused=True)
                    if grads is None:
                        grads = []
                        for g in new_grads:
                            if g is not None:
                                grads.append(g.to(device=X.device))
                            else:
                                grads.append(None)
                    else:
                        for gi in range(len(grads)):
                            if (grads[gi] is None) != (new_grads[gi] is None):
                                continue  # This can happen since bwd at iter-0 is different from following iters.
                            if grads[gi] is not None:
                                grads[gi] += new_grads[gi].to(X.device)
        return (deff_fwd, dfit_nopen_fwd, trace_fwd), grads

    @staticmethod
    def forward(
            ctx,
            kernel_args: torch.Tensor,
            penalty: torch.Tensor,
            M: torch.Tensor,
            X: torch.Tensor,
            Y: torch.Tensor,
            t: int,
            deterministic: bool,
            solve_options: FalkonOptions,
            solve_maxiter: int,
            gaussian_random: bool,
            use_stoch_trace: bool,
            warm_start: bool
    ):
        if StochasticDeffNoPenFitTrFn._last_t is not None and StochasticDeffNoPenFitTrFn._last_t != t:
            StochasticDeffNoPenFitTrFn._last_solve_y = None
            StochasticDeffNoPenFitTrFn._last_solve_z = None
            StochasticDeffNoPenFitTrFn._last_solve_ytilde = None
            StochasticDeffNoPenFitTrFn.last_alpha = None
        StochasticDeffNoPenFitTrFn._last_t = t
        if deterministic:
            torch.manual_seed(12)

        device, avail_mem = RegLossAndDeffv2.choose_device_mem(X.device, X.dtype, solve_options)

        with Timer(StochasticDeffNoPenFitTrFn.iter_times):
            # Initialize hutch trace estimation vectors (t of them)
            Z = init_random_vecs(X.shape[0], t, dtype=X.dtype, device=X.device,
                                 gaussian_random=gaussian_random)
            ZY = torch.cat((Z, Y), dim=1)
            M_dev = M.to(device, copy=False).requires_grad_(M.requires_grad)
            kernel_args_dev = kernel_args.to(device, copy=False).requires_grad_(kernel_args.requires_grad)
            penalty_dev = penalty.to(device, copy=False).requires_grad_(penalty.requires_grad)

            with Timer(StochasticDeffNoPenFitTrFn.solve_times):
                solve_zy, num_flk_iters = StochasticDeffNoPenFitTrFn.solve_flk_zy(
                    X, M_dev, Y, Z, penalty_dev, kernel_args_dev, solve_options, solve_maxiter, warm_start)
                StochasticDeffNoPenFitTrFn.num_flk_iters.append(num_flk_iters)

            with Timer(StochasticDeffNoPenFitTrFn.kmm_times):  # Move small matrices to the computation device
                solve_zy_dev = solve_zy.to(device, copy=False)

                with torch.autograd.no_grad():
                    kernel = GaussianKernel(kernel_args_dev.detach(), opt=solve_options)
                    knm_solvey = kernel.mmv(X, M_dev.detach(), solve_zy_dev[:, t:].contiguous())  # also called y_tilde
                with torch.autograd.enable_grad():
                    kmm = full_rbf_kernel(M_dev, M_dev, kernel_args_dev)
                    kmm_solve_zy = kmm @ solve_zy_dev  # Mx(T+1)
                with torch.autograd.no_grad():
                    mm_eye = torch.eye(M_dev.shape[0], device=device, dtype=M_dev.dtype) * EPS
                    kmm_chol, info = torch.linalg.cholesky_ex(kmm + mm_eye, check_errors=False)

            with Timer(StochasticDeffNoPenFitTrFn.solve_times):
                # Solve Falkon part 2 (alpha_tilde = H^{-1} @ k_nm.T @ y_tilde
                solve_ytilde, _ = StochasticDeffNoPenFitTrFn.solve_flk_ytilde(
                    X, M, Ytilde=knm_solvey, penalty=penalty_dev, kernel_args=kernel_args_dev,
                    solve_options=solve_options, solve_maxiter=solve_maxiter, warm_start=warm_start)

            fwd, grads = StochasticDeffNoPenFitTrFn.direct_wsplit(
                X, M_dev, Y, penalty_dev, kernel_args_dev, kmm, kmm_chol, ZY, solve_zy_dev,
                kmm_solve_zy, solve_ytilde, t, StochasticDeffNoPenFitTrFn.coef_nm, device, avail_mem, use_stoch_trace,
                ctx.needs_input_grad)

        deff_fwd, dfit_fwd, trace_fwd = fwd
        ctx.grads = grads
        print(f"Stochastic: D-eff {deff_fwd:.3e} No-Penalty-Data-Fit {dfit_fwd:.3e} Trace {trace_fwd:.3e}")
        return (deff_fwd + dfit_fwd + trace_fwd).to(X.device)

    @staticmethod
    def backward(ctx, out):
        grads_out = []
        for g in ctx.grads:
            if g is not None:
                g = g * out
            grads_out.append(g)
        return tuple(grads_out)

    @staticmethod
    def grad_check():
        torch.manual_seed(3)
        X = torch.randn(50, 6, dtype=torch.float64)
        w = torch.randn(X.shape[1], 1, dtype=torch.float64)
        Y = X @ w
        M = X[:10].clone().detach().requires_grad_()
        s = torch.tensor([10.0], dtype=X.dtype).requires_grad_()
        p = torch.tensor(1e-2, dtype=X.dtype).requires_grad_()

        torch.autograd.gradcheck(
            lambda sigma, pen, centers:
            StochasticDeffNoPenFitTrFn.apply(sigma, pen, centers, X, Y, 20, True, FalkonOptions(), 30, False,
                                   True, False),
            (s, p, M))
        torch.autograd.gradcheck(
            lambda sigma, pen, centers:
            StochasticDeffNoPenFitTrFn.apply(sigma, pen, centers, X, Y, 20, True, FalkonOptions(), 30, False,
                                   False, False),
            (s, p, M))


def creg_plainfit(kernel_args, penalty, centers, X, Y, num_estimators, deterministic, solve_options,
                  solve_maxiter, gaussian_random, use_stoch_trace, warm_start: bool = True):
    return StochasticDeffNoPenFitTrFn.apply(
        kernel_args, penalty, centers, X, Y, num_estimators, deterministic, solve_options,
        solve_maxiter, gaussian_random, use_stoch_trace, warm_start
    )
