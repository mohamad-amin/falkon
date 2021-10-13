import torch

from falkon import FalkonOptions
from falkon.hypergrad.common import full_rbf_kernel, get_scalar, cholesky
from falkon.hypergrad.complexity_reg import NystromKRRModelMixinN, HyperOptimModel
from falkon.hypergrad.stochastic_creg_penfit import (
    creg_penfit, creg_plainfit,
    RegLossAndDeffv2, StochasticDeffNoPenFitTrFn
)
from falkon.kernels import GaussianKernel


EPS = 5e-5


def do_chol(mat):
    eye = torch.eye(mat.shape[0], device=mat.device, dtype=mat.dtype)
    epsilons = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]
    last_exception = None
    for eps in epsilons:
        try:
            out = cholesky(mat + eye * eps)
            return out
        except RuntimeError as e:
            print("Matrix has NaNs", (~torch.isfinite(mat)).sum())
            last_exception = e
    raise last_exception


class CompDeffPenFitTr(HyperOptimModel):
    def __init__(
            self,
            sigma_init,
            penalty_init,
            centers_init,
            opt_centers,
            opt_sigma,
            opt_penalty,
            cuda: bool,
            flk_opt: FalkonOptions,
            num_trace_est: int = 20,
            flk_maxiter: int = 10,
            nystrace_ste: bool = False,
    ):
        super(CompDeffPenFitTr, self).__init__()
        self.stoch_model = StochasticDeffPenFitTr(sigma_init.clone(), penalty_init, centers_init.clone(), opt_centers, opt_sigma, opt_penalty, cuda, flk_opt, num_trace_est, flk_maxiter, nystrace_ste)
        self.det_model = DeffPenFitTr(sigma_init.clone(), penalty_init, centers_init.clone(), opt_centers, opt_sigma, opt_penalty, cuda)
        self.use_model = "stoch"

    def parameters(self):
        if self.use_model == "stoch":
            return self.stoch_model.parameters()
        else:
            return self.det_model.parameters()

    def named_parameters(self):
        if self.use_model == "stoch":
            return self.stoch_model.named_parameters()
        else:
            return self.det_model.named_parameters()

    def eval(self):
        self.stoch_model.eval()
        self.det_model.eval()

    def hp_loss(self, X, Y):
        if self.use_model == "stoch":
            self.det_model.centers = self.stoch_model.centers
            self.det_model.penalty = self.stoch_model.penalty
            self.det_model.sigma = self.stoch_model.sigma
        else:
            self.stoch_model.centers = self.det_model.centers
            self.stoch_model.penalty = self.det_model.penalty
            self.stoch_model.sigma = self.det_model.sigma

        ndeff, datafit, trace = self.det_model.hp_loss(X, Y)
        stoch_loss = self.stoch_model.hp_loss(X, Y)
        print(f"Deterministic: D-eff {ndeff:.2e} Data-Fit {datafit:.2e} Trace {trace:.2e}")

        if self.use_model == "stoch":
            return stoch_loss
        else:
            return [ndeff + datafit + trace]

    def predict(self, X):
        return self.det_model.predict(X)

    @property
    def centers(self):
        if self.use_model == "stoch":
            return self.stoch_model.centers
        return self.det_model.centers

    @property
    def sigma(self):
        if self.use_model == "stoch":
            return self.stoch_model.sigma
        return self.det_model.sigma

    @property
    def penalty(self):
        if self.use_model == "stoch":
            return self.stoch_model.penalty
        return self.det_model.penalty

    @property
    def loss_names(self):
        if self.use_model == "stoch":
            return ["stoch-creg-penfit"]
        return ["det-creg-penfit"]

    def __repr__(self):
        return f"CompDeffPenFitTr()"


class StochasticDeffPenFitTr(NystromKRRModelMixinN, HyperOptimModel):
    def __init__(
            self,
            sigma_init,
            penalty_init,
            centers_init,
            opt_centers,
            opt_sigma,
            opt_penalty,
            cuda: bool,
            flk_opt: FalkonOptions,
            num_trace_est: int = 20,
            flk_maxiter: int = 10,
            nystrace_ste: bool = False,
    ):
        super().__init__(
            penalty=penalty_init,
            sigma=sigma_init,
            centers=centers_init,
            cuda=cuda,
            verbose=True,
        )
        self.opt_sigma, self.opt_centers, self.opt_penalty = opt_sigma, opt_centers, opt_penalty
        if opt_sigma:
            self.register_parameter("sigma", self.sigma_.requires_grad_(True))
        if opt_penalty:
            self.register_parameter("penalty", self.penalty_.requires_grad_(True))
        if opt_centers:
            self.register_parameter("centers", self.centers_.requires_grad_(True))

        self.flk_opt = flk_opt
        self.num_trace_est = num_trace_est
        self.flk_maxiter = flk_maxiter
        self.nystrace_ste = nystrace_ste

    def hp_loss(self, X, Y):
        loss = creg_penfit(kernel_args=self.sigma, penalty=self.penalty, centers=self.centers,
                           X=X, Y=Y, num_estimators=self.num_trace_est, deterministic=True,
                           solve_options=self.flk_opt, solve_maxiter=self.flk_maxiter,
                           gaussian_random=True, use_stoch_trace=self.nystrace_ste, warm_start=True)
        return [loss]

    def predict(self, X):
        if RegLossAndDeffv2.last_alpha is None:
            raise RuntimeError("Call hp_loss before calling predict.")

        kernel = GaussianKernel(sigma=self.sigma.detach(), opt=self.flk_opt)
        with torch.autograd.no_grad():
            return kernel.mmv(X, self.centers, RegLossAndDeffv2.last_alpha)

    def print_times(self):
        RegLossAndDeffv2.print_times()

    @property
    def last_beta(self):
        return RegLossAndDeffv2._last_solve_y

    @property
    def loss_names(self):
        return ["stoch-creg-penfit"]

    def __repr__(self):
        return f"StochasticDeffPenFitTr(sigma={get_scalar(self.sigma)}, penalty={get_scalar(self.penalty)}, " \
               f"num_centers={self.centers.shape[0]}, opt_centers={self.opt_centers}, " \
               f"opt_sigma={self.opt_sigma}, opt_penalty={self.opt_penalty}, t={self.num_trace_est}, " \
               f"flk_iter={self.flk_maxiter})"


class DeffPenFitTr(NystromKRRModelMixinN, HyperOptimModel):
    def __init__(
            self,
            sigma_init,
            penalty_init,
            centers_init,
            opt_centers,
            opt_sigma,
            opt_penalty,
            cuda: bool,
            div_trace_by_lambda: bool = False,
            div_deff_by_lambda: bool = False,
            special_one: bool = False,
    ):
        super().__init__(
            penalty=penalty_init,
            sigma=sigma_init,
            centers=centers_init,
            cuda=cuda,
            verbose=True,
        )
        self.opt_sigma, self.opt_centers, self.opt_penalty = opt_sigma, opt_centers, opt_penalty
        if opt_sigma:
            self.register_parameter("sigma", self.sigma_.requires_grad_(True))
        if opt_penalty:
            self.register_parameter("penalty", self.penalty_.requires_grad_(True))
        if opt_centers:
            self.register_parameter("centers", self.centers_.requires_grad_(True))

        self.L, self.LB, self.c = None, None, None
        self.noise_estimate = None
        self.div_trace_by_lambda = div_trace_by_lambda
        self.div_deff_by_lambda = div_deff_by_lambda
        self.special_one = special_one

    def hp_loss(self, X, Y):
        variance = self.penalty * X.shape[0]
        sqrt_var = torch.sqrt(variance)
        Kdiag = X.shape[0]
        m = self.centers.shape[0]
        kmn = full_rbf_kernel(self.centers, X, self.sigma)
        kmm = full_rbf_kernel(self.centers, self.centers, self.sigma)
        self.L = do_chol(kmm)  # L @ L.T = kmm
        # A = L^{-1} K_mn / (sqrt(n*pen))
        A = torch.triangular_solve(kmn, self.L, upper=False).solution  # / sqrt_var
        AAT = A @ A.T  # m*n @ n*m = m*m in O(n * m^2), equivalent to kmn @ knm.
        # B = A @ A.T + I
        B = AAT / variance + torch.eye(AAT.shape[0], device=X.device, dtype=X.dtype)
        self.LB = do_chol(B)  # LB @ LB.T = B
        AY = A @ Y  # m*1
        self.c = torch.triangular_solve(AY, self.LB, upper=False).solution  # m*p

        C = torch.triangular_solve(A, self.LB, upper=False).solution  # m*n

        # Complexity (nystrom-deff)
        datafit = (torch.square(Y).sum() - torch.square(self.c / sqrt_var).sum())
        ndeff = (C / sqrt_var).square().sum()
        trace = (Kdiag - torch.trace(AAT))

        if self.div_trace_by_lambda:
            trace = trace / variance
        if self.div_deff_by_lambda:
            ndeff = ndeff / variance
        if self.special_one:
            trace = trace * datafit / (variance * X.shape[0])

        return (ndeff, datafit, trace)

    def predict(self, X):
        if self.L is None or self.LB is None or self.c is None:
            raise RuntimeError("Call hp_loss before calling predict.")

        with torch.autograd.no_grad():
            tmp1 = torch.triangular_solve(self.c, self.LB, upper=False, transpose=True).solution
            tmp2 = torch.triangular_solve(tmp1, self.L, upper=False, transpose=True).solution
            kms = full_rbf_kernel(self.centers, X, self.sigma)
            return kms.T @ tmp2

    @property
    def loss_names(self):
        return "nys-deff", "data-fit", "trace"

    def __repr__(self):
        return f"DeffPenFitTr(sigma={get_scalar(self.sigma)}, penalty={get_scalar(self.penalty)}, num_centers={self.centers.shape[0]}, " \
               f"opt_centers={self.opt_centers}, opt_sigma={self.opt_sigma}, opt_penalty={self.opt_penalty}, " \
               f"divtr={self.div_trace_by_lambda}, divdeff={self.div_deff_by_lambda}, special_one={self.special_one})"


class CompDeffNoPenFitTr(HyperOptimModel):
    def __init__(
            self,
            sigma_init,
            penalty_init,
            centers_init,
            opt_centers,
            opt_sigma,
            opt_penalty,
            cuda: bool,
            flk_opt: FalkonOptions,
            num_trace_est: int = 20,
            flk_maxiter: int = 10,
            nystrace_ste: bool = False,
    ):
        super(CompDeffNoPenFitTr, self).__init__()
        self.stoch_model = StochasticDeffNoPenFitTr(sigma_init.clone(), penalty_init, centers_init.clone(), opt_centers, opt_sigma, opt_penalty, cuda, flk_opt, num_trace_est, flk_maxiter, nystrace_ste)
        self.det_model = DeffNoPenFitTr(sigma_init.clone(), penalty_init, centers_init.clone(), opt_centers, opt_sigma, opt_penalty, cuda)
        self.use_model = "stoch"

    def parameters(self):
        if self.use_model == "stoch":
            return self.stoch_model.parameters()
        else:
            return self.det_model.parameters()

    def named_parameters(self):
        if self.use_model == "stoch":
            return self.stoch_model.named_parameters()
        else:
            return self.det_model.named_parameters()

    def eval(self):
        self.stoch_model.eval()
        self.det_model.eval()

    def hp_loss(self, X, Y):
        if self.use_model == "stoch":
            self.det_model.centers = self.stoch_model.centers
            self.det_model.penalty = self.stoch_model.penalty
            self.det_model.sigma = self.stoch_model.sigma
        else:
            self.stoch_model.centers = self.det_model.centers
            self.stoch_model.penalty = self.det_model.penalty
            self.stoch_model.sigma = self.det_model.sigma

        ndeff, datafit, trace = self.det_model.hp_loss(X, Y)
        stoch_loss = self.stoch_model.hp_loss(X, Y)
        print(f"Deterministic: D-eff {ndeff:.2e} Data-Fit {datafit:.2e} Trace {trace:.2e}")

        if self.use_model == "stoch":
            return stoch_loss
        else:
            return [ndeff + datafit + trace]

    def predict(self, X):
        return self.det_model.predict(X)

    @property
    def centers(self):
        if self.use_model == "stoch":
            return self.stoch_model.centers
        return self.det_model.centers

    @property
    def sigma(self):
        if self.use_model == "stoch":
            return self.stoch_model.sigma
        return self.det_model.sigma

    @property
    def penalty(self):
        if self.use_model == "stoch":
            return self.stoch_model.penalty
        return self.det_model.penalty

    @property
    def loss_names(self):
        if self.use_model == "stoch":
            return ["stoch-creg-plainfit"]
        return ["det-creg-plainfit"]

    def __repr__(self):
        return f"CompDeffNoPenFitTr()"


class StochasticDeffNoPenFitTr(NystromKRRModelMixinN, HyperOptimModel):
    def __init__(
            self,
            sigma_init,
            penalty_init,
            centers_init,
            opt_centers,
            opt_sigma,
            opt_penalty,
            cuda: bool,
            flk_opt: FalkonOptions,
            num_trace_est: int = 20,
            flk_maxiter: int = 10,
            nystrace_ste: bool = False,
    ):
        super().__init__(
            penalty=penalty_init,
            sigma=sigma_init,
            centers=centers_init,
            cuda=cuda,
            verbose=True,
        )
        self.opt_sigma, self.opt_centers, self.opt_penalty = opt_sigma, opt_centers, opt_penalty
        if opt_sigma:
            self.register_parameter("sigma", self.sigma_.requires_grad_(True))
        if opt_penalty:
            self.register_parameter("penalty", self.penalty_.requires_grad_(True))
        if opt_centers:
            self.register_parameter("centers", self.centers_.requires_grad_(True))

        self.flk_opt = flk_opt
        self.num_trace_est = num_trace_est
        self.flk_maxiter = flk_maxiter
        self.nystrace_ste = nystrace_ste

    def hp_loss(self, X, Y):
        loss = creg_plainfit(kernel_args=self.sigma, penalty=self.penalty, centers=self.centers,
                             X=X, Y=Y, num_estimators=self.num_trace_est, deterministic=False,
                             solve_options=self.flk_opt, solve_maxiter=self.flk_maxiter,
                             gaussian_random=False, use_stoch_trace=self.nystrace_ste, warm_start=True)
        return [loss]

    def predict(self, X):
        if StochasticDeffNoPenFitTrFn.last_alpha is None:
            raise RuntimeError("Call hp_loss before calling predict.")
        kernel = GaussianKernel(self.sigma.detach(), opt=self.flk_opt)
        with torch.autograd.no_grad():
            return kernel.mmv(X, self.centers, StochasticDeffNoPenFitTrFn.last_alpha)

    @property
    def loss_names(self):
        return ["stoch-creg-plainfit"]

    def print_times(self):
        StochasticDeffNoPenFitTrFn.print_times()

    def __repr__(self):
        return f"StochasticDeffNoPenFitTr(sigma={get_scalar(self.sigma)}, penalty={get_scalar(self.penalty)}, " \
               f"num_centers={self.centers.shape[0]}, opt_centers={self.opt_centers}, " \
               f"opt_sigma={self.opt_sigma}, opt_penalty={self.opt_penalty}, t={self.num_trace_est}, " \
               f"flk_iter={self.flk_maxiter})"


class DeffNoPenFitTr(NystromKRRModelMixinN, HyperOptimModel):
    def __init__(
            self,
            sigma_init,
            penalty_init,
            centers_init,
            opt_centers,
            opt_sigma,
            opt_penalty,
            cuda: bool,
            div_trace_by_lambda: bool = False,
            div_trdeff_by_lambda: bool = False,
            div_mul_lambda: bool = False,
            div_deff_by_lambda: bool = False,
    ):
        super().__init__(
            penalty=penalty_init,
            sigma=sigma_init,
            centers=centers_init,
            cuda=cuda,
            verbose=True,
        )
        self.opt_sigma, self.opt_centers, self.opt_penalty = opt_sigma, opt_centers, opt_penalty
        if opt_sigma:
            self.register_parameter("sigma", self.sigma_.requires_grad_(True))
        if opt_penalty:
            self.register_parameter("penalty", self.penalty_.requires_grad_(True))
        if opt_centers:
            self.register_parameter("centers", self.centers_.requires_grad_(True))

        self.alpha = None
        self.div_trace_by_lambda = div_trace_by_lambda
        self.div_trdeff_by_lambda = div_trdeff_by_lambda
        self.div_mul_lambda = div_mul_lambda
        self.div_deff_by_lambda = div_deff_by_lambda

    def hp_loss(self, X, Y):
        variance = self.penalty * X.shape[0]
        Kdiag = X.shape[0]

        m = self.centers.shape[0]
        kmn = full_rbf_kernel(self.centers, X, self.sigma)
        kmm = full_rbf_kernel(self.centers, self.centers, self.sigma)
        L = do_chol(kmm)  # L @ L.T = kmm
        # A = L^{-1} K_mn / (sqrt(n*pen))
        A = torch.triangular_solve(kmn, L, upper=False).solution
        AAT = A @ A.T
        # B = A @ A.T + I
        B = AAT + torch.eye(AAT.shape[0], device=X.device, dtype=X.dtype) * variance
        LB = do_chol(B)  # LB @ LB.T = B
        AY = A @ Y
        c = torch.triangular_solve(AY, LB, upper=False).solution
        dfit_t1 = torch.triangular_solve(c, LB, upper=False, transpose=True).solution
        dfit_vec = Y - A.T @ dfit_t1

        self.alpha = torch.triangular_solve(dfit_t1, L, upper=False, transpose=True).solution
        C = torch.triangular_solve(A, LB, upper=False).solution

        ndeff = C.square().sum()  # = torch.trace(C.T @ C)
        datafit = torch.square(dfit_vec).sum()
        trace = (Kdiag - torch.trace(AAT))

        if self.div_trace_by_lambda:
            trace = trace / variance
        if self.div_deff_by_lambda:
            ndeff = ndeff / variance
        if self.div_trdeff_by_lambda:
            trace = trace / variance
            ndeff = ndeff / variance
        if self.div_mul_lambda:
            ndeff /= variance
            datafit *= variance
            trace /= variance

        return ndeff, datafit, trace

    def predict(self, X):
        if self.alpha is None:
            raise RuntimeError("Call hp_loss before calling predict.")

        with torch.autograd.no_grad():
            kms = full_rbf_kernel(self.centers, X, self.sigma)
            return kms.T @ self.alpha

    @property
    def loss_names(self):
        return "nys-deff", "data-fit", "trace"

    def __repr__(self):
        return f"DeffNoPenFitTr(sigma={get_scalar(self.sigma)}, penalty={get_scalar(self.penalty)}, num_centers={self.centers.shape[0]}, " \
               f"opt_centers={self.opt_centers}, opt_sigma={self.opt_sigma}, opt_penalty={self.opt_penalty}, "\
               f"divtr={self.div_trace_by_lambda}, divdeff={self.div_deff_by_lambda}, divtrdeff={self.div_trdeff_by_lambda})"


class CregNoTrace(NystromKRRModelMixinN, HyperOptimModel):
    def __init__(
            self,
            sigma_init,
            penalty_init,
            centers_init,
            opt_centers,
            opt_sigma,
            opt_penalty,
            cuda: bool,
    ):
        super().__init__(
            penalty=penalty_init,
            sigma=sigma_init,
            centers=centers_init,
            cuda=cuda,
            verbose=True,
        )
        self.opt_sigma, self.opt_centers, self.opt_penalty = opt_sigma, opt_centers, opt_penalty
        if opt_sigma:
            self.register_parameter("sigma", self.sigma_.requires_grad_(True))
        if opt_penalty:
            self.register_parameter("penalty", self.penalty_.requires_grad_(True))
        if opt_centers:
            self.register_parameter("centers", self.centers_.requires_grad_(True))

        self.L, self.LB, self.c = None, None, None
        self.noise_estimate = None

    def hp_loss(self, X, Y):
        variance = self.penalty * X.shape[0]
        sqrt_var = torch.sqrt(variance)

        m = self.centers.shape[0]
        kmn = full_rbf_kernel(self.centers, X, self.sigma)
        kmm = full_rbf_kernel(self.centers, self.centers, self.sigma)
        self.L = do_chol(kmm)  # L @ L.T = kmm
        # A = L^{-1} K_mn / (sqrt(n*pen))
        A = torch.triangular_solve(kmn, self.L, upper=False).solution  # / sqrt_var
        AAT = A @ A.T  # m*n @ n*m = m*m in O(n * m^2), equivalent to kmn @ knm.
        # B = A @ A.T + I
        B = AAT / variance + torch.eye(AAT.shape[0], device=X.device, dtype=X.dtype)
        self.LB = do_chol(B)  # LB @ LB.T = B
        AY = A @ Y / sqrt_var  # m*1
        self.c = torch.triangular_solve(AY, self.LB, upper=False).solution / sqrt_var  # m*1

        C = torch.triangular_solve(A / sqrt_var, self.LB, upper=False).solution  # m*n

        # Complexity (nystrom-deff)
        datafit = (torch.square(Y).sum() - torch.square(self.c * sqrt_var).sum())
        ndeff = (C.square().sum())

        return ndeff, datafit

    def predict(self, X):
        if self.L is None or self.LB is None or self.c is None:
            raise RuntimeError("Call hp_loss before calling predict.")

        with torch.autograd.no_grad():
            tmp1 = torch.triangular_solve(self.c, self.LB, upper=False, transpose=True).solution
            tmp2 = torch.triangular_solve(tmp1, self.L, upper=False, transpose=True).solution
            kms = full_rbf_kernel(self.centers, X, self.sigma)
            return kms.T @ tmp2

    @property
    def loss_names(self):
        return "nys-deff", "data-fit"

    def __repr__(self):
        return f"CregNoTrace(sigma={get_scalar(self.sigma)}, penalty={get_scalar(self.penalty)}, num_centers={self.centers.shape[0]}, " \
               f"opt_centers={self.opt_centers}, opt_sigma={self.opt_sigma}, opt_penalty={self.opt_penalty})"
