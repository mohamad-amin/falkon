import abc
import time
from functools import reduce

import torch
import numpy as np

import falkon
try:
    import falkon.cuda.initialization
except:
    pass  # No GPU
from falkon import FalkonOptions
from falkon.center_selection import FixedSelector, CenterSelector
from falkon.hypergrad.common import full_rbf_kernel, test_train_predict, get_start_sigma, PositiveTransform
from falkon.hypergrad.leverage_scores import regloss_and_deff, RegLossAndDeff
from falkon.kernels import GaussianKernel
from benchmark.common.summary import get_writer


def sgpr_calc(X, Y, centers, sigma, penalty, compute_C: bool):
    """Follow calculations of SGPR:

    A = L^{-1} K_mn / (sqrt(n*pen))
    B = A @ A.T
    LB = B @ B.T
    c = LB^{-1} @ A @ Y / (sqrt(n*pen))
    C = LB^{-1} @ A

    Parameters
    ----------
    X
    Y
    centers
    sigma
    penalty
    compute_C

    Returns
    -------

    """
    variance = penalty
    sqrt_var = torch.sqrt(variance)

    kuf = full_rbf_kernel(centers, X, sigma)
    kuu = full_rbf_kernel(centers, centers, sigma) + torch.eye(centers.shape[0],
                                                               device=centers.device,
                                                               dtype=X.dtype) * 1e-6
    L = torch.cholesky(kuu)

    A = torch.triangular_solve(kuf, L, upper=False).solution / sqrt_var
    AAT = A @ A.T
    B = AAT + torch.eye(AAT.shape[0], device=AAT.device, dtype=AAT.dtype)
    LB = torch.cholesky(B)
    AY = A @ Y
    c = torch.triangular_solve(AY, LB, upper=False).solution / sqrt_var

    C = None
    if compute_C:
        C = torch.triangular_solve(A, LB, upper=False).solution

    return AAT, LB, c, C, L, A


def get_scalar(t: torch.Tensor) -> float:
    if t.dim() == 0:
        return t.item()
    return t.flatten()[0]


def report_losses(losses, loss_names, step):
    assert len(losses) == len(loss_names), f"Found {len(losses)} losses and {len(loss_names)} loss-names."
    writer = get_writer()
    report_str = "LOSSES: "
    loss_sum = 0
    for loss, loss_name in zip(losses, loss_names):
        # Report the value of the loss
        writer.add_scalar(f'optim/{loss_name}', loss, step)
        report_str += f"{loss_name}: {loss:.3e} - "
        loss_sum += loss
    report_str += f"tot: {loss_sum:.3e}"
    print(report_str, flush=True)


def report_hps(named_hparams, step):
    writer = get_writer()
    for hp_name, hp_val in named_hparams.items():
        # Report the hparam value
        writer.add_scalar(f'hparams/{hp_name}', get_scalar(hp_val), step)


def report_grads(named_hparams, grads, losses, loss_names, step):
    assert len(losses) == len(loss_names), f"Found {len(losses)} losses and {len(loss_names)} loss-names."
    assert len(grads) == len(losses), f"Found {len(grads)} grads and {len(losses)} losses."
    writer = get_writer()

    for i in range(grads):
        for j, (hp_name, hp_val) in enumerate(named_hparams.items()):
            # Report the gradient of a specific loss wrt a specific hparam
            writer.add_scalar(f'grads/{hp_name}/{loss_names[i]}', get_scalar(grads[i][j]), step)


def reporting(named_hparams, grads, losses, loss_names, verbose, step):
    report_losses(losses, loss_names, step)
    report_hps(named_hparams, step)
    if verbose:
        report_grads(named_hparams, grads, losses, loss_names, step)


class FakeTorchModelMixin(abc.ABC):
    def __init__(self):
        self._named_parameters = {}
        self._named_buffers = {}

    def register_parameter(self, name: str, param: torch.Tensor):
        if name in self._named_buffers:
            del self._named_buffers[name]
        self._named_parameters[name] = param

    def register_buffer(self, name: str, param: torch.Tensor):
        if name in self._named_parameters:
            del self._named_parameters[name]
        self._named_buffers[name] = param

    def parameters(self):
        return list(self._named_parameters.values())

    def named_parameters(self):
        return self._named_parameters

    def buffers(self):
        return list(self._named_buffers.values())

    def eval(self):
        pass


def hp_grad(model: FakeTorchModelMixin, *loss_terms, accumulate_grads=True, verbose=True, losses_are_grads=False):
    grads = []
    hparams = model.parameters()
    if not losses_are_grads:
        if verbose:
            for loss in loss_terms:
                grads.append(torch.autograd.grad(loss, hparams, retain_graph=True))
        else:
            loss = reduce(torch.add, loss_terms)
            grads.append(torch.autograd.grad(loss, hparams, retain_graph=False))
    else:
        grads = loss_terms

    if accumulate_grads:
        for g in grads:
            for i in range(len(hparams)):
                hp = hparams[i]
                if hp.grad is None:
                    hp.grad = torch.zeros_like(hp)
                if g[i] is not None:
                    hp.grad += g[i]
    return grads


class HyperOptimModel(FakeTorchModelMixin, abc.ABC):
    @abc.abstractmethod
    def predict(self, X):
        pass

    @abc.abstractmethod
    def hp_loss(self, X, Y):
        pass

    @property
    @abc.abstractmethod
    def loss_names(self):
        pass

    @property
    def losses_are_grads(self):
        return False


class NystromKRRModelMixinN(FakeTorchModelMixin, abc.ABC):
    def __init__(self, penalty, sigma, centers, cuda, verbose):
        super().__init__()
        if cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.penalty_transform = PositiveTransform(1e-9)
        #self.penalty_transform = ExpTransform()
        self.penalty = penalty
        self.register_buffer("penalty", self.penalty_)

        self.sigma = sigma
        self.register_buffer("sigma", self.sigma_)

        self.centers = centers
        self.register_buffer("centers", self.centers_)

        self.verbose = verbose

    @property
    def penalty(self):
        return self.penalty_transform(self.penalty_)

    @property
    def sigma(self):
        return self.sigma_

    @property
    def centers(self):
        return self.centers_

    @centers.setter
    def centers(self, value):
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value)
        self.centers_ = value.clone().detach().to(device=self.device)

    @sigma.setter
    def sigma(self, value):
        # sigma cannot be a 0D tensor.
        if isinstance(value, float):
            value = torch.tensor([value], dtype=self.penalty_.dtype)
        elif isinstance(value, torch.Tensor) and value.dim() == 0:
            value = torch.tensor([value.item()], dtype=value.dtype)
        elif not isinstance(value, torch.Tensor):
            value = torch.tensor(value)
        self.sigma_ = value.clone().detach().to(device=self.device)

    @penalty.setter
    def penalty(self, value):
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value)
        self.penalty_ = self.penalty_transform._inverse(value.clone().detach().to(device=self.device))


class KRRModelMixinN(FakeTorchModelMixin, abc.ABC):
    def __init__(self, penalty, sigma, cuda, verbose):
        super().__init__()
        if cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.penalty_transform = PositiveTransform(1e-9)
        #self.penalty_transform = ExpTransform()
        self.penalty = penalty
        self.register_buffer("penalty", self.penalty_)

        self.sigma = sigma
        self.register_buffer("sigma", self.sigma_)

        self.verbose = verbose

    @property
    def penalty(self):
        return self.penalty_transform(self.penalty_)

    @property
    def sigma(self):
        return self.sigma_

    @sigma.setter
    def sigma(self, value):
        # sigma cannot be a 0D tensor.
        if isinstance(value, float):
            value = torch.tensor([value], dtype=self.penalty_.dtype)
        elif isinstance(value, torch.Tensor) and value.dim() == 0:
            value = torch.tensor([value.item()], dtype=value.dtype)
        elif not isinstance(value, torch.Tensor):
            value = torch.tensor(value)
        self.sigma_ = value.clone().detach().to(device=self.device)

    @penalty.setter
    def penalty(self, value):
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value)
        self.penalty_ = self.penalty_transform._inverse(value.clone().detach().to(device=self.device))


class NystromKRRModelMixin(FakeTorchModelMixin):
    def __init__(self, penalty, sigma, centers, flk_opt, flk_maxiter, cuda, verbose, T=1):
        super().__init__()
        if cuda:
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        self.penalty_transform = PositiveTransform(1e-9)
        #self.penalty_transform = ExpTransform()
        self.penalty = self.penalty_transform._inverse(penalty.clone().detach().to(device=device))
        self.register_buffer("penalty", self.penalty)

        self.sigma = sigma.clone().detach().to(device=device)
        self.centers = centers.clone().detach().to(device=device)
        self.register_buffer("sigma", self.sigma)
        self.register_buffer("centers", self.centers)
        self.f_alpha = torch.zeros(self.centers.shape[0], T, requires_grad=False, device=device,
                                   dtype=centers.dtype)
        self.register_buffer("alpha", self.f_alpha)
        self.f_alpha_pc = torch.zeros(self.centers.shape[0], T, requires_grad=False, device=device,
                                      dtype=centers.dtype)
        self.register_buffer("alpha_pc", self.f_alpha_pc)

        self.flk_opt = flk_opt
        self.flk_maxiter = flk_maxiter

        self.model = None
        self.verbose = verbose

    def hp_grad(self, *loss_terms, accumulate_grads=True):
        grads = []
        hparams = self.parameters()
        if self.verbose:
            for l in loss_terms:
                grads.append(torch.autograd.grad(l, hparams, retain_graph=True))
        else:
            loss = reduce(torch.add, loss_terms)
            grads.append(torch.autograd.grad(loss, hparams, retain_graph=False))

        if accumulate_grads:
            for g in grads:
                for i in range(len(hparams)):
                    hp = hparams[i]
                    if hp.grad is None:
                        hp.grad = torch.zeros_like(hp)
                    if g[i] is not None:
                        hp.grad += g[i]

        return grads

    def adapt_alpha(self, X, Y):
        k = GaussianKernel(self.sigma.detach(), self.flk_opt)

        if X.is_cuda:
            fcls = falkon.InCoreFalkon
        else:
            fcls = falkon.Falkon

        model = fcls(k,
                     self.penalty_val.item(),# / X.shape[0],
                     M=self.centers.shape[0],
                     center_selection=FixedSelector(self.centers.detach()),
                     maxiter=self.flk_maxiter,
                     options=self.flk_opt)
        model.fit(X, Y)#, warm_start=self.f_alpha_pc)
        self.f_alpha = model.alpha_.detach()
        self.f_alpha_pc = model.beta_.detach()
        self.model = model

    @abc.abstractmethod
    def predict(self, X):
        pass

    def get_model(self):
        if self.model is None:
            raise RuntimeError("You must call `adapt_alpha` before getting the model.")
        k = GaussianKernel(self.sigma.detach(), self.flk_opt)
        model = self.model.__class__(
            kernel=k,
            penalty=self.penalty_val.item() / X.shape[0],
            M=self.centers.shape[0],
            center_selection=FixedSelector(self.centers.detach()),
            maxiter=self.flk_maxiter,
            options=self.flk_opt,
        )
        return model

    def eval(self):
        pass

    @property
    def penalty_val(self):
        return self.penalty_transform(self.penalty)



class SimpleFalkonComplexityReg(NystromKRRModelMixin):
    def __init__(
            self,
            penalty_init,
            sigma_init,
            centers_init,
            flk_opt,
            flk_maxiter,
            opt_centers,
            opt_sigma,
            opt_penalty,
            verbose_tboard: bool,
            cuda: bool,
            T: int,
            only_trace: bool = False,
    ):
        super().__init__(
            penalty=penalty_init,
            sigma=sigma_init,
            centers=centers_init,
            flk_opt=flk_opt,
            flk_maxiter=flk_maxiter,
            cuda=cuda,
            T=T,
            verbose=verbose_tboard,
        )
        if opt_sigma:
            self.register_parameter("sigma", self.sigma.requires_grad_(True))
        if opt_penalty:
            self.register_parameter("penalty", self.penalty.requires_grad_(True))
        if opt_centers:
            self.register_parameter("centers", self.centers.requires_grad_(True))

        print("Penalty: ", self.penalty)

        self.only_trace = only_trace

    def hp_loss(self, X, Y):
        variance = self.penalty_val
        sqrt_var = torch.sqrt(variance)
        Kdiag = X.shape[0]

        AAT, LB, c, C, L, _ = sgpr_calc(X, Y, self.centers, self.sigma, variance, compute_C=True)

        # NKRR + Trace
        # Complexity (nystrom effective-dimension)
        if not self.only_trace:
            deff = torch.trace(C.T @ C)
        else:
            deff = torch.tensor(0.0)
        # Data-fit
        if not self.only_trace:
            c = c * sqrt_var  # Correct c would be LB^-1 @ A @ Y
            datafit = - torch.square(Y).sum()
            datafit += torch.square(c).sum()
        else:
            datafit = torch.tensor(0.0)
        # Traces
        # Cannot remove variance in either of these!
        # Keeping or removing `0.5` does not have much effect
        trace = - 0.5 * Kdiag / variance
        trace += 0.5 * torch.diag(AAT).sum()

        #print("DataFit: %.2e" % (-datafit))
        #print("Effective dim: %.2e" % (deff))
        #print("Trace: %.2e" % (-trace * variance))
        if not self.only_trace:
            #deff = -torch.log(deff / variance) * X.shape[0]
            #deff = torch.log(deff) * X.shape[0]
            deff = deff #* X.shape[0]
        if not self.only_trace:
            #datafit = -datafit / variance
            datafit = -datafit
        #trace = -trace
        trace = - trace * variance


        #deff = torch.log(deff / variance); datafit = datafit / variance; trace = trace
        #deff = torch.log(deff) * X.shape[0]; datafit = datafit; trace = trace * variance
        #deff = -torch.log(deff); datafit = datafit; trace = trace * variance
        #deff = -torch.log(deff) * X.shape[0]; datafit = datafit; trace = trace * variance
        #deff = -torch.log(deff /variance); datafit = datafit / variance; trace = trace

        #deff = -torch.log(deff / variance)# * variance) #* X.shape[0]
        #deff = torch.log(deff / variance) * X.shape[0]

        return deff, datafit, trace

    @property
    def loss_names(self):
        return "d-eff", "data-fit", "trace"


class NoRegFalkonComplexityReg(SimpleFalkonComplexityReg):
    def __init__(
            self,
            penalty_init,
            sigma_init,
            centers_init,
            flk_opt,
            flk_maxiter,
            opt_centers,
            opt_sigma,
            opt_penalty,
            verbose_tboard: bool,
            cuda: bool,
            T: int,
            only_trace: bool = False,
    ):
        super().__init__(penalty_init, sigma_init, centers_init, flk_opt, flk_maxiter,
                         opt_centers, opt_sigma, opt_penalty, verbose_tboard, cuda, T,
                         only_trace)

    def hp_loss(self, X, Y):
        variance = self.penalty_val
        sqrt_var = torch.sqrt(variance)
        Kdiag = X.shape[0]

        AAT, LB, c, C, L, A = sgpr_calc(X, Y, self.centers, self.sigma, variance, compute_C=True)

        # NKRR + Trace
        # Complexity (nystrom effective-dimension)
        if not self.only_trace:
            deff = torch.trace(C.T @ C)
            #print("deff", deff)
            #print("Y std: %.2e" % (torch.std(Y)))
            deff = deff * torch.std(Y) * 2
            #print("Deff", deff)
        else:
            deff = torch.tensor(0.0)
        # Data-fit
        if not self.only_trace:
            c_tilde = c * sqrt_var
            d = A.T @ torch.triangular_solve(c_tilde, LB, upper=False, transpose=True).solution
            datafit = torch.square(Y).sum() - 2 * torch.square(c_tilde).sum() + torch.square(d).sum()
        else:
            datafit = torch.tensor(0.0)
        # Traces
        # Cannot remove variance in either of these!
        # Keeping or removing `0.5` does not have much effect
        trace = - 0.5 * Kdiag / variance
        trace += 0.5 * torch.diag(AAT).sum()
        #trace = torch.tensor(0.0)

        if not self.only_trace:
            deff = deff
        if not self.only_trace:
            pass
        #trace = -trace
        trace = - trace * variance

        return deff, datafit, trace

    @property
    def loss_names(self):
        return "d-eff", "data-fit(noreg)", "trace"



class FalkonComplexityReg(NystromKRRModelMixin):
    def __init__(
            self,
            penalty_init,
            sigma_init,
            centers_init,
            flk_opt,
            flk_maxiter,
            opt_centers,
            verbose_tboard: bool,
            precise_trace: bool,
            cuda: bool,
            T: int,
    ):
        super().__init__(
            penalty=penalty_init,
            sigma=sigma_init,
            centers=centers_init,
            flk_opt=flk_opt,
            flk_maxiter=flk_maxiter,
            cuda=cuda,
            T=T,
            verbose=verbose_tboard,
        )
        falkon.cuda.initialization.init(flk_opt)
        self.register_parameter("penalty", self.penalty.requires_grad_(True))
        self.register_parameter("sigma", self.sigma.requires_grad_(True))
        if opt_centers:
            self.register_parameter("centers", self.centers.requires_grad_(True))

        self.precise_trace = precise_trace

    def hp_loss(self, X, Y):
        deff, datafit, trace = regloss_and_deff(
            kernel_args=self.sigma,
            penalty=self.penalty_val,
            M=self.centers,
            X=X, Y=Y, t=200,
            deterministic=False,
            solve_options=self.flk_opt,
            use_precise_trace=self.precise_trace,
            solve_maxiter=self.flk_maxiter,
            gaussian_random=False,
        )
        return -deff, -datafit, -trace

    def adapt_alpha(self, X, Y):
        # Mostly copied from super-class but the more efficient shortcut is taken
        k = GaussianKernel(self.sigma.detach(), self.flk_opt)

        if X.is_cuda:
            fcls = falkon.InCoreFalkon
        else:
            fcls = falkon.Falkon

        model = fcls(k,
                     self.penalty_val.item() / X.shape[0],
                     M=self.centers.shape[0],
                     center_selection=FixedSelector(self.centers.detach()),
                     maxiter=self.flk_maxiter,
                     options=self.flk_opt)
        if RegLossAndDeff.last_alpha is not None:
            weights = RegLossAndDeff.last_alpha
            model.alpha_ = weights.detach()
            model.ny_points_ = self.centers.detach()
        else:
            model.fit(X, Y, warm_start=self.f_alpha_pc)
            self.f_alpha = model.alpha_.detach()
            self.f_alpha_pc = model.beta_.detach()

        self.model = model

    @property
    def loss_names(self):
        return "d-eff", "data-fit", "trace"


class GPComplexityReg(NystromKRRModelMixin):
    def __init__(
            self,
            sigma_init,
            penalty_init,
            centers_init,
            opt_centers,
            opt_sigma,
            opt_penalty,
            verbose_tboard: bool,
            flk_opt: FalkonOptions,
            flk_maxiter,
            cuda: bool,
            T: int,
            only_trace: bool = False,
    ):
        super().__init__(
            penalty=penalty_init,
            sigma=sigma_init,
            centers=centers_init,
            flk_opt=flk_opt,
            flk_maxiter=flk_maxiter,
            cuda=cuda,
            T=T,
            verbose=verbose_tboard,
        )
        if opt_sigma:
            self.register_parameter("sigma", self.sigma.requires_grad_(True))
        if opt_penalty:
            self.register_parameter("penalty", self.penalty.requires_grad_(True))
        if opt_centers:
            self.register_parameter("centers", self.centers.requires_grad_(True))

        self.only_trace = only_trace

    def hp_loss(self, X, Y):
        variance = self.penalty_val
        sqrt_var = torch.sqrt(variance)
        Kdiag = X.shape[0]

        AAT, LB, c, _, L, _ = sgpr_calc(X, Y, self.centers, self.sigma, variance, compute_C=False)
        self.AAT = AAT
        self.LB = LB
        self.c = c
        self.L = L

        # Complexity
        if not self.only_trace:
            deff = -torch.log(torch.diag(LB)).sum()
            deff += -0.5 * X.shape[0] * torch.log(variance)
        else:
            deff = torch.tensor(0.0)
        # Data-fit
        if not self.only_trace:
            datafit = -0.5 * torch.square(Y).sum() / variance
            datafit += 0.5 * torch.square(c).sum()
        else:
            datafit = torch.tensor(0.0)
        # Traces
        trace = -0.5 * Kdiag / variance
        trace += 0.5 * torch.diag(AAT).sum()

        const = -0.5 * X.shape[0] * torch.log(2 * torch.tensor(np.pi, dtype=X.dtype))

        return -deff, -datafit, -trace, -const

    def predict_closed_form(self, X):
        kus = full_rbf_kernel(self.centers, X, self.sigma)
        tmp1 = torch.triangular_solve(kus, self.L, upper=False).solution
        tmp2 = torch.triangular_solve(tmp1, self.LB, upper=False).solution
        return tmp2.T @ self.c

    @property
    def loss_names(self):
        return "log-det", "data-fit", "trace", "const"


class TrainableSGPR():
    def __init__(self,
                 sigma_init,
                 penalty_init,
                 centers_init,
                 opt_centers,
                 opt_sigma,
                 opt_penalty,
                 num_epochs,
                 learning_rate,
                 err_fn):
        from gpflow import set_trainable
        import gpflow

        self.penalty = penalty_init.item()
        self.sigma = sigma_init.item()
        self.centers = centers_init.numpy()

        self.kernel = gpflow.kernels.SquaredExponential(lengthscales=self.sigma, variance=1)
        set_trainable(self.kernel.variance, False)
        if not opt_sigma:
            set_trainable(self.kernel.lengthscales, False)

        self.opt_centers = opt_centers
        self.opt_penalty = opt_penalty
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.err_fn = err_fn

    def train(self, Xtr, Ytr, Xts, Yts):
        import tensorflow as tf
        import tensorflow_probability as tfp
        from gpflow import set_trainable
        import gpflow

        Xtr = Xtr.numpy()
        Ytr = Ytr.numpy()
        Xts, Yts = Xts.numpy(), Yts.numpy()

        self.model = gpflow.models.SGPR(
            (Xtr, Ytr),
            kernel=self.kernel,
            inducing_variable=self.centers,
            noise_variance=self.penalty)
        if not self.opt_centers:
            set_trainable(self.model.inducing_variable.Z, False)
        #self.model.likelihood.variance = gpflow.Parameter(self.penalty, transform=tfp.bijectors.Identity())
        if not self.opt_penalty:
            set_trainable(self.model.likelihood.variance, False)
        self.model.kernel.lengthscales = gpflow.Parameter(self.sigma, transform=tfp.bijectors.Identity())

        opt = tf.keras.optimizers.Adam(self.learning_rate, epsilon=1e-8)

        @tf.function
        def step_fn():
            opt.minimize(self.model.training_loss, var_list=self.model.trainable_variables)

        gpflow.utilities.print_summary(self.model)
        for step in range(self.num_epochs):
            step_fn()
            tr_err, err_name = self.err_fn(Ytr, self.predict(Xtr))
            val_err, err_name = self.err_fn(Yts, self.predict(Xts))
            #gpflow.utilities.print_summary(self.model)
            print(f"Epoch {step + 1} - train {err_name} = {tr_err:.4f} - test {err_name} {val_err:.4f}")
        gpflow.utilities.print_summary(self.model)

    def predict(self, X):
        return self.model.predict_y(X)[0]



def train_complexity_reg(
        Xtr, Ytr,
        Xts, Yts,
        penalty_init: float,
        sigma_type: str,
        sigma_init: float,
        num_centers: int,
        opt_centers: bool,
        falkon_centers: CenterSelector,
        num_epochs: int,
        learning_rate: float,
        cuda: bool,
        loss_every: int,
        err_fn,
        falkon_opt,
        falkon_maxiter: int,
        model_type: str,
):
    print("Starting Falkon-NKRR-HO - FIXED VERSION - optimization.")
    print(f"{num_epochs} epochs - {sigma_type} sigma ({sigma_init}) - penalty ({penalty_init}) - "
          f"{num_centers} centers. LR={learning_rate}")
    start_sigma = get_start_sigma(sigma_init, sigma_type, Xtr.shape[1])

    verbose = True
    model_type = model_type.lower()

    if model_type == "gp":
        model = GPComplexityReg(
            penalty_init=penalty_init,
            sigma_init=start_sigma,
            centers_init=falkon_centers.select(Xtr, Y=None, M=num_centers),
            opt_centers=opt_centers,
            flk_opt=falkon_opt,
            flk_maxiter=falkon_maxiter,
            verbose_tboard=verbose,
            cuda=cuda,
            T=Ytr.shape[1],
            only_trace=False,
            opt_sigma=True,
            opt_penalty=True
        )
    elif model_type == "deff-simple":
        model = SimpleFalkonComplexityReg(
            penalty_init=penalty_init,
            sigma_init=start_sigma,
            centers_init=falkon_centers.select(Xtr, Y=None, M=num_centers),
            opt_centers=opt_centers,
            flk_opt=falkon_opt,
            flk_maxiter=falkon_maxiter,
            verbose_tboard=verbose,
            cuda=cuda,
            T=Ytr.shape[1],
        )
    elif model_type in {"deff-precise", "deff-fast"}:
        precise_trace = model_type == "deff-precise"
        model = FalkonComplexityReg(
            penalty_init=penalty_init,
            sigma_init=start_sigma,
            centers_init=falkon_centers.select(Xtr, Y=None, M=num_centers),
            opt_centers=opt_centers,
            flk_opt=falkon_opt,
            flk_maxiter=falkon_maxiter,
            verbose_tboard=verbose,
            precise_trace=precise_trace,
            cuda=cuda,
            T=Ytr.shape[1],
        )
    else:
        raise RuntimeError(f"{model_type} model type not recognized!")

    if cuda:
        Xtr, Ytr, Xts, Yts = Xtr.cuda(), Ytr.cuda(), Xts.cuda(), Yts.cuda()

    opt_hp = torch.optim.Adam([
        {"params": model.parameters(), "lr": learning_rate},
    ])

    cum_time = 0
    cum_step = 0
    writer = get_writer()
    for epoch in range(num_epochs):
        e_start = time.time()
        opt_hp.zero_grad()
        losses = model.hp_loss(Xtr, Ytr)
        grads = model.hp_grad(*losses, accumulate_grads=True)
        reporting(model.named_parameters(), grads, losses, model.loss_names, verbose, cum_step)
        # Loss reporting before step() to optimize (avoid second flk)
        if epoch != 0 and (epoch + 1) % loss_every == 0:
            model.adapt_alpha(Xtr, Ytr)
            cum_time, train_err, test_err = test_train_predict(
                model=model, Xtr=Xtr, Ytr=Ytr, Xts=Xts, Yts=Yts,
                err_fn=err_fn, epoch=epoch, time_start=e_start, cum_time=cum_time)
            writer.add_scalar('error/train', train_err, cum_step)
            writer.add_scalar('error/test', test_err, cum_step)
        opt_hp.step()
        cum_step += 1

    return model.get_model()
