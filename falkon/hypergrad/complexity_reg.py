import time
from typing import List

import torch

import falkon
import falkon.cuda.initialization
from falkon import FalkonOptions
from falkon.center_selection import FixedSelector, CenterSelector
from falkon.hypergrad.common import full_rbf_kernel, test_train_predict, get_start_sigma
from falkon.hypergrad.leverage_scores import regloss_and_deff, RegLossAndDeff
from falkon.kernels import GaussianKernel
from summary import get_writer


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
    variance = torch.exp(-penalty) * X.shape[0]
    sqrt_var = torch.sqrt(variance)

    kuf = full_rbf_kernel(centers, X, sigma)
    kuu = full_rbf_kernel(centers, centers, sigma) + torch.eye(centers.shape[0],
                                                               device=centers.device) * 1e-6
    L = torch.cholesky(kuu)

    A = torch.triangular_solve(kuf, L, upper=False).solution / sqrt_var
    AAT = A @ A.T
    B = AAT + torch.eye(AAT.shape[0], device=AAT.device)
    LB = torch.cholesky(B)
    AY = A @ Y
    c = torch.triangular_solve(AY, LB, upper=False).solution / sqrt_var

    C = None
    if compute_C:
        C = torch.triangular_solve(A, LB, upper=False).solution

    return AAT, LB, c, C


def report_complexity_reg(
        deff: torch.Tensor,
        datafit: torch.Tensor,
        trace: torch.Tensor,
        hparams: List[torch.Tensor],
        penalty: torch.Tensor,
        sigma: torch.Tensor,
        step: int,
        verbose_tboard: bool,
        ):
    writer = get_writer()
    writer.add_scalar('optim/d_eff', -deff.item(), step)
    writer.add_scalar('optim/data_fit', -datafit.item(), step)
    writer.add_scalar('optim/trace', -trace.item(), step)
    writer.add_scalar('optim/loss', (-deff - datafit - trace).item(), step)
    writer.add_scalar('hparams/penalty', torch.exp(-penalty).item(), step)
    writer.add_scalar('hparams/sigma', sigma[0], step)
    print(f"VALUE        d_eff {deff:.5e} - loss {datafit:.5e} - trace {trace:.5e}")

    if verbose_tboard:
        grad_deff = torch.autograd.grad(-deff, hparams, retain_graph=True)
        grad_loss = torch.autograd.grad(-datafit, hparams, retain_graph=True)
        grad_trace = torch.autograd.grad(-trace, hparams, retain_graph=False, allow_unused=True)
        # print(f"GRADS(sigma) d_eff {grad_deff[1][0]:.5e} - loss {grad_loss[1][0]:.5e} - trace {grad_trace[1][0]:.5e}")
        print(
            f"GRADS(penalty)   d_eff {grad_deff[0]:.5e} - loss {grad_loss[0]:.5e} - trace {grad_trace[0]:.5e}")
        writer.add_scalar('grads/penalty/d_eff', grad_deff[0].item(), step)
        writer.add_scalar('grads/penalty/data_fit', grad_loss[0].item(), step)
        writer.add_scalar('grads/penalty/trace', grad_trace[0].item(), step)
        writer.add_scalar('grads/sigma/d_eff', grad_deff[1][0].item(), step)
        writer.add_scalar('grads/sigma/data_fit', grad_loss[1][0].item(), step)
        writer.add_scalar('grads/sigma/trace', grad_trace[1][0].item(), step)
    else:
        grad = torch.autograd.grad(-deff - datafit - trace, hparams, retain_graph=False)
        writer.add_scalar('grads/penalty/sum', grad[0].item(), step)
        writer.add_scalar('grads/sigma/sum', grad[1][0].item(), step)
        grad_deff = grad
        grad_loss = [None] * len(grad)
        grad_trace = [None] * len(grad)

    for hp, g1, g2, g3 in zip(hparams, grad_deff, grad_loss, grad_trace):
        if not hp.requires_grad:
            continue
        if hp.grad is None:
            hp.grad = torch.zeros_like(hp)
        if g1 is not None:
            hp.grad += g1
        if g2 is not None:
            hp.grad += g2
        if g3 is not None:
            hp.grad += g3


class FakeTorchModelMixin():
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

    def buffers(self):
        return list(self._named_buffers.values())


class NystromKRRModelMixin(FakeTorchModelMixin):
    def __init__(self, penalty, sigma, centers, flk_opt, flk_maxiter, cuda):
        super().__init__()
        if cuda:
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        self.penalty = torch.tensor(penalty).to(device=device)
        self.register_buffer("penalty", self.penalty)
        self.sigma = torch.tensor(sigma).to(device=device)
        self.register_buffer("sigma", self.sigma)
        self.centers = torch.tensor(centers).to(device=device)
        self.register_buffer("centers", self.centers)
        self.f_alpha = torch.zeros(self.centers.shape[0], 1, requires_grad=False, device=device)
        self.register_buffer("alpha", self.f_alpha)
        self.f_alpha_pc = torch.zeros(self.centers.shape[0], 1, requires_grad=False, device=device)
        self.register_buffer("alpha_pc", self.f_alpha_pc)

        self.flk_opt = flk_opt
        self.flk_maxiter = flk_maxiter

        self.model = None

    def adapt_alpha(self, X, Y):
        k = GaussianKernel(self.sigma.detach(), self.flk_opt)

        if X.is_cuda:
            fcls = falkon.InCoreFalkon
        else:
            fcls = falkon.Falkon

        model = fcls(k,
                     torch.exp(-self.penalty).item(),
                     M=self.centers.shape[0],
                     center_selection=FixedSelector(self.centers.detach()),
                     maxiter=self.flk_maxiter,
                     options=self.flk_opt)
        model.fit(X, Y, warm_start=self.f_alpha_pc)
        self.f_alpha = model.alpha_.detach()
        self.f_alpha_pc = model.beta_.detach()
        self.model = model

    def predict(self, X):
        return self.model.predict(X)

    def get_model(self):
        if self.model is None:
            raise RuntimeError("You must call `adapt_alpha` before getting the model.")
        k = GaussianKernel(self.sigma.detach(), self.flk_opt)
        model = self.model.__class__(
            kernel=k,
            penalty=torch.exp(-self.penalty).item(),
            M=self.centers.shape[0],
            center_selection=FixedSelector(self.centers.detach()),
            maxiter=self.flk_maxiter,
            options=self.flk_opt,
        )
        return model

    def eval(self):
        pass


class SimpleFalkonComplexityReg(NystromKRRModelMixin):
    def __init__(
            self,
            penalty_init,
            sigma_init,
            centers_init,
            flk_opt,
            flk_maxiter,
            opt_centers,
            verbose_tboard: bool,
            cuda: bool,
    ):
        super().__init__(
            penalty=penalty_init,
            sigma=sigma_init,
            centers=centers_init,
            flk_opt=flk_opt,
            flk_maxiter=flk_maxiter,
            cuda=cuda,
        )
        self.register_parameter("penalty", self.penalty.requires_grad_(True))
        self.register_parameter("sigma", self.sigma.requires_grad_(True))
        if opt_centers:
            self.register_parameter("centers", self.centers.requires_grad_(True))

        self.verbose_tboard = verbose_tboard
        self.writer = get_writer()

    def adapt_hps(self, X, Y, step):
        AAT, LB, c, C = sgpr_calc(X, Y, self.centers, self.sigma, self.penalty, compute_C=True)

        variance = torch.exp(-self.penalty) * X.shape[0]
        sqrt_var = torch.sqrt(variance)
        Kdiag = X.shape[0]

        # NKRR + Trace
        # Complexity
        deff = -torch.trace(C.T @ C)
        # Data-fit
        c = c * sqrt_var  # Correct c would be LB^-1 @ A @ Y
        datafit = - torch.square(Y).sum()
        datafit += torch.square(c).sum()
        # Traces
        # Cannot remove variance in either of these!
        # Keeping or removing `0.5` does not have much effect
        trace = - 0.5 * Kdiag / (variance)
        trace += 0.5 * torch.diag(AAT).sum()

        report_complexity_reg(
            deff=deff,
            datafit=datafit,
            trace=trace,
            hparams=self.parameters(),
            penalty=self.penalty,
            sigma=self.sigma,
            step=step,
            verbose_tboard=self.verbose_tboard,
        )
        return deff + datafit + trace


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
    ):
        super().__init__(
            penalty=penalty_init,
            sigma=sigma_init,
            centers=centers_init,
            flk_opt=flk_opt,
            flk_maxiter=flk_maxiter,
            cuda=cuda,
        )
        falkon.cuda.initialization.init(flk_opt)
        self.register_parameter("penalty", self.penalty.requires_grad_(True))
        self.register_parameter("sigma", self.sigma.requires_grad_(True))
        if opt_centers:
            self.register_parameter("centers", self.centers.requires_grad_(True))

        self.verbose_tboard = verbose_tboard
        self.precise_trace = precise_trace
        self.writer = get_writer()

    def adapt_hps(self, X, Y, step):
        deff, datafit, trace = regloss_and_deff(
            kernel_args=self.sigma,
            penalty=self.penalty,
            M=self.centers,
            X=X, Y=Y, t=20,
            deterministic=False,
            solve_options=self.flk_opt,
            use_precise_trace=self.precise_trace,
            solve_maxiter=self.flk_maxiter,
            gaussian_random=False,
        )

        report_complexity_reg(
            deff=deff,
            datafit=datafit,
            trace=trace,
            hparams=self.parameters(),
            penalty=self.penalty,
            sigma=self.sigma,
            step=step,
            verbose_tboard=self.verbose_tboard,
        )
        return deff + datafit + trace

    def adapt_alpha(self, X, Y):
        # Mostly copied from super-class but the more efficient shortcut is taken
        k = GaussianKernel(self.sigma.detach(), self.flk_opt)

        if X.is_cuda:
            fcls = falkon.InCoreFalkon
        else:
            fcls = falkon.Falkon

        model = fcls(k,
                     torch.exp(-self.penalty).item(),  # / X.shape[0],
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


class GPComplexityReg(NystromKRRModelMixin):
    def __init__(
            self,
            sigma_init,
            penalty_init,
            centers_init,
            opt_centers,
            verbose_tboard: bool,
            flk_opt: FalkonOptions,
            flk_maxiter,
            cuda: bool,
    ):
        super().__init__(
            penalty=penalty_init,
            sigma=sigma_init,
            centers=centers_init,
            flk_opt=flk_opt,
            flk_maxiter=flk_maxiter,
            cuda=cuda,
        )
        self.register_parameter("penalty", self.penalty.requires_grad_(True))
        self.register_parameter("sigma", self.sigma.requires_grad_(True))
        if opt_centers:
            self.register_parameter("centers", self.centers.requires_grad_(True))

        self.verbose_tboard = verbose_tboard
        self.writer = get_writer()

    def adapt_hps(self, X, Y, step):
        AAT, LB, c, _ = sgpr_calc(X, Y, self.centers, self.sigma, self.penalty, compute_C=False)

        variance = torch.exp(-self.penalty) * X.shape[0]
        Kdiag = X.shape[0]

        # Complexity
        deff = -torch.log(torch.diag(LB)).sum()
        deff += -0.5 * X.shape[0] * torch.log(variance)
        # Data-fit
        datafit = -0.5 * torch.square(Y).sum() / variance
        datafit += 0.5 * torch.square(c).sum()
        # Traces
        trace = -0.5 * Kdiag / variance
        trace += 0.5 * torch.diag(AAT).sum()

        report_complexity_reg(
            deff=deff,
            datafit=datafit,
            trace=trace,
            hparams=self.parameters(),
            penalty=self.penalty,
            sigma=self.sigma,
            step=step,
            verbose_tboard=self.verbose_tboard,
        )
        return deff + datafit + trace


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
        loss = -model.adapt_hps(Xtr, Ytr, cum_step)
        # Loss reporting before step() to optimize (avoid second flk)
        if epoch != 0 and (epoch + 1) % loss_every == 0:
            model.adapt_alpha(Xtr, Ytr)
            cum_time, train_err, test_err = test_train_predict(
                model=model, Xtr=Xtr, Ytr=Ytr, Xts=Xts, Yts=Yts,
                err_fn=err_fn, epoch=epoch, time_start=e_start, cum_time=cum_time)
            writer.add_scalar('Error/train', train_err, cum_step)
            writer.add_scalar('Error/test', test_err, cum_step)
        opt_hp.step()
        cum_step += 1
    return model.get_model()
