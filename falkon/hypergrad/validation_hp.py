import time
from typing import List, Union
import functools

import torch

from summary import get_writer
import falkon
from falkon import FalkonOptions
from falkon.kernels import GaussianKernel
from falkon.optim import ConjugateGradient
from falkon.kernels.diff_rbf_kernel import DiffGaussianKernel
from falkon.hypergrad.common import get_start_sigma, test_train_predict, cg
from falkon.center_selection import FixedSelector, CenterSelector


class FalkonHyperGradient():
    def __init__(
            self,
            penalty_init: float,
            sigma_init: torch.Tensor,
            centers_init: torch.Tensor,
            opt_centers: bool,
            flk_opt: FalkonOptions,
            val_loss_type: str,
            cuda: bool,
    ):
        if cuda:
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        self.penalty = torch.tensor(penalty_init, requires_grad=True, device=device)
        self.sigma = torch.tensor(sigma_init, requires_grad=True, device=device)
        self.centers = torch.tensor(centers_init, requires_grad=opt_centers, device=device)

        self.f_alpha = torch.zeros(self.centers.shape[0], 1, requires_grad=True, device=device)
        self.f_alpha_pc = torch.zeros(self.centers.shape[0], 1, requires_grad=False, device=device)

        self.flk_opt = flk_opt
        self.flk_maxiter = 10
        self.use_hvp = True
        self.val_loss_type = val_loss_type
        self.model: Union[None, falkon.InCoreFalkon, falkon.Falkon] = None

    def params(self) -> List[torch.Tensor]:
        return [self.f_alpha]

    def hparams(self) -> List[torch.Tensor]:
        hps = [self.penalty, self.sigma]
        if self.centers.requires_grad:
            hps.append(self.centers)
        return hps

    def adapt_hps(self, Xtr, Ytr, Xval, Yval, step):
        # grad_outer_hparams is the 'direct gradient'.
        grad_outer_params, grad_outer_hparams = self.val_loss_grads(Xval, Yval)

        # First derivative of training loss wrt params
        first_diff = self.tr_loss_p_grad(Xtr, Ytr)

        # Calculate the Hessian multiplied by the outer-gradient wrt alpha
        if self.use_hvp:
            hvp_func = functools.partial(self.hessian_vector_product, Xtr, first_diff)
            vs = cg(hvp_func, grad_outer_params,
                    max_iter=self.flk_maxiter, epsilon=self.flk_opt.cg_epsilon_32)
        else:
            vs = self.solve_hessian(Xtr, grad_outer_params)

        # Multiply the mixed inner gradient by `vs`
        hyper_grads = torch.autograd.grad(
            first_diff, self.hparams(), grad_outputs=vs, allow_unused=True)

        final_grads = []
        for dg, hg in zip(grad_outer_hparams, hyper_grads):
            if dg is not None:
                final_grads.append(dg - hg)
            else:
                final_grads.append(-hg)

        writer = get_writer()
        writer.add_scalar('hparams/penalty', torch.exp(-self.penalty).item(), step)
        writer.add_scalar('hparams/sigma', self.sigma[0], step)
        writer.add_scalar('grads/penalty/hyper-grad', hyper_grads[0].item(), step)
        writer.add_scalar('grads/sigma/hyper-grad', hyper_grads[1][0].item(), step)
        writer.add_scalar('grads/sigma/direct-grad', grad_outer_hparams[1][0].item(), step)

        for hp, g in zip(self.hparams(), final_grads):
            if hp.grad is None:
                hp.grad = torch.zeros_like(hp)
            if g is not None:
                hp.grad += g

    def _val_loss_mse(self, Xval, Yval):
        kernel = DiffGaussianKernel(self.sigma, self.flk_opt)
        preds = kernel.mmv(Xval, self.centers, self.f_alpha)
        return torch.mean((preds - Yval) ** 2)

    def _val_loss_penalized_mse(self, Xval, Yval):
        kernel = DiffGaussianKernel(self.sigma, self.flk_opt)
        preds = kernel.mmv(Xval, self.centers, self.f_alpha)
        pen = (torch.exp(-self.penalty)) * (
                self.f_alpha.T @ kernel.mmv(self.centers, self.centers, self.f_alpha)
        )
        return torch.mean((preds - Yval) ** 2) + pen

    def val_loss(self, Xval: torch.Tensor, Yval: torch.Tensor):
        if self.val_loss_type == "penalized-mse":
            return self._val_loss_penalized_mse(Xval, Yval)
        elif self.val_loss_type == "mse":
            return self._val_loss_mse(Xval, Yval)
        else:
            raise RuntimeError("Loss %s unrecognized" % (self.val_loss_type))

    def val_loss_grads(self, Xval, Yval):
        """Gradients of the validation loss with respect to params (alpha) and hparams."""
        o_loss = self.val_loss(Xval, Yval)
        return (
            torch.autograd.grad(o_loss, self.params(), allow_unused=True, create_graph=False,
                                retain_graph=True),
            torch.autograd.grad(o_loss, self.hparams(), allow_unused=True, create_graph=False,
                                retain_graph=False)
        )

    def tr_loss_p_grad(self, Xtr, Ytr):
        """Gradients of the training loss with respect to params (alpha)"""
        # Must be differentiable wrt hyper-params
        kernel = DiffGaussianKernel(self.sigma, self.flk_opt)

        # 2/N * (K_MN(K_NM @ alpha - Y)) + 2*lambda*(K_MM @ alpha)
        _penalty = torch.exp(-self.penalty)
        return (
                kernel.mmv(self.centers, Xtr,
                           kernel.mmv(Xtr, self.centers, self.f_alpha) - Ytr) / Xtr.shape[0] +
                _penalty * kernel.mmv(self.centers, self.centers, self.f_alpha)
        )

    def solve_hessian(self, Xtr, vector):
        if self.model is None:
            raise RuntimeError("Model not initialized (call adapt_alpha first).")
        with torch.autograd.no_grad():
            _penalty = torch.exp(-self.penalty.detach())
            vector = vector[0].detach()
            N = Xtr.shape[0]

            cg = ConjugateGradient(self.flk_opt)

            kernel = self.model.kernel
            precond = self.model.precond
            centers = self.centers.detach()

            # NOTE: Different normalization constant from Falkon.
            B = precond.apply_t(vector)

            def mmv(sol):
                v = precond.invA(sol)
                cc = kernel.dmmv(Xtr, centers, precond.invT(v), None)
                return precond.invAt(precond.invTt(cc / N) + _penalty * v)

            d = cg.solve(X0=None, B=B, mmv=mmv, max_iter=self.flk_maxiter)
            c = precond.apply(d)

            return [c]

    def hessian_vector_product(self, Xtr, tr_loss_p_der, vector):
        vector = vector[0]
        with torch.autograd.no_grad():
            k = GaussianKernel(self.sigma.detach(), self.flk_opt)
            _penalty = torch.exp(-self.penalty)
            N = Xtr.shape[0]

            out = (k.mmv(self.centers, Xtr, k.mmv(Xtr, self.centers, vector)) / N +
                    _penalty * k.mmv(self.centers, self.centers, vector))
            return [out]


    def adapt_alpha(self, X, Y):
        with torch.autograd.no_grad():
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
            self.f_alpha = model.alpha_.clone().requires_grad_()
            self.f_alpha_pc = model.beta_.clone()
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


def train_hypergrad(Xtr, Ytr, Xval, Yval, Xts, Yts,
                    penalty_init: float,
                    sigma_type: str,
                    sigma_init: float,
                    num_centers: int,
                    opt_centers: bool,
                    falkon_centers: CenterSelector,
                    num_epochs: int,
                    learning_rate: float,
                    val_loss_type: str,
                    cuda: bool,
                    loss_every: int,
                    err_fn,
                    falkon_opt,
                    ):
    start_sigma = get_start_sigma(sigma_init, sigma_type, Xtr.shape[1])

    model = FalkonHyperGradient(
        penalty_init=penalty_init,
        sigma_init=start_sigma,
        centers_init=falkon_centers.select(Xtr, Y=None, M=num_centers),
        opt_centers=opt_centers,
        flk_opt=falkon_opt,
        val_loss_type=val_loss_type,
        cuda=cuda,
    )
    if cuda:
        Xtr, Ytr, Xts, Yts = Xtr.cuda(), Ytr.cuda(), Xts.cuda(), Yts.cuda()
        Xval, Yval = Xval.cuda(), Yval.cuda()

    outer_opt = torch.optim.Adam(lr=learning_rate, params=model.hparams())

    cum_time = 0
    cum_step = 0
    writer = get_writer()
    for epoch in range(num_epochs):
        e_start = time.time()
        outer_opt.zero_grad()
        # Run inner opt (falkon) to obtain the preconditioner
        model.adapt_alpha(Xtr, Ytr)
        # Run outer opt (Adam) to fill-in the .grad field of hps
        model.adapt_hps(Xtr, Ytr, Xval, Yval, cum_step)
        # Loss reporting before step() (so that our original model stays valid!)
        if epoch != 0 and (epoch + 1) % loss_every == 0:
            cum_time, train_err, test_err, val_err = test_train_predict(
                    model, Xts=Xts, Yts=Yts, Xtr=Xtr, Ytr=Ytr,
                    err_fn=err_fn, epoch=epoch, time_start=e_start,
                    cum_time=cum_time, Xval=Xval, Yval=Yval
            )
            writer.add_scalar('Error/train', train_err, cum_step)
            writer.add_scalar('Error/test', test_err, cum_step)
            writer.add_scalar('Error/validation', val_err, cum_step)
        # Update hp values
        outer_opt.step()
        cum_step += 1
    return model.get_model()
