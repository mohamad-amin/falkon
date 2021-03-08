from functools import partial
from typing import Sequence, Tuple, List, Dict
import time

import torch

from falkon.hypergrad.common import AbsHypergradModel

def direct_trgrad(params, hparams, model, set_grad):
    time_s = time.time()
    # call `detach` to avoid any residual gradient in the parameters to affect
    # the validation loss gradients
    params = {k: w.detach().requires_grad_(True) for k, w in params.items()}
    grad_outer_params, grad_outer_hparams = model.val_loss_grads(params, hparams)
    val_time = time.time()

    final_grads = []
    for ohp in grad_outer_hparams:
        if ohp is not None:
            final_grads.append(ohp)
        else:
            raise RuntimeError("Validation error must depend on all hparams for the direct method")

    if set_grad:
        for l, g in zip(list(hparams.values()), final_grads):
            if l.grad is None:
                l.grad = torch.zeros_like(l)
            if g is not None:
                l.grad += g

    return model.val_loss(params, hparams), final_grads


def direct_valgrad(params, hparams, model, set_grad):
    time_s = time.time()
    # call `detach` to avoid any residual gradient in the parameters to affect
    # the validation loss gradients
    params = {k: w.detach().requires_grad_(True) for k, w in params.items()}
    grad_outer_params, grad_outer_hparams = model.val_loss_grads(params, hparams)
    val_time = time.time()

    final_grads = []
    for ohp in grad_outer_hparams:
        if ohp is not None:
            final_grads.append(ohp)
        else:
            raise RuntimeError("Validation error must depend on all hparams for the direct method")

    if set_grad:
        for l, g in zip(list(hparams.values()), final_grads):
            if l.grad is None:
                l.grad = torch.zeros_like(l)
            if g is not None:
                l.grad += g

    return model.val_loss(params, hparams), final_grads



def compute_hypergrad(params: Dict[str, torch.Tensor],
                      hparams: Dict[str, torch.Tensor],
                      model: AbsHypergradModel,
                      cg_steps: int,
                      cg_tol: float = 1e-4,
                      set_grad: bool = True,
                      timings: bool = False,
                      ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    r"""Compute the gradient of the model with respect to the hyperparameters on the validation loss

    We are given a training loss $\mathcal{L}_T$ and a validation loss $\mathcal{L}_V$, a set of
    parameters which solve the training problem (exactly, or approximately), and a set of hyperparameters,
    with respect to which we calculate the gradients.
    The gradient of the validation loss with respect to hyperparameters, going through the training
    procedure can be simplified with the implicit function theorem.

    Parameters
    ----------
    params : Dict[str, torch.Tensor]
        List of model parameter vectors
    hparams : Dict[str, torch.Tensor]
        List of hyperparameter vectors for the model
    model : AbsHypergradModel
        The hypergradient-enabled model. The model implements all the necessary functions to calculate
        the hyperparameter gradient.
    cg_steps : int
        The number of conjugate gradient steps for calculating the hessian-vector product
        needed by the hypergradient procedure.
    cg_tol : float
        The tolerance in the conjugate gradient algorithm. If the residual-norm changes less than
        this value between iterations, the algorithm will be early-stopped.
    set_grad : bool
        Whether to set the `grad` attribute on the hyperparameter tensors after it has been
        calculated.

    Returns
    -------
    val_loss : scalar torch.Tensor
        The validation loss obtained by the current set of hyperparameters
    h_grads : List[torch.Tensor]
        The hyperparameter gradients
    """
    time_s = time.time()
    # call `detach` to avoid any residual gradient in the parameters to affect
    # the validation loss gradients
    params = {k: w.detach().requires_grad_(True) for k, w in params.items()}
    grad_outer_params, grad_outer_hparams = model.val_loss_grads(params, hparams)
    val_time = time.time()

    # First derivative of training loss wrt params
    first_diff = model.param_derivative(params, hparams)
    first_diff_time = time.time()

    # Calculate the Hessian multiplied by the outer-gradient wrt alpha
    vs, cg_iter_completed, hvp_time = model.solve_hessian(params, hparams, vector=grad_outer_params, max_iter=cg_steps, cg_tol=cg_tol)
    #hvp = partial(model.hessian_vector_product, params, first_diff)
    #vs, cg_iter_completed, hvp_time = cg(hvp, grad_outer_params, max_iter=cg_steps, epsilon=cg_tol)
    cg_time = time.time()

    # Multiply the mixed inner gradient by `vs`
    grads = model.mixed_vector_product(hparams, first_derivative=first_diff, vector=vs)
    mixed_time = time.time()

    final_grads = []
    for ohp, g in zip(grad_outer_hparams, grads):
        if ohp is not None:
            final_grads.append(ohp - g)
        else:
            final_grads.append(-g)

    if set_grad:
        for l, g in zip(list(hparams.values()), final_grads):
            if l.grad is None:
                l.grad = torch.zeros_like(l)
            if g is not None:
                l.grad += g
    end_time = time.time()
    if timings:
        print(f"Total time: {end_time - time_s:.2f}s - val-grad {val_time - time_s:.2f}s - "
              f"param-diff {first_diff_time - val_time:.2f}s - cg-time({cg_iter_completed}) {cg_time - first_diff_time:.2f}s (1 hvp: {hvp_time:.2f}s) - "
              f"mixed-time {mixed_time - cg_time:.2f}s")

    return model.val_loss(params, hparams), final_grads


def cg(Ax, b, x0=None, max_iter=100, epsilon=1.0e-5):
    """ Conjugate Gradient
      Args:
        Ax: function, takes list of tensors as input
        b: list of tensors
      Returns:
        x_star: list of tensors
    """
    app_times = []
    if x0 is None:
        x_last = [torch.zeros_like(bb) for bb in b]
        r_last = [torch.zeros_like(bb).copy_(bb) for bb in b]
    else:
        x_last = x0
        mmvs = Ax(x0)
        r_last = [bb - mmmvs for (bb, mmmvs) in zip(b, mmvs)]
    p_last = [torch.zeros_like(rr).copy_(rr) for rr in r_last]
    for ii in range(max_iter):
        t_s = time.time()
        Ap = Ax(p_last)
        app_times.append(time.time() - t_s)
        Ap_vec = cat_list_to_tensor(Ap)
        p_last_vec = cat_list_to_tensor(p_last)
        r_last_vec = cat_list_to_tensor(r_last)
        rTr = torch.sum(r_last_vec * r_last_vec)
        pAp = torch.sum(p_last_vec * Ap_vec)
        alpha = rTr / pAp

        x = [xx + alpha * pp for xx, pp in zip(x_last, p_last)]
        r = [rr - alpha * pp for rr, pp in zip(r_last, Ap)]
        r_vec = cat_list_to_tensor(r)

        if float(torch.norm(r_vec)) < epsilon:
            break

        beta = torch.sum(r_vec * r_vec) / rTr
        p = [rr + beta * pp for rr, pp in zip(r, p_last)]

        x_last = x
        p_last = p
        r_last = r

    return x_last, ii, min(app_times)


def cat_list_to_tensor(list_tx):
    return torch.cat([xx.view([-1]) for xx in list_tx])
