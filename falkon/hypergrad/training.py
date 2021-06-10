import time
from typing import Optional, Dict, List, Any

import torch
from dataclasses import dataclass

from falkon.hypergrad.creg import DeffNoPenFitTr, DeffPenFitTr
from falkon.hypergrad.hgrad import NystromClosedFormHgrad, NystromIFTHgrad
from falkon.hypergrad.common import get_start_sigma, get_scalar
from falkon.hypergrad.complexity_reg import HyperOptimModel, hp_grad
from falkon.hypergrad.gcv import NystromGCV
from falkon.hypergrad.loocv import NystromLOOCV
from falkon.hypergrad.sgpr import GPR, SGPR
from benchmark.common.summary import get_writer

__all__ = [
    "train_complexity_reg",
    "init_model",
    "run_on_grid",
    "HPGridPoint",
]


def report_losses(losses, loss_names, step) -> Dict[str, float]:
    assert len(losses) == len(
        loss_names), f"Found {len(losses)} losses and {len(loss_names)} loss-names."
    writer = get_writer()
    report_str = "LOSSES: "
    report_dict = {}
    loss_sum = 0
    for loss, loss_name in zip(losses, loss_names):
        # Report the value of the loss
        writer.add_scalar(f'optim/{loss_name}', loss, step)
        report_str += f"{loss_name}: {loss:.3e} - "
        report_dict[f"loss_{loss_name}"] = loss
        loss_sum += loss
    if len(losses) > 1:
        report_str += f"tot: {loss_sum:.3e}"
    report_dict["loss"] = loss_sum
    print(report_str, flush=True)
    return report_dict


def report_hps(named_hparams, step) -> Dict[str, float]:
    writer = get_writer()
    report_dict = {}
    for hp_name, hp_val in named_hparams.items():
        # Report the hparam value
        writer.add_scalar(f'hparams/{hp_name}', get_scalar(hp_val), step)
        report_dict[f"hp_{hp_name}"] = get_scalar(hp_val)
    return report_dict


def report_grads(named_hparams, grads, losses, loss_names, step) -> Dict[str, float]:
    assert len(losses) == len(
        loss_names), f"Found {len(losses)} losses and {len(loss_names)} loss-names."
    assert len(grads) == len(losses), f"Found {len(grads)} grads and {len(losses)} losses."
    writer = get_writer()
    report_dict = {}
    for i in range(grads):
        for j, (hp_name, hp_val) in enumerate(named_hparams.items()):
            # Report the gradient of a specific loss wrt a specific hparam
            writer.add_scalar(f'grads/{hp_name}/{loss_names[i]}', get_scalar(grads[i][j]), step)
            report_dict[f"grad_{hp_name}_{loss_names[i]}"] = get_scalar(grads[i][j])
    return report_dict


def grad_loss_reporting(named_hparams, grads, losses, loss_names, verbose, step, losses_are_grads):
    report_dicts = []
    if not losses_are_grads:
        report_dicts.append(report_losses(losses, loss_names, step))
    report_dicts.append(report_hps(named_hparams, step))
    if verbose:
        report_dicts.append(report_grads(named_hparams, grads, losses, loss_names, step))
    report_dict = {}
    for rd in report_dicts:
        report_dict.update(rd)
    return report_dict


def pred_reporting(model: HyperOptimModel,
                   Xts, Yts,
                   Xtr, Ytr,
                   err_fn: callable,
                   epoch: int,
                   time_start: float,
                   cum_time: float,
                   Xval: Optional[torch.Tensor] = None,
                   Yval: Optional[torch.Tensor] = None,
                   ) -> Dict[str, float]:
    writer = get_writer()
    t_elapsed = time.time() - time_start  # Stop the time
    cum_time += t_elapsed
    model.eval()

    test_preds = model.predict(Xts)
    train_preds = model.predict(Xtr)
    test_err, err_name = err_fn(Yts.detach().cpu(), test_preds.detach().cpu())
    train_err, err_name = err_fn(Ytr.detach().cpu(), train_preds.detach().cpu())
    out_str = (f"Epoch {epoch} ({cum_time:5.2f}s) - "
               f"Sigma {get_scalar(model.sigma):.3f} - Penalty {get_scalar(model.penalty):.2e} - "
               f"Tr  {err_name} = {train_err:6.4f} - "
               f"Ts  {err_name} = {test_err:6.4f}")
    writer.add_scalar(f'error/{err_name}/train', train_err, epoch)
    writer.add_scalar(f'error/{err_name}/test', test_err, epoch)

    ret = [cum_time, train_err, test_err]
    if Xval is not None and Yval is not None:
        val_preds = model.predict(Xval)
        val_err, err_name = err_fn(Yval.detach().cpu(), val_preds.detach().cpu())
        out_str += f" - Val {err_name} = {val_err:6.4f}"
        ret.append(val_err)
        writer.add_scalar(f'error/{err_name}/val', val_err, epoch)
    print(out_str, flush=True)
    return {
        "cum_time": cum_time,
        f"train_{err_name}": train_err,
        f"test_{err_name}": test_err,
    }


def train_complexity_reg(
        Xtr: torch.Tensor,
        Ytr: torch.Tensor,
        Xts: torch.Tensor,
        Yts: torch.Tensor,
        model: HyperOptimModel,
        err_fn,
        learning_rate: float,
        num_epochs: int,
        cuda: bool,
        verbose: bool,
        loss_every: int,
) -> List[Dict[str, float]]:
    if cuda:
        Xtr, Ytr, Xts, Yts = Xtr.cuda(), Ytr.cuda(), Xts.cuda(), Yts.cuda()
    opt_hp = torch.optim.Adam([
        {"params": model.parameters(), "lr": learning_rate},
    ])
    print(f"Starting hyperparameter optimization on model {model}.")
    print(f"Will run for {num_epochs} epochs with {opt_hp} optimizer.")

    logs = []
    cum_time = 0
    for epoch in range(num_epochs):
        e_start = time.time()
        opt_hp.zero_grad()
        losses = model.hp_loss(Xtr, Ytr)
        # Loss reporting before grad() to optimize
        pred_dict = {}
        if epoch != 0 and (epoch + 1) % loss_every == 0:
            pred_dict = pred_reporting(
                model=model, Xtr=Xtr, Ytr=Ytr, Xts=Xts, Yts=Yts,
                err_fn=err_fn, epoch=epoch, time_start=e_start, cum_time=cum_time)
            cum_time = pred_dict["cum_time"]
        grads = hp_grad(*losses, accumulate_grads=True, losses_are_grads=model.losses_are_grads)
        loss_dict = grad_loss_reporting(model.named_parameters(), grads, losses, model.loss_names,
                                        verbose, epoch, losses_are_grads=model.losses_are_grads)
        opt_hp.step()
        loss_dict.update(pred_dict)
        logs.append(loss_dict)
    return logs


def init_model(model_type, data, penalty_init, sigma_init, centers_init, opt_penalty, opt_sigma,
               opt_centers, sigma_type, cuda, val_pct, cg_tol):
    start_sigma = get_start_sigma(sigma_init, sigma_type, data['X'].shape[1])
    if model_type in {"hgrad-closed", "hgrad-ift"}:
        if val_pct <= 0 or val_pct >= 100:
            raise RuntimeError("val_pct must be between 1 and 99")
        tot = data['X'].shape[0]
        n_val = int(tot * val_pct)
        all_idx = torch.randperm(tot)
        val_idx = all_idx[:n_val]
        tr_idx = all_idx[n_val:]

    if model_type == "gpr":
        model = GPR(sigma_init=start_sigma, penalty_init=penalty_init,
                    opt_sigma=opt_sigma, opt_penalty=opt_penalty,
                    cuda=cuda)
    elif model_type == "sgpr":
        model = SGPR(sigma_init=start_sigma, penalty_init=penalty_init, centers_init=centers_init,
                     opt_sigma=opt_sigma, opt_penalty=opt_penalty, opt_centers=opt_centers,
                     cuda=cuda)
    elif model_type == "gcv":
        model = NystromGCV(sigma_init=start_sigma, penalty_init=penalty_init,
                           centers_init=centers_init,
                           opt_sigma=opt_sigma, opt_penalty=opt_penalty, opt_centers=opt_centers,
                           cuda=cuda)
    elif model_type == "loocv":
        model = NystromLOOCV(sigma_init=start_sigma, penalty_init=penalty_init,
                             centers_init=centers_init,
                             opt_sigma=opt_sigma, opt_penalty=opt_penalty, opt_centers=opt_centers,
                             cuda=cuda)
    elif model_type == "hgrad-closed":
        # noinspection PyUnboundLocalVariable
        model = NystromClosedFormHgrad(sigma_init=sigma_init, penalty_init=penalty_init,
                                       centers_init=centers_init, opt_centers=opt_centers,
                                       opt_sigma=opt_sigma, opt_penalty=opt_penalty, cuda=cuda,
                                       tr_indices=tr_idx, ts_indices=val_idx)
    elif model_type == "hgrad-ift":
        # noinspection PyUnboundLocalVariable
        model = NystromIFTHgrad(sigma_init=sigma_init, penalty_init=penalty_init,
                                centers_init=centers_init, opt_centers=opt_centers,
                                opt_sigma=opt_sigma, opt_penalty=opt_penalty, cuda=cuda,
                                tr_indices=tr_idx, ts_indices=val_idx, cg_tol=cg_tol)
    elif model_type == "creg-penfit":
        model = DeffPenFitTr(sigma_init=start_sigma, penalty_init=penalty_init,
                             centers_init=centers_init, opt_sigma=opt_sigma,
                             opt_penalty=opt_penalty, opt_centers=opt_centers, cuda=cuda)
    elif model_type == "creg-nopenfit":
        model = DeffNoPenFitTr(sigma_init=start_sigma, penalty_init=penalty_init,
                               centers_init=centers_init, opt_sigma=opt_sigma,
                               opt_penalty=opt_penalty, opt_centers=opt_centers, cuda=cuda)
    else:
        raise RuntimeError(f"{model_type} model type not recognized!")

    return model


@dataclass
class HPGridPoint:
    attributes: Dict[str, Any]
    results: Optional[Dict[str, float]] = None


def set_grid_point(model: HyperOptimModel, grid_point: HPGridPoint):
    for attr_name, attr_val in grid_point.attributes.items():
        setattr(model, attr_name, attr_val)


def run_on_grid(
        Xtr: torch.Tensor,
        Ytr: torch.Tensor,
        Xts: torch.Tensor,
        Yts: torch.Tensor,
        model: HyperOptimModel,
        grid_spec: List[HPGridPoint],
        err_fn,
        cuda: bool):
    if cuda:
        Xtr, Ytr, Xts, Yts = Xtr.cuda(), Ytr.cuda(), Xts.cuda(), Yts.cuda()

    print(f"Starting grid-search on model {model}.")
    print(f"Will run for {len(grid_spec)} points.")

    loss_report = {}
    cum_time = 0
    for i, grid_point in enumerate(grid_spec):
        e_start = time.time()
        set_grid_point(model, grid_point)
        losses = model.hp_loss(Xtr, Ytr)
        if not model.losses_are_grads:
            loss_report = report_losses(losses, model.loss_names, i)
        pred_report = pred_reporting(
            model=model, Xtr=Xtr, Ytr=Ytr, Xts=Xts, Yts=Yts,
            err_fn=err_fn, epoch=i, time_start=e_start, cum_time=cum_time)
        cum_time = pred_report["cum_time"]
        grid_point.results = {}
        grid_point.results.update(loss_report)
        grid_point.results.update(pred_report)

    return grid_spec
