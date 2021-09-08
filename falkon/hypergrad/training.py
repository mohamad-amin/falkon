import math
import time
from functools import reduce
from typing import Optional, Dict, List, Any

import numpy as np
import torch
from dataclasses import dataclass

from falkon import FalkonOptions, Falkon, InCoreFalkon
from falkon.hypergrad.creg import (
    DeffNoPenFitTr, DeffPenFitTr, StochasticDeffPenFitTr,
    StochasticDeffNoPenFitTr, CompDeffPenFitTr, CompDeffNoPenFitTr,
)
from falkon.hypergrad.hgrad import NystromClosedFormHgrad, NystromIFTHgrad, FalkonClosedFormHgrad
from falkon.hypergrad.common import get_start_sigma, get_scalar
from falkon.hypergrad.complexity_reg import HyperOptimModel, hp_grad
from falkon.hypergrad.gcv import NystromGCV, StochasticNystromGCV
from falkon.hypergrad.loocv import NystromLOOCV
from falkon.hypergrad.sgpr import GPR, SGPR
from falkon.hypergrad.svgp import SVGP
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
        _loss = get_scalar(loss)
        # Report the value of the loss
        writer.add_scalar(f'optim/{loss_name}', _loss, step)
        report_str += f"{loss_name}: {_loss:.3e} - "
        report_dict[f"loss_{loss_name}"] = _loss
        loss_sum += _loss
    if len(losses) > 1:
        report_str += f"tot: {loss_sum:.3e}"
    report_dict["loss"] = loss_sum
    print(report_str, flush=True)
    return report_dict


def report_hps(named_hparams, step) -> Dict[str, float]:
    writer = get_writer()
    report_dict = {}
    for hp_name, hp_val in named_hparams.items():
        hp_val_ = get_scalar(hp_val)
        # Report the hparam value
        writer.add_scalar(f'hparams/{hp_name}', hp_val_, step)
        report_dict[f"hp_{hp_name}"] = hp_val_
    return report_dict


def report_grads(named_hparams, grads, losses, loss_names, step) -> Dict[str, float]:
    assert len(losses) == len(
        loss_names), f"Found {len(losses)} losses and {len(loss_names)} loss-names."
    assert len(grads) == len(losses), f"Found {len(grads)} grads and {len(losses)} losses."
    writer = get_writer()
    report_dict = {}
    for i in range(len(grads)):
        for j, (hp_name, hp_val) in enumerate(named_hparams.items()):
            grad_ = get_scalar(grads[i][j])
            # Report the gradient of a specific loss wrt a specific hparam
            writer.add_scalar(f'grads/{hp_name}/{loss_names[i]}', grad_, step)
            report_dict[f"grad_{hp_name}_{loss_names[i]}"] = grad_
    return report_dict


def grad_loss_reporting(named_hparams, grads, losses, loss_names, verbose, step, losses_are_grads):
    report_dicts = []
    if not losses_are_grads and losses is not None:
        report_dicts.append(report_losses(losses, loss_names, step))
    report_dicts.append(report_hps(named_hparams, step))
    if verbose and grads is not None:
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
                   resolve_model: bool = False,
                   mb_size: Optional[int] = None,
                   ) -> Dict[str, float]:
    writer = get_writer()
    t_elapsed = time.time() - time_start  # Stop the time
    cum_time += t_elapsed
    model.eval()
    sigma, penalty, centers = model.sigma, model.penalty, model.centers

    if resolve_model:
        flk_opt = FalkonOptions(use_cpu=not torch.cuda.is_available(), cg_tolerance=1e-4)
        from falkon.kernels import GaussianKernel
        from falkon.center_selection import FixedSelector
        kernel = GaussianKernel(sigma.detach().flatten(), flk_opt)
        center_selector = FixedSelector(centers.detach())
        if Xtr.is_cuda:
            flk_model = InCoreFalkon(kernel, penalty.item(), M=centers.shape[0],
                                     center_selection=center_selector, maxiter=30,
                                     seed=1312, error_fn=err_fn, error_every=None, options=flk_opt)
        else:
            flk_model = Falkon(kernel, penalty.item(), M=centers.shape[0],
                               center_selection=center_selector, maxiter=30,
                               seed=1312, error_fn=err_fn, error_every=None, options=flk_opt)
        if Xval is not None and Yval is not None:
            Xtr_full, Ytr_full = torch.cat((Xtr, Xval), dim=0), torch.cat((Ytr, Yval), dim=0)
        else:
            Xtr_full, Ytr_full = Xtr, Ytr
        flk_model.fit(Xtr_full, Ytr_full, Xts, Yts)
        model = flk_model

    # Predict in mini-batches
    test_preds, train_preds = [], []
    c_mb_size = mb_size or Xts.shape[0]
    for i in range(0, Xts.shape[0], c_mb_size):
        test_preds.append(model.predict(Xts[i: i + c_mb_size]).detach().cpu())
    c_mb_size = mb_size or Xtr.shape[0]
    for i in range(0, Xtr.shape[0], c_mb_size):
        train_preds.append(model.predict(Xtr[i: i + c_mb_size]).detach().cpu())
    test_preds = torch.cat(test_preds, dim=0)
    train_preds = torch.cat(train_preds, dim=0)
    test_err, err_name = err_fn(Yts.detach().cpu(), test_preds)
    train_err, err_name = err_fn(Ytr.detach().cpu(), train_preds)
    out_str = (f"Epoch {epoch} ({cum_time:5.2f}s) - "
               f"Sigma {get_scalar(sigma):.3f} - Penalty {get_scalar(penalty):.2e} - "
               f"Tr  {err_name} = {train_err:7.5f} - "
               f"Ts  {err_name} = {test_err:7.5f}")
    writer.add_scalar(f'error/{err_name}/train', train_err, epoch)
    writer.add_scalar(f'error/{err_name}/test', test_err, epoch)

    ret = [cum_time, train_err, test_err]
    if Xval is not None and Yval is not None:
        val_preds = model.predict(Xval).detach().cpu()
        val_err, err_name = err_fn(Yval.detach().cpu(), val_preds)
        out_str += f" - Val {err_name} = {val_err:6.4f}"
        ret.append(val_err)
        writer.add_scalar(f'error/{err_name}/val', val_err, epoch)
    print(out_str, flush=True)
    return {
        "cum_time": cum_time,
        f"train_{err_name}": train_err,
        f"test_{err_name}": test_err,
    }


def create_optimizer(opt_type: str, model: HyperOptimModel, learning_rate: float):
    center_lr_div = 1
    named_params = model.named_parameters()
    print("Creating optimizer with the following parameters:")
    for k, v in named_params.items():
        print(f"\t{k} : {v.shape}")
    if opt_type == "adam":
        if 'penalty' not in named_params:
            opt_modules = [
                {"params": named_params.values(), 'lr': learning_rate}
            ]
        else:
            opt_modules = [
                {"params": named_params['penalty'], 'lr': learning_rate},
                {"params": named_params['sigma'], 'lr': learning_rate},
            ]
            if 'centers' in model.named_parameters():
                opt_modules.append({
                    "params": named_params['centers'], 'lr': learning_rate / center_lr_div})
        opt_hp = torch.optim.Adam(opt_modules)
        # schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_hp, factor=0.5, patience=1)
        schedule = torch.optim.lr_scheduler.StepLR(opt_hp, step_size=100, gamma=0.5)
    elif opt_type == "lbfgs":
        if model.losses_are_grads:
            raise ValueError("L-BFGS not valid for model %s" % (model))
        opt_hp = torch.optim.LBFGS(model.parameters(), lr=learning_rate,
                                   history_size=100, )
        schedule = None
    elif opt_type == "rmsprop":
        opt_hp = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
        schedule = None
    else:
        raise ValueError("Optimizer type %s not recognized" % (opt_type))

    return opt_hp, schedule


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
        optimizer: str,
        retrain_nkrr: bool = False,
) -> List[Dict[str, float]]:
    if cuda:
        Xtr, Ytr, Xts, Yts = Xtr.cuda(), Ytr.cuda(), Xts.cuda(), Yts.cuda()
    loss_every = 5
    opt_hp, schedule = create_optimizer(optimizer, model, learning_rate)
    print(f"Starting hyperparameter optimization on model {model}.")
    print(f"Will run for {num_epochs} epochs with {opt_hp} optimizer.")

    logs = []
    cum_time = 0
    t_start = time.time()
    for epoch in range(num_epochs):
        grads: Any = None
        losses: Any = None

        def closure():
            opt_hp.zero_grad()
            nonlocal grads, losses
            losses = model.hp_loss(Xtr, Ytr)
            grads = hp_grad(model, *losses, accumulate_grads=True,
                            losses_are_grads=model.losses_are_grads)
            loss = reduce(torch.add, losses)
            return float(loss)
        opt_hp.step(closure)

        pred_dict = {}
        if epoch != 0 and (epoch + 1) % loss_every == 0:
            pred_dict = pred_reporting(
                model=model, Xtr=Xtr, Ytr=Ytr, Xts=Xts, Yts=Yts,
                err_fn=err_fn, epoch=epoch, time_start=t_start, cum_time=cum_time,
                resolve_model=True)
            cum_time = pred_dict["cum_time"]
            t_start = time.time()
        loss_dict = grad_loss_reporting(model.named_parameters(), grads, losses, model.loss_names,
                                        verbose, epoch, losses_are_grads=model.losses_are_grads)
        loss_dict.update(pred_dict)
        logs.append(loss_dict)
        if schedule is not None:
            if isinstance(schedule, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if 'train_NRMSE' in loss_dict:
                    schedule.step(loss_dict['train_NRMSE'])
            else:
                schedule.step()
        del grads, losses
    if retrain_nkrr:
        print(f"Final retrain after {num_epochs} epochs:")
        pred_dict = pred_reporting(
            model=model, Xtr=Xtr, Ytr=Ytr, Xts=Xts, Yts=Yts,
            err_fn=err_fn, epoch=num_epochs, time_start=time.time(), cum_time=cum_time,
            resolve_model=True)
        logs.append(pred_dict)

    return logs


def train_complexity_reg_mb(
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
        optimizer: str,
        minibatch: int,
        retrain_nkrr: bool = False,
) -> List[Dict[str, float]]:
    Xtrc, Ytrc, Xtsc, Ytsc = Xtr, Ytr, Xts, Yts
    if cuda:
        Xtrc, Ytrc, Xtsc, Ytsc = Xtr.cuda(), Ytr.cuda(), Xts.cuda(), Yts.cuda()
    opt_hp, schedule = create_optimizer(optimizer, model, learning_rate)
    print(f"Starting hyperparameter optimization on model {model}.")
    print(f"Will run for {num_epochs} epochs with {opt_hp} optimizer, "
          f"mini-batch size {minibatch}.")

    logs = []
    cum_time = 0
    t_start = time.time()
    mb_indices = np.arange(Xtr.shape[0])
    for epoch in range(num_epochs):
        np.random.shuffle(mb_indices)
        for mb_start in range(0, Xtr.shape[0], minibatch):
            print("%d " % mb_start, end='', flush=True)
            Xtr_batch = (Xtr[mb_indices[mb_start: mb_start + minibatch], :]).contiguous()
            Ytr_batch = (Ytr[mb_indices[mb_start: mb_start + minibatch], :]).contiguous()
            if cuda:
                Xtr_batch, Ytr_batch = Xtr_batch.cuda(), Ytr_batch.cuda()

            #def closure():
            #    opt_hp.zero_grad()
            #    losses = model.hp_loss(Xtr_batch, Ytr_batch)
                #hp_grad(model, *losses, accumulate_grads=True,
                #        losses_are_grads=model.losses_are_grads)
            #    loss = reduce(torch.add, losses)
            #    loss.backward()
            #    return loss
            #opt_hp.step(closure)

            opt_hp.zero_grad()
            loss = model.hp_loss(Xtr_batch, Ytr_batch)[0]
            loss.backward()
            opt_hp.step()

        pred_dict = {}
        if epoch != 0 and (epoch + 1) % loss_every == 0:
            pred_dict = pred_reporting(
                model=model, Xtr=Xtrc, Ytr=Ytrc, Xts=Xtsc, Yts=Ytsc,
                err_fn=err_fn, epoch=epoch, time_start=t_start, cum_time=cum_time,
                resolve_model=True, mb_size=minibatch)
            cum_time = pred_dict["cum_time"]
            t_start = time.time()
        logs.append(pred_dict)
        if schedule is not None:
            if isinstance(schedule, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if 'train_NRMSE' in pred_dict or 'train_MSE' in pred_dict or 'train_c-error' in pred_dict:
                    schedule.step(pred_dict['train_NRMSE'])
            else:
                schedule.step()
    if retrain_nkrr:
        print(f"Final retrain after {num_epochs} epochs:")
        pred_dict = pred_reporting(
            model=model, Xtr=Xtrc, Ytr=Ytrc, Xts=Xtsc, Yts=Ytsc,
            err_fn=err_fn, epoch=num_epochs, time_start=time.time(), cum_time=cum_time,
            resolve_model=True)
        logs.append(pred_dict)

    return logs


def init_model(model_type, data, penalty_init, sigma_init, centers_init, opt_penalty, opt_sigma,
               opt_centers, sigma_type, cuda, val_pct, cg_tol, num_trace_vecs=20, flk_maxiter=10,
               nystrace_ste=False):
    start_sigma = get_start_sigma(sigma_init, sigma_type, data['X'].shape[1])
    if model_type in {"hgrad-closed", "hgrad-ift", "flk-hgrad-closed"}:
        if val_pct <= 0 or val_pct >= 100:
            raise RuntimeError("val_pct must be between 1 and 99")
        tot = data['X'].shape[0]
        n_val = int(tot * val_pct)
        all_idx = torch.randperm(tot)
        val_idx = all_idx[:n_val]
        tr_idx = all_idx[n_val:]
        print(f"Validation split ({val_pct}): {len(tr_idx)} training points, "
              f"{len(val_idx)} validation points")

    flk_opt = FalkonOptions(cg_tolerance=cg_tol, use_cpu=not torch.cuda.is_available(),
                            cg_full_gradient_every=10, cg_epsilon_32=1e-6)

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
    elif model_type == "stoch-creg-penfit":
        model = StochasticDeffPenFitTr(sigma_init=start_sigma, penalty_init=penalty_init,
                                       centers_init=centers_init, opt_sigma=opt_sigma,
                                       opt_penalty=opt_penalty, opt_centers=opt_centers, cuda=cuda,
                                       flk_opt=flk_opt, num_trace_est=num_trace_vecs,
                                       flk_maxiter=flk_maxiter, nystrace_ste=nystrace_ste)
    elif model_type == "comp-creg-penfit":
        model = CompDeffPenFitTr(sigma_init=start_sigma, penalty_init=penalty_init,
                                 centers_init=centers_init, opt_sigma=opt_sigma,
                                 opt_penalty=opt_penalty, opt_centers=opt_centers, cuda=cuda,
                                 flk_opt=flk_opt, num_trace_est=num_trace_vecs,
                                 flk_maxiter=flk_maxiter, nystrace_ste=nystrace_ste)
    elif model_type == "stoch-creg-nopenfit":
        model = StochasticDeffNoPenFitTr(sigma_init=start_sigma, penalty_init=penalty_init,
                                         centers_init=centers_init, opt_sigma=opt_sigma,
                                         opt_penalty=opt_penalty, opt_centers=opt_centers,
                                         cuda=cuda, flk_opt=flk_opt, num_trace_est=num_trace_vecs,
                                         flk_maxiter=flk_maxiter, nystrace_ste=nystrace_ste)
    elif model_type == "comp-creg-nopenfit":
        model = CompDeffNoPenFitTr(sigma_init=start_sigma, penalty_init=penalty_init,
                                   centers_init=centers_init, opt_sigma=opt_sigma,
                                   opt_penalty=opt_penalty, opt_centers=opt_centers, cuda=cuda,
                                   flk_opt=flk_opt, num_trace_est=num_trace_vecs,
                                   flk_maxiter=flk_maxiter, nystrace_ste=nystrace_ste)
    elif model_type == "stoch-gcv":
        model = StochasticNystromGCV(sigma_init=start_sigma, penalty_init=penalty_init,
                                     centers_init=centers_init, opt_sigma=opt_sigma,
                                     opt_penalty=opt_penalty, opt_centers=opt_centers, cuda=cuda,
                                     flk_opt=flk_opt, num_trace_est=num_trace_vecs,
                                     flk_maxiter=flk_maxiter)
    elif model_type == "flk-hgrad-closed":
        # noinspection PyUnboundLocalVariable
        model = FalkonClosedFormHgrad(sigma_init=sigma_init, penalty_init=penalty_init,
                                      centers_init=centers_init, opt_centers=opt_centers,
                                      opt_sigma=opt_sigma, opt_penalty=opt_penalty, cuda=cuda,
                                      tr_indices=tr_idx, ts_indices=val_idx, flk_opt=flk_opt,
                                      flk_maxiter=flk_maxiter)
    elif model_type == "svgp":
        model = SVGP(sigma_init=start_sigma, penalty_init=penalty_init, centers_init=centers_init,
                     opt_sigma=opt_sigma, opt_penalty=opt_penalty, opt_centers=opt_centers,
                     cuda=cuda, num_data=data['X'].shape[0], multi_class=data['Y'].shape[1])
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
