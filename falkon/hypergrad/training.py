import time
from dataclasses import dataclass
from functools import reduce
from typing import Optional, Dict, List, Any

import numpy as np
import torch

from benchmark.common.summary import get_writer
from falkon import FalkonOptions, Falkon, InCoreFalkon
from falkon.hypergrad.common import get_start_sigma, get_scalar
from falkon.hypergrad.complexity_reg import HyperOptimModel, hp_grad
from falkon.hypergrad.creg import (
    DeffNoPenFitTr, DeffPenFitTr, StochasticDeffPenFitTr,
    StochasticDeffNoPenFitTr, CompDeffPenFitTr, CompDeffNoPenFitTr,
    CregNoTrace,
)
from falkon.hypergrad.gcv import NystromGCV, StochasticNystromGCV
from falkon.hypergrad.hgrad import NystromClosedFormHgrad, NystromIFTHgrad, FalkonClosedFormHgrad
from falkon.hypergrad.loocv import NystromLOOCV
from falkon.hypergrad.sgpr import GPR, SGPR
from falkon.hypergrad.svgp import SVGP

__all__ = [
    "train_complexity_reg",
    "init_model",
    "run_on_grid",
    "fetch_loss",
    "HPGridPoint",
]

LOSS_EVERY = 5
EARLY_STOP_EPOCHS = 31


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
    report_dict = {}
    # Losses
    if not losses_are_grads and losses is not None:
        report_dict.update(report_losses(losses, loss_names, step))
    # Hyperparameters
    report_dict.update(report_hps(named_hparams, step))
    # Gradients
    if verbose and grads is not None:
        report_dict.update(report_grads(named_hparams, grads, losses, loss_names, step))
    return report_dict


def pred_reporting(model: HyperOptimModel,
                   Xts, Yts,
                   Xtr, Ytr,
                   err_fn: callable,
                   epoch: int,
                   cum_time: float,
                   Xval: Optional[torch.Tensor] = None,
                   Yval: Optional[torch.Tensor] = None,
                   resolve_model: bool = False,
                   mb_size: Optional[int] = None,
                   ) -> Dict[str, float]:
    writer = get_writer()
    model.eval()
    sigma, penalty, centers = model.sigma, model.penalty, model.centers

    if resolve_model:
        flk_opt = FalkonOptions(use_cpu=not torch.cuda.is_available(), cg_tolerance=1e-4, cg_epsilon_32=1e-6)
        from falkon.kernels import GaussianKernel
        from falkon.center_selection import FixedSelector
        kernel = GaussianKernel(sigma.detach().flatten(), flk_opt)
        center_selector = FixedSelector(centers.detach())
        flk_cls = InCoreFalkon if Xtr.is_cuda else Falkon
        flk_model = flk_cls(kernel, penalty.item(), M=centers.shape[0],
                            center_selection=center_selector, maxiter=100,
                            seed=1312, error_fn=err_fn, error_every=None, options=flk_opt)
        Xtr_full, Ytr_full = Xtr, Ytr
        if Xval is not None and Yval is not None:
            Xtr_full, Ytr_full = torch.cat((Xtr, Xval), dim=0), torch.cat((Ytr, Yval), dim=0)
        warm_start = None
        if hasattr(model, "last_beta"):
            warm_start = model.last_beta.to(Xtr_full.device)
        flk_model.fit(Xtr_full, Ytr_full, warm_start=warm_start)#, Xts, Yts)
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
               f"Tr  {err_name} = {train_err:9.7f} - "
               f"Ts  {err_name} = {test_err:9.7f}")
    writer.add_scalar(f'error/{err_name}/train', train_err, epoch)
    writer.add_scalar(f'error/{err_name}/test', test_err, epoch)

    if Xval is not None and Yval is not None:
        val_preds = model.predict(Xval).detach().cpu()
        val_err, err_name = err_fn(Yval.detach().cpu(), val_preds)
        out_str += f" - Val {err_name} = {val_err:6.4f}"
        writer.add_scalar(f'error/{err_name}/val', val_err, epoch)
    print(out_str, flush=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        f"train_{err_name}": train_err,
        f"test_{err_name}": test_err,
        "train_error": train_err,
        "test_error": test_err,
    }


def create_optimizer(opt_type: str, model: HyperOptimModel, learning_rate: float):
    center_lr_div = 1
    schedule = None
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
            opt_modules = []
            if 'sigma' in named_params:
                opt_modules.append({"params": named_params['sigma'], 'lr': learning_rate})
            if 'penalty' in named_params:
                opt_modules.append({"params": named_params['penalty'], 'lr': learning_rate})
            if 'centers' in named_params:
                opt_modules.append({
                    "params": named_params['centers'], 'lr': learning_rate / center_lr_div})
        opt_hp = torch.optim.Adam(opt_modules)
        # schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_hp, factor=0.5, patience=1)
        schedule = torch.optim.lr_scheduler.StepLR(opt_hp, step_size=200, gamma=0.3)
    elif opt_type == "sgd":
        opt_hp = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif opt_type == "lbfgs":
        if model.losses_are_grads:
            raise ValueError("L-BFGS not valid for model %s" % (model))
        opt_hp = torch.optim.LBFGS(model.parameters(), lr=learning_rate,
                                   history_size=100, )
    elif opt_type == "rmsprop":
        opt_hp = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    else:
        raise ValueError("Optimizer type %s not recognized" % (opt_type))

    return opt_hp, schedule


class EarlyStop(Exception):
    def __init__(self, msg):
        super(EarlyStop, self).__init__(msg)


def epoch_bookkeeping(
        epoch: int,
        model: HyperOptimModel,
        data: Dict[str, torch.Tensor],
        err_fn,
        grads,
        losses,
        loss_every: int,
        early_stop_patience: Optional[int],
        schedule,
        minibatch: Optional[int],
        logs: list,
        cum_time: float,
        verbose):
    Xtr, Ytr, Xts, Yts = data['Xtr'], data['Ytr'], data['Xts'], data['Yts']

    loss_dict = grad_loss_reporting(model.named_parameters(), grads, losses, model.loss_names,
                                    verbose, epoch, losses_are_grads=model.losses_are_grads)
    if epoch != 0 and (epoch + 1) % loss_every == 0:
        pred_dict = pred_reporting(
            model=model, Xtr=Xtr, Ytr=Ytr, Xts=Xts, Yts=Yts,
            err_fn=err_fn, epoch=epoch, cum_time=cum_time,
            resolve_model=True, mb_size=minibatch)
        if hasattr(model, "print_times"):
            model.print_times()
        loss_dict.update(pred_dict)
    logs.append(loss_dict)
    # Learning rate schedule
    if schedule is not None:
        if isinstance(schedule, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if 'train_error' in loss_dict:
                schedule.step(loss_dict['train_error'])
        else:
            schedule.step()
    # Early stop if no training-error improvement in the past `early_stop_patience` epochs.
    if early_stop_patience is not None and len(logs) >= early_stop_patience:
        if "train_error" in logs[-1]:
            past_errs = []
            past_logs = logs[-early_stop_patience:]  # Last n logs from most oldest to most recent
            for plog in past_logs:
                if 'train_error' in plog:
                    past_errs.append(abs(plog['train_error']))
            if np.argmin(past_errs) == 0:  # The minimal error in the oldest log
                raise EarlyStop(f"Early stopped at epoch {epoch} with past errors: {past_errs}.")


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
    loss_every = LOSS_EVERY
    early_stop_epochs = EARLY_STOP_EPOCHS
    opt_hp, schedule = create_optimizer(optimizer, model, learning_rate)
    print(f"Starting hyperparameter optimization on model {model}.")
    print(f"Will run for {num_epochs} epochs with {opt_hp} optimizer.")

    logs = []
    cum_time = 0
    with torch.autograd.profiler.profile(enabled=False) as prof:
        for epoch in range(num_epochs):
            t_start = time.time()
            grads: Any = None
            losses: Any = None

            def closure():
                opt_hp.zero_grad()
                nonlocal grads, losses
                losses = model.hp_loss(Xtr, Ytr)
                grads = hp_grad(model, *losses, accumulate_grads=True,
                                losses_are_grads=model.losses_are_grads, verbose=False)
                loss = reduce(torch.add, losses)
                return float(loss)
            opt_hp.step(closure)

            cum_time += time.time() - t_start
            try:
                epoch_bookkeeping(epoch=epoch, model=model, data={'Xtr': Xtr, 'Ytr': Ytr, 'Xts': Xts, 'Yts': Yts},
                                  err_fn=err_fn, grads=grads, losses=losses, loss_every=loss_every,
                                  early_stop_patience=early_stop_epochs, schedule=schedule, minibatch=None,
                                  logs=logs, cum_time=cum_time, verbose=verbose)
            except EarlyStop as e:
                print(e)
                break
            finally:
                del grads, losses
    if prof is not None:
        print(prof.key_averages().table())
    if retrain_nkrr:
        print(f"Final retrain after {num_epochs} epochs:")
        pred_dict = pred_reporting(
            model=model, Xtr=Xtr, Ytr=Ytr, Xts=Xts, Yts=Yts,
            err_fn=err_fn, epoch=num_epochs, cum_time=cum_time,
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
    loss_every = LOSS_EVERY
    early_stop_epochs = EARLY_STOP_EPOCHS
    opt_hp, schedule = create_optimizer(optimizer, model, learning_rate)
    print(f"Starting hyperparameter optimization on model {model}.")
    print(f"Will run for {num_epochs} epochs with {opt_hp} optimizer, "
          f"mini-batch size {minibatch}.")

    logs = []
    cum_time = 0
    mb_indices = np.arange(Xtr.shape[0])
    for epoch in range(num_epochs):
        t_start = time.time()
        np.random.shuffle(mb_indices)
        for mb_start in range(0, Xtr.shape[0], minibatch):
            Xtr_batch = (Xtr[mb_indices[mb_start: mb_start + minibatch], :]).contiguous()
            Ytr_batch = (Ytr[mb_indices[mb_start: mb_start + minibatch], :]).contiguous()
            if cuda:
                Xtr_batch, Ytr_batch = Xtr_batch.cuda(), Ytr_batch.cuda()

            opt_hp.zero_grad()
            loss = model.hp_loss(Xtr_batch, Ytr_batch)[0]  # There is only one loss!
            loss.backward()
            opt_hp.step()

        cum_time += time.time() - t_start
        try:
            epoch_bookkeeping(epoch=epoch, model=model, data={'Xtr': Xtrc, 'Ytr': Ytrc, 'Xts': Xtsc, 'Yts': Ytsc},
                              err_fn=err_fn, grads=None, losses=None, loss_every=loss_every,
                              early_stop_patience=early_stop_epochs, schedule=schedule, minibatch=minibatch,
                              logs=logs, cum_time=cum_time, verbose=verbose)
        except EarlyStop as e:
            print(e)
            break
    if retrain_nkrr:
        print(f"Final retrain after {num_epochs} epochs:")
        pred_dict = pred_reporting(
            model=model, Xtr=Xtrc, Ytr=Ytrc, Xts=Xtsc, Yts=Ytsc,
            err_fn=err_fn, epoch=num_epochs, cum_time=cum_time,
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
                     cuda=cuda, no_log_det=False)
    elif model_type == "sgpr-nologdet":
        model = SGPR(sigma_init=start_sigma, penalty_init=penalty_init, centers_init=centers_init,
                     opt_sigma=opt_sigma, opt_penalty=opt_penalty, opt_centers=opt_centers,
                     cuda=cuda, no_log_det=True)
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
    elif model_type == "creg-notrace":
        model = CregNoTrace(sigma_init=start_sigma, penalty_init=penalty_init,
                            centers_init=centers_init, opt_sigma=opt_sigma,
                            opt_penalty=opt_penalty, opt_centers=opt_centers, cuda=cuda)
    elif model_type == "creg-penfit":
        model = DeffPenFitTr(sigma_init=start_sigma, penalty_init=penalty_init,
                             centers_init=centers_init, opt_sigma=opt_sigma,
                             opt_penalty=opt_penalty, opt_centers=opt_centers, cuda=cuda)
    elif model_type == "creg-nopenfit":
        model = DeffNoPenFitTr(sigma_init=start_sigma, penalty_init=penalty_init,
                               centers_init=centers_init, opt_sigma=opt_sigma,
                               opt_penalty=opt_penalty, opt_centers=opt_centers, cuda=cuda,
                               div_trace_by_lambda=False, div_trdeff_by_lambda=False)
    elif model_type == "creg-nopenfit-divtr":
        model = DeffNoPenFitTr(sigma_init=start_sigma, penalty_init=penalty_init,
                               centers_init=centers_init, opt_sigma=opt_sigma,
                               opt_penalty=opt_penalty, opt_centers=opt_centers, cuda=cuda,
                               div_trace_by_lambda=True, div_trdeff_by_lambda=False)
    elif model_type == "creg-nopenfit-divdeff":
        model = DeffNoPenFitTr(sigma_init=start_sigma, penalty_init=penalty_init,
                               centers_init=centers_init, opt_sigma=opt_sigma,
                               opt_penalty=opt_penalty, opt_centers=opt_centers, cuda=cuda,
                               div_deff_by_lambda=True)
    elif model_type == "creg-nopenfit-divtrdeff":
        model = DeffNoPenFitTr(sigma_init=start_sigma, penalty_init=penalty_init,
                               centers_init=centers_init, opt_sigma=opt_sigma,
                               opt_penalty=opt_penalty, opt_centers=opt_centers, cuda=cuda,
                               div_trdeff_by_lambda=True, div_trace_by_lambda=False)
    elif model_type == "creg-nopenfit-divmul":
        model = DeffNoPenFitTr(sigma_init=start_sigma, penalty_init=penalty_init,
                               centers_init=centers_init, opt_sigma=opt_sigma,
                               opt_penalty=opt_penalty, opt_centers=opt_centers, cuda=cuda,
                               div_trdeff_by_lambda=False, div_trace_by_lambda=False,
                               div_mul_lambda=True)
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
        minibatch: Optional[int],
        err_fn,
        cuda: bool):
    if cuda:
        Xtr, Ytr, Xts, Yts = Xtr.cuda(), Ytr.cuda(), Xts.cuda(), Yts.cuda()

    print(f"Starting grid-search on model {model}.")
    print(f"Will run for {len(grid_spec)} points.")

    if minibatch is None or minibatch <= 0:
        minibatch = Xtr.shape[0]
    cum_time = 0
    for i, grid_point in enumerate(grid_spec):
        e_start = time.time()
        set_grid_point(model, grid_point)
        losses = [0.0] * len(model.loss_names)
        for mb_start in range(0, Xtr.shape[0], minibatch):
            Xtr_batch = Xtr[mb_start: mb_start + minibatch, :]
            Ytr_batch = Ytr[mb_start: mb_start + minibatch, :]
            mb_losses = model.hp_loss(Xtr_batch, Ytr_batch)
            for lidx in range(len(mb_losses)):
                losses[lidx] += get_scalar(mb_losses[lidx])
        cum_time += time.time() - e_start
        grid_point.results = pred_reporting(
            model=model, Xtr=Xtr, Ytr=Ytr, Xts=Xts, Yts=Yts, resolve_model=True,
            err_fn=err_fn, epoch=i, cum_time=cum_time, mb_size=minibatch)
        if not model.losses_are_grads:
            grid_point.results.update(report_losses(losses, model.loss_names, i))
    return grid_spec


def fetch_loss(
        Xtr: torch.Tensor,
        Ytr: torch.Tensor,
        Xts: torch.Tensor,
        Yts: torch.Tensor,
        model: HyperOptimModel,
        ):
    train_losses = model.hp_loss(Xtr, Ytr)
    return train_losses, model.loss_names
