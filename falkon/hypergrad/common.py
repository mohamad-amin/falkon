import abc
import time
from typing import Sequence, Optional, Tuple, Union, Dict

import numpy as np
import torch
import torch.distributions.constraints as constraints


class TransformedParameter():
    def __init__(self, value, transform):
        self.transform = transform
        self.value = self.transform.forward(value)

    def __get__(self):
        return self.value

    def __repr__(self):
        return self.value.__repr__()

    def __str__(self):
        return self.value.__repr__()


class ExpTransform(torch.distributions.transforms.Transform):
    _cache_size = 0
    domain = constraints.real
    codomain = constraints.positive

    def __init__(self):
        super().__init__()

    def __eq__(self, other):
        return isinstance(other, ExpTransform)

    def _call(self, x):
        return torch.exp(x)

    def _inverse(self, y):
        return torch.log(y)



class PositiveTransform(torch.distributions.transforms.Transform):
    _cache_size = 0
    domain = constraints.real
    codomain = constraints.positive

    def __init__(self, lower_bound=0.0):
        super().__init__()
        self.lower_bound = lower_bound

    def __eq__(self, other):
        if not isinstance(other, PositiveTransform):
            return False
        return other.lower_bound == self.lower_bound

    def _call(self, x):
        # softplus and then shift
        y = torch.nn.functional.softplus(x)
        y = y + self.lower_bound
        return y

    def _inverse(self, y):
        # https://github.com/tensorflow/probability/blob/v0.12.2/tensorflow_probability/python/math/generic.py#L456-L507
        x = y - self.lower_bound

        threshold = torch.log(torch.tensor(torch.finfo(y.dtype).eps, dtype=y.dtype)) + 2.
        is_too_small = x < torch.exp(threshold)
        is_too_large = x > -threshold
        too_small_val = torch.log(x)
        too_large_val = x

        x = torch.where(is_too_small | is_too_large, torch.tensor(1.0, dtype=y.dtype), x)
        x = x + torch.log(-torch.expm1(-x))
        return torch.where(is_too_small,
                           too_small_val,
                           torch.where(is_too_large, too_large_val, x))


class FastTensorDataLoader:
    def __init__(self, *tensors, batch_size, shuffle=False, drop_last=False, cuda=False):
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors
        self.num_points = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.cuda = cuda

        n_batches, remainder = divmod(self.num_points, self.batch_size)
        if remainder > 0 and not drop_last:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            self.indices = torch.randperm(self.num_points)
        else:
            self.indices = None
        self.i = 0
        return self

    def __next__(self):
        try:
            if self.i >= self.n_batches:  # This should handle drop_last correctly
                raise StopIteration()
        except AttributeError:
            raise RuntimeError(
                "Make sure you make the tensor data-loader an iterator before iterating over it!")

        if self.indices is not None:
            indices = self.indices[self.i * self.batch_size: (self.i + 1) * self.batch_size]
            batch = tuple(t[indices] for t in self.tensors)
        else:
            batch = tuple(
                t[self.i * self.batch_size: (self.i + 1) * self.batch_size] for t in self.tensors)
        if self.cuda:
            batch = tuple(t.cuda() for t in batch)
        self.i += 1
        return batch

    def __len__(self):
        return self.n_batches


class AbsHypergradModel(abc.ABC):
    @abc.abstractmethod
    def val_loss(self,
                 params: Dict[str, torch.Tensor],
                 hparams: Dict[str, torch.Tensor]) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def param_derivative(self,
                         params: Dict[str, torch.Tensor],
                         hparams: Dict[str, torch.Tensor]) -> Sequence[torch.Tensor]:
        pass

    @abc.abstractmethod
    def hessian_vector_product(self,
                               params: Dict[str, torch.Tensor],
                               first_derivative: Sequence[torch.Tensor],
                               vector: Union[torch.Tensor, Sequence[torch.Tensor]]) -> \
                                Sequence[torch.Tensor]:
        pass

    @abc.abstractmethod
    def mixed_vector_product(self,
                             hparams: Dict[str, torch.Tensor],
                             first_derivative: Sequence[torch.Tensor],
                             vector: Union[torch.Tensor, Sequence[Optional[torch.Tensor]], None]) -> \
                                Sequence[Optional[torch.Tensor]]:
        pass

    @abc.abstractmethod
    def val_loss_grads(self,
                       params: Dict[str, torch.Tensor],
                       hparams: Dict[str, torch.Tensor]) -> \
                        Tuple[Sequence[Optional[torch.Tensor]], Sequence[Optional[torch.Tensor]]]:
        pass


@torch.jit.script
def squared_euclidean_distance(x1, x2):
    x1_norm = torch.norm(x1, p=2, dim=-1, keepdim=True).pow(2)
    x2_norm = torch.norm(x2, p=2, dim=-1, keepdim=True).pow(2)
    res = torch.addmm(x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
    res = res.clamp_min_(1e-30)
    return res


@torch.jit.script
def full_rbf_kernel(X1, X2, sigma):
    pairwise_dists = squared_euclidean_distance(X1 / sigma, X2 / sigma)
    return torch.exp(-0.5 * pairwise_dists)


def test_train_predict(model,
                       Xts, Yts,
                       Xtr, Ytr,
                       err_fn: callable,
                       epoch: int,
                       time_start: float,
                       cum_time: float,
                       Xval: Optional[torch.Tensor]=None,
                       Yval: Optional[torch.Tensor]=None,
                       ):
    t_elapsed = time.time() - time_start  # Stop the time
    cum_time += t_elapsed
    model.eval()

    test_preds = model.predict(Xts)
    train_preds = model.predict(Xtr)
    test_err, err_name = err_fn(Yts.detach().cpu(), test_preds.detach().cpu())
    train_err, err_name = err_fn(Ytr.detach().cpu(), train_preds.detach().cpu())
    out_str = (f"Epoch {epoch} ({cum_time:5.2f}s) - "
               f"Sigma {model.sigma[0].item():.3f} - Penalty {model.penalty_val.item():.2e} - "
               f"Tr  {err_name} = {train_err:6.4f} - "
               f"Ts  {err_name} = {test_err:6.4f}")
    ret = [cum_time, train_err, test_err]
    if Xval is not None and Yval is not None:
        val_preds = model.predict(Xval)
        val_err, err_name = err_fn(Yval.detach().cpu(), val_preds.detach().cpu())
        out_str += f" - Val {err_name} = {val_err:6.4f}"
        ret.append(val_err)
    print(out_str, flush=True)
    return ret


def get_start_sigma(sigma_init: float, sigma_type: str, d: int = None) -> torch.Tensor:
    if sigma_type == 'single':
        start_sigma = torch.tensor([sigma_init])
    elif sigma_type == 'diag':
        if d is None:
            raise RuntimeError("Dimension d must be specified for diagonal sigma.")
        start_sigma = torch.tensor([sigma_init] * d)
    else:
        raise ValueError("sigma_type %s unrecognized" % (sigma_type))
    return start_sigma

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

    return x_last#, ii, min(app_times)


def cat_list_to_tensor(list_tx):
    return torch.cat([xx.view([-1]) for xx in list_tx])

