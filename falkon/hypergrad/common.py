import abc
import time
from typing import Sequence, Optional, Tuple, Union, Dict

import numpy as np
import torch


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
                       ):
    t_elapsed = time.time() - time_start  # Stop the time
    cum_time += t_elapsed
    model.eval()

    test_preds = model.predict(Xts)
    train_preds = model.predict(Xtr)
    test_err, err_name = err_fn(Yts.detach().cpu(), test_preds.detach().cpu())
    train_err, err_name = err_fn(Ytr.detach().cpu(), train_preds.detach().cpu())
    print(f"Epoch {epoch} ({cum_time:5.2f}s) - "
          f"Tr {err_name} = {train_err:6.4f} , "
          f"Ts {err_name} = {test_err:6.4f} -- "
          f"Sigma {model.sigma[0].item():.3f} - Penalty {np.exp(-model.penalty.item()):.2e}")
    return cum_time, train_err, test_err


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