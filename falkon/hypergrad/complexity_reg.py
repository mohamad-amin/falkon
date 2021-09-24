import abc
from functools import reduce

import torch

try:
    import falkon.cuda.initialization
except:
    pass  # No GPU
from falkon.hypergrad.common import full_rbf_kernel, PositiveTransform
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
    return torch.flatten(t)[0]


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
    #verbose = True  # TODO: Remove
    grads = []
    hparams = model.parameters()
    if not losses_are_grads:
        if verbose:
            for loss in loss_terms:
                grads.append(torch.autograd.grad(loss, hparams, retain_graph=True, allow_unused=True))
        else:
            loss = reduce(torch.add, loss_terms)
            grads.append(torch.autograd.grad(loss, hparams, retain_graph=False))
    else:
        grads = loss_terms

    #print(f"Lambda grads: deff={grads[0][1]:.2e} dfit={grads[1][1]:.2e} trace={grads[2][1]:.2e}")

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

        self.sigma_transform = PositiveTransform(1e-4)
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
        return self.sigma_transform(self.sigma_)

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
        self.sigma_ = self.sigma_transform._inverse(value.clone().detach().to(device=self.device))

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

        self.penalty_transform = PositiveTransform(1e-8)
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
