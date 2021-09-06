import dataclasses
from typing import Tuple, Optional, List, Any

import numpy as np
import torch

from falkon.options import BaseOptions
from falkon.utils import devices, PropagatingThread
from falkon.utils.devices import DeviceInfo
from falkon.utils.fake_queue import FakeQueue
from falkon.utils.tensor_helpers import is_contig, extract_same_stride

__all__ = ("_setup_opt", "_check_contiguity", "_get_gpu_info", "_get_cpu_ram",
           "_start_wait_processes", "_gpu_tns_same_memory", "_call_direct",
           "ensure_batch_dim", "_extract_flat", "_is_incore", "_dev_from_id",
           )


def _setup_opt(opt: Optional[BaseOptions], is_cpu=False) -> BaseOptions:
    if opt is None:
        opt = BaseOptions()
    return dataclasses.replace(opt, use_cpu=is_cpu)


def _check_contiguity(*args: Tuple[Optional[torch.Tensor], str]) -> None:
    for tensor, name in args:
        if tensor is not None and not is_contig(tensor):
            raise ValueError(f"Tensor '{name}' must be memory contiguous")


def _get_gpu_info(opt: BaseOptions, slack: float = 0.9) -> List[DeviceInfo]:
    # List available devices, get their relative speed and split
    # computations based on device relative speed.
    gpu_info = [v for k, v in devices.get_device_info(opt).items() if v.isGPU]
    for g in gpu_info:
        g.usable_ram = min(g.free_memory * slack, opt.max_gpu_mem * slack)
    return gpu_info


def _get_cpu_ram(opt: BaseOptions, slack: float = 0.9) -> float:
    cpu_info = devices.get_device_info(opt)[-1]
    avail_mem = min(cpu_info.free_memory, opt.max_cpu_mem - cpu_info.used_memory)
    return avail_mem * slack


def _start_wait_processes(target, args) -> List[Any]:
    processes, outputs = [], []
    for i, a in enumerate(args):
        args_queue = FakeQueue()
        args_queue.put(a[0])
        new_args_tuple = (i, args_queue, a[1])
        # PropagatingThread throws any exception which happened in the thread on join
        process = PropagatingThread(target=target, name=f'GPU-{a[1]}', args=new_args_tuple)
        processes.append(process)
    for p in processes:
        p.start()
    for p in processes:
        outputs.append(p.join())
    return outputs


def _call_direct(target, arg):
    args_queue = FakeQueue()
    args_queue.put(arg[0])
    new_args_tuple = (0, args_queue, arg[1])
    return target(*new_args_tuple)


def _gpu_tns_same_memory(A: torch.Tensor, B: torch.Tensor) -> bool:
    # noinspection PyArgumentList
    return (A.dtype == B.dtype) and \
           (A.shape == B.shape) and \
           (A.data_ptr() == B.data_ptr()) and \
           (A.stride() == B.stride())


def ensure_batch_dim(*args: Optional[torch.Tensor]):
    for tensor in args:
        if tensor is None:
            yield tensor
        elif tensor.dim() == 3:
            yield tensor
        elif tensor.dim() == 2:
            yield tensor.unsqueeze(0)
        else:
            raise ValueError("Cannot ensure batch dimension on tensor with %d dimensions" % (tensor.dim()))


def _extract_flat(flat_tn, size, other, offset):
    struct_tn = extract_same_stride(flat_tn, size=size, other=other, offset=offset)
    offset += np.prod(struct_tn.shape)
    return struct_tn, offset


def _is_incore(computation_device: torch.device, data_device: torch.device) -> bool:
    return computation_device.type == data_device.type


def _dev_from_id(device_id: int) -> torch.device:
    if device_id < 0:
        return torch.device('cpu')
    return torch.device('cuda:%d' % device_id)
