"""Device resolution for the experiment harness.

Policy (per orchestrator instruction): GPU is the default and a silent CPU
fallback is treated as a bug, not a convenience. If --device is left at its
default ('cuda') and no CUDA device is visible, this raises instead of
quietly running on CPU (which would make a full sweep ~10x slower without
anyone noticing). Pass --device cpu explicitly to opt out.
"""
import platform

import torch


def resolve_device(device_arg="cuda"):
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "device='cuda' (the default) but torch.cuda.is_available() is False. "
                "This harness hard-fails instead of silently falling back to CPU. "
                "Pass --device cpu explicitly if you really want CPU."
            )
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    return torch.device(device_arg)


def device_info(device):
    info = {
        "device_arg": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "python_version": platform.python_version(),
    }
    if device.type == "cuda":
        idx = torch.cuda.current_device()
        info["gpu_name"] = torch.cuda.get_device_name(idx)
        info["gpu_capability"] = ".".join(str(x) for x in torch.cuda.get_device_capability(idx))
    return info


def reset_peak_memory(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_mb(device):
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return 0.0


def free_memory(device, *tensors_or_models):
    for obj in tensors_or_models:
        del obj
    if device.type == "cuda":
        torch.cuda.empty_cache()
