# picks the compute dtype for training and eval


def pick_dtype():
    """bf16 on GPUs that support it natively (Ampere+, e.g. A100), otherwise fp16
    so older cards like the P100 can at least run eval.

    Needs including_emulation=False, otherwise it also says True on pre-Ampere
    cards that only emulate bf16. This is just the 4-bit compute/ mixed precision
    dtype, the DTW weights are always done in fp32 (see train/sft_train.py)."""
    import torch
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported(
            including_emulation=False):
        return torch.bfloat16
    return torch.float16


def describe_device(dtype):
    """One line with the GPU name and dtype, printed at startup."""
    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no CUDA"
    return f"device: {gpu}  compute dtype: {str(dtype).replace('torch.', '')}"
