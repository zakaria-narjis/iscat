import torch.distributed as dist

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0