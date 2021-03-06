"""Train utils"""

import torch
from torch import optim

from adpkd_segmentation.config.config_utils import ObjectGetter


class TorchLRScheduler:
    def __init__(self, name, param_dict, step_type):
        self.param_dict = param_dict
        self.scheduler = getattr(optim.lr_scheduler, name)
        self.step_type = step_type

    def __call__(self, optimizer):
        return self.scheduler(optimizer, **self.param_dict)


class OptimGetter:
    def __init__(self, module_name, name, param_dict):
        self.optim = ObjectGetter(module_name, name)()
        self.param_dict = param_dict

    def __call__(self, model_params):
        return self.optim(model_params, **self.param_dict)


def save_model_data(path, model, global_step):
    print("saving checkpoint to {}".format(path))
    torch.save(
        {"global_step": global_step, "model_state_dict": model.state_dict()},
        path,
    )


def load_model_data(path, model, new_format=False):
    print("loading checkpoint {}".format(path))
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    global_step = 0
    if not new_format:
        model.load_state_dict(checkpoint)
    else:
        global_step = checkpoint["global_step"]
        model.load_state_dict(checkpoint["model_state_dict"])
    return global_step
