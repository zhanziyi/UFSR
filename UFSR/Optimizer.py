import torch
import UFSR.Error
from typing import Any


def get_optimizer_1(model_parameters: Any, 
                    optimizer_dir: dict[str,int|bool|float|str|dict|None]) -> torch.optim.Adam:
    all_optimizer = {
        "Adam": torch.optim.Adam
        }
    if optimizer_dir["name"] in all_optimizer:
        return all_optimizer[optimizer_dir["name"]](model_parameters, **optimizer_dir["parameter"])
    else:
        UFSR.Error.error_exit_1(f"Optimizer Name: {optimizer_dir["name"]} Does Not Exist")