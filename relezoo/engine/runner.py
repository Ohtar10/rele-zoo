import os
import importlib
from typing import Any

from omegaconf import DictConfig


def run(cfg: DictConfig) -> Any:
    workdir = os.getcwd()
    print(f"Algorithm was {cfg.algorithm}")
    algo_class = cfg.algorithm['class']
    print(f"algorithm class: {algo_class}")
    module_name = ".".join(algo_class.split(".")[:-1])
    class_name = algo_class.split(".")[-1]
    algo_class = getattr(importlib.import_module(module_name), class_name)
    # algo = algo_class()
    print(algo_class)
