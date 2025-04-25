# -*- coding: utf-8 -*-
# Time      :2025/3/29 10:27
# Author    :Hui Huang
from omegaconf import OmegaConf, DictConfig


def load_config(config_path: str) -> DictConfig:
    """Loads a configuration file and optionally merges it with a base configuration.

    Args:
    config_path (Path): Path to the configuration file.
    """
    # Load the initial configuration from the given path
    config = OmegaConf.load(config_path)

    # Check if there is a base configuration specified and merge if necessary
    if config.get("base_config", None) is not None:
        base_config = OmegaConf.load(config["base_config"])
        config = OmegaConf.merge(base_config, config)

    return config
