import pytest
from hydra import initialize, compose, initialize_config_module

from relezoo.cli import hcli


def test_cli_composition_smoke() -> None:
    with initialize(config_path="../../../relezoo/conf"):
        # config is relative to a module
        cfg = compose(config_name="config")
        assert cfg is not None
        assert "environment" in cfg
        assert "algorithm" in cfg
        assert "network" in cfg
        assert "logger" in cfg


def test_cli_full() -> None:
    with initialize_config_module(config_module="relezoo.conf"):
        cfg = compose(config_name="config")
        try:
            hcli.run(cfg)
        except Exception as e:
            pytest.fail(f"It should not have failed. {e}")
