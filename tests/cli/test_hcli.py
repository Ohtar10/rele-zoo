import pytest
from hydra import initialize, compose


@pytest.mark.cli
def test_cli_composition_smoke() -> None:
    with initialize(config_path="../../relezoo/conf"):
        # config is relative to a module
        cfg = compose(config_name="config")
        assert cfg is not None
        assert "env_train" in cfg
        assert "env_test" in cfg
        assert "algorithm" in cfg
        assert "network" in cfg
        assert "logger" in cfg
