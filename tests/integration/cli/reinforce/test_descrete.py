import os

import pytest
from hydra import compose, initialize_config_module

from relezoo.algorithms.reinforce.discrete import ReinforceDiscretePolicy
from relezoo.cli import hcli
from tests.utils.common import BASELINES_PATH, BASE_PATH


def test_reinforce_discrete_train() -> None:
    with initialize_config_module(config_module="relezoo.conf"):
        cfg = compose(config_name="config")
        try:
            # test for only three episodes instead of the default
            cfg.episodes = 5
            hcli.hrelezoo(cfg)
            checkpoints = os.path.join(os.getcwd(), cfg.checkpoints)
            expected_cp = os.path.join(checkpoints, f"{ReinforceDiscretePolicy.__name__}.cpt")
            assert os.path.exists(expected_cp)
        except Exception as e:
            pytest.fail(f"It should not have failed. {e}")


def test_reinforce_discrete_play() -> None:
    with initialize_config_module(config_module="relezoo.conf"):
        cfg = compose(config_name="config")
        try:
            # test for only three episodes instead of the default
            cfg.episodes = 5
            cfg.mode = 'play'
            cfg.checkpoints = os.path.join(BASELINES_PATH, "reinforce", "cartpole.cpt")
            hcli.hrelezoo(cfg)
        except Exception as e:
            pytest.fail(f"It should not have failed. {e}")
