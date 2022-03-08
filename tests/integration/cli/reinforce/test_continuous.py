import os

import pytest
from hydra import compose, initialize_config_module

from relezoo.algorithms.reinforce.continuous import ReinforceContinuousPolicy
from relezoo.cli import hcli
from tests.utils.common import MAX_TEST_EPISODES


def test_reinforce_continuous() -> None:
    with initialize_config_module(config_module="relezoo.conf"):
        cfg = compose(config_name="config",
                      overrides=[
                          "environment=pendulum",
                          "algorithm=reinforce-continuous"
                      ])
        try:
            # test for only three episodes instead of the default
            cfg.episodes = MAX_TEST_EPISODES
            hcli.hrelezoo(cfg)
            checkpoints = os.path.join(os.getcwd(), cfg.checkpoints)
            expected_cp = os.path.join(checkpoints, f"{ReinforceContinuousPolicy.__name__}.cpt")
            assert os.path.exists(expected_cp)
        except Exception as e:
            pytest.fail(f"It should not have failed. {e}")

