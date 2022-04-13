import os

import pytest
from hydra import compose, initialize_config_module

from relezoo.algorithms.reinforce.discrete import ReinforceDiscretePolicy
from relezoo.cli import hcli
from tests.utils.common import BASELINES_PATH, MAX_TEST_EPISODES


@pytest.mark.cli
class TestReinforceDiscreteCli:
    def test_reinforce_discrete_train(self) -> None:
        with initialize_config_module(config_module="relezoo.conf"):
            cfg = compose(config_name="config")
            try:
                # test for only three episodes instead of the default
                cfg.episodes = MAX_TEST_EPISODES
                hcli.hrelezoo(cfg)
                checkpoints = os.path.join(os.getcwd(), cfg.checkpoints)
                expected_cp = os.path.join(checkpoints, f"{ReinforceDiscretePolicy.__name__}.cpt")
                assert os.path.exists(expected_cp)
            except Exception as e:
                pytest.fail(f"It should not have failed. {e}")

    @pytest.mark.parametrize(
        "environment",
        [
            "cartpole",
            "acrobot"
        ]
    )
    def test_reinforce_discrete_play(self, environment) -> None:
        with initialize_config_module(config_module="relezoo.conf"):
            cfg = compose(config_name="config",
                          overrides=[
                              f"environments@env_train={environment}",
                              f"environments@env_test={environment}"
                          ]
                          )
            try:
                # test for only three episodes instead of the default
                cfg.episodes = MAX_TEST_EPISODES
                cfg.mode = 'play'
                cfg.checkpoints = os.path.join(BASELINES_PATH, "reinforce", environment, f"{environment}.cpt")
                hcli.hrelezoo(cfg)
            except Exception as e:
                pytest.fail(f"It should not have failed. {e}")
