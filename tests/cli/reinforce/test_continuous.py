import os

import pytest
from hydra import compose, initialize_config_module

from relezoo.algorithms.reinforce.continuous import ReinforceContinuousPolicy
from relezoo.cli import hcli
from tests.utils.common import MAX_TEST_EPISODES, BASELINES_PATH


@pytest.mark.cli
class TestReinforceContinuousCli:
    def test_reinforce_continuous(self) -> None:
        with initialize_config_module(config_module="relezoo.conf"):
            cfg = compose(config_name="config",
                          overrides=[
                              "environments@env_train=pendulum",
                              "environments@env_test=pendulum",
                              "algorithm=reinforce-continuous"
                          ])
            try:
                # test for only three episodes instead of the default
                cfg.context.episodes = MAX_TEST_EPISODES
                hcli.hrelezoo(cfg)
                checkpoints = os.path.join(os.getcwd(), cfg.context.checkpoints)
                expected_cp = os.path.join(checkpoints, f"{ReinforceContinuousPolicy.__name__}.cpt")
                assert os.path.exists(expected_cp)
            except Exception as e:
                pytest.fail(f"It should not have failed. {e}")

    @pytest.mark.skip(reason="Baseline not ready")
    def test_reinforce_continuous_play(self) -> None:
        with initialize_config_module(config_module="relezoo.conf"):
            cfg = compose(config_name="config",
                          overrides=[
                              "environments@env_train=pendulum",
                              "environments@env_test=pendulum",
                              "algorithm=reinforce-continuous"
                          ])
            try:
                # test for only three episodes instead of the default
                cfg.context.episodes = MAX_TEST_EPISODES
                cfg.context.mode = "play"
                cfg.context.checkpoints = os.path.join(BASELINES_PATH, "reinforce", "pendulum", "pendulum.cpt")
                hcli.hrelezoo(cfg)
            except Exception as e:
                pytest.fail(f"It should not have failed. {e}")

    def test_reinforce_continuous_with_render(self) -> None:
        with initialize_config_module(config_module="relezoo.conf"):
            cfg = compose(config_name="config",
                          overrides=[
                              "environments@env_train=pendulum",
                              "environments@env_test=pendulum",
                              "algorithm=reinforce-continuous"
                          ])
            try:
                cfg.context.episodes = MAX_TEST_EPISODES
                cfg.context.render = True
                cfg.context.eval_every = 1
                hcli.hrelezoo(cfg)
                checkpoints = os.path.join(os.getcwd(), cfg.context.checkpoints)
                expected_cp = os.path.join(checkpoints, f"{ReinforceContinuousPolicy.__name__}.cpt")
                assert os.path.exists(expected_cp)
            except Exception as e:
                pytest.fail(f"It should not have failed. {e}")
