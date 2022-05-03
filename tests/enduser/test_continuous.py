import os

import pytest
from hydra import compose, initialize_config_module

from relezoo.algorithms.reinforce.continuous import ReinforceContinuousPolicy
from relezoo.cli import hcli
from tests.utils.common import MAX_TEST_EPISODES, BASELINES_PATH


@pytest.mark.cli
@pytest.mark.parametrize(
    "algorithm",
    [
        "reinforce-continuous"
    ]
)
class TestReinforceContinuousCli:

    def test_train(self, algorithm) -> None:
        with initialize_config_module(config_module="relezoo.conf"):
            cfg = compose(config_name="config",
                          overrides=[
                              "environments@env_train=pendulum",
                              "environments@env_test=pendulum",
                              f"algorithm={algorithm}"
                          ])
            try:
                # test for only three episodes instead of the default
                cfg.context.epochs = MAX_TEST_EPISODES
                hcli.hrelezoo(cfg)
                checkpoints = os.path.join(os.getcwd(), cfg.context.checkpoints)
                expected_cp = os.path.join(checkpoints, f"{ReinforceContinuousPolicy.__name__}.cpt")
                assert os.path.exists(expected_cp)
            except Exception as e:
                pytest.fail(f"It should not have failed. {e}")

    def test_train_with_seed(self, algorithm) -> None:
        with initialize_config_module(config_module="relezoo.conf"):
            cfg = compose(config_name="config",
                          overrides=[
                              "environments@env_train=pendulum",
                              "environments@env_test=pendulum",
                              f"algorithm={algorithm}",
                              "context.seed=123"
                          ])
            try:
                # test for only three episodes instead of the default
                cfg.context.epochs = MAX_TEST_EPISODES
                hcli.hrelezoo(cfg)
                checkpoints = os.path.join(os.getcwd(), cfg.context.checkpoints)
                expected_cp = os.path.join(checkpoints, f"{ReinforceContinuousPolicy.__name__}.cpt")
                assert os.path.exists(expected_cp)
            except Exception as e:
                pytest.fail(f"It should not have failed. {e}")

    def test_train_with_parallel_env(self, algorithm) -> None:
        with initialize_config_module(config_module="relezoo.conf"):
            cfg = compose(config_name="config",
                          overrides=[
                              f"algorithm={algorithm}",
                              "environments@env_train=parallel-pendulum",
                              "environments@env_test=pendulum"
                          ])
            try:
                # test for only three episodes instead of the default
                cfg.context.epochs = MAX_TEST_EPISODES
                hcli.hrelezoo(cfg)
                checkpoints = os.path.join(os.getcwd(), cfg.context.checkpoints)
                expected_cp = os.path.join(checkpoints, f"{ReinforceContinuousPolicy.__name__}.cpt")
                assert os.path.exists(expected_cp)
            except Exception as e:
                pytest.fail(f"It should not have failed. {e}")

    @pytest.mark.skip(reason="Baseline not ready")
    def test_play(self, algorithm) -> None:
        with initialize_config_module(config_module="relezoo.conf"):
            cfg = compose(config_name="config",
                          overrides=[
                              "environments@env_train=pendulum",
                              "environments@env_test=pendulum",
                              f"algorithm={algorithm}"
                          ])
            try:
                # test for only three episodes instead of the default
                cfg.context.epochs = MAX_TEST_EPISODES
                cfg.context.mode = "play"
                cfg.context.checkpoints = os.path.join(BASELINES_PATH, "reinforce", "pendulum", "pendulum.cpt")
                hcli.hrelezoo(cfg)
            except Exception as e:
                pytest.fail(f"It should not have failed. {e}")

    def test_train_with_render(self, algorithm) -> None:
        with initialize_config_module(config_module="relezoo.conf"):
            cfg = compose(config_name="config",
                          overrides=[
                              "environments@env_train=pendulum",
                              "environments@env_test=pendulum",
                              f"algorithm={algorithm}"
                          ])
            try:
                cfg.context.epochs = MAX_TEST_EPISODES
                cfg.context.render = True
                cfg.context.eval_every = 1
                hcli.hrelezoo(cfg)
                checkpoints = os.path.join(os.getcwd(), cfg.context.checkpoints)
                expected_cp = os.path.join(checkpoints, f"{ReinforceContinuousPolicy.__name__}.cpt")
                assert os.path.exists(expected_cp)
            except Exception as e:
                pytest.fail(f"It should not have failed. {e}")
