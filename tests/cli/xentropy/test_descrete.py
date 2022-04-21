import os

import pytest
from hydra import compose, initialize_config_module

from relezoo.algorithms.xentropy.discrete import CrossEntropyDiscretePolicy
from relezoo.cli import hcli
from tests.utils.common import BASELINES_PATH, MAX_TEST_EPISODES


@pytest.mark.cli
class TestXEntropyDiscreteCli:
    def test_train(self) -> None:
        with initialize_config_module(config_module="relezoo.conf"):
            cfg = compose(config_name="config",
                          overrides=[
                              "algorithm=xentropy-discrete"
                          ])
            try:
                # test for only three episodes instead of the default
                cfg.context.epochs = MAX_TEST_EPISODES
                hcli.hrelezoo(cfg)
                checkpoints = os.path.join(os.getcwd(), cfg.context.checkpoints)
                expected_cp = os.path.join(checkpoints, f"{CrossEntropyDiscretePolicy.__name__}.cpt")
                assert os.path.exists(expected_cp)
            except Exception as e:
                pytest.fail(f"It should not have failed. {e}")

    def test_train_with_parallel_env(self) -> None:
        with initialize_config_module(config_module="relezoo.conf"):
            cfg = compose(config_name="config",
                          overrides=[
                              "algorithm=xentropy-discrete",
                              "environments@env_train=parallel-cartpole"
                          ])
            try:
                # test for only three episodes instead of the default
                cfg.context.epochs = MAX_TEST_EPISODES
                hcli.hrelezoo(cfg)
                checkpoints = os.path.join(os.getcwd(), cfg.context.checkpoints)
                expected_cp = os.path.join(checkpoints, f"{CrossEntropyDiscretePolicy.__name__}.cpt")
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
    def test_play(self, environment) -> None:
        with initialize_config_module(config_module="relezoo.conf"):
            cfg = compose(config_name="config",
                          overrides=[
                              "algorithm=xentropy-discrete",
                              f"environments@env_train={environment}",
                              f"environments@env_test={environment}"
                          ]
                          )
            try:
                # test for only three episodes instead of the default
                cfg.context.epochs = MAX_TEST_EPISODES
                cfg.context.mode = 'play'
                cfg.context.checkpoints = os.path.join(BASELINES_PATH, "xentropy", environment, f"{environment}.cpt")
                hcli.hrelezoo(cfg)
            except Exception as e:
                pytest.fail(f"It should not have failed. {e}")

    def test_train_with_render(self) -> None:
        with initialize_config_module(config_module="relezoo.conf"):
            cfg = compose(config_name="config",
                          overrides=[
                              "algorithm=xentropy-discrete"
                          ])
            try:
                # test for only three episodes instead of the default
                cfg.context.epochs = MAX_TEST_EPISODES
                cfg.context.render = True
                cfg.context.eval_every = 1
                hcli.hrelezoo(cfg)
                checkpoints = os.path.join(os.getcwd(), cfg.context.checkpoints)
                expected_cp = os.path.join(checkpoints, f"{CrossEntropyDiscretePolicy.__name__}.cpt")
                assert os.path.exists(expected_cp)
            except Exception as e:
                pytest.fail(f"It should not have failed. {e}")
