import os

import pytest
from hydra import compose, initialize_config_module

from relezoo.cli import hcli
from tests.utils.common import BASELINES_PATH, MAX_TEST_EPISODES


@pytest.fixture(autouse=True)
def run_around_tests(tmpdir):
    tmp_folder = tmpdir.mkdir('output')
    os.chdir(tmp_folder)
    yield


@pytest.mark.enduser
@pytest.mark.parametrize(
    "algorithm",
    [
        "xentropy-discrete",
        "reinforce-discrete"
    ]
)
class TestDiscreteAlgorithmsCli:
    def test_train(self, algorithm) -> None:
        with initialize_config_module(config_module="relezoo.conf"):
            cfg = compose(config_name="config",
                          overrides=[
                              f"algorithm={algorithm}"
                          ])
            try:
                # test for only three episodes instead of the default
                cfg.context.epochs = MAX_TEST_EPISODES
                hcli.hrelezoo(cfg)
                checkpoints = os.path.join(os.getcwd(), cfg.context.checkpoints)
                assert os.path.exists(checkpoints)
                assert len(os.listdir(checkpoints)) > 0
                logs = os.path.join(os.getcwd(), cfg.logger.log_dir)
                assert os.path.exists(logs)
                assert len(os.listdir(logs)) > 0
            except Exception as e:
                pytest.fail(f"It should not have failed. {e}")

    def test_train_with_seed(self, algorithm) -> None:
        with initialize_config_module(config_module="relezoo.conf"):
            cfg = compose(config_name="config",
                          overrides=[
                              f"algorithm={algorithm}",
                              "context.seed=123"
                          ])
            try:
                # test for only three episodes instead of the default
                cfg.context.epochs = MAX_TEST_EPISODES
                hcli.hrelezoo(cfg)
                checkpoints = os.path.join(os.getcwd(), cfg.context.checkpoints)
                assert os.path.exists(checkpoints)
                assert len(os.listdir(checkpoints)) > 0
                logs = os.path.join(os.getcwd(), cfg.logger.log_dir)
                assert os.path.exists(logs)
                assert len(os.listdir(logs)) > 0
            except Exception as e:
                pytest.fail(f"It should not have failed. {e}")

    def test_train_with_parallel_env(self, algorithm) -> None:
        with initialize_config_module(config_module="relezoo.conf"):
            cfg = compose(config_name="config",
                          overrides=[
                              f"algorithm={algorithm}",
                              "environments@env_train=classic_control/parallel-cartpole"
                          ])
            try:
                # test for only three episodes instead of the default
                cfg.context.epochs = MAX_TEST_EPISODES
                hcli.hrelezoo(cfg)
                checkpoints = os.path.join(os.getcwd(), cfg.context.checkpoints)
                assert os.path.exists(checkpoints)
                assert len(os.listdir(checkpoints)) > 0
                logs = os.path.join(os.getcwd(), cfg.logger.log_dir)
                assert os.path.exists(logs)
                assert len(os.listdir(logs)) > 0
            except Exception as e:
                pytest.fail(f"It should not have failed. {e}")

    @pytest.mark.parametrize(
        "environment",
        [
            "classic_control/cartpole",
            "classic_control/acrobot"
        ]
    )
    def test_play(self, environment, algorithm) -> None:
        checkpoint = os.path.join(
                    BASELINES_PATH, algorithm.split('-')[0], environment
                )
        if not (os.path.exists(checkpoint)):
            pytest.skip("Baseline not available")

        with initialize_config_module(config_module="relezoo.conf"):
            cfg = compose(config_name="config",
                          overrides=[
                              f"algorithm={algorithm}",
                              f"environments@env_train={environment}",
                              f"environments@env_test={environment}"
                          ]
                          )
            try:
                # test for only three episodes instead of the default
                cfg.context.epochs = MAX_TEST_EPISODES
                cfg.context.mode = 'play'
                cfg.context.checkpoints = checkpoint
                hcli.hrelezoo(cfg)
            except Exception as e:
                pytest.fail(f"It should not have failed. {e}")

    def test_train_with_render(self, algorithm) -> None:
        with initialize_config_module(config_module="relezoo.conf"):
            cfg = compose(config_name="config",
                          overrides=[
                              f"algorithm={algorithm}"
                          ])
            try:
                # test for only three episodes instead of the default
                cfg.context.epochs = MAX_TEST_EPISODES
                cfg.context.render = True
                cfg.context.eval_every = 1
                hcli.hrelezoo(cfg)
                checkpoints = os.path.join(os.getcwd(), cfg.context.checkpoints)
                assert os.path.exists(checkpoints)
                assert len(os.listdir(checkpoints)) > 0
                logs = os.path.join(os.getcwd(), cfg.logger.log_dir)
                assert os.path.exists(logs)
                assert len(os.listdir(logs)) > 0
            except Exception as e:
                pytest.fail(f"It should not have failed. {e}")
