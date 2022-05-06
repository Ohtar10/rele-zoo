import pytest
from hydra import compose, initialize_config_module

from relezoo.cli import hcli


@pytest.mark.skip(reason="Debug/Manual only Test")
class TestAdhoc:
    """Test Adhoc.

    This test class is meant to be used
    to run the program with custom, manual tweaks
    for debugging purposes in Adhoc mode. It is
    not intended to be part of the normal test
    suite, hence the skip mark.
    """

    @pytest.mark.parametrize(
        ("env_train", "env_test", "algorithm"),
        [
            ("box2d/parallel-lunar-lander", "box2d/lunar-lander", "reinforce-discrete")
        ]
    )
    def test_train(self, env_train, env_test, algorithm, tmpdir):
        with initialize_config_module(config_module="relezoo.conf"):
            out_folder = tmpdir.mkdir('output')
            cfg = compose(config_name="config", overrides=[
                f"environments@env_train={env_train}",
                f"environments@env_test={env_test}",
                f"algorithm={algorithm}",
                "logger=tensorboard"
            ])
            try:
                cfg.context.epochs = 5
                cfg.context.render = True
                cfg.algorithm.batch_size = 5000
                result = hcli.hrelezoo(cfg)
                print(result)
            except Exception as e:
                pytest.fail(f"It should not have failed. {e}")
