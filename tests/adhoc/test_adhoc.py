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
        ("environment", "algorithm"),
        [
            ("cartpole", "xentropy-discrete")
        ]
    )
    def test_train(self, environment, algorithm):
        with initialize_config_module(config_module="relezoo.conf"):
            cfg = compose(config_name="config", overrides=[
                f"environments@env_train=parallel-{environment}",
                f"environments@env_test={environment}",
                f"algorithm={algorithm}"
            ])
            try:
                cfg.context.epochs = 20
                cfg.context.render = False
                cfg.context.eval_every = 5
                cfg.algorithm.batch_size = 16
                result = hcli.hrelezoo(cfg)
                print(result)
            except Exception as e:
                pytest.fail(f"It should not have failed. {e}")
