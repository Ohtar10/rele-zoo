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
            ("pendulum", "reinforce-continuous")
        ]
    )
    def test_train(self, environment, algorithm):
        with initialize_config_module(config_module="relezoo.conf"):
            cfg = compose(config_name="config", overrides=[
                f"environments@env_train={environment}",
                f"environments@env_test={environment}",
                f"algorithm={algorithm}"
            ])
            try:
                cfg.episodes = 5
                hcli.hrelezoo(cfg)
            except Exception as e:
                pytest.fail(f"It should not have failed. {e}")
