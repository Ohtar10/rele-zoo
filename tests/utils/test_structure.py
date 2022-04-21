import pytest
from hydra import initialize_config_module, compose

from relezoo.utils.structure import Context


class TestContext:

    def test_create_context_from_dict(self):
        config = {
            "a": 1,
            "b": "hello",
            "c": {"sub_a": 1, "sub_b": 2},
            "d": [1, 2, 3]
        }
        context = Context(config)
        assert context.a == 1
        assert context['a'] == 1
        assert context.b == "hello"
        assert context['b'] == "hello"
        assert context.c == {"sub_a": 1, "sub_b": 2}
        assert context['c'] == {"sub_a": 1, "sub_b": 2}
        assert context.d == [1, 2, 3]
        assert context['d'] == [1, 2, 3]

    def test_create_context_from_omegaconf(self):
        with initialize_config_module(config_module="relezoo.conf"):
            cfg = compose(config_name="config")
            try:
                context = Context(cfg.context)
                assert context.mode == "train"
                assert context['mode'] == "train"
                assert context.eval_every == 10
                assert context['eval_every'] == 10
                assert context.epochs == 50
                assert context['epochs'] == 50

            except Exception as e:
                pytest.fail(f"It should not have failed. {e}")
