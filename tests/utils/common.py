import os

from relezoo.utils.structure import Context

BASE_PATH = os.path.dirname(__file__)
BASELINES_PATH = os.path.join(BASE_PATH, "../../baselines/")
RENDER = False
MAX_TEST_EPISODES = 3
TEST_SEED = 456


def get_test_context():
    return Context({
        "epochs": MAX_TEST_EPISODES,
        "render": False,
        "gpu": False,
        "start_at_step": 0,
        "render_every": 1,
        "eval_every": 1,
        "mean_reward_window": 100
    })


def get_fqcn(clazz: object):
    return ".".join([clazz.__module__, clazz.__name__])
