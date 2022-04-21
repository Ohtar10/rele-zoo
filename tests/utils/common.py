import os

BASE_PATH = os.path.dirname(__file__)
BASELINES_PATH = os.path.join(BASE_PATH, "../../baselines/")
RENDER = False
MAX_TEST_EPISODES = 3


def get_fqcn(clazz: object):
    return ".".join([clazz.__module__, clazz.__name__])
