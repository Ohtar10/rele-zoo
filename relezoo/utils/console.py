from functools import wraps
from typing import Optional


def _embed_ipython_shell(ns: Optional[dict] = None):
    if ns is None:
        ns = {}

    from IPython.terminal.embed import InteractiveShellEmbed
    from IPython.terminal.ipapp import load_default_config

    @wraps(_embed_ipython_shell)
    def wrapper(namespace=ns, banner=''):
        config = load_default_config()
        InteractiveShellEmbed.clear_instance()
        shell = InteractiveShellEmbed.instance(
            banner1=banner, user_ns=namespace, config=config
        )
        shell()

    return wrapper


def start_python_console(namespace: Optional[dict] = None, banner: str = ''):
    if namespace is None:
        namespace = {}

    try:
        shell = _embed_ipython_shell()
        shell(namespace, banner)
    except SystemExit:  # raised when invoking exit() hence safe to ignore
        pass
