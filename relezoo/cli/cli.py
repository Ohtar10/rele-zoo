import logging

import click

from relezoo.__version__ import __version__
from relezoo.utils.console import start_python_console


def docstring_parameter(*sub):
    """Decorate the main click command to format the docstring."""
    def dec(obj):
        obj.__doc__ = obj.__doc__.format(*sub)
        return obj
    return dec


@click.group()
@click.option("--debug/--no-debug", default=False, help="Enable debug output.")
@click.option("-wd", "--work-dir", default=".", help="Working directory to drop output.")
@click.pass_context
@docstring_parameter(__version__)
def relezoo(ctx, debug, work_dir):
    """ReleZoo {0}"""
    logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s ",
                        level=logging.INFO if not debug else logging.DEBUG)
    ctx.ensure_object(dict)
    ctx.obj['WORK_DIR'] = work_dir


@relezoo.command()
@click.pass_context
def shell(ctx):
    """Run interactive shell with preloaded module resources."""
    start_python_console(banner='ReleZoo shell')


