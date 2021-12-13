import mock
from click.testing import CliRunner
from relezoo.cli import *


def test_relezoo_command():
    runner = CliRunner()
    result = runner.invoke(relezoo, ['--help'])
    assert result.exit_code == 0
    assert all(item in result.output.split() for item in relezoo.__doc__.split())


@mock.patch("relezoo.cli.start_python_console")
def test_shell_command(mock_start_shell):
    runner = CliRunner()
    result = runner.invoke(relezoo, ['shell'])
    assert result.exit_code == 0
    mock_start_shell.assert_called_once_with(banner="ReleZoo shell")
