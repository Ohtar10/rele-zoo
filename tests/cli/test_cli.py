import mock
import pytest
from click.testing import CliRunner
from relezoo.cli import relezoo


@pytest.mark.cli
class TestCli:
    def test_relezoo_command(self):
        runner = CliRunner()
        result = runner.invoke(relezoo, ['--help'])
        assert result.exit_code == 0
        assert all(item in result.output.split() for item in relezoo.__doc__.split())

    @mock.patch("relezoo.cli.cli.start_python_console")
    def test_shell_command(self, mock_start_shell):
        runner = CliRunner()
        result = runner.invoke(relezoo, ['shell'])
        assert result.exit_code == 0
        mock_start_shell.assert_called_once_with(banner="ReleZoo shell")
