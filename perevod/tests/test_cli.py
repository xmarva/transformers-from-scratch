import io
import subprocess
import sys

from perevod import __version__
from perevod.cli import main


def run_python(*args):
    command = [sys.executable, *args]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def test_the_version():
    assert run_python("-m", "perevod.cli", "--version") == f"perevod {__version__}"


def test_importing_the_package_does_not_load_pytorch():
    assert run_python("-c", "import sys, perevod; print('torch' in sys.modules)") == "False"


def test_each_argument_gives_one_line(tiny_dir, capsys):
    main(["--model", str(tiny_dir), "--device", "cpu", "tom reads", "i think"])
    assert len(capsys.readouterr().out.splitlines()) == 2


def test_one_line_in_one_line_out(tiny_dir, capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO("tom reads\n\ni think\n"))
    main(["--model", str(tiny_dir), "--device", "cpu"])
    lines = capsys.readouterr().out.splitlines()
    # an empty line stays empty: line n of the output is the translation of line n
    assert len(lines) == 3
    assert lines[1] == ""
