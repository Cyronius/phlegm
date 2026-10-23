"""Dependency audits must retain actual header paths on both host platforms."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'open_kernels'))
from compiler_deps import dependencies


def test_windows_separators_are_not_shell_escapes():
    path = r'C:\work\generated\gemv_tab.h'
    assert str(Path(path)) in dependencies('object.o: ' + path)


def test_make_escaped_spaces_and_continuations():
    tokens = dependencies('object.o: /tmp/private\\ header.h \\\n /tmp/second.h')
    assert str(Path('/tmp/private header.h')) in tokens
    assert str(Path('/tmp/second.h')) in tokens
    assert str(Path('/tmp/header.h')) not in tokens
