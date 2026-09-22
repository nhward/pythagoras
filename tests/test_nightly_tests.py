"""Nightly runner notifications without invoking pytest or sending real mail."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location('nightly_tests', Path(__file__).parents[1] / 'scripts/nightly_tests.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
pytestmark = pytest.mark.unit


@pytest.mark.parametrize('code', [0, 1, 2, 5])
def test_reports_and_emails_only_failed_runs(tmp_path, monkeypatch, code):
    monkeypatch.setattr(m, 'REPORTS', tmp_path)
    monkeypatch.setattr(m.sys, 'argv', ['nightly_tests.py'])
    launches, emails = [], []
    def launch(command, **kwargs):
        launches.append((command, kwargs))
        return SimpleNamespace(wait=lambda timeout: code)
    monkeypatch.setattr(m.subprocess, 'Popen', launch)
    monkeypatch.setattr(m, 'email_failure', lambda log, status: emails.append((log, status)))
    assert m.main() == code
    assert launches[0][0][1:4] == ['-m', 'pytest', 'tests']
    assert len(emails) == bool(code)
    assert f'Exit code: {code}' in next(tmp_path.glob('*.log')).read_text()


def test_email_failure_preserves_failed_test_status(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'REPORTS', tmp_path)
    monkeypatch.setattr(m.sys, 'argv', ['nightly_tests.py'])
    monkeypatch.setattr(m.subprocess, 'Popen', lambda *a, **kw: SimpleNamespace(wait=lambda timeout: 1))
    def fail(*args):
        raise OSError('Mail unavailable')
    monkeypatch.setattr(m, 'email_failure', fail)
    assert m.main() == 1
    assert 'Email notification failed' in next(tmp_path.glob('*.log')).read_text()
