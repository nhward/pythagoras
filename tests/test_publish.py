"""Publish gating with fake test/export/upload commands; never contacts a server."""
import importlib.util
from pathlib import Path
import subprocess

import pytest

spec = importlib.util.spec_from_file_location('publish_runner', Path(__file__).parents[1] / 'scripts/publish.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
pytestmark = pytest.mark.unit


@pytest.fixture
def project(tmp_path, monkeypatch):
    for name in ['app', 'tests', 'site']:
        (tmp_path / name).mkdir()
    (tmp_path / 'app/main.py').write_text('initial')
    (tmp_path / 'site/index.html').write_text('index')
    monkeypatch.setattr(m.importlib.metadata, 'distributions', lambda: [])
    return tmp_path


def runner(root, calls, fail=None):
    def run(command, **kwargs):
        calls.append(command)
        assert kwargs == {'cwd': root, 'check': True}
        if command == fail:
            raise subprocess.CalledProcessError(1, command)
        if command == ['make', 'shinylive-force']:
            (root / 'site/app.json').write_text((root / 'app/main.py').read_text())
        if command == m.DEPLOY:
            metadata = root / 'site/rsconnect-python'
            metadata.mkdir(exist_ok=True)
            (metadata / 'deployment.json').write_text('CLI-created metadata')
    return run


def test_publish_order_no_change_skip_and_test_only_change(project):
    calls = []
    run = runner(project, calls)
    m.publish(project, run)
    assert calls == [['make', 'test-force'], ['make', 'shinylive-force'], m.DEPLOY]
    calls.clear()
    m.publish(project, run)
    assert calls == []
    (project / 'tests/new_test.py').write_text('test addition')
    m.publish(project, run)
    assert calls == [['make', 'test-force'], ['make', 'shinylive-force']]


@pytest.mark.parametrize('stage', [['make', 'test-force'], ['make', 'shinylive-force'], m.DEPLOY])
def test_failure_never_records_success(project, stage):
    calls = []
    with pytest.raises(subprocess.CalledProcessError):
        m.publish(project, runner(project, calls, fail=stage))
    assert calls[-1] == stage
    assert not (project / '.make/publish-state.json').exists()


def test_changed_source_republishes_and_deleted_file_is_detected(project):
    calls = []
    m.publish(project, runner(project, calls))
    calls.clear()
    (project / 'app/main.py').write_text('changed')
    m.publish(project, runner(project, calls))
    assert calls[-1] == m.DEPLOY
    before = m.inputs(project)
    (project / 'app/main.py').unlink()
    assert m.inputs(project) != before


def test_changes_during_tests_block_export(project):
    def run(command, **kwargs):
        (project / 'app/main.py').write_text('edited while tests ran')
    with pytest.raises(RuntimeError, match='during tests'):
        m.publish(project, run)
    assert not (project / '.make/publish-state.json').exists()
