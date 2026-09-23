"""Test, export, and publish changed content; record only successful states."""
import fcntl
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ACCOUNT = 'nickward'
APP_ID = '01a0c2e4-a9ea-54ad-1602-0f024b4d556a'
DEPLOY = ['.venv/bin/rsconnect', 'deploy', 'html', '--connect-cloud',
          '--account', ACCOUNT, '--app-id', APP_ID, 'site']
IGNORED = {'__pycache__', '.pytest_cache', '.DS_Store', '.shinylive-built', 'rsconnect-python'}


def fingerprint(root, names):
    digest = hashlib.sha256()
    for name in sorted(names):
        path = root / name
        files = sorted(path.rglob('*')) if path.is_dir() else [path]
        digest.update(name.encode() + b'\0')
        for file in files:
            relative = file.relative_to(root)
            if any(part in IGNORED for part in relative.parts) or file.suffix in {'.pyc', '.pyo'}:
                continue
            if file.is_file():
                digest.update(str(relative).encode() + b'\0')
                digest.update(hashlib.sha256(file.read_bytes()).digest())
    return digest.hexdigest()


def inputs(root):
    files = fingerprint(root, ['app', 'tests', 'scripts', 'markdown', 'Makefile',
                              'pyproject.toml', 'pytest.ini', 'README.md', '_quarto.yml',
                              '.python-version'])
    environment = sorted((dist.metadata['Name'], dist.version)
                         for dist in importlib.metadata.distributions())
    return hashlib.sha256(json.dumps([files, sys.version, environment, DEPLOY]).encode()).hexdigest()


def publish(root=ROOT, run=subprocess.run):
    state_dir = root / '.make'
    state_dir.mkdir(exist_ok=True)
    with (state_dir / 'publish.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another publish is already running.')
        state_file = state_dir / 'publish-state.json'
        try:
            previous = json.loads(state_file.read_text())
        except (FileNotFoundError, ValueError):
            previous = {}
        before = inputs(root)
        bundle = fingerprint(root, ['site'])
        if previous == {'inputs': before, 'bundle': bundle, 'deploy': DEPLOY}:
            print('Publish skipped: inputs and exported content are unchanged.')
            return
        # Sequential subprocesses keep the test gate intact even under make -j.
        run(['make', 'test-force'], cwd=root, check=True)
        if inputs(root) != before:
            raise RuntimeError('Inputs changed during tests; rerun make publish.')
        run(['make', 'shinylive-force'], cwd=root, check=True)
        if inputs(root) != before:
            raise RuntimeError('Inputs changed during export; rerun make publish.')
        if not (root / 'site/index.html').is_file() or not (root / 'site/app.json').is_file():
            raise RuntimeError('Shinylive export is incomplete; refusing to publish.')
        bundle = fingerprint(root, ['site'])
        if previous.get('bundle') != bundle or previous.get('deploy') != DEPLOY:
            run(DEPLOY, cwd=root, check=True)
        else:
            print('Tests passed; exported content is unchanged, so upload skipped.')
        if inputs(root) != before or fingerprint(root, ['site']) != bundle:
            raise RuntimeError('Files changed during publishing; success was not cached. Rerun make publish.')
        temporary = state_file.with_suffix('.tmp')
        temporary.write_text(json.dumps({'inputs': before, 'bundle': bundle, 'deploy': DEPLOY}) + '\n')
        temporary.replace(state_file)


if __name__ == '__main__':
    try:
        publish()
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(f'Publish stopped: {error}', file=sys.stderr)
        sys.exit(1)
