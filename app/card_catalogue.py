"""Persistent card descriptions, populated by real use or an isolated worker."""
from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
from copy import deepcopy
import hashlib
from html.parser import HTMLParser
import importlib
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import threading

SCHEMA_VERSION = 1
EXCLUDED = {'__init__', 'sys_bookmark', 'template'}
LOG = logging.getLogger('pythagoras')


def discover_cards(root):
    return {p.stem: p for p in sorted((Path(root) / 'cards').glob('*.py')) if p.stem not in EXCLUDED}


class _PlainText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def plain_text(value):
    if not isinstance(value, str):
        return None
    parser = _PlainText()
    parser.feed(value)
    return ' '.join(' '.join(parser.parts).split()) or None


def source_hash(path):
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def valid_entries(payload):
    if not isinstance(payload, dict) or payload.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('Unsupported card catalogue format')
    entries = payload.get('cards')
    if not isinstance(entries, dict):
        raise ValueError('Card catalogue must contain a cards object')
    result = {}
    for name, entry in entries.items():
        if not isinstance(name, str) or not name.isidentifier() or not isinstance(entry, dict):
            continue
        if entry.get('module') != name or not isinstance(entry.get('source_hash'), str):
            continue
        if any(entry.get(k) is not None and not isinstance(entry[k], str) for k in ('long_name', 'description')):
            continue
        if any(entry.get(k) is not None and type(entry[k]) is not bool for k in ('mutable', 'pipeline_compatible')):
            continue
        result[name] = {k: entry.get(k) for k in ('module', 'long_name', 'description', 'mutable', 'pipeline_compatible', 'source_hash')}
    return result


class CardCatalogue:
    def __init__(self, root, path=None, *, persistent=True):
        self.root = Path(root)
        self.path = Path(path) if path is not None else self.root / 'data' / 'card_catalogue.json'
        self.persistent = persistent
        self._lock = threading.RLock()
        self._entries = self._read()
        self.write_error = None

    def _read(self):
        try:
            return valid_entries(json.loads(self.path.read_text(encoding='utf-8')))
        except FileNotFoundError:
            return {}
        except (OSError, ValueError, TypeError) as error:
            LOG.warning('Cannot read card catalogue %s: %s; using filename fallbacks', self.path, error)
            return {}

    def snapshot(self):
        with self._lock:
            return {'schema_version': SCHEMA_VERSION, 'cards': deepcopy(self._entries)}

    def entry(self, name):
        path = discover_cards(self.root).get(name)
        with self._lock:
            entry = self._entries.get(name)
            if path and entry and entry['source_hash'] == source_hash(path):
                return deepcopy(entry)
        return {'module': name, 'long_name': None, 'description': None,
                'mutable': None, 'pipeline_compatible': None, 'source_hash': None}

    def choices(self, names=None):
        choices = {}
        for name in (discover_cards(self.root) if names is None else names):
            entry = self.entry(name)
            title = entry['long_name'] or name
            if title != name:
                title += f' ({name})'
            choices[name] = title + (f" — {entry['description']}" if entry['description'] else '')
        return choices

    @contextmanager
    def _disk_lock(self):
        # Native macOS/Linux processes share a lock; browser runtimes never write.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix('.json.lock').open('a') as handle:
            try:
                import fcntl
            except ImportError:
                fcntl = None
            if fcntl:
                fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl:
                    fcntl.flock(handle, fcntl.LOCK_UN)

    def update(self, entries):
        entries = valid_entries({'schema_version': SCHEMA_VERSION, 'cards': entries})
        with self._lock:
            changed = {name: entry for name, entry in entries.items() if self._entries.get(name) != entry}
            if not changed:
                return False
            self._entries.update(changed)
            if not self.persistent:
                return True
            temporary = None
            try:
                with self._disk_lock():
                    # Merge another worker's recent updates rather than replace them.
                    merged = self._read()
                    merged.update(changed)
                    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=self.path.parent,
                                                     prefix='.card-catalogue-', delete=False) as handle:
                        temporary = Path(handle.name)
                        json.dump({'schema_version': SCHEMA_VERSION, 'cards': merged}, handle, ensure_ascii=False, indent=2, sort_keys=True)
                        handle.write('\n')
                    os.replace(temporary, self.path)
                    temporary = None
                    self._entries.update(merged)
                    self.write_error = None
            except OSError as error:
                if self.write_error != str(error):
                    LOG.warning('Card catalogue updated in memory but could not be saved: %s', error)
                self.write_error = str(error)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            return True

    def capture(self, card):
        file = getattr(card, 'file', None)
        if file is None:
            return False
        file = Path(file).resolve()
        name = file.stem
        candidate = discover_cards(self.root).get(name)
        if candidate is None or candidate.resolve() != file:
            return False
        digest = source_hash(file)
        if digest is None:
            return False
        return self.update({name: {
            'module': name, 'long_name': plain_text(getattr(card, 'long_name', None)),
            'description': plain_text(getattr(card, 'description', None)),
            'mutable': getattr(card, 'mutable', None) if type(getattr(card, 'mutable', None)) is bool else None,
            'pipeline_compatible': getattr(card, 'pipeline_compatible', None) if type(getattr(card, 'pipeline_compatible', None)) is bool else None,
            'source_hash': digest,
        }})


async def rebuild_catalogue(catalogue, *, timeout=180):
    """Construct cards only in a child process, without UI/server startup."""
    with tempfile.TemporaryDirectory(prefix='pythagoras-card-catalogue-') as directory:
        output = Path(directory) / 'result.json'
        environment = dict(os.environ, PYTHAGORAS_CATALOGUE_WORKER='1')
        process = await asyncio.create_subprocess_exec(
            sys.executable, str(Path(__file__).resolve()), '--worker', '--root', str(catalogue.root.resolve()),
            '--output', str(output), cwd=catalogue.root, env=environment,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
        try:
            _, stderr = await asyncio.wait_for(process.communicate(), timeout)
            if process.returncode:
                raise RuntimeError('Catalogue worker failed: ' + stderr.decode(errors='replace')[-2000:])
            payload = json.loads(output.read_text(encoding='utf-8'))
            entries = valid_entries(payload)
            # Ignore results if their source changed while the worker was running.
            paths = discover_cards(catalogue.root)
            entries = {name: entry for name, entry in entries.items()
                       if name in paths and entry['source_hash'] == source_hash(paths[name])}
            catalogue.update(entries)
            return {'updated': len(entries), 'errors': payload.get('errors', {}), 'write_error': catalogue.write_error}
        finally:
            if process.returncode is None:
                process.kill()
                await process.communicate()


def build_catalogue(root):
    root = Path(root).resolve()
    sys.path.insert(0, str(root))
    os.chdir(root)
    from module import Module
    catalogue = CardCatalogue(root, persistent=False)
    # Eager output contains freshly inspected entries only. Failed cards retain
    # their previous entries when the parent merges this result.
    catalogue._entries = {}
    errors = {}
    for name in discover_cards(root):
        try:
            card = importlib.import_module(f'cards.{name}').instance()
            if not catalogue.capture(card):
                raise ValueError('Factory did not return a card with a matching source file')
        except Exception as error:
            errors[name] = f'{type(error).__name__}: {error}'
        finally:
            # This is the isolated worker's registry, never the live app registry.
            Module.Instances.clear()
    return dict(catalogue.snapshot(), errors=errors)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    os.environ['PYTHAGORAS_CATALOGUE_WORKER'] = '1'
    output_path = arguments.output.resolve()
    payload = build_catalogue(arguments.root)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
