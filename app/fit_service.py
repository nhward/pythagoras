"""Session-only fitted artifacts and cross-process, training-priority fit leases.

No paths, fitted models or predictions are added to portable search evidence.
SQLite coordinates independent search pools and ensemble workers; it never holds
large model/data blobs. Native workers exchange private artifact IDs, not models.
"""
from __future__ import annotations

import atexit
import hashlib
import inspect
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import threading
import time
import uuid
import weakref
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache, wraps
from importlib import metadata
from pathlib import Path

import joblib
from fit_responses import collect_responses
from sklearn.base import clone

_LOADED = {}
_VERSION_LOCK = threading.Lock()
_VERSIONS = None


def dependency_identity():
    global _VERSIONS
    with _VERSION_LOCK:
        if _VERSIONS is None:
            # IDE runners can append their bundled library paths again when
            # spawn re-executes them. Repeated distributions do not change the
            # environment and must not change parent/worker fit identities.
            _VERSIONS = tuple(sorted({(d.metadata['Name'].lower().replace('_', '-'), d.version)
                                      for d in metadata.distributions() if d.metadata['Name']}))
    return _VERSIONS


@lru_cache(maxsize=256)
def _source_digest(path, modified, size):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pipeline_implementation(pipeline):
    paths = set()
    for value in (pipeline, *pipeline.get_params(deep=True).values()):
        try:
            path = inspect.getsourcefile(value if callable(value) else type(value))
            if path and Path(path).is_file():
                paths.add(Path(path).resolve())
        except TypeError:
            pass
    return tuple((str(p), _source_digest(str(p), p.stat().st_mtime_ns, p.stat().st_size)) for p in sorted(paths))


def fit_key(context, pipeline, train, validation):
    # Context already fingerprints stable data, roles, target, task and seed.
    return joblib.hash((context, pipeline, train, validation, dependency_identity(), pipeline_implementation(pipeline)))


def _alive(pid, created):
    try:
        import psutil
        process = psutil.Process(pid)
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE and process.create_time() == created
    except (ProcessLookupError, OSError, psutil.Error):
        return False


def _process():
    import psutil
    return os.getpid(), psutil.Process().create_time()


def _closed_cleanup(method):
    """Late card/task cleanup is a no-op after the session cache is removed."""
    @wraps(method)
    def cleanup(self, *args, **kwargs):
        path = Path(self.root, 'index.sqlite')
        if not path.exists():
            return None
        try:
            return method(self, *args, **kwargs)
        except sqlite3.OperationalError:
            if path.exists():
                raise  # Do not hide database faults in a live session.
            return None
    return cleanup


@dataclass(frozen=True)
class FitRuntime:
    root: str
    capacity: int
    memory_budget: int
    cache_budget: int = 2 * 1024**3
    training_ensemble_limit: int = 1

    @contextmanager
    def database(self):
        # Never recreate an empty database if a late callback follows deletion.
        connection = sqlite3.connect(Path(self.root, 'index.sqlite').as_uri() + '?mode=rw', uri=True, timeout=30)
        try:
            connection.execute('PRAGMA busy_timeout=30000')
            connection.execute('BEGIN IMMEDIATE')
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def enabled(self):
        with self.database() as db:
            return bool(db.execute("SELECT value FROM settings WHERE name='reuse'").fetchone()[0])

    def enable(self):
        with self.database() as db:
            db.execute("UPDATE settings SET value=1 WHERE name='reuse'")

    def count(self, name, amount=1):
        with self.database() as db:
            db.execute('INSERT INTO counters VALUES (?,?) ON CONFLICT(name) DO UPDATE SET value=value+excluded.value', (name, amount))

    def _reap(self, db):
        owners = db.execute('SELECT DISTINCT pid,created FROM leases UNION SELECT pid,created FROM searches').fetchall()
        for pid, created in owners:
            if not _alive(pid, created):
                db.execute('DELETE FROM leases WHERE pid=? AND created=?', (pid, created))
                db.execute('DELETE FROM searches WHERE pid=? AND created=?', (pid, created))

    def register_search(self, token):
        pid, created = _process()
        with self.database() as db:
            db.execute('INSERT OR REPLACE INTO searches VALUES (?,?,?)', (token, pid, created))

    @_closed_cleanup
    def finish_search(self, token):
        with self.database() as db:
            db.execute('DELETE FROM searches WHERE token=?', (token,))
            db.execute('DELETE FROM leases WHERE search=?', (token,))

    def ensemble_width(self):
        with self.database() as db:
            self._reap(db)
            training = db.execute('SELECT count(*) FROM searches').fetchone()[0]
            limit = min(self.capacity, self.training_ensemble_limit) if training else self.capacity
            return self._background_limit(db, limit)

    def _background_limit(self, db, limit):
        # Assessment of the passed-through model must not queue behind a slow
        # speculative ensemble member. Keep one non-training slot available
        # while an assessment card is present, within the same resource budget.
        if limit > 1 and db.execute('SELECT 1 FROM assessments LIMIT 1').fetchone():
            return limit - 1
        return limit

    @contextmanager
    def lease(self, kind, estimate=0, cancelled=None, search=''):
        token = uuid.uuid4().hex
        pid, created = _process()
        with self.database() as db:
            db.execute('INSERT INTO leases VALUES (?,?,?,?,?,?,?)',
                       (token, pid, created, kind, 'waiting', estimate, search))
        started = time.monotonic()
        try:
            while True:
                if cancelled is not None and cancelled.is_set():
                    raise InterruptedError('Fit cancelled while waiting for capacity.')
                with self.database() as db:
                    self._reap(db)
                    if db.execute("SELECT value FROM settings WHERE name='closed'").fetchone()[0]:
                        raise InterruptedError('Fit session closed.')
                    rows = db.execute('SELECT kind,bytes FROM leases WHERE state="running"').fetchall()
                    training = db.execute('SELECT count(*) FROM searches').fetchone()[0]
                    waiting_train = db.execute('SELECT count(*) FROM leases WHERE kind="train" AND state="waiting"').fetchone()[0]
                    limit = min(self.capacity, self.training_ensemble_limit) if training else self.capacity
                    allowed = len(rows) < self.capacity and (not rows or sum(r[1] for r in rows) + estimate <= self.memory_budget)
                    if kind in {'ensemble', 'assessment'}:
                        allowed = allowed and not waiting_train and sum(r[0] in {'ensemble', 'assessment'} for r in rows) < limit
                    if kind == 'ensemble':
                        allowed = allowed and sum(r[0] == 'ensemble' for r in rows) < self._background_limit(db, limit)
                    if allowed:
                        db.execute('UPDATE leases SET state="running" WHERE token=?', (token,))
                        break
                time.sleep(.025)
            self.count(kind + '_queue_seconds', time.monotonic() - started)
            yield
        finally:
            try:
                with self.database() as db:
                    db.execute('DELETE FROM leases WHERE token=?', (token,))
            except Exception:  # noqa: BLE001, S110
                pass  # Session closure can remove the private cache directory.

    def path(self, key):
        if len(key) != 32 or any(c not in '0123456789abcdef' for c in key):
            raise ValueError('Invalid private fit artifact ID.')
        return Path(self.root) / (key + '.joblib')

    def put(self, key, model, responses, owner, search=None, trial=None):
        temporary = Path(self.root) / (uuid.uuid4().hex + '.tmp')
        started = time.monotonic()
        try:
            joblib.dump((model, responses), temporary)
            # Round-trip now: some providers serialize but cannot deserialize.
            joblib.load(temporary)
            with self.database() as db:
                path = self.path(key)
                os.replace(temporary, path)
                db.execute('INSERT OR REPLACE INTO artifacts VALUES (?,?,?)', (key, path.stat().st_size, time.time()))
                db.execute('INSERT OR IGNORE INTO pins VALUES (?,?)', (owner, key))
                if search is not None:
                    db.execute('INSERT OR IGNORE INTO tags VALUES (?,?,?)', (search, trial, key))
            self.count('serialization_seconds', time.monotonic() - started)
            self.prune()
            return None
        except Exception as error:  # noqa: BLE001
            # Artifact reuse is optional: a provider/disk failure must not turn
            # an otherwise successful training fold into a failed trial.
            try:
                self.count('serialization_failures')
            except Exception:  # noqa: BLE001, S110
                pass
            return f'{type(error).__name__}: {error}'
        finally:
            temporary.unlink(missing_ok=True)

    def remember(self, key, model, responses):
        try:
            model._pythagoras_validation_responses = responses
            _LOADED.setdefault(self.root, weakref.WeakValueDictionary())[key] = model
        except (TypeError, AttributeError):
            pass

    def get(self, key, owner):
        loaded = _LOADED.setdefault(self.root, weakref.WeakValueDictionary())
        local = loaded.get(key)
        # Nonserializable providers may still be reused while a runtime ensemble
        # holds their fitted object. No strong global memory cache is introduced.
        if local is not None and hasattr(local, '_pythagoras_validation_responses'):
            with self.database() as db:
                if db.execute('SELECT key FROM artifacts WHERE key=?', (key,)).fetchone():
                    db.execute('INSERT OR IGNORE INTO pins VALUES (?,?)', (owner, key))
            self.count('cache_hits')
            return local, local._pythagoras_validation_responses
        # Pin under the same transaction as lookup: pruning cannot race a load.
        with self.database() as db:
            if db.execute('SELECT key FROM artifacts WHERE key=?', (key,)).fetchone() is None:
                return None
            db.execute('INSERT OR IGNORE INTO pins VALUES (?,?)', (owner, key))
            db.execute('UPDATE artifacts SET accessed=? WHERE key=?', (time.time(), key))
        loaded = _LOADED.setdefault(self.root, weakref.WeakValueDictionary())
        try:
            model = loaded.get(key)
            if model is None:
                model, responses = joblib.load(self.path(key))
                try:
                    loaded[key] = model
                    model._pythagoras_validation_responses = responses
                except (TypeError, AttributeError):
                    pass
            else:
                responses = getattr(model, '_pythagoras_validation_responses', None)
                if responses is None:
                    _, responses = joblib.load(self.path(key))
            self.count('cache_hits')
            return model, responses
        except Exception:  # noqa: BLE001
            with self.database() as db:
                db.execute('DELETE FROM artifacts WHERE key=?', (key,))
                db.execute('DELETE FROM pins WHERE key=?', (key,))
            self.path(key).unlink(missing_ok=True)
            self.count('cache_misses_corrupt')
            return None

    def register_review(self, owner):
        with self.database() as db:
            db.execute('INSERT OR IGNORE INTO reviewers VALUES (?)', (owner,))

    @_closed_cleanup
    def remove_review(self, owner):
        with self.database() as db:
            db.execute('DELETE FROM reviewers WHERE owner=?', (owner,))
            db.execute('DELETE FROM pins WHERE owner=?', (owner,))
            if not db.execute('SELECT 1 FROM reviewers LIMIT 1').fetchone():
                db.execute("DELETE FROM pins WHERE owner LIKE 'pending:%'")
        self.prune()

    def retire_unreviewed(self, search, trial):
        with self.database() as db:
            if not db.execute('SELECT 1 FROM reviewers LIMIT 1').fetchone():
                db.execute('DELETE FROM pins WHERE owner=?', ('pending:' + search + ':' + trial,))
        self.prune()

    def pin_review(self, owner, search, winners, reviewed):
        if not self.enabled():
            return
        with self.database() as db:
            db.execute('DELETE FROM pins WHERE owner=?', (owner,))
            for trial in winners:
                db.execute('INSERT OR IGNORE INTO pins SELECT ?,key FROM tags WHERE search=? AND trial=?', (owner, search, trial))
            losing = []
            for trial in reviewed:
                pending = 'pending:' + search + ':' + trial
                db.execute('DELETE FROM pins WHERE owner=?', (pending,))
                if trial not in winners:
                    losing.extend(row[0] for row in db.execute('SELECT key FROM tags WHERE search=? AND trial=?', (search, trial)))
            self._remove_unpinned(db, losing)
        self.prune()

    def _remove_unpinned(self, db, keys):
        for key in keys:
            if db.execute('SELECT 1 FROM pins WHERE key=? LIMIT 1', (key,)).fetchone() is None:
                self.path(key).unlink(missing_ok=True)
                db.execute('DELETE FROM artifacts WHERE key=?', (key,))
                db.execute('DELETE FROM tags WHERE key=?', (key,))

    @_closed_cleanup
    def release(self, owner):
        with self.database() as db:
            db.execute('DELETE FROM pins WHERE owner=?', (owner,))
        self.prune()

    @_closed_cleanup
    def release_search(self, search):
        with self.database() as db:
            db.execute('DELETE FROM pins WHERE owner LIKE ?', ('pending:' + search + ':%',))
        self.prune()

    @_closed_cleanup
    def prune(self):
        with self.database() as db:
            rows = db.execute('SELECT a.key,a.bytes FROM artifacts a WHERE NOT EXISTS (SELECT 1 FROM pins p WHERE p.key=a.key) ORDER BY accessed DESC').fetchall()
            total = 0
            for key, size in rows:
                total += size
                if total > self.cache_budget:
                    self._remove_unpinned(db, [key])

    def put_context(self, key, frame, target, owner):
        path = Path(self.root) / ('context-' + key + '.joblib')
        with self.database() as db:
            if not path.exists():
                temporary = Path(self.root) / (uuid.uuid4().hex + '.tmp')
                try:
                    joblib.dump((frame, target), temporary)
                    os.replace(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
            db.execute('INSERT OR IGNORE INTO contexts VALUES (?,?)', (owner, str(path)))
        return str(path)

    @_closed_cleanup
    def release_context(self, owner):
        with self.database() as db:
            paths = [r[0] for r in db.execute('SELECT path FROM contexts WHERE owner=?', (owner,))]
            db.execute('DELETE FROM contexts WHERE owner=?', (owner,))
            for path in paths:
                if db.execute('SELECT 1 FROM contexts WHERE path=?', (path,)).fetchone() is None:
                    Path(path).unlink(missing_ok=True)

    def snapshot(self):
        with self.database() as db:
            self._reap(db)
            result = dict(db.execute('SELECT name,value FROM counters'))
            result['pinned_bytes'] = db.execute('SELECT COALESCE(sum(bytes),0) FROM artifacts WHERE key IN (SELECT key FROM pins)').fetchone()[0]
            result['evictable_bytes'] = db.execute('SELECT COALESCE(sum(bytes),0) FROM artifacts WHERE key NOT IN (SELECT key FROM pins)').fetchone()[0]
            result['context_bytes'] = sum(p.stat().st_size for p in Path(self.root).glob('context-*.joblib'))
            result['training_searches'] = db.execute('SELECT count(*) FROM searches').fetchone()[0]
            for state in ('waiting', 'running'):
                result['ensemble_' + state] = db.execute('SELECT count(*) FROM leases WHERE kind="ensemble" AND state=?', (state,)).fetchone()[0]
            return result


def _load_context(path):
    # Disk/OS page cache shares the serialized context without retaining large
    # frames in idle pool processes outside the active-fit memory allowance.
    return joblib.load(path)


def ensemble_fit_job(runtime, context_path, key, pipeline, train, validation, task, estimate, owner):
    from threadpoolctl import threadpool_limits
    frame, target = _load_context(context_path)
    with runtime.lease('ensemble', estimate), threadpool_limits(1):
        if runtime.get(key, owner) is not None:
            runtime.count('ensemble_reused_after_wait')
            return key, None, False
        started = time.monotonic()
        model = clone(pipeline).fit(frame.iloc[train].copy(), frame[target].iloc[train].copy())
        runtime.count('ensemble_fits')
        runtime.count('ensemble_fit_seconds', time.monotonic() - started)
        responses = collect_responses(model, frame.iloc[validation], task)
        runtime.count('prediction_batches')
        error = runtime.put(key, model, responses, owner)
        return key, error, True


class MemoryFitService:
    """Sequential browser fallback; only current members/snapshots are retained."""
    runtime = None
    def __init__(self):
        self.entries = {}
    def close(self):
        self.entries.clear()


class FitService:
    def __init__(self, *, cpus=None, memory_budget=None, cache_budget=2 * 1024**3, directory=None):
        self.root = tempfile.mkdtemp(prefix='pythagoras-fit-', dir=directory)
        cpus = joblib.cpu_count() if cpus is None else cpus
        if memory_budget is None:
            import psutil
            memory_budget = psutil.virtual_memory().total // 4
        self.runtime = FitRuntime(self.root, max(1, cpus - 1), int(memory_budget), cache_budget, max(1, cpus // 5))
        self.executor = None
        self._lock = threading.Lock()
        self._snapshots = {}
        self._consumers = set()
        self.local_only = set()
        self._closed = False
        sqlite3.connect(str(Path(self.root, 'index.sqlite'))).close()
        with self.runtime.database() as db:
            for statement in (
                'CREATE TABLE settings(name TEXT PRIMARY KEY,value INTEGER)',
                'CREATE TABLE artifacts(key TEXT PRIMARY KEY,bytes INTEGER,accessed REAL)',
                'CREATE TABLE pins(owner TEXT,key TEXT,PRIMARY KEY(owner,key))',
                'CREATE TABLE tags(search TEXT,trial TEXT,key TEXT,PRIMARY KEY(search,trial,key))',
                'CREATE TABLE searches(token TEXT PRIMARY KEY,pid INTEGER,created REAL)',
                'CREATE TABLE reviewers(owner TEXT PRIMARY KEY)',
                'CREATE TABLE assessments(owner TEXT PRIMARY KEY)',
                'CREATE TABLE leases(token TEXT PRIMARY KEY,pid INTEGER,created REAL,kind TEXT,state TEXT,bytes INTEGER,search TEXT)',
                'CREATE TABLE contexts(owner TEXT,path TEXT,PRIMARY KEY(owner,path))',
                'CREATE TABLE counters(name TEXT PRIMARY KEY,value REAL)'):
                db.execute(statement)
            db.execute("INSERT INTO settings VALUES ('reuse',0)")
            db.execute("INSERT INTO settings VALUES ('closed',0)")
        Path(self.root, 'owner.json').write_text(json.dumps(_process()))
        atexit.register(self.close)

    def enable(self, consumer='default', *, assessment=False):
        self._consumers.add(consumer)
        if assessment:
            with self.runtime.database() as db:
                db.execute('INSERT OR IGNORE INTO assessments VALUES (?)', (consumer,))
        self.runtime.enable()

    def remove_consumer(self, consumer):
        self._consumers.discard(consumer)
        owner = self._snapshots.pop(consumer, None)
        if self._closed:
            return
        with self.runtime.database() as db:
            db.execute('DELETE FROM assessments WHERE owner=?', (consumer,))
        if owner:
            self.runtime.release(owner)
        if not self._consumers:
            with self.runtime.database() as db:
                db.execute("UPDATE settings SET value=0 WHERE name='reuse'")
                db.execute('DELETE FROM pins')
            self.runtime.prune()

    def submit(self, *args):
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        with self._lock:
            if self._closed:
                raise InterruptedError('Fit session closed.')
            if self.executor is None:
                self.executor = ProcessPoolExecutor(max_workers=self.runtime.capacity, mp_context=mp.get_context('spawn'))
            return self.executor.submit(ensemble_fit_job, self.runtime, *args)

    def publish_snapshot(self, card, owner):
        previous = self._snapshots.get(card)
        self._snapshots[card] = owner
        if previous and previous != owner:
            self.runtime.release(previous)

    def stop(self):
        if not self._closed:
            with self.runtime.database() as db:
                db.execute("UPDATE settings SET value=1 WHERE name='closed'")

    def close(self):
        with self._lock:
            if self._closed:
                return
            self.stop()
            self._closed = True
            if self.executor is not None:
                processes = list((self.executor._processes or {}).values())
                self.executor.shutdown(wait=False, cancel_futures=True)
                for process in processes:
                    process.join(timeout=.1)
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=.5)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=.5)
            _LOADED.pop(self.root, None)
            shutil.rmtree(self.root, ignore_errors=True)
        atexit.unregister(self.close)


def cleanup_abandoned(directory=None):
    for path in Path(directory or tempfile.gettempdir()).glob('pythagoras-fit-*'):
        try:
            pid, created = json.loads((path / 'owner.json').read_text())
            if not _alive(pid, created):
                shutil.rmtree(path)
        except (OSError, ValueError, TypeError):
            continue


def session_fit_service(session):
    root = session.root_scope()
    service = getattr(root, '_pythagoras_fit_service', None)
    if service is None:
        if sys.platform == 'emscripten':
            service = MemoryFitService()
        else:
            cleanup_abandoned()
            service = FitService()
        root._pythagoras_fit_service = service
        # Training/card callbacks close their work first; session destruction is
        # the final boundary for the shared service and its private artifacts.
        if isinstance(service, FitService):
            root.on_ended(service.stop)
        root.on_destroy(service.close)
    return service
