"""Bounded review summaries; only unfinished trials need temporary loss arrays.

Completed checkpoints are file-free. Portable unfinished checkpoints embed their
compressed arrays, never host paths. Browser restoration uses compressed bytes
instead of the native temporary-file backend. No outcomes, predictions, row IDs,
class matrices, fitted estimators or executable serialization are retained here.
"""
import base64
import copy
import hashlib
import io
import re
import sys
import tempfile
import uuid
import zipfile
from pathlib import Path

import numpy as np

_TEMP = None
_MEMORY = {}
_KEY = re.compile(r'[a-f0-9]{32}_[a-f0-9]{32}')
CURVE_POINTS = 101


def directory():
    global _TEMP
    if sys.platform == 'emscripten':
        return None
    if _TEMP is None:
        _TEMP = tempfile.TemporaryDirectory(prefix='pythagoras-losses-')
    return _TEMP.name


def _path(key, root=None):
    if not isinstance(key, str) or not _KEY.fullmatch(key):
        raise ValueError('Invalid temporary evidence key.')
    return Path(root or directory()) / (key + '.npz')


def _digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def write_losses(losses, root=None, namespace=None):
    key = (namespace or uuid.uuid4().hex) + '_' + uuid.uuid4().hex
    values = np.asarray(losses, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Invalid validation loss array.')
    if sys.platform == 'emscripten':
        stream = io.BytesIO()
        np.savez_compressed(stream, losses=values)
        payload = stream.getvalue()
        _MEMORY[key] = payload
        digest = hashlib.sha256(payload).hexdigest()
    else:
        path = _path(key, root)
        try:
            np.savez_compressed(path, losses=values)
            digest = _digest(path)
        except BaseException:
            path.unlink(missing_ok=True)
            raise
    return {'key': key, 'sha256': digest}


def read_losses(reference, root=None):
    try:
        if sys.platform == 'emscripten':
            payload = _MEMORY.get(reference['key'])
            if payload is None:
                raise FileNotFoundError
            checksum = hashlib.sha256(payload).hexdigest()
            source = io.BytesIO(payload)
        else:
            source = _path(reference['key'], root)
            checksum = _digest(source)
    except FileNotFoundError as exc:
        raise ValueError('Unfinished prediction evidence is unavailable; reload its portable checkpoint.') from exc
    if checksum != reference['sha256']:
        raise ValueError('Prediction-evidence checksum does not match.')
    with np.load(source, allow_pickle=False) as archive:
        values = archive['losses']
    if values.ndim != 1 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Invalid validation loss array.')
    return values


def references(state):
    return {fold['evidence']['key']: fold['evidence']
            for trial in (state or {}).get('trials', []) for fold in trial['folds']
            if 'key' in fold.get('evidence', {})}


def remove(keys, root=None):
    for key in keys:
        if sys.platform == 'emscripten':
            _MEMORY.pop(key, None)
        else:
            _path(key, root).unlink(missing_ok=True)


def prune_namespace(namespace, keep=(), root=None):
    if not namespace:
        return
    if sys.platform == 'emscripten':
        remove([k for k in _MEMORY if k.startswith(namespace + '_') and k not in keep])
    else:
        for path in Path(root or directory()).glob(namespace + '_*.npz'):
            if path.stem not in keep:
                path.unlink(missing_ok=True)


def portable_checkpoint(state):
    """Explicit save/export boundary; completed history contains no binary data."""
    if state is None:
        return None
    result = copy.deepcopy(state)
    blobs = {}
    for key, reference in references(state).items():
        # Saving does not need to materialize the validation arrays.
        try:
            payload = _MEMORY[key] if sys.platform == 'emscripten' else _path(key).read_bytes()
        except (KeyError, FileNotFoundError) as exc:
            raise ValueError('Unfinished prediction evidence is unavailable; reload its portable checkpoint.') from exc
        if hashlib.sha256(payload).hexdigest() != reference['sha256']:
            raise ValueError('Prediction-evidence checksum does not match.')
        blobs[key] = base64.b64encode(payload).decode('ascii')
    if blobs:
        result['pending_evidence'] = {'version': 1, 'blobs': blobs}
    return result


def restore_checkpoint(checkpoint, root=None):
    """Rekey each import, preventing configuration switches from sharing ownership."""
    if checkpoint is None:
        return None
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get('trials'), list):
        raise TypeError('Invalid search checkpoint.')
    if any(not isinstance(t, dict) or not isinstance(t.get('folds'), list) for t in checkpoint['trials']):
        raise ValueError('Invalid trial records.')
    if any(not isinstance(f, dict) or not isinstance(f.get('evidence', {}), dict)
           for t in checkpoint['trials'] for f in t['folds']):
        raise ValueError('Invalid validation diagnostics.')
    # Old configurations can contain gigabytes of redundant row lists. Do not
    # duplicate those lists merely to discard them; preserve their score history.
    legacy = any({'predicted', 'observed', 'probabilities'} & fold.get('evidence', {}).keys()
                 for trial in checkpoint.get('trials', []) for fold in trial['folds'])
    if legacy:
        result = {k: copy.deepcopy(v) for k, v in checkpoint.items() if k != 'trials'}
        result['trials'] = []
        for trial in checkpoint['trials']:
            item = {k: copy.deepcopy(v) for k, v in trial.items() if k != 'folds'}
            item['folds'] = []
            for fold in trial['folds']:
                scores = {k: copy.deepcopy(v) for k, v in fold.items() if k != 'evidence'}
                scores['evidence'] = {'error': 'Legacy row-level predictions discarded; start a new search for compact diagnostics.'}
                item['folds'].append(scores)
            result['trials'].append(item)
        result['legacy_evidence_removed'] = True
    else:
        result = copy.deepcopy(checkpoint)
    package = result.pop('pending_evidence', None)
    refs = references(result)
    for key, reference in refs.items():
        if (not isinstance(key, str) or not _KEY.fullmatch(key) or type(reference.get('count')) is not int or reference['count'] < 1
                or not isinstance(reference.get('sha256'), str) or not re.fullmatch('[a-f0-9]{64}', reference['sha256'])):
            raise ValueError('Invalid unfinished evidence reference.')
    if package is None:
        # Runtime checkpoints must resolve in the explicitly shared worker store.
        if any((key not in _MEMORY if sys.platform == 'emscripten' else not _path(key, root).exists()) for key in refs):
            raise ValueError('Unfinished prediction evidence is unavailable; reload its portable checkpoint.')
        return result
    if not isinstance(package, dict) or package.get('version') != 1 or not isinstance(package.get('blobs'), dict) or set(package['blobs']) != set(refs):
        raise ValueError('Invalid unfinished-evidence package.')
    created = []
    try:
        for key, encoded in package['blobs'].items():
            payload = base64.b64decode(encoded, validate=True)
            if hashlib.sha256(payload).hexdigest() != refs[key]['sha256']:
                raise ValueError('Prediction-evidence checksum does not match.')
            # Enforce the declared uncompressed size before loading an untrusted archive.
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                if archive.namelist() != ['losses.npy'] or archive.infolist()[0].file_size > 1024 + refs[key]['count'] * 8:
                    raise ValueError('Invalid or oversized unfinished loss archive.')
                with archive.open('losses.npy') as stream:
                    version = np.lib.format.read_magic(stream)
                    if version != (1, 0):
                        raise ValueError('Unsupported loss-array format.')
                    shape, order, dtype = np.lib.format.read_array_header_1_0(stream)
                    if shape != (refs[key]['count'],) or dtype.kind != 'f' or dtype.itemsize != 8 or order:
                        raise ValueError('Unfinished evidence has the wrong validation count or type.')
            # Import compressed bytes directly; do not decompress every pending
            # fold merely to switch configurations, especially in shinylive.
            new = {'key': uuid.uuid4().hex + '_' + uuid.uuid4().hex, 'sha256': refs[key]['sha256']}
            if sys.platform == 'emscripten':
                _MEMORY[new['key']] = payload
            else:
                _path(new['key'], root).write_bytes(payload)
            created.append(new['key'])
            for trial in result['trials']:
                for fold in trial['folds']:
                    if fold.get('evidence', {}).get('key') == key:
                        fold['evidence'].update(new)
    except Exception as exc:
        remove(created, root)
        if isinstance(exc, (ValueError, OSError)):
            raise
        raise ValueError('Invalid unfinished prediction-evidence archive.') from exc
    return result


def summarize_trial(trial, spec, root=None):
    """Exact repeat-wise P95, plus the same 101-point curves used by review.

    At most one repeat's losses are materialized. Completed trials discard their
    loss references; only bounded summaries and scalar fold diagnostics remain.
    """
    evidence = [f.get('evidence', {}) for f in trial['folds']]
    retired = [e['key'] for e in evidence if 'key' in e]
    try:
        if not evidence or any('count' not in e for e in evidence):
            raise ValueError('Validation diagnostics are unavailable for one or more folds.')
        probability_loss = spec['task'] == 'classification' and all(e['loss_kind'] == 'log_loss' for e in evidence)
        kind = 'absolute_error' if spec['task'] == 'regression' else 'log_loss' if probability_loss else 'classification_error'
        maximum = max(e.get('maximum', 0.) for e in evidence) if kind != 'classification_error' else 1.
        thresholds = np.linspace(0., maximum, CURVE_POINTS)
        descriptors = spec.get('evaluation', {}).get('split_descriptors', [])
        groups = {}
        for fold_id, e in zip(trial['fold_ids'], evidence):
            repeat = descriptors[fold_id]['repeat'] if len(descriptors) == spec['folds'] else 1
            groups.setdefault(repeat, []).append(e)
        curves, tails, errors, means = [], [], [], []
        for group in groups.values():
            count = sum(e['count'] for e in group)
            if count < 1:
                raise ValueError('Empty validation diagnostics.')
            if spec['task'] == 'classification':
                errors.append(sum(e['error_count'] for e in group) / count)
            if kind == 'classification_error':
                curves.append(np.where(thresholds < 1, errors[-1], 0.))
            else:
                values = np.empty(count, dtype=np.float64)
                offset = 0
                for e in group:
                    chunk = read_losses(e, root)
                    if len(chunk) != e['count']:
                        raise ValueError('Temporary evidence has the wrong validation count.')
                    values[offset:offset + len(chunk)] = chunk
                    offset += len(chunk)
                    del chunk
                values.sort()
                curves.append(1 - np.searchsorted(values, thresholds, side='right') / len(values))
                if kind == 'absolute_error':
                    tails.append(float(np.quantile(values, .95, overwrite_input=True)))
                else:
                    means.append(float(values.mean()))
        summary = {'version': 2, 'loss_kind': kind, 'maximum': float(maximum),
                   'exceedance': np.mean(curves, axis=0).tolist()}
        if tails:
            summary['p95_absolute_error'] = float(np.mean(tails))
        if errors:
            summary['error_rate'] = float(np.mean(errors))
            summary['log_loss'] = float(np.mean(means)) if means else None
        trial['diagnostics'] = summary
    except (ValueError, OSError, KeyError) as exc:
        trial['diagnostics'] = {'error': str(exc)}
    for e in evidence:
        for field in ('key', 'sha256', 'maximum', 'context'):
            e.pop(field, None)
    return retired
