"""Publish completed search evidence periodically without slowing worker updates."""
import math
from copy import deepcopy

import joblib


class CompletedTrialPublication:
    def __init__(self):
        self.snapshot = None
        self._revision = None
        self._published_at = None

    def update(self, checkpoint, interval, now, *, force=False):
        """Return (changed, retry_after); progress never restarts the interval."""
        if not isinstance(interval, (int, float)) or not math.isfinite(interval) or interval < 0:
            raise ValueError('Choose a finite, nonnegative result refresh interval.')
        candidate = None if checkpoint is None else {
            'specification': checkpoint['specification'],
            'trials': [trial for trial in checkpoint['trials'] if trial['state'] == 'complete'],
        }
        revision = joblib.hash(None if candidate is None else {
            'fingerprint': candidate['specification']['fingerprint'],
            'trials': candidate['trials'],
        })
        if revision == self._revision:
            return False, None
        previous = self.snapshot
        reset = (candidate is None or previous is None
                 or candidate['specification']['fingerprint'] != previous['specification']['fingerprint']
                 or not {t['id'] for t in previous['trials']} <= {t['id'] for t in candidate['trials']})
        remaining = interval - (now - self._published_at) if self._published_at is not None else 0
        if not (force or reset or checkpoint['status'] != 'running') and remaining > 0:
            return False, remaining
        self.snapshot = deepcopy(candidate)
        self._revision = revision
        self._published_at = now
        return True, None
