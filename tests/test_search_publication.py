from copy import deepcopy

import pytest
from search_publication import CompletedTrialPublication


def checkpoint(count, *, status='running', fingerprint='search'):
    return {'specification': {'fingerprint': fingerprint}, 'status': status,
            'trials': [{'id': i, 'state': 'complete', 'folds': [i]} for i in range(count)]}


def test_continuous_completions_publish_latest_at_interval_without_waiting_for_quiet():
    publication = CompletedTrialPublication()
    assert publication.update(checkpoint(0), 5, 0) == (True, None)
    for now in range(1, 5):
        assert publication.update(checkpoint(now), 5, now) == (False, 5 - now)
        assert publication.snapshot['trials'] == []
    latest = checkpoint(5)
    assert publication.update(latest, 5, 5) == (True, None)
    latest['trials'].clear()
    assert len(publication.snapshot['trials']) == 5
    assert publication.update(checkpoint(6), 5, 6) == (False, 4)


def test_fold_only_progress_does_not_publish_or_start_another_timer():
    publication = CompletedTrialPublication()
    initial = checkpoint(1)
    publication.update(initial, 5, 0)
    current = deepcopy(initial)
    current.update(elapsed=99, error='')
    current['trials'].append({'id': 1, 'state': 'running', 'folds': [1, 2]})
    assert publication.update(current, 5, 1) == (False, None)


@pytest.mark.parametrize('status', ['exhausted', 'paused', 'budget reached', 'error'])
def test_stopped_search_publishes_final_evidence_without_waiting(status):
    publication = CompletedTrialPublication()
    publication.update(checkpoint(0), 60, 0)
    assert publication.update(checkpoint(3, status=status), 60, 1) == (True, None)
    assert len(publication.snapshot['trials']) == 3


def test_interval_changes_acceptance_resets_and_upstream_identity():
    publication = CompletedTrialPublication()
    publication.update(checkpoint(0), 30, 0)
    assert publication.update(checkpoint(1), 30, 2) == (False, 28)
    assert publication.update(checkpoint(1), 1, 2) == (True, None)
    assert publication.update(checkpoint(2), 30, 3, force=True) == (True, None)
    assert publication.update(checkpoint(0), 30, 4) == (True, None)
    assert publication.update(checkpoint(1, fingerprint='new'), 30, 5) == (True, None)
    assert publication.update(None, 30, 6) == (True, None)


@pytest.mark.parametrize('interval', [-1, float('nan'), float('inf'), None])
def test_invalid_interval_is_rejected(interval):
    with pytest.raises(ValueError, match='refresh interval'):
        CompletedTrialPublication().update(checkpoint(0), interval, 0)
