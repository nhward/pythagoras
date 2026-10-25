"""Source coverage and conservative package discovery contracts."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'app/data/estimator_capabilities'


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_audit_accounts_for_every_source_entry():
    builder = script('build_estimator_catalogue')
    registry = builder.load_registry()
    snapshot = json.loads((DIRECTORY / 'audit_sources.json').read_text())
    expected = {(source['id'], project['name']) for source in snapshot['sources']
                for project in source['projects']}
    references = [tuple(ref) for row in registry['packages'] for ref in row['source_entries']]
    assert len(references) == len(set(references))
    assert set(references) == expected
    catalogue = json.loads((DIRECTORY / 'catalogue.json').read_text())
    assert set(catalogue['packages']) == {
        row['distribution'] for row in registry['packages'] if row['status'] == 'included'
    }


@pytest.mark.unit
def test_source_extraction_handles_multiline_links_and_ignores_inline_links():
    audit = script('audit_estimator_packages')
    rst = '''- `One <https://example.org/one>`_ uses `other <https://example.org/other>`_.
- `Two
  <https://example.org/two>`_ description
Translations of scikit-learn documentation
- `Translation <https://example.org/translation>`_
'''
    assert audit.extract_projects('related_projects', rst) == [
        {'name': 'One', 'url': 'https://example.org/one'},
        {'name': 'Two', 'url': 'https://example.org/two'},
    ]


@pytest.mark.unit
def test_new_source_project_is_reported_without_automatic_approval():
    audit = script('audit_estimator_packages')
    snapshot = {'scope': 'Test', 'sources': [{'id': 'test', 'url': 'https://example.org',
                'retrieved_at': '2026-10-24', 'projects': [{'name': 'New package'}]}]}
    catalogue = {'estimators': [], 'profiles': [], 'specialists': [], 'packages': {}, 'discovery_issues': []}
    report = audit.report({'packages': []}, snapshot, catalogue)
    assert 'Unreviewed source entries: 1.' in report
    assert '- test: New package' in report


@pytest.mark.unit
def test_discovery_excludes_abstract_classes_without_requiring_base_estimator(monkeypatch):
    from abc import ABC, abstractmethod
    from types import SimpleNamespace

    class Abstract(ABC):
        @abstractmethod
        def fit(self, X, y=None): ...

        def get_params(self): return {}

    class Concrete:
        def fit(self, X, y=None): return self
        def get_params(self): return {}

    Abstract.__module__ = Concrete.__module__ = 'audit_test.models'
    builder = script('build_estimator_catalogue')
    monkeypatch.setattr(builder.importlib, 'import_module', lambda name: SimpleNamespace(
        Abstract=Abstract, Concrete=Concrete))
    assert builder.discover('audit_test', [''], []) == {'audit_test.Concrete': Concrete}


@pytest.mark.unit
def test_class_level_estimator_type_property_is_not_serialized():
    class Configurable:
        @property
        def _estimator_type(self): return 'regressor'
        def fit(self, X, y): return self

    entry = script('build_estimator_catalogue').describe('example.Configurable', Configurable, 'example', '1')
    assert entry['kinds'] == []
    json.dumps(entry)
