"""Metadata fallbacks, lazy persistence, and isolated eager discovery."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from card_catalogue import CardCatalogue, discover_cards, rebuild_catalogue, source_hash


def make_root(tmp_path):
    (tmp_path / 'cards').mkdir()
    (tmp_path / 'cards' / '__init__.py').write_text('')
    return tmp_path


def make_card(root, name='sample', **properties):
    path = root / 'cards' / f'{name}.py'
    path.write_text('# card source\n')
    return SimpleNamespace(file=str(path), long_name='Readable name', description='<b>Helpful</b> description',
                           mutable=True, pipeline_compatible=False, **properties)


@pytest.mark.unit
def test_lazy_capture_persists_and_refreshes_stale_entries(tmp_path):
    root = make_root(tmp_path)
    card = make_card(root)
    catalogue = CardCatalogue(root)
    assert catalogue.choices() == {'sample': 'sample'}
    assert catalogue.capture(card)
    entry = catalogue.entry('sample')
    assert entry['description'] == 'Helpful description'
    assert entry['mutable'] is True and entry['pipeline_compatible'] is False
    modified = catalogue.path.stat().st_mtime_ns
    assert not catalogue.capture(card)
    assert catalogue.path.stat().st_mtime_ns == modified
    assert CardCatalogue(root).entry('sample') == entry
    Path(card.file).write_text('# updated card\n')
    assert catalogue.choices()['sample'] == 'sample'
    card.long_name = 'New name'
    assert catalogue.capture(card)
    assert catalogue.entry('sample')['long_name'] == 'New name'


@pytest.mark.unit
def test_multiple_catalogue_writers_merge_and_ignore_invalid_files(tmp_path):
    root = make_root(tmp_path)
    one, two = make_card(root, 'one'), make_card(root, 'two')
    first, second = CardCatalogue(root), CardCatalogue(root)
    first.capture(one)
    second.capture(two)
    assert set(CardCatalogue(root).snapshot()['cards']) == {'one', 'two'}
    first.path.write_text('{broken json')
    assert CardCatalogue(root).choices() == {'one': 'one', 'two': 'two'}
    first.path.write_text(json.dumps({'schema_version': 999, 'cards': {}}))
    assert CardCatalogue(root).snapshot()['cards'] == {}


@pytest.mark.unit
def test_browser_and_read_only_runtime_keep_metadata_in_memory(tmp_path):
    root = make_root(tmp_path)
    card = make_card(root)
    browser = CardCatalogue(root, persistent=False)
    browser.capture(card)
    assert browser.entry('sample')['long_name'] == 'Readable name'
    assert not browser.path.exists()
    read_only = CardCatalogue(root, path=root / 'blocked' / 'catalogue.json')
    (root / 'blocked').write_text('not a directory')
    assert read_only.capture(card)
    assert read_only.write_error
    assert read_only.entry('sample')['description'] == 'Helpful description'


@pytest.mark.unit
def test_missing_properties_and_discovery_exclusions(tmp_path):
    root = make_root(tmp_path)
    card = make_card(root)
    for name in ('template', 'sys_bookmark'):
        make_card(root, name)
    catalogue = CardCatalogue(root)
    card.long_name = card.description = None
    assert catalogue.capture(card)
    assert list(discover_cards(root)) == ['sample']
    assert catalogue.choices() == {'sample': 'sample'}
    snapshot = catalogue.snapshot()
    snapshot['cards']['sample']['mutable'] = 'not a boolean'
    catalogue.path.write_text(json.dumps(snapshot))
    assert CardCatalogue(root).entry('sample')['mutable'] is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rebuild_is_a_subprocess_and_leaves_live_instances_untouched(tmp_path, monkeypatch):
    from module import Module
    root = make_root(tmp_path)
    (root / 'cards' / 'sample.py').write_text('''from types import SimpleNamespace
from module import Module
def instance():
    Module.Instances['child_only'] = object()
    return SimpleNamespace(file=__file__, long_name='Worker card', description='Isolated discovery', mutable=False, pipeline_compatible=True)
''')
    (root / 'cards' / 'broken.py').write_text("raise RuntimeError('intentional constructor failure')\n")
    live = {'existing': object()}
    monkeypatch.setattr(Module, 'Instances', live)
    catalogue = CardCatalogue(root)
    result = await rebuild_catalogue(catalogue, timeout=30)
    assert result['updated'] == 1
    assert 'broken' in result['errors']
    assert Module.Instances is live and list(live) == ['existing']
    assert catalogue.entry('sample')['long_name'] == 'Worker card'
    assert catalogue.entry('sample')['source_hash'] == source_hash(root / 'cards' / 'sample.py')
    assert catalogue.choices()['broken'] == 'broken'


@pytest.mark.unit
def test_card_ui_captures_metadata_after_factory_assignments(tmp_path, monkeypatch):
    from card import Card
    from module import Module
    root = make_root(tmp_path)
    path = root / 'cards' / 'sample.py'
    path.write_text('# metadata example')
    catalogue = CardCatalogue(root)
    monkeypatch.setattr(Module, 'card_catalogue', catalogue)
    monkeypatch.setattr(Module, 'Instances', {})
    card = Card(file=str(path), mutable=True, pipeline_compatible=False)
    assert catalogue.entry('sample')['long_name'] is None
    card.long_name = 'Assigned after construction'
    card.description = 'Final factory description'
    card.call_ui()
    assert catalogue.entry('sample')['long_name'] == 'Assigned after construction'
    assert catalogue.entry('sample')['pipeline_compatible'] is False
