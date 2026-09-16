import pytest
from playwright.sync_api import expect
from shiny.pytest import create_app_fixture
app = create_app_fixture(app='../scenarios/var_encode_logical.py',scope='function')
pytestmark = pytest.mark.ui


def test_flip_side_tracks_enabled_encoding(page,app):
    page.goto(app.url)
    card = page.locator('.card').first
    namespace = card.get_attribute('id').partition('-')[0]
    def by_id(name): return page.locator(f'#{namespace}-{name}')
    card.hover()
    card.locator('button.flip-btn').click()
    expect(by_id('AuditSummary')).to_contain_text('No encodings selected',timeout=30000)
    by_id('Encode').locator('input[value="logical"]').check()
    expect(by_id('AuditTable')).to_contain_text('flag__integer',timeout=30000)
    expect(by_id('AuditSummary')).to_contain_text('Predictors: 1 → 1 · Originals removed: 1 · Pipeline steps added: 1')
    by_id('Encode').locator('input[value="logical"]').uncheck()
    expect(by_id('AuditSummary')).to_contain_text('No encodings selected',timeout=30000)
    expect(by_id('AuditTable')).not_to_contain_text('flag__integer')
