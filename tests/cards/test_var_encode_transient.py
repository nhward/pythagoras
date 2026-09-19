import pytest
from playwright.sync_api import expect
from shiny.pytest import create_app_fixture

app = create_app_fixture(app='../scenarios/var_encode_transient.py', scope='function')
pytestmark = pytest.mark.ui

@pytest.fixture(scope='session')
def browser_context_args():
    return {'viewport': {'width': 1700, 'height': 1100}}


def test_audit_recovers_from_pending_and_missing_data(page, app):
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.on('console', lambda message: errors.append(message.text) if message.type == 'error' else None)
    page.goto(app.url)
    card = page.locator('.card').first
    namespace = card.get_attribute('id').partition('-')[0]
    def by_id(name):
        return page.locator(f'#{namespace}-{name}')
    card.hover()
    card.locator('button.flip-btn').click()
    expect(by_id('AuditSummary')).to_contain_text('Originals removed: 0', timeout=30000)
    by_id('Encode').locator('input[value="logical"]').check()
    expect(by_id('AuditSummary')).to_contain_text('Predictors: 1 → 1, Originals removed: 1', timeout=30000)
    for _ in range(2):
        by_id('SourceState').select_option('missing')
        expect(by_id('SourceProbe')).to_have_text('missing')
        expect(by_id('AuditSummary')).to_be_empty()
        by_id('SourceState').select_option('pending')
        expect(by_id('SourceProbe')).to_have_text('pending')
        card.hover()
        card.locator('button.flip-btn').click()
        card.locator('button.flip-btn').click()
        expect(by_id('AuditSummary')).to_contain_text('Predictors: 2 → 2, Originals removed: 1', timeout=30000)
        expect(by_id('AuditTable')).to_contain_text('flag__integer', timeout=30000)
        by_id('SourceState').select_option('ready')
        expect(by_id('AuditSummary')).to_contain_text('Predictors: 1 → 1, Originals removed: 1', timeout=30000)
    assert not errors, errors
