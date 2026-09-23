"""Browser window controls remain usable without a Python round trip."""
from pathlib import Path

import pytest
from playwright.sync_api import expect

pytestmark = pytest.mark.ui


@pytest.fixture
def controls(page):
    source = (Path(__file__).parents[1] / 'app/www/pythagoras.js').read_text()
    source = source.split('    // Application window controls:')[1].split('    // End application window controls.')[0]
    source = source[source.index('\n'):]
    page.set_content('<button id="FullScreen"><span>Expand</span></button><button id="Quit">Quit</button>')
    page.evaluate("""() => {
        window.registeredHandlers = {};
        window.Shiny = {addCustomMessageHandler: (name, handler) => {
            if (typeof handler !== 'function' || handler.length !== 1) {
                throw new Error('handler must be a function that takes one argument.');
            }
            registeredHandlers[name] = handler;
        }};
    }""")
    page.add_script_tag(content=source)
    assert page.evaluate("typeof registeredHandlers.quit_app") == 'function'
    return page


def test_fullscreen_calls_api_during_click_and_toggles(controls):
    page = controls
    page.evaluate('''() => {
        window.calls = [];
        document.documentElement.requestFullscreen = () => {
            calls.push('enter');
            Object.defineProperty(document, 'fullscreenElement', {configurable:true, value:document.documentElement});
            return Promise.resolve();
        };
        document.exitFullscreen = () => { calls.push('exit'); return Promise.resolve(); };
        document.getElementById('FullScreen').addEventListener('click', () => calls.push('button'));
    }''')
    page.locator('#FullScreen span').click()
    page.locator('#FullScreen span').click()
    assert page.evaluate('calls') == ['enter', 'button', 'exit', 'button']


def test_fullscreen_rejection_is_explained(controls):
    controls.evaluate("() => { document.documentElement.requestFullscreen = () => Promise.reject(new Error('Denied')); }")
    controls.locator('#FullScreen').click()
    expect(controls.get_by_role('status')).to_contain_text('Full screen is unavailable')
    controls.get_by_role('button', name='Dismiss').click()
    expect(controls.get_by_role('status')).to_have_count(0)


def test_blocked_quit_preserves_shiny_click_and_explains_manual_close(controls):
    controls.evaluate('''() => {
        window.close = () => {};
        window.quitEvents = 0;
        document.getElementById('Quit').addEventListener('click', () => quitEvents++);
    }''')
    controls.locator('#Quit').click()
    expect(controls.get_by_role('status')).to_contain_text('close the browser tab')
    assert controls.evaluate('quitEvents') == 1
