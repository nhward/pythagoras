"""Exercise sync browser -> asyncio -> sync browser in the same pytest run."""
import asyncio

import pytest


@pytest.mark.ui
def test_browser_before_async(page):
    page.set_content('<p>Before async</p>')
    assert page.locator('p').inner_text() == 'Before async'


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_after_browser():
    assert await asyncio.to_thread(lambda: 42) == 42


@pytest.mark.ui
def test_browser_after_async(page):
    page.set_content('<p>After async</p>')
    assert page.locator('p').inner_text() == 'After async'
