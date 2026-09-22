"""Keep synchronous Playwright and pytest-asyncio event loops isolated."""
import pytest
from pytest_playwright import pytest_playwright as playwright_plugin

# The plugin's session-scoped sync driver leaves its asyncio loop running in a
# greenlet between tests. A subsequent pytest-asyncio test then cannot enter its
# Runner. Close the whole dependent fixture chain after each browser test.
# Reuse the plugin implementations to preserve launch options, device settings,
# remote connections, and teardown behavior instead of duplicating those here.

playwright = pytest.fixture(scope="function")(playwright_plugin.playwright.__wrapped__)
browser_type = pytest.fixture(scope="function")(playwright_plugin.browser_type.__wrapped__)
launch_browser = pytest.fixture(scope="function")(playwright_plugin.launch_browser.__wrapped__)
browser = pytest.fixture(scope="function")(playwright_plugin.browser.__wrapped__)
browser_context_args = pytest.fixture(scope="function")(playwright_plugin.browser_context_args.__wrapped__)
