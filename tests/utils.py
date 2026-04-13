from playwright.sync_api import Page
from typing import Literal, List

def find_shiny_ids_by_suffix(page: Page, suffix: str, *, kind: Literal["output", "input", "any"] = "any") -> List[str]:
    """
    Return a list of Shiny input/output IDs whose *full* id ends with `suffix`.

    This searches the DOM for [id$="suffix"]

    Parameters
    ----------
    page : Playwright Page
        The active page in your Shiny app test.
    suffix : str
        The trailing part of the id you know, e.g. "Summary", "Packages", "Refresh".

    Returns
    -------
    List[str]
        A list of matching full ids (e.g. ["config-Summary"]).
    """
    ids: List[str] = []
    out_loc = page.locator(f'[id$="{suffix}"]')
    for i in range(out_loc.count()):
        val = out_loc.nth(i).get_attribute("id")
        if val is not None:
            ids.append(val)
    return ids


def get_single_shiny_id_by_suffix(page: Page, suffix: str, *, kind: Literal["output", "input", "any"] = "any") -> str:
    """
    Like find_shiny_ids_by_suffix(), but assert that there is exactly one match.
    Raises AssertionError if there are zero or multiple matches.
    """
    ids = find_shiny_ids_by_suffix(page, f"-{suffix}")
    if not ids:
        raise AssertionError(f"No Shiny ids found with suffix '-{suffix}'")
    if len(ids) > 1:
        raise AssertionError(
            f"Multiple Shiny ids found with suffix '-{suffix}': {ids}"
        )
    return ids[0]


def find_id(page: Page, suffix: str):
    return get_single_shiny_id_by_suffix(page, suffix)


def hover_solo_card(page: Page):
# Hover the card so its hover children (like header buttons) become visible
    card = page.locator('[id$="-Card"].hover-card')
    card.hover()


def click_button(page: Page, suffix: str):
    eles = page.locator(f'[id$="-{suffix}"]').all()
    if not eles:
        raise AssertionError(f"No UI elements found with suffix '-{suffix}'")
    if len(eles) > 1:
        raise AssertionError(
            f"Multiple UI elements found with suffix '-{suffix}'"
        )
    button = eles[0]
    button.click()
    
