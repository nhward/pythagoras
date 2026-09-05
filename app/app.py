###########################
## application           ##
###########################

## This the app for a shiny application called Pythagoras
## It provides:
##    Creating sections for cards (i.e. a grouping structure)
##    Dynamically creating the card instances for a current section
##    Reactively connecting card outputs to subsequent card inputs
##    Invoking the shiny app either in Positron or in Python via the last lines 


import importlib
import json
import logging
import os
import re
import sys
import tempfile
import threading
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

#TODO: provide a guide button for the sections and buttons
#TODO: provide a info button for the whole app               

# Ensure local modules and packages are resolved from the app directory.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import cards  # noqa: F401
from faicons import icon_svg as icon
from jsonschema import SchemaError, ValidationError, validate
from module import Module
from shiny import App, reactive, req, ui

log = logging.getLogger("pythagoras")
if not log.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%H:%M:%S"
    ))
    log.addHandler(h)
log.propagate = False
log.setLevel(logging.DEBUG)


jslog = logging.getLogger("<Javascript>")
h = logging.StreamHandler(sys.stdout)
h.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%H:%M:%S"
))
jslog.addHandler(h)
jslog.propagate = False
jslog.setLevel(logging.DEBUG)
if Module.log_handler not in jslog.handlers:
    jslog.addHandler(Module.log_handler)

config = Module.config
CONFIG_PATH = ROOT / "config" / "pythagoras.json"
SCHEMA_PATH = ROOT / "config" / "pythagoras.schema.json"
CONFIG_WRITE_LOCK = threading.Lock()
SECTION_NAME_PATTERN = re.compile(r"^[A-Za-z0-9]+(?: [A-Za-z0-9]+)*$")
RESERVED_SECTION_NAMES = frozenset({"start"})
SECTION_ID_PREFIX = "section_"
TEST_SHOW_START_ENV = "PYTHAGORAS_TEST_SHOW_START"
START_SECTION_ID = "start"
SECTIONS_NAV_ID = "sections"
WELCOME_ICON_TAG_PATTERN = re.compile(
    r"<i\b(?P<attributes>[^>]*)>\s*</i>",
    flags=re.IGNORECASE,
)
HTML_CLASS_PATTERN = re.compile(
    r"\bclass\s*=\s*([\"'])(?P<classes>.*?)\1",
    flags=re.IGNORECASE | re.DOTALL,
)
FONT_AWESOME_STYLE_CLASSES = frozenset({
    "fa-brands",
    "fa-duotone",
    "fa-light",
    "fa-regular",
    "fa-sharp",
    "fa-solid",
    "fa-thin",
})


@dataclass
class CardNode:
    card: Module
    upstream: reactive.Value
    output: Callable[[], object]


def wire_card_nodes(
    order: tuple[str, ...],
    nodes: dict[str, CardNode],
) -> None:
    """Connect card outputs to subsequent inputs without reading any data."""
    source = None
    for namespace in order:
        node = nodes.get(namespace)
        if node is None:
            continue
        node.card.resume()
        node.upstream.set(None if source is None else source.output)
        source = node


def configuration_from_card_state(
    base_config: Mapping[str, object],
    *,
    section_order: Sequence[str] | None = None,
    visited_sections: Sequence[str],
    section_orders: Mapping[str, Sequence[str]],
    card_modules: Mapping[str, str],
    show_start: bool | None = None,
) -> dict[str, object]:
    """Return a configuration containing the current non-empty section layout."""
    candidate = deepcopy(dict(base_config))
    visited = set(visited_sections)
    layout = candidate.get("layout")
    if not isinstance(layout, list):
        raise TypeError("Configuration layout must be a list")

    configured_groups = {}
    for group in layout:
        if not isinstance(group, dict):
            raise TypeError("Each configuration layout entry must be an object")
        configured_groups[group.get("section")] = group

    ordered_sections = (
        list(section_order)
        if section_order is not None
        else [group["section"] for group in layout]
    )
    saved_layout = []
    for section in ordered_sections:
        group = deepcopy(configured_groups.get(section, {
            "section": section,
            "cards": [],
        }))
        if section in visited:
            if section not in section_orders:
                raise ValueError(f"Card order is not ready for section {section!r}")
            group["cards"] = [
                {"module": card_modules[namespace]}
                for namespace in section_orders[section]
                if namespace in card_modules
            ]
        cards_in_section = group.get("cards")
        if not isinstance(cards_in_section, list):
            raise TypeError(f"Cards for section {section!r} must be a list")
        if cards_in_section:
            saved_layout.append(group)

    if not saved_layout:
        raise ValueError("At least one non-empty section is required")
    candidate["layout"] = saved_layout
    if show_start is not None:
        if not isinstance(show_start, bool):
            raise TypeError("show_start must be a boolean")
        settings = candidate.get("settings")
        if not isinstance(settings, dict):
            raise TypeError("Configuration settings must be an object")
        settings["show_start"] = show_start
    return candidate


def validated_section_name(
    value: object,
    existing_sections: Sequence[str],
) -> str:
    """Return a canonical, schema-safe and unique section name."""
    if not isinstance(value, str):
        raise TypeError("Section name must be text")
    name = " ".join(value.split())
    if not name:
        raise ValueError("Section name is required")
    if not SECTION_NAME_PATTERN.fullmatch(name):
        raise ValueError("Use letters, numbers and single spaces only")
    if name.casefold() in RESERVED_SECTION_NAMES:
        raise ValueError(f"{name!r} is reserved")

    normalized = Module.section_normalise(name).casefold()
    for existing in existing_sections:
        if name.casefold() == existing.casefold():
            raise ValueError(f"Section {name!r} already exists")
        if Module.section_normalise(existing).casefold() == normalized:
            raise ValueError(
                f"Section {name!r} conflicts with existing section {existing!r}"
            )
    return name


def inserted_section_order(
    sections: Sequence[str],
    *,
    current: str,
    new: str,
    position: str,
) -> tuple[str, ...]:
    """Insert a section immediately before or after the current section."""
    order = list(sections)
    if current not in order:
        raise ValueError(f"Current section {current!r} does not exist")
    if position not in {"before", "after"}:
        raise ValueError("Section position must be 'before' or 'after'")
    index = order.index(current) + (position == "after")
    order.insert(index, new)
    return tuple(order)


def write_validated_configuration(
    candidate: Mapping[str, object],
    *,
    config_path: Path = CONFIG_PATH,
    schema_path: Path = SCHEMA_PATH,
) -> None:
    """Validate and atomically persist a complete Pythagoras configuration."""
    config_path = Path(config_path)
    schema_path = Path(schema_path)
    temporary_path: Path | None = None
    with CONFIG_WRITE_LOCK:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        validate(instance=candidate, schema=schema)
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=config_path.parent,
                prefix=f".{config_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                json.dump(candidate, temporary, indent=2, ensure_ascii=False)
                temporary.write("\n")
                temporary.flush()
                os.fsync(temporary.fileno())
            if config_path.exists():
                temporary_path.chmod(config_path.stat().st_mode & 0o7777)
            os.replace(temporary_path, config_path)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)


def replace_welcome_icons(html: str) -> str:
    """Replace decorative Font Awesome placeholders with inline SVGs."""
    def replacement(match: re.Match[str]) -> str:
        class_match = HTML_CLASS_PATTERN.search(match.group("attributes"))
        if class_match is None:
            return match.group(0)

        classes = class_match.group("classes").split()
        if "fa-solid" not in classes:
            return match.group(0)
        icon_class = next(
            (
                value
                for value in classes
                if value.startswith("fa-")
                and value not in FONT_AWESOME_STYLE_CLASSES
            ),
            None,
        )
        if icon_class is None:
            return match.group(0)

        name = icon_class.removeprefix("fa-")
        try:
            return str(icon(name, a11y="deco"))
        except ValueError:
            log.warning("Welcome icon %r is not available in faicons", name)
            return match.group(0)

    return WELCOME_ICON_TAG_PATTERN.sub(replacement, html)


def welcome():
    """Read and safely embed the body of the generated welcome document."""
    html_file = Path("www/markdown/welcome.html")
    if html_file.exists():
        try:
            text = html_file.read_text(encoding="utf-8")
            body_match = re.search(
                r"<body\b[^>]*>(.*?)</body\s*>",
                text,
                flags=re.IGNORECASE | re.DOTALL,
            )
            body = body_match.group(1) if body_match else text
            body = re.sub(
                r"<script\b[^>]*>.*?</script\s*>",
                "",
                body,
                flags=re.IGNORECASE | re.DOTALL,
            )
            body = replace_welcome_icons(body)
            return ui.HTML(body.strip())
        except Exception:  # noqa: BLE001
            log.error(f"Error reading {html_file} file")
            return ""
    else:
        log.error(f"File {html_file} not found")
        return ""


def show_start_enabled() -> bool:
    """Return the configured Start-page state, with a test-only override."""
    configured = bool(config.get("settings", {}).get("show_start", False))
    if os.environ.get("SHINY_TESTMODE") != "1":
        return configured

    override = os.environ.get(TEST_SHOW_START_ENV)
    if override is None:
        return configured
    normalized = override.strip().casefold()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(
        f"{TEST_SHOW_START_ENV} must be true or false, not {override!r}"
    )


def sections() -> list[str]:
    """
    return a list of section names
    """
    return [section["section"] for section in config["layout"]
]


def configured_section_id(index: int) -> str:
    """Return the stable per-session ID for a configured section."""
    return f"{SECTION_ID_PREFIX}{index}"


def section_contents(section_id: str, name: str, *, empty: bool):
    """Create a section's empty-state region and sortable card container."""
    return ui.TagList(
        ui.div(
            ui.tags.p("This section has no cards."),
            ui.tags.button(
                "Add a card",
                type="button",
                class_="btn btn-primary btn-sm section-add-card",
                data_section_id=section_id,
            ),
            ui.tags.span(
                " or delete the section using its title control.",
                class_="text-muted ms-2",
            ),
            id=f"{section_id}-empty-state",
            class_="section-empty-state text-center py-4",
            hidden=None if empty else True,
            data_section_id=section_id,
        ),
        ui.div(
            id=f"{section_id}-cards-container",
            class_="cards-grid",
            data_section_id=section_id,
        ),
    )


def section_panel(
    section_id: str,
    name: str,
    *,
    group_style: str,
    empty: bool = False,
):
    """Create a tab or accordion panel for one section."""
    contents = section_contents(section_id, name, empty=empty)
    if group_style == "tab":
        return ui.nav_panel(name, contents, value=section_id)
    if group_style == "accordion":
        return ui.accordion_panel(name, contents, value=section_id)
    raise ValueError(f"Unsupported section style {group_style!r}")


def start_panel():
    """Create the optional Start navigation panel."""
    return ui.nav_panel(
        "Start",
        ui.div(
            welcome(),
            ui.input_action_button(
                id="GuideButton",
                label="Guide me",
                icon=icon(
                    "eye",
                    title="Take a guided tour of this card",
                    a11y="sem",
                ),
                class_="btn rounded-pill hover-btn btn-sm guide-btn",
                style="border: 0px; box-shadow: none;",
                aria_label="Take a guided tour of this card",
            ),
            id="Start-cards-container",
            class_="",
        ),
        value=START_SECTION_ID,
    )


def create_sections():
    group_style = config.get("settings", {}).get("section_style")
    panels = []
    if show_start_enabled():
        panels.append(start_panel())
    if group_style == "tab":
        for index, group in enumerate(config["layout"]):
            panels.append(section_panel(
                configured_section_id(index),
                group["section"],
                group_style=group_style,
                empty=not group["cards"],
            ))
    elif group_style == "accordion":
        panels.append(
            ui.nav_panel(
                "",
                ui.accordion(
                    *[
                        section_panel(
                            configured_section_id(index),
                            group["section"],
                            group_style=group_style,
                            empty=not group["cards"],
                        )
                        for index, group in enumerate(config["layout"])
                    ],
                    multiple=False,
                    id="Accordion",
                ),
                value=SECTIONS_NAV_ID,
            )
        )
    return panels

def application():
    """
    Create a shiny app.
    This involves creating UI (app_ui) and SERVER (server) functions and
    passing these to shiny.app.
    """
    # main ui object for the app
    app_ui = ui.page_fillable(
        ui.head_content(
            ui.tags.link(rel="icon", type="image/x-icon", href="favicon.ico"),
            [ui.include_js(script, method="inline") for script in dict.fromkeys(Module.script_list)], # iterate through unique js scripts
            [ui.include_css(css, method="inline") for css in dict.fromkeys(Module.css_list)], # iterate through unique CSS documents
            [ui.tags.script(type = "module", src=script) for script in dict.fromkeys(Module.mjs_list)] # iterate through unique mjs modules
        ),
        ui.busy_indicators.options(spinner_type = "bars2"),
        ui.busy_indicators.use(),
        ui.page_navbar(
            *create_sections(),  # << This is the important call here
            ui.nav_spacer(),
            ui.nav_control(
                ui.input_action_button(
                    id = "ManageCardSection",
                    label= None, 
                    icon = icon("wrench", title = "Manage card or section", a11y = "sem"),
                    class_ = "btn rounded-pill btn-sm fa-xl",
                    style = "border: 0px; box-shadow: none; display: block;"
                )
            ),
            ui.nav_control(
                ui.input_action_button(
                    id = "SaveConfiguration",
                    label = None,
                    icon = icon("bookmark", title = "Save card and section layout", a11y = "sem"),
                    class_ = "btn rounded-pill btn-sm fa-xl",
                    style = "border: 0px; box-shadow: none; display: block;"
                )
            ),
            ui.nav_control(
                ui.input_action_button(
                    id = "FullScreen",  
                    label= None, 
                    icon = icon("expand", title = "Toggle full screen", a11y = "sem"),
                    class_ = "btn rounded-pill btn-sm fa-xl",
                    style = "border: 0px; box-shadow: none; display: block;"
                )
            ),
            ui.nav_control(
                ui.input_action_button(
                    id = "Quit",  
                    label = None, 
                    icon = icon("stop", title = "Quit session", a11y = "sem"),
                    class_ = "btn rounded-pill btn-sm fa-xl",
                    style = "border: 0px; box-shadow: none; display: block;"
                )
            ),

            id = "Navbar",
            title = ui.tooltip(
                ui.TagList(ui.tags.img(src="favicon.ico", style="height:2em; margin-right:0.5em;"), ui.span("Pythagoras", class_ = "text-primary")),
                '"All is number."',
                placement = "bottom"
            )
        )
    )

    # main server function for the app
    def server(input, output, session):
        Module.ModSession = session

        card_nodes: dict[str, CardNode] = {}
        section_definitions = {
            configured_section_id(index): deepcopy(group)
            for index, group in enumerate(config.get("layout", []))
        }
        SectionOrder = reactive.value(tuple(section_definitions))
        SectionsVisited = reactive.value(())
        TopologyVersion = reactive.value(0)
        ShowStart = reactive.value(show_start_enabled())
        next_section_number = len(section_definitions)

        def section_name(section_id: str) -> str:
            return section_definitions[section_id]["section"]

        def new_section_id() -> str:
            nonlocal next_section_number
            while True:
                candidate = configured_section_id(next_section_number)
                next_section_number += 1
                if candidate not in section_definitions:
                    return candidate

        def bump_topology():
            with reactive.isolate():
                version = TopologyVersion.get()
            TopologyVersion.set(version + 1)

        def ordered_visited(values: Sequence[str]) -> tuple[str, ...]:
            visited = set(values)
            with reactive.isolate():
                order = SectionOrder.get()
            return tuple(section for section in order if section in visited)

        def unregister_card(namespace: str):
            card_nodes.pop(namespace, None)
            bump_topology()

        @reactive.effect
        async def startup():
            if ShowStart():
                await session.send_custom_message("animate", {"id" : session.ns("GuideButton"), "animation" : "bounce", "delay" : 500})

        @reactive.calc
        def currentSection():
            """
            Uses the group_style to determine how to assess the current group name.
            The name is tested in case it is not a valid group name (i.e. a nav_bar button)
            """
            section_style = config.get("settings", {}).get("section_style")
            if section_style == "tab":
                current = input.Navbar()
            else:
                current = input.Accordion()
                req(current)
                req(len(current) == 1)
                current = current[0]
            valid = SectionOrder()
            #Check that the current tab-item is not a button etc
            req(current in valid)
            log.debug(
                "🔀 Section switched to %r (%s) using %s style",
                section_name(current),
                current,
                section_style,
            )
            return current

        def create_card(name: str):
            module_name = f"cards.{name}"
            try:
                module = importlib.import_module(module_name)
                if not hasattr(module, "instance"):
                    raise AttributeError(f"{module_name} does not define instance()")
                module = module.instance()
                module.log.info(msg=f"✅ Card instantiated ({module.namespace})")
                return module
            except Exception:
                log.exception(f"⚠️ Failed to instantiate card {module_name}")
                return None

        @reactive.effect
        def create_section_cards():
            current = currentSection()
            if current not in SectionsVisited.get():
                model_group = section_definitions.get(current)
                req(model_group is not None)
                for card in model_group["cards"]:
                    instance = create_card(card["module"])
                    if instance is None:
                        continue
                    ui.insert_ui(ui = instance.call_ui(), selector = f"#{current}-cards-container", where = "beforeEnd")
                    upstream = reactive.Value(None)
                    card_output = instance.call_server(
                        input,
                        output,
                        session,
                        upstream=upstream,
                        on_remove=unregister_card,
                    )
                    card_nodes[instance.namespace] = CardNode(
                        card=instance,
                        upstream=upstream,
                        output=card_output,
                    )
                    instance.resume()
                    card_id = instance.ns("Card")
                    instance.section = current
                    async def after_flush(card_id=card_id):
                        await session.send_custom_message("init_card", {"id": card_id})
                    session.on_flushed(after_flush, once=True)
                async def after_flush2(current=current):
                    container = f"{current}-cards-container"
                    imp_id = f"{current}_CardOrder"
                    await session.send_custom_message("MakeSortable", {"id": container, "input_id": imp_id})
                session.on_flushed(after_flush2, once=True)
                visited = (*SectionsVisited.get(), current)
                SectionsVisited.set(ordered_visited(visited))


        @reactive.calc
        def available_cards():
            """
            Return {display_name: file_path} for all card files in cards_dir.
            Assumes each card file defines instance().
            This is not filesystem-reactive but instead uses invalidation to pick up file changes (eventually)
            """
            cardDict = {}
            cards_dir = Module.ROOT / "cards"
            for path in sorted(cards_dir.glob("*.py")):
                if path.name == "__init__.py":
                    continue
                cardDict[path.stem] = path
            # re-evaluate every hour
            reactive.invalidate_later(3600)
            return cardDict
        
        ModalSection = reactive.Value(None)

        def add_item_modal(
            cardDict: dict[str, Path],
            *,
            current_name: str,
            show_start: bool,
            selected="card",
        ):
            """Build the card and section management modal."""
            action_labels = json.dumps({
                "card": "Add card",
                "section": "Add section",
                "rename": "Rename section",
            })
            return ui.modal(
                ui.div(
                    ui.span("The current section: "),
                    ui.tags.strong(current_name),
                    class_="current-section-context mb-3",
                ),
                ui.input_checkbox(
                    id="ShowStartSection",
                    label='Show "Start" section',
                    value=show_start,
                ),
                ui.navset_tab(
                    ui.nav_panel(
                        "New card",
                        ui.input_select(
                            id="CardPicker_selected",
                            label="Choose a card to insert",
                            choices=list(cardDict.keys()),
                            selected=(
                                next(iter(cardDict.keys())) if cardDict else None
                            ),
                        ),
                        value="card",
                    ),
                    ui.nav_panel(
                        "New section",
                        ui.input_text(
                            id="NewSectionName",
                            label="Section name",
                            placeholder="For example: Modelling",
                        ),
                        ui.input_radio_buttons(
                            id="SectionPosition",
                            label="Position relative to the current section",
                            choices={"before": "Before", "after": "After"},
                            selected="after",
                            inline=True,
                        ),
                        ui.p(
                            "Section names may contain letters, numbers and spaces.",
                            class_="form-text",
                        ),
                        value="section",
                    ),
                    ui.nav_panel(
                        "Rename section",
                        ui.input_text(
                            id="RenameSectionName",
                            label="New section name",
                            value=current_name,
                        ),
                        ui.p(
                            "Only the displayed name changes; cards and their "
                            "reactive connections remain in place.",
                            class_="form-text",
                        ),
                        value="rename",
                    ),
                    id="AddItemType",
                    selected=selected,
                ),
                title="Manage cards and sections",
                footer=ui.div(
                    ui.input_action_button(
                        id = "CardPicker_cancel", 
                        label = "Cancel", 
                        class_ = "btn btn-secondary"),
                    ui.input_action_button(
                        id = "CardPicker_ok", 
                        label = {
                            "card": "Add card",
                            "section": "Add section",
                            "rename": "Rename section",
                        }.get(selected, "Continue"),
                        class_ = "btn btn-primary ms-2",
                        data_action_labels=action_labels,
                    ),
                    class_ = "d-flex justify-content-end"
                ),
                easy_close=True
            )

        def show_add_modal(*, section: str, selected="card"):
            ModalSection.set(section)
            ui.modal_show(add_item_modal(
                available_cards(),
                current_name=section_name(section),
                show_start=ShowStart(),
                selected=selected,
            ))

        @reactive.effect
        @reactive.event(input.ManageCardSection)
        def showPicker():
            """Open the combined creation modal at its card tab."""
            if input.Navbar() == START_SECTION_ID:
                section = SectionOrder()[0]
            else:
                section = currentSection()
            show_add_modal(section=section, selected="card")

        @reactive.effect
        @reactive.event(input.ShowStartSection)
        def update_start_section():
            """Immediately add or remove the session's optional Start panel."""
            desired = bool(input.ShowStartSection())
            if desired == ShowStart():
                return

            group_style = config.get("settings", {}).get("section_style")
            sections_target = (
                SectionOrder()[0]
                if group_style == "tab"
                else SECTIONS_NAV_ID
            )
            if desired:
                ui.insert_nav_panel(
                    id="Navbar",
                    nav_panel=start_panel(),
                    target=sections_target,
                    position="before",
                    select=False,
                )
            else:
                if input.Navbar() == START_SECTION_ID:
                    ui.update_navset(id="Navbar", selected=sections_target)
                ui.remove_nav_panel(id="Navbar", target=START_SECTION_ID)
            ShowStart.set(desired)

        @reactive.effect
        @reactive.event(input.AddCardToSection)
        def showPickerForSection():
            """Open card creation from an empty section's invitation."""
            payload = input.AddCardToSection()
            section = payload.get("section") if isinstance(payload, dict) else None
            if section not in SectionOrder():
                return
            show_add_modal(section=section, selected="card")


        @reactive.effect
        @reactive.event(input.CardPicker_cancel)
        async def _cancel_picker():
            ui.modal_remove()

        @reactive.effect
        @reactive.event(input.CardPicker_ok)
        async def _confirm_picker():
            """Create the selected card or section after validation."""
            mode = input.AddItemType()
            reference = ModalSection()
            if reference not in SectionOrder():
                ui.notification_show(
                    "The reference section no longer exists.",
                    type="error",
                )
                return

            if mode == "section":
                try:
                    name = validated_section_name(
                        input.NewSectionName(),
                        [section_name(value) for value in SectionOrder()],
                    )
                    position = input.SectionPosition()
                    if position not in {"before", "after"}:
                        raise ValueError(
                            "Section position must be 'before' or 'after'"
                        )
                except (TypeError, ValueError) as error:
                    ui.notification_show(str(error), type="error", duration=8)
                    return

                created = new_section_id()
                section_definitions[created] = {"section": name, "cards": []}
                new_order = inserted_section_order(
                    SectionOrder(),
                    current=reference,
                    new=created,
                    position=position,
                )
                SectionOrder.set(new_order)
                SectionsVisited.set(ordered_visited(
                    (*SectionsVisited.get(), created)
                ))
                group_style = config.get("settings", {}).get("section_style")
                panel = section_panel(
                    created,
                    name,
                    group_style=group_style,
                    empty=True,
                )
                if group_style == "tab":
                    ui.insert_nav_panel(
                        id="Navbar",
                        nav_panel=panel,
                        target=reference,
                        position=position,
                        select=True,
                    )
                else:
                    ui.insert_accordion_panel(
                        id="Accordion",
                        panel=panel,
                        target=reference,
                        position=position,
                    )
                    ui.update_accordion(id="Accordion", show=created)

                async def initialize_section(section=created):
                    await session.send_custom_message("MakeSortable", {
                        "id": f"{section}-cards-container",
                        "input_id": f"{section}_CardOrder",
                    })

                session.on_flushed(initialize_section, once=True)
                ui.modal_remove()
                bump_topology()
                return

            if mode == "rename":
                try:
                    name = validated_section_name(
                        input.RenameSectionName(),
                        [
                            section_name(value)
                            for value in SectionOrder()
                            if value != reference
                        ],
                    )
                except (TypeError, ValueError) as error:
                    ui.notification_show(str(error), type="error", duration=8)
                    return

                section_definitions[reference]["section"] = name
                await session.send_custom_message("RenameSection", {
                    "section_id": reference,
                    "name": name,
                })
                ui.modal_remove()
                ui.notification_show(
                    f"Section renamed to {name!r}.",
                    type="message",
                    duration=5,
                )
                return

            name = input.CardPicker_selected()
            if not name or name not in available_cards():
                ui.notification_show("Choose a valid card.", type="error")
                return
            current = reference
            instance = create_card(name)
            if instance is None:
                ui.notification_show(
                    f"Card {name!r} could not be created.",
                    type="error",
                )
                return
            ui.insert_ui(ui = instance.call_ui(), selector = f"#{current}-cards-container", where = "beforeEnd")
            upstream = reactive.Value(None)
            card_output = instance.call_server(
                input,
                output,
                session,
                upstream=upstream,
                on_remove=unregister_card,
            )
            card_nodes[instance.namespace] = CardNode(
                card=instance,
                upstream=upstream,
                output=card_output,
            )
            instance.section = current
            instance.resume()
            card_id = instance.ns("Card")
            container = f"{current}-cards-container"
            imp_id = f"{current}_CardOrder"
            async def after_flush(card_id=card_id, container = container, container_id = imp_id):
                await session.send_custom_message("init_card", {"id": card_id})
                await session.send_custom_message("UpdateCardOrder", {"id": container, "input_id": container_id})
            session.on_flushed(after_flush, once=True)
            ui.modal_remove()

        @reactive.effect
        @reactive.event(input.DeleteSection)
        def DeleteSection():
            """Delete an empty section after rechecking its server-side state."""
            payload = input.DeleteSection()
            section = payload.get("section") if isinstance(payload, dict) else None
            order = SectionOrder()
            if section not in order:
                return
            if len(order) <= 1:
                ui.notification_show(
                    "The final section cannot be deleted.",
                    type="error",
                )
                return

            definition = section_definitions.get(section, {})
            not_instantiated = section not in SectionsVisited.get()
            if any(
                node.card.section == section
                for node in card_nodes.values()
            ) or (not_instantiated and definition.get("cards")):
                ui.notification_show(
                    "Only an empty section can be deleted.",
                    type="error",
                )
                return

            index = order.index(section)
            neighbor = order[index - 1] if index > 0 else order[index + 1]
            group_style = config.get("settings", {}).get("section_style")
            if group_style == "tab":
                ui.update_navset(id="Navbar", selected=neighbor)
                ui.remove_nav_panel(id="Navbar", target=section)
            else:
                ui.update_accordion(id="Accordion", show=neighbor)
                ui.remove_accordion_panel(id="Accordion", target=section)

            new_order = tuple(value for value in order if value != section)
            SectionOrder.set(new_order)
            SectionsVisited.set(tuple(
                value for value in SectionsVisited.get() if value != section
            ))
            section_definitions.pop(section, None)
            bump_topology()

        @reactive.effect
        @reactive.event(input.SaveConfiguration)
        def SaveConfiguration():
            """Persist the current card order for every instantiated section."""
            if Module.IS_SHINYLIVE:
                ui.notification_show(
                    "Configuration cannot be persisted from Shinylive.",
                    type="error",
                    duration=8,
                )
                return

            try:
                visited = tuple(SectionsVisited.get())
                section_orders: dict[str, tuple[str, ...]] = {}
                for section_id in visited:
                    input_id = f"{section_id}_CardOrder"
                    raw_order = input[input_id]()
                    if raw_order is None:
                        raise ValueError(
                            "Card order is not ready for section "
                            f"{section_name(section_id)!r}"
                        )
                    section_orders[section_name(section_id)] = tuple(
                        value.removesuffix("-Card") for value in raw_order
                    )

                candidate = configuration_from_card_state(
                    config,
                    section_order=tuple(
                        section_name(value) for value in SectionOrder()
                    ),
                    visited_sections=tuple(
                        section_name(value) for value in visited
                    ),
                    section_orders=section_orders,
                    card_modules={
                        namespace: node.card.name
                        for namespace, node in card_nodes.items()
                    },
                    show_start=ShowStart(),
                )
                write_validated_configuration(candidate)
            except (OSError, ValueError, SchemaError, ValidationError) as error:
                log.exception("Could not save Pythagoras configuration")
                ui.notification_show(
                    f"Configuration was not saved: {error}",
                    type="error",
                    duration=None,
                )
                return

            log.info("💾 Configuration saved to %s", CONFIG_PATH)
            ui.notification_show(
                "Card layout saved. It will be used on the next app start.",
                type="message",
                duration=6,
            )
        

        @reactive.effect
        @reactive.event(input.FullScreen)
        async def FullScreen():
            """
            This makes the browser go full screen.
            Because full-screen is browser specific this may be unreliable.
            The implememtation is in pythagoras.js
            """
            log.info("🙏 Full-screen app requested")
            await session.send_custom_message("fullscreen_app", None)
        

        @reactive.effect
        @reactive.event(input.Quit)
        async def Quit():
            """
            This shuts the browser session down - just like a conventional application.
            Because closing tabs is browser specific this may be unreliable.
            The implememtation is in pythagoras.js
            """
            log.info("🙏 Quit app requested")
            await session.send_custom_message("quit_app", None)
            await session.close()  # in case the window close is ignored
        

        # redirect browser console to the python console
        @reactive.effect
        @reactive.event(input.Console_log)
        def Redirect():
            """
            Java-script console messages are redirected to the python log.
            """
            message = input.Console_log()
            level = message['level'].upper()
            if level =="ERROR":
                jslog.error(msg = f"☢️ {message['text']}")
            elif level == "INFO":
                jslog.info(msg = f"ℹ️ {message['text']}")
            elif level == "WARNING":
                jslog.warning(msg = f"⚠️ {message['text']}")
            else:
                jslog.debug(msg = f"🪲 {message['text']}")


        @reactive.calc
        def FlowOrder() -> tuple[str, ...]:
            TopologyVersion.get()
            visited = set(SectionsVisited.get())
            order: list[str] = []
            for section_id in SectionOrder():
                if section_id not in visited:
                    continue
                input_id = f"{section_id}_CardOrder"
                section_order = input[input_id]()
                req(section_order is not None)
                order.extend(
                    value.removesuffix("-Card") for value in section_order
                )
            return tuple(order)

        @reactive.effect
        @reactive.event(FlowOrder)
        def wire_flow():
            log.debug(msg="𑙬 Card flow topology changed")
            wire_card_nodes(FlowOrder(), card_nodes)

    return App(
        ui = app_ui, 
        server = server, 
        static_assets = {
            "/html": Module.ROOT / "www" / "markdown",
            "/": Module.ROOT / "www"
        }
        )


app = application()  # This MUST be called "app" for shiny-mode of IDE integration

def main():
    app.run(
        host = "127.0.0.1",
        port = 3277,
        log_level = "info",
        launch_browser = True,
        dev_mode = False,
    )


if __name__ == "__main__":
    """
    This avoids IDE thread conflicts when calling as a python file by
    choosing between foreground and background threads.
    """
    if "ipykernel" in sys.modules:
        t = threading.Thread(target=main, daemon=True)
        log.info("🔙 Shiny (running in background thread)")
        t.start()
    else:
        log.info("➬ Shiny (running in foreground thread)")
        main()  # normal script behavior
else:
    app  # noqa: B018
