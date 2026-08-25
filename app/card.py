###########################
## Class: Card           ##
###########################
##
## Card is the base class for Pythagoras bslib-style analysis cards.
## It inherits from Module.
##
## It provides:
##    A front-side card body
##    Optional back-side card body for flip views
##    Optional right-hand settings sidebar
##    Optional footer
##    Standard card header controls:
##        Flip front/back
##        Show information modal from www/markdown/{card_name}.html
##        Start Shepherd guide
##        Show recorded code modal
##        Expand/contract card
##        Confirm and remove card
##    Namespaced UI via shiny.module.ui/server
##    Basic import -> export passthrough for non-mutable cards
##    Reactive helpers:
##        isFullScreen
##        isFront
##        hasSidebar
##        hasFlipSide
##        hasFooter
##
## Implementations are expected to:
##    Set self.front, and optionally self.back, self.settings, self.footer
##    Implement server(input, output, session)

import base64
from pathlib import Path

import plotly.graph_objects as go
from faicons import icon_svg as icon
from module import Module
from shiny import module, reactive, render, ui

# TODO: add a SibebarActive reactive

class Card(Module):
    
    def __init__(self, file, long_name = None, allow_full_screen = True, mutable = False, *args, **kwargs): # will be inherited by child classes
        if file is None:
            raise ValueError("Filename is required — stopping.")
        name = Path(file).resolve().stem
        super().__init__(name, *args, **kwargs)
        self.file = file
        self.allow_full_screen = allow_full_screen
        self.mutable = mutable
        self.initially_hidden = False
        self.description = None
        self.long_name = long_name
        self._front = None
        self._back = None
        self._settings = None
        self._footer = None

    max_height = Module.config.get("settings", {}).get("max_card_height")
    SHADOW_PREFIX = "shadow__"

    def empty_figure(
        message: str,
        *,
        icon_name: str = "eye-slash",
        icon_colour: str = "#198754",
    ) -> go.Figure:
        icon_tag = icon(icon_name, title=message, a11y="sem")
        svg = icon_tag.get_html_string()
        svg = svg.replace("<svg ", '<svg xmlns="http://www.w3.org/2000/svg" ',  1)
        svg = svg.replace('preserveAspectRatio="none"', 'preserveAspectRatio="xMidYMid meet"')
        svg = svg.replace("fill:currentColor", f"fill:{icon_colour}")
        encoded_svg = base64.b64encode(svg.encode("utf-8")).decode("ascii")
        icon_source = f"data:image/svg+xml;base64,{encoded_svg}"
        figure = go.Figure()
        figure.add_layout_image(
            source=icon_source,
            x=0.5,
            y=0.62,
            xref="paper",
            yref="paper",
            sizex=0.28,
            sizey=0.28,
            xanchor="center",
            yanchor="middle",
            sizing="contain",
            opacity=0.25,
            layer="above",
        )
        figure.add_annotation(
            text=message,
            x=0.5,
            y=0.35,
            xref="paper",
            yref="paper",
            showarrow=False,
            align="center",
            font={"size": 16, "color": "#6c757d"},
        )
        figure.update_layout(
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#e5ecf6",
            margin={"l": 20, "r": 20, "t": 20, "b": 20},
            xaxis={"visible": False, "fixedrange": True},
            yaxis={"visible": False, "fixedrange": True},
        )
        return figure
    def fetch(self, value):
        if value is None:
            return None
        if callable(value):
            return value()
        else:
            return value

    @property
    def front(self):
        return self.fetch(self._front) 
    @front.setter
    def front(self, value):
        self._front = value

    @property
    def back(self):
        return self.fetch(self._back) 
    @back.setter
    def back(self, value):
        self._back = value

    @property
    def footer(self):
        return self.fetch(self._footer) 
    @footer.setter
    def footer(self, value):
        self._footer = value

    @property
    def settings(self):
        return self.fetch(self._settings) 
    @settings.setter
    def settings(self, value):
        self._settings = value

    def hasSidebar(self) -> bool:
        return self._settings is not None

    def hasFlipSide(self) -> bool:
        return self._back is not None

    def hasFooter(self) -> bool:
        return self._footer is not None

    def call_ui(self):

        @module.ui
        def ui_cardfunc():

            def header():
                # flip button
                if not self.hasFlipSide():
                    flip_button = None
                else:
                    flip_button = ui.input_action_button(
                        id = "FlipButton",
                        label = None,
                        icon = icon("arrows-rotate", title = "Flip this card", a11y = "sem"),
                        class_ = "btn rounded-pill hover-btn btn-sm flip-btn",
                        style = "border: 0px; box-shadow: none;",
                        aria_label = "Flip this card",
                        guide = self,
                        title = "Flip button", priority = 13, position = "bottom",
                        text = "This button flips the card over. It is available when the card is in use. The reverse side generally shows evidence for the cards visualisations and tables. Clicking the button again will return the card to the front again. The [esc] key can also be used."
                    )


                # Info button
                card_file = Path(self.file).resolve()
                html_file = card_file.parents[1] / "www" / "markdown" / card_file.with_suffix(".html").name
                if not html_file.exists():
                    info_button = None
                else:
                    info_button = ui.input_action_button(
                        id = "InfoButton",  
                        label = None,
                        icon = icon("info", title = "Information about this card", a11y = "sem"),
                        class_ = "btn rounded-pill hover-btn btn-sm info-btn",
                        style = "border: 0px; box-shadow: none;",
                        aria_label = "Information about this card",
                        guide = self, title = "Info button", priority = 12, position = "bottom",
                        text = "The info button displays a discussion of the card's significance and use."
                        )


                # Guide button
                guide_button = ui.input_action_button(
                    id = "GuideButton", 
                    label= None, 
                    icon = icon("eye", title = "Take a guided tour of this card", a11y = "sem"), 
                    class_ = "btn rounded-pill hover-btn btn-sm guide-btn",
                    style = "border: 0px; box-shadow: none;",
                    aria_label = "Take a guided tour of this card",
                    guide = self, title = "Guide button", priority = 11, position = "bottom",
                    text = "The guide button starts a tour of the card's features. The tour can be controlled by the keyboard through the arrow and [esc] keys. The tour will take you through any tabs in the card and through any sidebar settings."
                )

                # Code button
                code_button = ui.input_action_button(
                    id = "CodeButton",  
                    label= None, 
                    icon = icon("code", title = "View the code associated with this card", a11y = "sem"),
                    class_ = "btn rounded-pill hover-btn btn-sm code-btn",
                    style = "border: 0px; box-shadow: none;",
                    aria_label = "View the code associated with this card",
                    guide = self, priority = 10, title = "Code button", position = "bottom",
                    text = "The code button lists the python code employed to generate the tables and charts <em>that you have viewed</em>. This code can be copied and used in a python notebook."
                )


                if self.allow_full_screen:
                    expand_button = ui.input_action_button(
                        id = "ExpandButton",  
                        label= None, 
                        icon = icon("maximize", title = "Expand this card", a11y = "sem"),
                        class_ = "btn rounded-pill hover-btn btn-sm expand-btn",
                        style = "border: 0px; box-shadow: none; display: block;",
                        guide = self, priority = 10, title = "Expand button", position = "bottom",
                        text = "The expand button enlarges the card to full-screen."
                    )
                    contract_button = ui.input_action_button(
                        id = "ContractButton",  
                        label = None, 
                        icon = icon("minimize", title = "Restore this card", a11y = "sem"),
                        class_ = "btn rounded-pill hover-btn btn-sm contract-btn",
                        style = "border: 0px; box-shadow: none;",
                        guide = self, priority = 10, title = "Contract button", position = "bottom",
                        text = "The contract button restores the card to its normal size. The [esc] key can also be used."
                    )

                else:
                    expand_button = None
                    contract_button = None

                close_button = ui.input_action_button(
                    id = "CloseButton",  
                    label = None, 
                    icon = icon("xmark", title = "Close this card", a11y = "sem"),
                    class_ = "btn rounded-pill hover-btn btn-sm close-btn",
                    style = "border: 0px; box-shadow: none;",
                    guide = self, priority = 10, title = "Close button", position = "bottom",
                    text = "The close button removes the card."
                )

                return ui.card_header(
                    ui.div(
                        class_="drag-tab drag-handle hover-btn shadow",
                        title="Drag",
                        role="button",
                        aria_label="Drag card",
                        tabindex="0"
                    ),
                    ui.div(
                        ui.tags.img(src="favicon.ico", style="height:2em; margin-right:0.5em;"), #attrs = str({"title": "Tetractys", "a11y": "sem"})),  #Tetractys
                        self.long_name,
                        ui.output_text(id = "Name", inline = True),
                        class_="d-flex align-items-center gap-1"
                    ),
                    # Right side: buttons
                    ui.div(
                        flip_button,
                        info_button,
                        guide_button,
                        code_button,
                        expand_button,
                        contract_button,
                        close_button,
                        class_="d-flex align-items-center gap-2"
                    ),
                    class_="d-flex justify-content-between align-items-center fs-6 bg-info bg-opacity-25 px-3 py-2"
                )

            def front_back():
                return ui.card_body(
                    ui.div(
                        ui.div(
                            self.front,
                            id = self.ns("Front"), # The decorator misses divs
                            class_ = "front html-fill-container html-fill-item"
                        ),
                        ui.div(
                            self.back,
                            id = self.ns("Back"), # The decorator misses divs
                            class_ = "back html-fill-container html-fill-item"
                        ),
                        class_ = "flippable html-fill-container html-fill-item"
                    ),
                    id = self.ns("CardBody"),  # The decorator misses divs
                    class_ = "flip-container html-fill-container html-fill-item",
                    fillable = True,
                    fill = True,
                    gap = 0,
                    padding = 10
                )

            def myfooter():
                if self.hasFooter():
                    return ui.card_footer(self.footer, class_ = "text-center bg-info bg-opacity-25")
                return None
                

            if not self.hasSidebar():
                return ui.card(
                    header(),
                    front_back(),
                    myfooter(),
                    id = "Card",
                    fill   = True,
                    class_ = "shadow hover-card p-0 m-2 hidden" if self.initially_hidden else "shadow hover-card p-0 m-2",
                    height = self.max_height,
                    min_height = "250px",
                    max_height = self.max_height,
                    gap = 0,
                    padding = 0
                )
            else:
                sb = ui.sidebar(
                    ui.card_header("Settings", class_ = "w-100 text-end text-primary sidebar-title"),
                    self.settings,
                    id = "Sidebar", 
                    width = "30%",
                    position = "right", 
                    open = "closed",
                    padding = [12,5,5,0], #top right bottom left
                    bg = "lightgrey"
                )

                return ui.card(
                    ui.layout_sidebar(
                        sb,
                        header(),
                        front_back(),
                        myfooter(),
                        padding = [0,25,0,0],
                        gap = 0
                    ),    
                    id = "Card",
                    fill   = True,
                    class_ = "shadow hover-card p-0 m-2 hidden" if self.initially_hidden else "shadow hover-card p-0 m-2",
                    height = self.max_height,
                    min_height = "250px",
                    max_height = self.max_height
                )
        return ui_cardfunc(id = self.namespace)


    # Read html from a file 
    def information(self):
        card_file = Path(self.file).resolve()
        html_file = self.ROOT / "www" / "markdown" / card_file.with_suffix(".html").name
        print(f"  ROOT={self.ROOT}  ")
        if html_file.exists():
            try:
                text = html_file.read_text(encoding="utf-8")
                return text
            except Exception:  # noqa: BLE001
                return f"<br>Error reading {html_file} file"
        else:
            return f"<br>File {html_file} not found"

    def call_server(self, input, output, session):

        @module.server
        def server_func(input, output, session):


            # isFullScreen
            @self.suspendable(calc = True)
            def isFullScreen():
                if isinstance(input.Card_full_screen(), bool):
                    return input.Card_full_screen()
                return False
            self.isFullScreen = reactive.calc(isFullScreen)

            # isFront
            @self.suspendable(calc = True)
            def isFront():
                if self.back is None:
                    return True
                is_front = input.Card_is_front()
                if isinstance(is_front, bool):
                    return is_front
                return True
            self.isFront = reactive.calc(isFront)


            # Info button event
            @self.suspendable(triggers = [input.InfoButton])
            def show_info():
                ui.modal_show(
                    ui.modal(
                        ui.card(
                            ui.card_header(
                                self.long_name,
                                class_="d-flex justify-content-between fs-6 bg-info bg-opacity-25",
                                style = "display:inline-block; margin-right:8px;"
                            ),
                            ui.card_body(
                                self.description,
                                ui.HTML(self.information())
                            )
                        ),
                        fade = True,
                        easy_close = True,
                        size = "l"
                    )
                )


            # Guide button event
            @self.suspendable(triggers = [input.GuideButton])
            async def GuideButton():
                await self.create_run_tour(session)


            # Code button event
            @self.suspendable(triggers = [input.CodeButton])
            def show_Code():
                ui.modal_show(
                    ui.modal(
                        ui.head_content(
                            ui.include_js(path = "www/clipboard.js", method = "inline")
                        ),
                        ui.card(
                            ui.card_header(
                                ui.div(
                                    ui.tags.img(src="favicon.ico", style="height:2em; margin-right:0.5em;"),
                                    f"{self.long_name} code",
                                    class_="d-flex align-items-center"
                                ),
                                ui.HTML("""
                                    <button class='btn btn-default action-button btn rounded-pill btn-sm clipboard-btn' type='button' aria-label='Copy to the clipboard' style='border: 0px; box-shadow: none;'>
                                    <svg viewBox='0 0 384 512' preserveAspectRatio='none' aria-label='Copy to the clipboard' role='img' class='fa' style='fill:currentColor;height:1em;width:0.75em;margin-left:auto;margin-right:0.2em;position:relative;vertical-align:-0.125em;overflow:visible;'>
                                    <title>Copy to the clipboard</title>
                                    <path d='M192 0c-41.8 0-77.4 26.7-90.5 64H48C21.5 64 0 85.5 0 112V464c0 26.5 21.5 48 48 48H336c26.5 0 48-21.5 48-48V112c0-26.5-21.5-48-48-48H282.5C269.4 26.7 233.8 0 192 0zm0 128c-17.7 0-32-14.3-32-32s14.3-32 32-32s32 14.3 32 32s-14.3 32-32 32zm-80 64H272c8.8 0 16 7.2 16 16s-7.2 16-16 16H112c-8.8 0-16-7.2-16-16s7.2-16 16-16z'></path>
                                    </svg></button>
                                """),  # see www/clipboard.js for click event.
                                class_="d-flex justify-content-between fs-6 bg-info bg-opacity-25 clipboard-btn",
                                style = "display:inline-block; margin-right:8px;"
                            ),
                            ui.card_body(
                                self.code_text(),
                                class_ = "clipboard-text",
                                style = "white-space: pre-wrap; font-family: monospace;"
                            )
                        ),
                        fade = True,
                        easy_close = True,
                        size = "xl"
                    )
                )


            @self.suspendable(triggers=[input.CloseButton])
            def _confirm_remove_card():
                ui.modal_show(
                    ui.modal(
                        ui.card_header(ui.tags.h3(self.long_name)),
                        ui.tags.h5("Remove this card?"),
                        ui.p("This action cannot be easily undone."),
                        ui.div(
                            ui.input_action_button(id = "ConfirmRemove", label = "Yes, remove", class_="btn-danger"),
                            ui.input_action_button(id = "CancelRemove", label = "Cancel"),
                            class_="d-flex justify-content-end gap-2"
                        ),
                        easy_close=True,
                        footer=None
                    )
                )

            @self.suspendable(triggers=[input.ConfirmRemove])
            def _remove_card():
                ui.modal_remove()
                self.suspend()
                id = self.ns('Card')
                self.reset()
                ui.remove_ui(selector=f"#{id}")

            @self.suspendable(triggers=[input.CancelRemove])
            def _cancel():
                ui.modal_remove()


            @reactive.effect
            def passthrough():
                if not self.mutable:
                    if self._imports.is_set():
                        self._exports.set(self._imports.get())
                    else:
                        self._exports.unset()

            @output
            @render.text
            def Name():
                if self._imports.is_set():
                    return f"of \"{self._imports.get().name}\""
                elif self._exports.is_set():
                    return f"of \"{self._exports.get().name}\""
                else:
                    return ""
            

            return self.server(input, output, session)

        return server_func(self.namespace)
