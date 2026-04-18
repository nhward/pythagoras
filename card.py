###########################
## Class: Card           ##
###########################

## This is a class for a bslib card-based design
## This class inherits from Module, Meta and Guide
## It provides:
##    Front side card
##    Optional back side card
##    Optional Right-sided sidebar for settings
##    Header with buttons 
##        Implementation of front-to-back flipping
##        Implementation of Info modal dialogue based on markdown
##    Between card/module communication
##    Update_element() function for updating class and/or style

from module  import Module
from shiny   import ui, module, reactive, render, req
from faicons import icon_svg as icon
from pathlib import Path


# TODO: add a SibebarActive reactive
# TODO: make it possible to edit the raw markdown in situ and save it.


class Card(Module):
    
    def __init__(self, name, logger = None, long_name = None, allow_full_screen = True, max_height = "450px", mutable = False, *args, **kwargs): # will be inherited by child classes
        super().__init__(name=name, logger=logger, *args, **kwargs)
        self.allow_full_screen = allow_full_screen
        self.max_height = max_height
        self.mutable = mutable
        self.initially_hidden = False
        self.description = None
        self.long_name = long_name or self.name
        self.script_list.append(self.WWW / "pythagoras.js")
        self.script_list.append(self.WWW / "markdown_tabsets.js")
        self.css_list.append(self.WWW / "animate.css")
        self.css_list.append(self.WWW / "pythagoras.css")

    def front(self):
        return None
    def back(self):
        return None
    def settings(self):
        return None
    def footer(self):
        return None

    def hasSidebar(self):
        return self.settings() is not None

    def hasFlipSide(self):
        return self.back() is not None

    def hasFooter(self):
        return self.footer() is not None

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
                        text = "This button flips the card over. t is available when the card is in use. The reverse side generally shows evidence for the the cards visualisations and tables. Clicking the button again will return the card to the front again. The [esc] key can also be used."
                    )


                # Info button
                file = Path(self.ROOT / "markdown" / f"{self.name}.md")
                if not file.exists():
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
                        text = "The info buttom displays a discussion of the card's significance and use."
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
                    text = "The guide button starts a tour of the cards features. The tour can be controlled by the keyboard through the arrow and [esc] keys. The tour will take you through any tabs in the card and through any sidebar settings."
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

                expand_button = ui.input_action_button(
                    id = "ExpandButton",  
                    label= None, 
                    icon = icon("maximize", title = "Expand this card", a11y = "sem"),
                    class_ = "btn rounded-pill hover-btn btn-sm expand-btn",
                    style = "border: 0px; box-shadow: none; display: block;",
                    guide = self, priority = 10, title = "Expand button", position = "bottom",
                    text = "The expand button enlarges the card to full-screen."
                )

                if self.allow_full_screen:
                    contract_button = ui.input_action_button(
                        id = "ContractButton",  
                        label = None, 
                        icon = icon("minimize", title = "Restore this card", a11y = "sem"),
                        class_ = "btn rounded-pill hover-btn btn-sm contract-btn",
                        style = "border: 0px; box-shadow: none;",
                        guide = self, priority = 10, title = "Contract button", position = "bottom",
                        text = "The contract button restores the card to its normal size. The [esc] key can also be used."
                    )

                    close_button = ui.input_action_button(
                        id = "CloseButton",  
                        label = None, 
                        icon = icon("xmark", title = "Close this card", a11y = "sem"),
                        class_ = "btn rounded-pill hover-btn btn-sm close-btn",
                        style = "border: 0px; box-shadow: none;",
                        guide = self, priority = 10, title = "Close button", position = "bottom",
                        text = "The close button removes the card."
                    )

                else:
                    contract_button = None
                    close_button = None
                
                return ui.card_header(
                    ui.div(
                        class_ = "drag-tab drag-handle hover-btn shadow ", 
                        title = "Drag",     
                        attrs = str({"role": "button", "aria-label": "Drag card", "tabindex": "0"})
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
                            self.front(),
                            id = self.ns("Front"), # The decorator misses divs
                            class_ = "front html-fill-container html-fill-item"
                        ),
                        ui.div(
                            self.back(),
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

            def _footer():
                if self.hasFooter():
                    return ui.card_footer(self.footer(), class_ = "text-center bg-info bg-opacity-25")
                return None
                

            if not self.hasSidebar():
                return ui.card(
                    header(),
                    front_back(),
                    _footer(),
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
                    self.settings(),
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
                        _footer(),
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


    # Read markdown from a file 
    def information(self):
        # Path to your markdown file
        #import markdown

        file = Path(self.ROOT / "markdown" / f"{self.name}.md")
        try:
            # Read the markdown content
            with file.open(encoding="utf-8") as f:
                text = f.read()
            # Convert markdown to HTML - see the markdown extensions in use
            # return ui.HTML(markdown.markdown(text, extensions=["extra", "tables", "fenced_code", "markdown_katex"]))
            return ui.markdown(text)  # TODO check this change works okay - otherwise use commented-out line above
        except Exception:
            return None

    async def show(self, session):
        await session.send_custom_message("toggle_visibility", {"id": session.ns("Card"), "visible": True})

    async def hide(self, session):
        await session.send_custom_message("toggle_visibility", {"id": session.ns("Card"), "visible": False})


    def call_server(self, input, output, session):

        @module.server
        def server_func(input, output, session):


            # isFullScreen
            @reactive.calc
            def isFullScreen():
                if isinstance(input.Card_full_screen(), bool):
                    return input.Card_full_screen()
                return False
            self.isFullScreen = reactive.calc(isFullScreen)

            # isFront
            @reactive.calc
            def isFront():
                return input.FlipButton() % 2 == 0
            self.isFront = reactive.calc(isFront)


            # Info button event
            @self.suspendable(triggers = [input.InfoButton])
            def show_info():
                ui.modal_show(
                    ui.modal(
                        # the following head_contents() 'seems' to be necessary to add here as modal documents do not reliably inherit from the main document
                        ui.head_content(
                            ui.tags.script(path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js", method = "inline"), # enables equations
                            ui.tags.script(path = "www/markdown_tabsets.js", method = "inline")   # enables tabsets
                        ),
                        ui.card(
                            ui.card_header(
                                self.long_name,
                                # ui.input_action_button(
                                #     id = "Edit", 
                                #     label = None,
                                #     icon = icon("pen-to-square", title = "Edit the document", a11y = "sem"),
                                #     class_ = "btn rounded-pill btn-sm",
                                #     style = "border: 0px; box-shadow: none;"
                                # ),
                                class_="d-flex justify-content-between fs-6 bg-info bg-opacity-25",
                                style = "display:inline-block; margin-right:8px;"
                            ),
                            ui.card_body(
                                self.description,
                                self.information()
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


            @self.suspendable(triggers = [input.Copy])
            async def CopyClick():
                self.debug("Copying")
                await session.send_custom_message("copyText", str(self.code_text()))

            @reactive.calc
            def CurrentTabCardOrder(): #given in namespaces
                req(input.CardOrder)
                order = input.CardOrder()
                order = [s.removesuffix("-Card") for s in order]
                return order

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
                async def after_flush():
                    await session.send_custom_message("UpdateCardOrder", None)
                session.on_flushed(after_flush, once=True)

            @self.suspendable(triggers=[input.CancelRemove])
            def _remove_card():
                ui.modal_remove()

            result = self.server(input, output, session)
 
            @reactive.effect
            def name_passthrough():
                if not self.mutable:
                    if self._imports['name'].is_set():
                        self.debug(f"import -> export: {self._imports['name'].get()}")
                        self._exports["name"].set(self._imports["name"].get())
                    else:
                        self._exports["name"].unset()

            @reactive.effect
            def data_passthrough():
                if not self.mutable:
                    if self._imports['data'].is_set():
                        self._exports["data"].set(self._imports["data"].get())
                    else:
                        self._exports["data"].unset()

            @output
            @render.text
            def Name():
                if self._exports["name"].is_set():
                    return f"of \"{self._exports['name'].get()}\""
                else:
                    return ""

            return result
        return server_func(self.namespace)
