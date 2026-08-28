from __future__ import annotations

import os
import sys
from pathlib import Path

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    # Ensure local modules and packages are resolved from the app directory.
    os.chdir(ROOT)
    root_string = str(ROOT)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)

import importlib.metadata
import os
import platform
import sys

import pandas as pd
import session_info
from card import Card
from faicons import icon_svg as icon
from module import Module
from shiny import render, req, ui


def instance():
    """
    Creates an instance of Card configured as "configuration".
    """
    this = Card(file=__file__, mutable=False) # "mutable" means it can change the pxd - probably with a commit button
    this.long_name = "Configuration"
    this.description = "This card records the host-system configuration."


    def front():
        return ui.navset_bar(
            ui.nav_panel("Summary",  ui.output_table(id = "Summary",  guide = this, title = "Summary",  position = "top", text = "This table briefly lists the host, URL, Python executable/version, Platform, and packages (installed/Loaded).")),
            ui.nav_panel("Url",      ui.output_table(id = "Url",      guide = this, title = "URL",      position = "top", text = "This table lists the components of the URL for the current web page.")),
            ui.nav_panel("Packages", ui.output_table(id = "Packages", guide = this, title = "Packages", position = "top", text = "This table lists the loaded package names, and versions, that have been loaded upto now.")),
            ui.nav_panel("Folders",  ui.output_table(id = "Folders",  guide = this, title = "Folders",  position = "top", text = "This table lists the Pythagoras directory paths and the file count of each.")),
            title = None,
            id = "Navset", 
            padding = 0, 
            fillable = True
        )
    this.front = front

    def back():
        return ui.output_text_verbatim(id = "Session", guide = this, title = "Session", priority = -10, position = "bottom", text = "This listing on the flip-side shows the traditional system configuration output")
    this.back = back
  
    def footer():
        return ui.input_action_button(
            id = "Refresh", 
            label = 'Refresh', 
            icon = icon("arrows-rotate", title = "Refesh the information", a11y = "sem"),
            width = "250px", 
            class_ = "btn rounded-pill btn-sm d-block mx-auto btn-primary",
            style = "border: 0px; box-shadow: none;",
            guide = this, 
            title = "Refresh button",
            text = "This button refreshes the list of currently loaded modules.",
            position = "top"
        )
    this.footer = footer


    def server(input, output, session):

        @this.record_code
        def get_loaded_packages():
            loaded = {}
            for name, module in list(sys.modules.items()):
                version = None
                if module:
                    try:
                        version = importlib.metadata.version(name)
                    except Exception:  # noqa: BLE001, S110
                        pass  # ignore modules without metadata
                if isinstance(version, str):
                    loaded[name] = version

            df = pd.DataFrame(list(loaded.items()), columns=["Loaded package", "Version"])
            return df.sort_values("Loaded package", key=lambda s: s.str.lower()).reset_index(drop=True)


        @output
        @render.table
        @this.record_code
        def Summary():
            input.Refresh()
            s = Module.ModSession
            req(s)
            if s.clientdata.url_hostname() == "localhost" or s.clientdata.url_hostname() == "127.0.0.1":
                local = "Yes" 
            else:
                local = "No" 

            data = {
                "Running locally": local,
                "Python executable": sys.executable,
                "Python version": sys.version.replace("\n", " "),
                "Platform": platform.platform(),
                "Pixel ratio": s.clientdata.pixelratio(),
                "Installed packages": len(list(importlib.metadata.distributions())),
                "Loaded packages": get_loaded_packages().shape[0]
            }
            return pd.DataFrame(list(data.items()), columns=["Property", "Value"])

        @output
        @render.table
        #Do not record code as this is all shiny specific
        def Url():
            input.Refresh()
            s = Module.ModSession
            req(s)
            data = {
                "Host name": s.clientdata.url_hostname(),
                "Path name": s.clientdata.url_pathname(),
                "Port": s.clientdata.url_port(),
                "Protocol": s.clientdata.url_protocol(),
                "Search": s.clientdata.url_search(),
                "URL hash initial": s.clientdata.url_hash_initial(),
                "URL hash": s.clientdata.url_hash(),
            }
            return pd.DataFrame(list(data.items()), columns=["Property", "Value"])


        @output
        @render.table
        @this.record_code
        def Folders():
            input.Refresh()
            dirs = ["." , "./www", "./www/markdown", "./cards", "./config"]
            rows = []
            for label, d in zip(["home", "www", "markdown", "cards"], dirs):
                abs_path = os.path.abspath(d)
                count = len(os.listdir(d)) if os.path.exists(d) else 0
                rows.append({"Name": label, "Path": abs_path, "Files": count})
            return pd.DataFrame(rows)


        @output
        @render.table
        @this.record_code
        def Packages():
            input.Refresh()
            return get_loaded_packages()


        @output
        @render.text
        @this.capture_print
        @this.record_code
        def Session():
            input.Refresh()
            session_info.show(cpu = True, dependencies = this.isFullScreen(), std_lib = this.isFullScreen(), private = this.isFullScreen(), html = False)  # writes to stdout


    this.server = server

    return this


if Module.running_directly(name =__name__):
    this = instance()
    this.run()
