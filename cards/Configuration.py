from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shiny import ui, render, req  # noqa: E402
from card import Card  # noqa: E402
from module import Module  # noqa: E402
from faicons import icon_svg as icon  # noqa: E402

def instance():
    this = Card(name = "configuration", mutable = False)
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
            import pandas as pd
            import importlib.metadata
            import sys

            loaded = {}
            for name, module in sys.modules.items():
                version = None
                if module:
                    try:
                        version = importlib.metadata.version(name)
                    except Exception:
                        pass  # ignore modules without metadata
                if isinstance(version, str):
                    loaded[name] = version

            df = pd.DataFrame(list(loaded.items()), columns=["Loaded package", "Version"])
            return df.sort_values("Loaded package", key=lambda s: s.str.lower()).reset_index(drop=True)


        @output
        @render.table
        @this.record_code
        def Summary():
            import pandas as pd
            import importlib.metadata
            import sys
            import platform
            input.Refresh()
            s = Module.ModSession
            req(s)
            if s.clientdata.url_hostname() == "localhost":
                local = "Yes" 
            elif s.clientdata.url_hostname() == "127.0.0.1":
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
            import pandas as pd
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
            import pandas as pd
            import os
            input.Refresh()
            dirs = ["." , "./www", "./markdown", "./cards"]
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
            import session_info
            input.Refresh()
            session_info.show(cpu = True, dependencies = this.isFullScreen(), std_lib = this.isFullScreen(), private = this.isFullScreen(), html = False)  # writes to stdout


    this.server = server

    return this

app = None
if Module.running_under_tests():
    this = instance()
    app = Module.app(modules = {this.ns: this})
elif Module.running_directly(name =__name__):
    this = instance()
    Module.run(modules = {this.ns: this})