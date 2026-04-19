from pathlib import Path
import sys
import pandas as pd  # needed for test / solo modes

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shiny import ui, render, reactive, req  # noqa: E402
from module import Module  # noqa: E402
from card import Card  # noqa: E402
from roles import Role, RoleMap  # noqa: E402
from faicons import icon_svg as icon  # noqa: E402
from proxyData import ProxyData as Pxy  # noqa: E402


def instance():
    this = Card(name = "roleAssign", mutable = False)
    this.long_name = "Role Assignment"
    this.description = "This card enables the variables to be assigned to roles."

    def settings():
        return ui.TagList(
            ui.input_text(
                id = "Separator", 
                label = "Between-identifier-variable separator", 
                value = "|",
                guide = this,
                text = 'Since tab, semi-colon and comma characters are common in identifiers, this string specifies the type of separation to employ.',
                position = "left"),
            ui.input_slider(
                id = "MaxObs", 
                label = "Maximum observations to check for negatives, cardinality etc", 
                min = 3,
                max = 7,
                value = 4,
                ticks = True,
                pre = "10^",
                guide = this,
                text = 'Limit to number of observations to chart to ensure responsiveness (logarithmic scale).',
                position = "left")
        )

    this.settings = settings

    def front():
        grid = ui.div(class_ = "roles-grid")
        for role in [r.value for r in Role]:
            bucket = ui.div(
                ui.div(role.title(), class_="role-title"),
                    ui.div(
                        id = f"role-{role}",
                        class_ = "role-list sortable-role",
                        **{"data-role": role},
                    ),
                class_ = "role-box"
            )
            grid.append(bucket)
        return this.guidedDiv(
            grid, 
            id = "Roles", 
            class_ = "roles-layout",
            guide = this, 
            title = "Role assignments",
            text = "This drag-and-drop dialogue allows the variables to be placed in the appropriate role boxes. You can scroll to the right to access all the roles. This dialogue is best in full screen.",
            position = "top",
            priority = 0
        )

    this.front = front
    
    def back():
        return ui.output_table(id = "Assignments")

    this.back = back

    def footer():
        return ui.TagList(
            ui.input_action_button(
                id = "Commit", 
                label = 'Commit Assignments', 
                icon = icon("gavel", title = "Commit the role assignments", a11y = "sem"),
                disabled = True, 
                width = "250px", 
                class_ = "btn rounded-pill btn-sm d-block mx-auto btn-primary",
                style = "border: 0px; box-shadow: none;",
                guide = this, 
                title = "Commit button",
                text = "This button commits the role assignments. It bounces momentarily when it is ready to be clicked.",
                position = "top"
            ),
            ui.output_ui(
                id = "Check",
                guide = this, 
                title = "Card status",
                text = "This contains a single line of colour coded information about the role validation.",
                position = "top")
        )

    this.footer = footer

    
    def server(input, output, session):

        @reactive.calc()
        def incommingProxyData():
            req(this._imports.is_set())
            return this._imports()
 
        @reactive.calc()
        def PreparedData():
            pxd = incommingProxyData()
            return pxd.sample(n = 10**input.MaxObs(), mode = "random", keep_geometry = True)
            
        @this.suspendable(triggers = [PreparedData])
        async def PopulateRoles():
            messages = ValidateMap()
            if len(messages) > 0: #invalid
                pxd = PreparedData()
                rm = pxd.role_map.to_primitive()
                await session.send_custom_message("PopulateRoles", {"card": session.ns("Card"), "role_map": rm})

        @output
        @render.table
        @this.record_code
        def Assignments():
            orm = this._exports().role_map
            return orm.roles_to_frame()

        @reactive.calc
        @this.record_code
        def ValidateMap():
            pxd = PreparedData()
            this.debug("Validating changes")
            if input.role_map() is None:
                return pxd.validate(separator = input.Separator())
            else: 
                # Convert the json to the RoleMap class
                rm = RoleMap.from_primitive(input.role_map())
                msgs = pxd.validate(role_map = rm, separator = input.Separator()) 
                return msgs

        #### Update commit button ----
        @this.suspendable(triggers = [ValidateMap])
        async def Gatekeeper():
            messages = ValidateMap()
            ok = len(messages) == 0
            ui.update_action_button(
                id = "Commit",
                disabled = not ok
            )
            if ok:
                await session.send_custom_message("animate", {"id" : session.ns("Commit"), "animation" : "bounce", "delay" : 500})

        #### Committed  event ----
        @reactive.calc
        def Committed():
            req(input.role_map())
            pxd = PreparedData()
            return pxd.with_roles(input.role_map()) 
            

        #### Commit event ----
        @this.suspendable(triggers = [input.Commit])
        def CommitEvent():
            this._exports.set(Committed())

        @output
        @render.ui
        async def Check():
            messages = ValidateMap()
            if len(messages) == 0:
                if (this._exports.is_set()) and (this._exports().role_map == Committed().role_map):
                    return ui.span("Assignments applied", class_ = "text-success")
                else:
                    return ui.span("Assignments ready to commit", class_ = "text-primary")
            else:
                i = len(messages)
                return ui.span(i,": ", messages[i-1], class_ = "text-danger")

    this.server = server

    return this


if Module.running_under_tests():
    this = instance()
    df = pd.DataFrame(
        {
            "y": [1, 0, 1, 0],
            "x1": [10.0, 11.0, 12.0, 13.0],
            "x2": ["A", "B", "A", "B"],
            "id": [100, 101, 102, 103],
            "part": ["Train", "Train", "Test", "Test"],
        }
    )
    pxd = Pxy.from_native(df)
    pxd.name = "Test"
    this._imports.set(pxd)
    app = Module.app(modules = {this.ns: this})
elif Module.running_directly(name =__name__):
    this = instance()
    df = pd.read_csv( Card.ROOT / "data" / "Ass2.csv")
    pxd = Pxy.from_native(df)
    pxd.name = "Ass2"
    this._imports.set(pxd)
    Module.run(modules = {this.ns: this})
