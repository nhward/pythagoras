from pathlib import Path
import sys
import pandas as pd  # needed for test / solo modes

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shiny import ui, render, req  # noqa: E402
from module import Module  # noqa: E402
from card import Card  # noqa: E402
from roles import Role, RoleMap  # noqa: E402
from faicons import icon_svg as icon  # noqa: E402
from proxyData import ProxyData as Pxy  # noqa: E402


def instance():
    this = Card(name = "roleAssign", mutable = True) # "mutable" means it can change the pxd - probably with a commit button
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
                id = "CardinalityThreshold", 
                label = "Maximum cardinality of of \"Low Cardinality\" roles", 
                min = 3,
                max = 50,
                value = 4,
                ticks = True,
                guide = this,
                text = 'Limit the cardinality of certain roles to be less than this - specificially: Sensitive, Stratifier and Treatment roles. This setting is used in role validation.',
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

        @this.suspendable(calc = True)
        def incomingProxyData():
            this._imports.get()
            req(this._imports.is_set())
            return this._imports.get()
 
        @this.suspendable(calc = True)
        def PreparedData():
            pxd = incomingProxyData()
            return pxd.sample(n = 10**input.MaxObs(), mode = "random", keep_geometry = True)
            
        @this.suspendable(triggers = [PreparedData])
        async def PopulateRoles():
            this._imports.get()
            if not this._imports.is_set():
                rm = RoleMap().to_primitive()
                await session.send_custom_message("PopulateRoles", {"card": session.ns("Card"), "role_map": rm})
            else:
                try:
                    input.role_map()
                    messages = ValidateMap() # using input.role_map
                    if len(messages) > 0: # not valid
                        pxd = PreparedData()
                        rm = pxd.role_map.to_primitive()
                        await session.send_custom_message("PopulateRoles", {"card": session.ns("Card"), "role_map": rm})
                except (Exception):
                    pxd = PreparedData()
                    rm = pxd.role_map.to_primitive()
                    await session.send_custom_message("PopulateRoles", {"card": session.ns("Card"), "role_map": rm})

        @output
        @render.table
        @this.record_code
        def Assignments():
            orm = this._exports.get().role_map
            return orm.roles_to_frame()

        @this.suspendable(calc = True)
        @this.record_code
        def ValidateMap():
            this._imports.get()
            req(this._imports.is_set())
            req(input.role_map())
            this.log.debug("Validating changes")
            # Convert the json to the RoleMap class
            rm = RoleMap.from_primitive(input.role_map())
            pxd = PreparedData()
            msgs = pxd.validate(role_map = rm, separator = input.Separator(), low_cardinality = input.CardinalityThreshold()) 
            return msgs


        #### Committed  event ----
        @this.suspendable(calc = True)
        def Committed():
            req(input.role_map())
            return Pxy(_df = incomingProxyData().to_native(), _roles = input.role_map(), _name = incomingProxyData().name) 
            

        #### Commit event ----
        @this.suspendable(triggers = [input.Commit])
        def CommitEvent():
            this._exports.set(Committed())

        @this.suspendable()
        def allowAutoCommit():
            if not this._exports.is_set() and this._imports.is_set():
                this._exports.set(this._imports.get())
            elif this._imports.is_set() and this._imports.get().role_map == this._exports().role_map:
                this._exports.set(this._imports.get())

        @output
        @render.ui
        async def Check():
            messages = ValidateMap()
            ok = len(messages) == 0
            ui.update_action_button(id = "Commit", disabled = not ok)
            if ok:
                if (this._exports.is_set()) and (this._exports.get() == Committed()):
                    return ui.span("Assignments applied", class_ = "text-success")
                else:
                    await session.send_custom_message("animate", {"id" : session.ns("Commit"), "animation" : "bounce", "delay" : 500})
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
    pxd = Pxy(_df = df, _name = "Test")
    this._imports.set(pxd)
    app = this.application()
elif Module.running_directly(name =__name__):
    this = instance()
    df = pd.read_csv( Card.ROOT / "data" / "Ass2.csv")
    pxd = Pxy(_df = df, _name = "Ass2")
    this._imports.set(pxd)
    this.run()
