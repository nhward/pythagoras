from shiny import ui, render, reactive, req
from module import Module
from card import Card
from roles import Role  #, RoleMap
from faicons import icon_svg as icon
import json


__version__ = "0.1.0"


def instance():
    this = Card(name = "RoleMapping", mutable = False)
    this.long_name = "Role Assignment"
    this.description = "This card enables the variables to be assigned to roles."


    COLUMNS = [
        "age", "income", "postcode", "outcome", "weight",
        "lat", "lon", "customer_id", "date"
    ]

    ROLES = [r.value for r in Role]
 
    def front():
        grid = ui.div(
            {"class": "roles-grid"},
        )
        for role in ROLES:
            inner = ui.div(
                    id=f"role-{role}",
                    class_="role-list sortable-role",
                    **{"data-role": role},
                )
            bucket = ui.div(
                {"class": "role-box"},
                ui.div(role.title(), class_="role-title"),
                inner
            )
            if role == "predictor":
                inner.append(
                    *[ui.div(
                        col,
                        class_="var-chip",
                        **{"data-varname": col}
                    ) for col in COLUMNS]
                )
            grid.append(bucket)
        this.info("Calling Sortable Roles")
        return ui.div(
            {"class": "roles-layout", "style": "overflow-x: auto;"},
            grid
        )
 
    this.front = front
    
    def back():
        return ui.output_text_verbatim(id = "Assignments")

    this.back = back

    def footer():
        return ui.input_action_button(
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
        )

    this.footer = footer

    
    def server(input, output, session):
        
        @output
        @render.ui
        def Mapping():
            grid = ui.div(
                {"class": "roles-grid"},
            )
            for role in ROLES:
                inner = ui.div(
                        id=f"role-{role}",
                        class_="role-list sortable-role",
                        **{"data-role": role},
                    )
                bucket = ui.div(
                    {"class": "role-box"},
                    ui.div(role.title(), class_="role-title"),
                    inner
                )
                if role == "predictor":
                    inner.append(
                        *[ui.div(
                            col,
                            class_="var-chip",
                            **{"data-varname": col}
                        ) for col in COLUMNS]
                    )
                grid.append(bucket)
            this.info("Calling Sortable Roles")
            return ui.div(
                {"class": "roles-layout", "style": "overflow-x: auto;"},
                grid
            )

        

        # @reactive.Effect
        # async def custom():
        #     this.info("Triggering sortable-roles")
        #     await session.send_custom_message("SortableRoles", {"card" : session.ns("Card"), "input" : session.ns("role_map")})    

        @reactive.calc
        def data_ready():
            return this._imports["data"].is_set()

        @reactive.calc()
        def GetData():
            req(data_ready())
            return this._imports["data"]()

        @render.text
        def Assignments():
            # req(data_ready())
            return json.dumps(input.role_map(), indent=2)
        
      
        @reactive.calc
        def ValidateMap():
            return True #tempoarary
        

        #### Update commit button ----
        @this.suspendable(triggers = [ValidateMap])
        def GateKeeper():
            ui.update_action_button(
                id = "Commit",
                disabled = not ValidateMap()
            )

        #### Commit event ----
        @this.suspendable(triggers = [input.Commit])
        def CommitEvent():
            pass
            # CommittedData.set(GetData2())
            # this._exports["data"].set(CommittedData())

        # CurrentColumns = reactive.Value(None)

        # @reactive.calc()
        # def changedBaseRoleMap():
        #     data = GetData()
        #     req(CurrentColumns() is None or set(CurrentColumns()) != set(data.columns))
        #     CurrentColumns.set(data.columns)
        #     this.debug(msg = "New-data roles:")
        #     for r in data._roles:
        #         this.debug(msg = f"\t {r}")  
        #     return data._roles


        # @output
        # @render.ui
        # def RoleMapWidget():
        #     # req(changedBaseRoleMap() is not None)

        #     def role_column(role_value: str, label: str, cols_in_role: list[str]):
        #         # role_value is either a Role.value (e.g. "target") or UNASSIGNED_SENTINEL
        #         return ui.div(
        #             ui.h6(label, class_="text-center"),
        #             ui.tags.ul(
        #                 *[
        #                     ui.tags.li(
        #                         col,
        #                         **{
        #                             "data-col": col,
        #                             "class": "var-pill list-group-item list-group-item-action py-1 px-2 m-1",
        #                         },
        #                     )
        #                     for col in cols_in_role
        #                 ],
        #                 **{
        #                     "data-role": role_value,
        #                     "class": "role-zone list-unstyled border rounded p-2 min-vh-25",
        #                 },
        #             ),
        #             class_="col"
        #         )
        #     with reactive.isolate():
        #         rolemap = GetData()._roles
        #         print(rolemap)

        #     return ui.div(
        #         ui.div(
        #             "testing",
        #             # *[
        #                 # role_column(r.value, r.name.title(), rolemap.columns_with_role(r)) for r in rolemap
        #             # ],
        #             class_="row g-3",
        #             id = this.ns("xxRoleMapWidget")  # Note: the actual Shiny input id we’ll set from JS is ns("RoleMap")
        #         )
        #     )


        # @reactive.calc
        # def role_mapXXX() -> RoleMap:
        #     """
        #     Reactive RoleMap derived from the drag-and-drop UI.
        #     """
        #     raw = input.RoleMap() or {}  # dict[str, list[str]] from JS
        #     # filter out "none" if you want unassigned to be "no entry" in RoleMap
        #     cleaned = {
        #         col: [r for r in roles if r != Role.NONE.value]
        #         for col, roles in raw.items()
        #     }
        #     return RoleMap.from_primitive(cleaned)

        # # Optionally expose on the card for other modules to use:
        # this.role_map = role_map

        # @reactive.effect
        # def _debug_role_map():
        #     if role_map().column_roles:
        #         print(f"[{this.name}] RoleMap updated: {role_map().to_primitive()}")

    this.server = server

    return this


if Module.running_under_tests():
    this = instance()
    app = Module.app(modules = {this.ns: this})
elif Module.running_directly(name =__name__):
    import pandas as pd
    this = instance()
    df = pd.read_csv("data/Ass2.csv")
    this._imports["data"].set(df)
    Module.run(modules = {this.ns: this})
