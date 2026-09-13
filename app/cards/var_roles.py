from __future__ import annotations

import os
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    # Ensure local modules and packages are resolved from the app directory.
    os.chdir(ROOT)
    root_string = str(ROOT)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)

import pandas as pd  # needed for test / solo modes
from card import Card
from faicons import icon_svg as icon
from module import Module
from proxy_data import proxy_data as Pxy
from roles import Role, RoleMap
from shiny import reactive, render, req, ui


def _retained_role_state(
    columns,
    role_map: RoleMap,
) -> tuple[list[object], RoleMap, list[object]]:
    """Return the columns and roles exported after applying the None role."""
    retained: list[object] = []
    removed: list[object] = []
    retained_roles = RoleMap()
    for column in columns:
        roles = role_map.roles_for(column)
        if Role.NONE in roles:
            removed.append(column)
            continue
        retained.append(column)
        retained_roles.set_roles(column, roles)
    return retained, retained_roles, removed


def _role_map_covers_columns(value, columns) -> bool:
    """Return whether a browser role map assigns every current column once."""
    if not isinstance(value, Mapping):
        return False
    current = {str(column) for column in columns}
    counts = {column: 0 for column in current}
    for role_name, assigned in value.items():
        try:
            Role.from_value(role_name)
        except (TypeError, ValueError):
            return False
        if isinstance(assigned, (str, bytes)) or not isinstance(assigned, Iterable):
            return False
        for raw_column in assigned:
            column = str(raw_column)
            if column not in counts:
                return False
            counts[column] += 1
    return all(count == 1 for count in counts.values())


def _reconcile_restored_role_map(
    saved,
    columns,
    fallback: RoleMap | Mapping[str, Iterable[str]],
) -> tuple[dict[str, list[str]], list[str]]:
    """Make saved sortable state structurally safe for the current dataset."""
    column_order = [str(column) for column in columns]
    current = set(column_order)
    role_order = [role.value for role in Role]
    restored = {role: [] for role in role_order}
    warnings: list[str] = []

    if isinstance(fallback, RoleMap):
        fallback = fallback.to_primitive()
    fallback_by_column: dict[str, list[str]] = {}
    if isinstance(fallback, Mapping):
        for role_name in role_order:
            for raw_column in fallback.get(role_name, []):
                column = str(raw_column)
                if column in current:
                    fallback_by_column.setdefault(column, []).append(role_name)

    saved_by_role = {role: [] for role in role_order}
    stale: set[str] = set()
    malformed_roles: list[str] = []
    if not isinstance(saved, Mapping):
        warnings.append("The saved role map is malformed; upstream roles were used.")
        saved = {}
    for raw_role, assigned in saved.items():
        try:
            role = Role.from_value(raw_role).value
        except (TypeError, ValueError):
            malformed_roles.append(str(raw_role))
            continue
        if isinstance(assigned, (str, bytes)) or not isinstance(assigned, Iterable):
            warnings.append(
                f"Saved role {role!r} did not contain a variable list and was ignored."
            )
            continue
        for raw_column in assigned:
            column = str(raw_column)
            if column not in current:
                stale.add(column)
                continue
            saved_by_role[role].append(column)

    if malformed_roles:
        warnings.append(
            "Unknown saved role(s) were ignored: "
            + ", ".join(sorted(malformed_roles))
            + "."
        )
    if stale:
        warnings.append(
            "Variables no longer present were removed from the saved roles: "
            + ", ".join(sorted(stale))
            + "."
        )

    occurrences: dict[str, list[str]] = {}
    for role in role_order:
        for column in saved_by_role[role]:
            roles = occurrences.setdefault(column, [])
            if role not in roles:
                roles.append(role)
    duplicated = {
        column for column, roles in occurrences.items() if len(roles) > 1
    }
    if duplicated:
        warnings.append(
            "Multiply assigned saved variables reverted to their upstream roles: "
            + ", ".join(sorted(duplicated))
            + "."
        )

    assigned: set[str] = set()
    for role in role_order:
        for column in saved_by_role[role]:
            if column in duplicated or column in assigned:
                continue
            restored[role].append(column)
            assigned.add(column)

    added = [column for column in column_order if column not in occurrences]
    if added and isinstance(saved, Mapping) and saved:
        warnings.append(
            "New variables used their upstream roles: " + ", ".join(added) + "."
        )

    for column in column_order:
        if column in assigned:
            continue
        fallback_roles = fallback_by_column.get(column, [])
        if len(fallback_roles) > 1:
            warnings.append(
                f"Variable {column!r} had multiple upstream roles; "
                f"{fallback_roles[0]!r} was used."
            )
        role = fallback_roles[0] if fallback_roles else Role.PREDICTOR.value
        restored[role].append(column)
        assigned.add(column)

    return restored, warnings


def instance():
    """
    Creates an instance of Card configured as "roleAssign".
    """
    this = Card(file=__file__, mutable=True) # "mutable" means it can change the pxd - probably with a commit button
    this.register_configuration_input("role_map")
    this.long_name = "Role Assignment"
    this.description = "This card enables variables to be assigned to roles and removes variables assigned the None role from downstream data."

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
                label = "Maximum cardinality of low-cardinality roles",
                min = 3,
                max = 50,
                value = 4,
                ticks = True,
                guide = this,
                text = "Sets the maximum number of distinct observed values permitted when validating Sensitive, Stratifier, and Treatment roles. Raise it only when a larger grouping remains analytically meaningful.",
                position = "left"),
            ui.input_slider(
                id = "MaxObs", 
                label = "Maximum observations to analyze",
                min = 3,
                max = 7,
                value = 4,
                ticks = True,
                pre = "10^",
                guide = this,
                text = "Sets a logarithmic cap of 10^n observations used to assess cardinality and missingness during validation. It does not remove observations from committed data.",
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
            text = "Drag each variable into one role container and scroll horizontally to reach every role. Full-screen mode provides more room. Assignments remain provisional until validation succeeds and Commit Assignments is clicked. Variables committed to None are omitted from this card's downstream output.",
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
                text = "This button commits the role assignments. Variables assigned to None are removed from the data passed downstream, but remain available here for later reassignment. The button bounces momentarily when it is ready to be clicked.",
                position = "top"
            ),
            ui.output_ui(
                id = "Check",
                guide = this, 
                title = "Card status",
                text = "Reports the most immediate role-validation problem, or confirms whether valid assignments are ready to commit or already applied.",
                position = "top")
        )

    this.footer = footer

    
    def server(input, output, session):

        OutputData = reactive.Value()
        restored_role_map = this.restored_configuration_input("role_map")
        restored_roles_consumed = False
        protected_columns: tuple[str, ...] | None = None
        protected_role_map: dict[str, list[str]] | None = None

        @this.suspendable(calc = True)
        def incomingproxy_data():
            return this.input_data()
 
        @this.settle(seconds=2)
        @this.suspendable(calc = True)
        def MaxObs():
            return 10**input.MaxObs()

        @this.suspendable(calc = True)
        def PreparedData():
            samp = incomingproxy_data().sample(n=MaxObs(), mode="random", keep_geometry=True)
            return samp

        @this.suspendable()
        def PxdChange():
            OutputData.set(incomingproxy_data())

        @this.suspendable(triggers = [PreparedData])
        async def PopulateRoles():
            nonlocal restored_roles_consumed, protected_columns, protected_role_map
            if not this.has_input_data():
                rm = RoleMap().to_primitive()
                await session.send_custom_message("PopulateRoles", {"card": session.ns("Card"), "role_map": rm})
                return

            pxd = PreparedData()
            columns = tuple(str(column) for column in pxd.frame.columns)
            if not restored_roles_consumed and restored_role_map is not None:
                role_map, warnings = _reconcile_restored_role_map(
                    restored_role_map,
                    columns,
                    pxd.role_map,
                )
                for warning in warnings:
                    this.log.warning("Role bookmark: %s", warning)
                restored_roles_consumed = True
                protected_columns = columns
                protected_role_map = role_map
                await session.send_custom_message(
                    "PopulateRoles",
                    {"card": session.ns("Card"), "role_map": role_map},
                )
                return

            try:
                current_role_map = input.role_map()
            except Exception:  # noqa: BLE001
                current_role_map = None
            if _role_map_covers_columns(current_role_map, columns):
                return
            if protected_columns == columns and protected_role_map is not None:
                await session.send_custom_message(
                    "PopulateRoles",
                    {"card": session.ns("Card"), "role_map": protected_role_map},
                )
                return

            protected_columns = None
            protected_role_map = None
            await session.send_custom_message(
                "PopulateRoles",
                {
                    "card": session.ns("Card"),
                    "role_map": pxd.role_map.to_primitive(),
                },
            )

        @output
        @render.table
        @this.record_code
        def Assignments():
            orm = OutputData.get().role_map
            return orm.roles_to_frame()

        @this.suspendable(calc = True)
        @this.record_code
        def ValidateMap():
            this.input_data()
            req(input.role_map())
            this.log.debug("☑️ Validating changes")
            # Convert the json to the RoleMap class
            rm = RoleMap.from_primitive(input.role_map())
            pxd = PreparedData()
            msgs = pxd.validate(role_map = rm, separator = input.Separator(), low_cardinality = input.CardinalityThreshold()) 
            return msgs


        #### Committed  event ----
        @this.suspendable(calc = True)
        def Committed():
            req(input.role_map())
            data = incomingproxy_data()
            role_map = RoleMap.from_primitive(input.role_map())
            retained, retained_roles, removed = _retained_role_state(
                data.frame.columns,
                role_map,
            )
            changes = []
            for variable in data.frame.columns:
                original_roles = sorted(
                    role.value for role in data.role_map.roles_for(variable)
                )
                new_roles = sorted(
                    role.value for role in role_map.roles_for(variable)
                )
                is_removed = variable in removed
                if original_roles != new_roles or is_removed:
                    change = {
                        "variable": str(variable),
                        "original_roles": original_roles,
                        "new_roles": new_roles,
                    }
                    if is_removed:
                        change["removed_downstream"] = True
                    changes.append(change)

            if changes:
                return data.with_cleaned_data(
                    data.frame.loc[:, retained],
                    card="var_roles",
                    operation="Assign variable roles",
                    parameters={
                        "changes": changes,
                        "removed_variables": [str(variable) for variable in removed],
                    },
                    role_map=retained_roles,
                )
            return data.with_inactive_step(
                stage="Cleaning",
                card="var_roles",
                operation="Assign variable roles",
                parameters={"changes": changes},
            )
            

        #### Commit event ----
        @this.suspendable(triggers = [input.Commit])
        def CommitEvent():
            OutputData.set(Committed())


        @output
        @render.ui
        async def Check():
            messages = ValidateMap()
            ok = len(messages) == 0
            ui.update_action_button(id = "Commit", disabled = not ok)
            if ok:
                desired_roles = RoleMap.from_primitive(input.role_map())
                desired_columns, desired_output_roles, _ = _retained_role_state(
                    incomingproxy_data().frame.columns,
                    desired_roles,
                )
                if (
                    OutputData.is_set()
                    and list(OutputData.get().frame.columns) == desired_columns
                    and OutputData.get().role_map == desired_output_roles
                ):
                    return ui.span("Assignments applied", class_ = "text-success")
                else:
                    await session.send_custom_message("animate", {"id" : session.ns("Commit"), "animation" : "bounce", "delay" : 500})
                    return ui.span("Assignments are ready to commit", class_ = "text-primary")
            else:
                i = len(messages)
                return ui.span(i,": ", messages[i-1], class_ = "text-danger")

        return OutputData

    this.server = server

    return this


if Module.running_directly(name =__name__):
    this = instance()
    df = pd.read_csv( Card.ROOT / "data" / "Ass2.csv")
    pxd = Pxy(_df = df, _name = "Ass2")
    this._imports.set(pxd)
    this.run()
