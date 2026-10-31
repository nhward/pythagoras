import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'app'))

import pandas as pd
from cards.var_nominal import instance
from proxy_data import proxy_data
from roles import Role, RoleMap
from shiny import render, ui

this = instance()
frame = pd.DataFrame({
    'Day': pd.Categorical(['Monday'] * 30 + ['Mon.'] * 3 + ['Mondya'] + ['Tuesday'] * 20),
    'Colour': pd.Categorical(['Blue'] * 40 + ['Blu'] * 3 + ['Red'] * 11),
})
frame['Outcome'] = frame.Day.copy()
frame['Stratum'] = frame.Day.copy()
frame['Dual'] = frame.Day.copy()
roles = RoleMap()
for column, role in [('Day', Role.PREDICTOR), ('Colour', Role.PREDICTOR),
                     ('Outcome', Role.TARGET), ('Stratum', Role.STRATIFIER)]:
    roles.set_roles(column, [role])
roles.set_roles('Dual', [Role.PREDICTOR, Role.TARGET])
this._imports.set(proxy_data(_df=frame, _roles=roles))

original_server, original_footer = this.server, this._footer
this.footer = lambda: ui.TagList(original_footer(), ui.output_text('Preview'), ui.output_text('PipelineProbe'), ui.output_text('TargetProbe'))
def server(input, output, session):
    result = original_server(input, output, session)
    @output
    @render.text
    def Preview():
        return 'Monday count: ' + str(int((result().frame['Day'] == 'Monday').sum()))
    @output
    @render.text
    def PipelineProbe():
        data = result()
        if not data.has_pipeline:
            return 'steps=none'
        replay = data.pipeline_for_training().fit_transform(data.clean_frame)
        return 'steps=' + ','.join(data.pipeline_steps) + '; replay=' + str(replay.equals(data.frame))
    @output
    @render.text
    def TargetProbe():
        data = result()
        return f"Outcome Mon.: {int(data.frame.Outcome.eq('Mon.').sum())}; Dual Mon.: {int(data.frame.Dual.eq('Mon.').sum())}"
    return result
this.server = server
app = this.application()
