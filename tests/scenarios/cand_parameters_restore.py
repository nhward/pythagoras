"""Restore an earlier commitment alongside later, uncommitted draft edits."""
import json
import runpy
from copy import deepcopy
from pathlib import Path

scenario = runpy.run_path(str(Path(__file__).with_name('cand_parameters.py')))

from shiny import reactive, render, ui
from sklearn.ensemble import RandomForestRegressor
from tuning import compile_spec, discover_source

this, source = scenario['this'], scenario['source']
rows = discover_source(source)
row = next(r for r in rows if r['path'] == 'miss_map__observation_threshold')
definitions = {row['path']: dict(row['definition'], owner=row['owner'], active=True)}
saved = {'spec': compile_spec(rows, definitions), 'drafts': deepcopy(definitions)}
# A prior session's exact fingerprint must not prevent replay of valid choices.
saved['spec']['fingerprint'] = 'previous-session-fingerprint'
draft = deepcopy(definitions)
draft[row['path']]['active'] = False
this.restore_configuration_state(json.loads(json.dumps({
    'tuning': {'drafts': draft, 'committed': saved},
})))
original_server, original_footer = this.server, this._footer
this.footer = lambda: ui.TagList(original_footer(),
                               ui.output_text('SavedState'),
                               ui.input_action_button('ChangePipeline', label='Change pipeline'))


def server(input, output, session):
    exported = original_server(input, output, session)

    @output
    @render.text
    def SavedState():
        saved = this.configuration_state()['tuning']['committed']
        expected = compile_spec(discover_source(this.input_data()), saved['drafts'])
        return 'Saved commitment is current' if saved['spec'] == expected else 'Saved commitment is stale'

    @reactive.effect
    @reactive.event(input.ChangePipeline)
    def change_pipeline():
        this._imports.set(source.with_estimator(RandomForestRegressor(n_estimators=123),
                                               label='Random forest', recipe_id='test'))

    return exported


this.server = server
app = this.application()
