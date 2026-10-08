"""Informative missingness with an invalid assigned weighting."""
from pathlib import Path
import runpy

scenario = runpy.run_path(str(Path(__file__).with_name('miss_informative.py')))
frame, roles, this = (scenario[name] for name in ('frame', 'roles', 'this'))
frame['importance'] = -1.
roles.set_roles('importance', [scenario['Role'].WEIGHTING])
this._imports.set(scenario['proxy_data'](_df=frame, _roles=roles, _name='Invalid weights'))
app = scenario['app']
