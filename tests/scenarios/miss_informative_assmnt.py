"""Assmnt's restored informative-missingness state."""
from pathlib import Path
import runpy
import pandas as pd

scenario = runpy.run_path(str(Path(__file__).with_name("miss_informative.py")))
root = Path(__file__).resolve().parents[2]
# Snapshot the relevant settings so this regression does not need a local bookmark.
role_map = {'geometry': [],
 'identifier': ['CODE'],
 'none': [],
 'partition': [],
 'predictor': ['POLITICS',
               'POPULATION',
               'AGE25_PROPTN',
               'AGE_MEDIAN',
               'AGE50_PROPTN',
               'POP_DENSITY',
               'GDP',
               'INFANT_MORT',
               'DOCS',
               'VAX_RATE',
               'HEALTHCARE_COST',
               'GROCERIES'],
 'sensitive': ['OBS_TYPE'],
 'sequence': [],
 'stratifier': ['HEALTHCARE_BASIS'],
 'target': ['DEATH_RATE'],
 'treatment': [],
 'weighting': []}
restored_state = {'inputs': {'CVFolds': 5,
            'InformativeOnly': True,
            'MaxObs': 4,
            'MaxVariables': 12,
            'MinBalancedAccuracy': 0.55,
            'MinMissProp': 0.05,
            'Shadow': ['POP_DENSITY'],
            'UseWeights': True}}
frame = pd.read_csv(root / "app/data/Assmnt.csv")
roles = scenario["RoleMap"].from_primitive(role_map)
this = scenario["this"]
settle = this.settle
this.settle = lambda seconds=2, **kwargs: settle(seconds=seconds, bypass_during_tests=False)
this._imports.set(scenario["proxy_data"](_df=frame, _roles=roles, _name="Assmnt"))
this.restore_configuration_state(restored_state)
app = this.application()
