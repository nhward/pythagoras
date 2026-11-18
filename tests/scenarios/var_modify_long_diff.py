"""Variable-modification card with a long committed structural report."""
import runpy
from pathlib import Path

import pandas as pd

scenario = runpy.run_path(str(Path(__file__).with_name('var_modify.py')))
this = scenario['this']
columns = [f'variable_{i:03}' for i in range(80)]
changes = [{'source': name, 'source_type': 'integer', 'name': 'renamed_' + name,
            'data_type': 'integer', 'order': []} for name in columns]
this.restore_configuration_state({'modifications': {
    'proposed': changes, 'committed': changes, 'source_columns': columns,
    'selected_source': columns[0]}})
this._imports.set(scenario['proxy_data'](_df=pd.DataFrame({name: [1, 2, 3] for name in columns}), _name='Long report'))
app = this.application()
