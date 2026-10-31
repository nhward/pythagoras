"""A placeholder label remains in dtype metadata after an upstream transform."""
import runpy
from pathlib import Path
from PlaceholderTransformer import PlaceholderTransformer

base = runpy.run_path(str(Path(__file__).with_name('miss_manual.py')))
source, this = base['source'], base['this']
frame = source.frame.copy()
frame['LABEL'] = frame.LABEL.cat.add_categories(['--', 'unused'])
frame.loc[frame.LABEL.isna(), 'LABEL'] = '--'
source = base['proxy_data'](_df=frame, _roles=source.role_map, _name='Placeholder pipeline example')
step = PlaceholderTransformer([('LABEL', 'str', '--')])
source = source.with_pipeline_step(step, name='miss_placeholders', preview_frame=step.fit_transform(frame))
this._imports.set(source)
app = this.application()
