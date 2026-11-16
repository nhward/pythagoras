"""Restore paused completed trials into a newly constructed browser session."""
import json
import runpy
from pathlib import Path

scenario = runpy.run_path(str(Path(__file__).with_name('cand_train.py')))
import prediction_evidence as evidence_store
from pipeline_search import fresh, prepare, run

this, source = scenario['this'], scenario['source']
problem = prepare(source, method='grid', execution='sequential', workers=1)
paused = False
def publish(state):
    global paused
    paused = sum(t['state'] == 'complete' for t in state['trials']) >= 2
state = run(problem, fresh(problem), trials=1, publish=publish, pause=lambda: paused)
state['status'] = 'paused'
restoration = {'inputs': {'Method': 'grid', 'Execution': 'sequential', 'Workers': 1,
                         'Seed': 1729, 'Metric': 'neg_root_mean_squared_error', 'Trials': 0},
               'search': {'checkpoint': evidence_store.portable_checkpoint(state),
                          'source_signature': source._tuning_evidence_signature(),
                          'fingerprint_components': {'legacy': 'discard-on-save'}}}
this.restore_configuration_state(json.loads(json.dumps(restoration)))
app = this.application()
