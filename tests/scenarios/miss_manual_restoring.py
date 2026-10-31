"""Restore reviewed healthcare rules and a draft, before mounting the UI."""
import json
import runpy
from pathlib import Path

# Reuse exactly the pipeline-bearing input and probe server from the main scenario.
base = runpy.run_path(str(Path(__file__).with_name('miss_manual.py')))
m, source = base['m'], base['source']
this = base['this']
proposal = m.propose_tree(source, 'HEALTHCARE_COST', 2)
proposal['source'] = m.source_fingerprint(source)
proposal['extra_level'] = ''
leaf = next(n for n in proposal['nodes'] if n['leaf'] and n['missing'])
rule = {'column': 'HEALTHCARE_COST', 'kind': 'integer', 'conditions': leaf['conditions'], 'value': 0, 'extra_level': ''}
this.restore_configuration_state({'inputs': {'MaxTreeDepth': 2, 'ExtraLevel': ''}, 'manual': {
    'fingerprint': proposal['source'], 'rules': [rule],
    'drafts': {f"{m.proposal_key(proposal)}_{leaf['id']}": {'enabled': True, 'value': 0}},
}})
app = this.application()
