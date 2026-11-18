"""Prepare an experimental package copy; never edit the installed dependency."""
import argparse
import json
import re
import shutil
from pathlib import Path

import shinywidgets

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('variant', choices=['retain-model', 'model-close-only', 'registry-guard', 'instrumented'])
args = parser.parse_args()
source = Path(shinywidgets.__file__).parent
root = Path('/tmp') / f'pythagoras-widget-cleanup-{args.variant}'
package = root / 'shinywidgets'
if source.resolve() == package.resolve():
    parser.error('Run this script with the original installed package, without the overlay on PYTHONPATH.')
shutil.copytree(source, package, dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__'))

if args.variant == 'retain-model':
    path = package / 'static/output.js'
    text = path.read_text()
    assert text.count('delete v.model;') == 1
    path.write_text(text.replace('delete v.model;', '/* Retain model reference through view removal. */'))

if args.variant in {'model-close-only', 'instrumented'}:
    path = package / 'static/output.js'
    modified = []
    def patch(match):
        value = json.loads(match.group(1))
        if 'Shiny.addCustomMessageHandler("shinywidgets_comm_close"' not in value:
            return match.group(0)
        start = value.index('        // Some widget views need explicit teardown')
        end = value.index('        // View removal updates _view_count.', start)
        modified.append(True)
        return 'eval(' + json.dumps(value[:start] + value[end:]) + ')'
    text = re.sub(r'eval\(("(?:[^"\\]|\\.)*")\)', patch, path.read_text())
    assert len(modified) == 1
    path.write_text(text)

if args.variant in {'registry-guard', 'instrumented'}:
    path = package / 'static/libembed-amd.js'
    text = path.read_text()
    token = 'delete t.model.views[r]'
    assert text.count(token) == 1
    replacement = ('t.model&&t.model.views&&delete t.model.views[r]' if args.variant == 'registry-guard' else
        'console.info("WIDGET_REMOVAL "+JSON.stringify({hasModel:!!t.model,hasViews:!!(t.model&&t.model.views)})),delete t.model.views[r]')
    path.write_text(text.replace(token, replacement))
print(root)
