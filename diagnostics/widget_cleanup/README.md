# Plotly widget cleanup reproduction

This app imports Shiny, shinywidgets and Plotly only. It does not load Pythagoras
cards, data, sections or preprocessing.

## Findings

Tested in WebKit with Shiny 1.9.0, shinywidgets 0.8.2, ipywidgets 8.1.9,
anywidget 0.11.0 and Plotly 7.1.0.

| Experiment | Update | Replace | Remove output | Explicit model close |
|---|---|---|---|---|
| Installed packages | OK | OK | OK | Exact reported error |
| Retain `view.model` in shinywidgets cleanup | — | — | — | Same error |
| Let `model.close()` perform view removal without manual teardown | OK | OK | OK | Same error |
| Guard the bundled manager's removal callback | OK | OK | OK | OK |

The instrumented model-close-only experiment logs:

```json
{"hasModel": true, "hasViews": false}
```

The view's model still exists. Its `views` registry has already been deleted by
`WidgetModel.close()` when the asynchronous view-removal callback runs. The
callback then executes `delete view.model.views[viewId]` and throws.

This corrects the initial suspicion that shinywidgets' explicit `delete v.model`
was the sole cause: removing that statement did not resolve this reproduction.

The successful experiment changes exactly one expression in the bundled
`libembed-amd.js`:

```javascript
// Original
 delete t.model.views[r]
// Experimental guard: the registry can already be gone during model closure.
 t.model && t.model.views && delete t.model.views[r]
```

The error is caught and written to the browser console; it is not an uncaught
page exception. Tests listening only for `pageerror` will miss it.

## Reproduce with installed packages

From the repository root:

```sh
.venv/bin/shiny run --launch-browser diagnostics/widget_cleanup/app.py
```

Click **Close model** and inspect the browser console. The other buttons isolate
ordinary mutation, widget replacement and output removal.

Automated reproduction (the Close case intentionally fails on affected packages):

```sh
.venv/bin/python -m pytest diagnostics/widget_cleanup/probe.py -s
```

The diagnostic probe is outside normal test discovery. Each case uses a fresh
app/session and records console errors, page errors, package versions, model IDs,
server import paths and the served JavaScript marker in
`/tmp/pythagoras-widget-cleanup-evidence.jsonl`.

## Controlled comparison without modifying installed packages

```sh
.venv/bin/python diagnostics/widget_cleanup/prepare_overlay.py registry-guard
PYTHONPATH=/tmp/pythagoras-widget-cleanup-registry-guard:app \
WIDGET_CLEANUP_VARIANT=registry-guard \
.venv/bin/python -m pytest diagnostics/widget_cleanup/probe.py -s
```

All four cases passed with this temporary package copy. The preparation script
also supports `retain-model`, `model-close-only` and `instrumented`; each uses a
separate `/tmp/pythagoras-widget-cleanup-VARIANT` directory. The instrumented
variant retains the failure and logs model/registry availability immediately
before the failing callback expression.

These overlays are investigation artifacts, not a production dependency fix.
The installed environment is unchanged. An upstream widget-manager correction
or a maintained dependency patch still needs to be integrated and tested in the
full application.
