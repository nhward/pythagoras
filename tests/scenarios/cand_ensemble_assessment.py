"""An unchecked, busy ensemble must not block assessment of the passed-through model."""
import threading
from concurrent.futures import ThreadPoolExecutor

from cards import cand_ensemble
from cards.cand_assessment import instance as assessment_card
from ensemble_scenario import example_source
from fit_service import FitService
from module import Module
from shiny import App, reactive, render, req, ui

source = example_source()
entered = threading.Event()


def slow_evaluate(source, *args, cancelled, fit_service):
    width = fit_service.runtime.ensemble_width()
    active = 0
    lock = threading.Lock()

    def hold_slot():
        nonlocal active
        with fit_service.runtime.lease('ensemble', cancelled=cancelled):
            with lock:
                active += 1
                if active == width:
                    entered.set()
            cancelled.wait(60)

    with ThreadPoolExecutor(width) as pool:
        futures = [pool.submit(hold_slot) for _ in range(width)]
        for future in futures:
            future.result()
    raise InterruptedError('Background ensemble cancelled')


cand_ensemble.evaluate_ensemble = slow_evaluate
app_ui = ui.page_fillable(
    ui.head_content(
        *[ui.include_js(p, method='inline') for p in dict.fromkeys(Module.script_list)],
        *[ui.include_css(p, method='inline') for p in dict.fromkeys(Module.css_list)]),
    ui.output_ui('Workflow'), ui.output_text('Background'))


def server(input, output, session):
    Module.ModSession = session
    service = FitService(cpus=3)
    session._pythagoras_fit_service = service
    session.on_destroy(service.close)
    ensemble, assessment = cand_ensemble.instance(), assessment_card()
    ensemble._imports.set(source)
    exported = ensemble.call_server(input, output, session)

    @reactive.calc
    def model_for_assessment():
        if not entered.is_set():
            reactive.invalidate_later(.05)
            req(False)
        return exported()

    assessment.call_server(input, output, session, upstream=reactive.Value(model_for_assessment))
    ensemble.resume()
    assessment.resume()

    @output
    @render.ui
    def Workflow():
        return ui.div(ensemble.call_ui(), assessment.call_ui(), class_='cards-grid')

    @output
    @render.text
    def Background():
        reactive.invalidate_later(.1)
        return 'ensemble running' if entered.is_set() else 'waiting'


app = App(app_ui, server, static_assets=Module.ROOT / 'www')
