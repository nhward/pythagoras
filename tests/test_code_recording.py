"""Run-only source capture across card callbacks, transformers and workers."""
import asyncio
import inspect
import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from code_recording import collect_job_code, recordable, recorded_job, recording_context
from joblib import Parallel, delayed
from LogicalEncodingTransformer import LogicalEncodingTransformer
from module import Module
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted

pytestmark = pytest.mark.unit


@recordable
def _double(value):
    return value * 2


@recordable
def _calculate(value, double=False):
    return _double(value) if double else value + 1


@recordable
@lru_cache(maxsize=2)
def _cached(value):
    return value + 1


def test_only_executed_helpers_are_recorded_without_callback_plumbing():
    first, second = {}, {}
    _calculate(2, double=True)  # Script use has no card owner.
    calculate = recording_context(first, _calculate)
    assert first == second == {}
    assert calculate(2) == 3
    assert set(first) == {"_calculate"}
    assert recording_context(second, _calculate)(2, double=True) == 4
    assert set(second) == {"_calculate", "_double"}
    assert set(first) == {"_calculate"}


def test_nested_card_contexts_restore_the_caller_even_after_error():
    upstream, downstream = {}, {}

    @recording_context_decorator(upstream)
    def fail():
        _double(4)
        raise ValueError("expected")

    @recording_context_decorator(downstream)
    def run():
        with pytest.raises(ValueError, match="expected"):
            fail()
        return _calculate(3)

    assert run() == 4
    assert set(upstream) == {"_double"}
    assert set(downstream) == {"_calculate"}
    _double(3)
    assert set(downstream) == {"_calculate"}


def recording_context_decorator(registry):
    return lambda function: recording_context(registry, function)


@pytest.mark.asyncio
async def test_async_callbacks_and_threads_keep_concurrent_cards_separate():
    first, second = {}, {}

    async def run(double):
        await asyncio.sleep(0)
        return await asyncio.to_thread(_calculate, 3, double=double)

    left = recording_context(first, run)
    right = recording_context(second, run)
    assert inspect.iscoroutinefunction(left)
    assert await asyncio.gather(left(False), right(True)) == [4, 6]
    assert set(first) == {"_calculate"}
    assert set(second) == {"_calculate", "_double"}


@pytest.mark.parametrize("backend", ["threading", "loky"])
def test_worker_sources_return_to_the_correct_card(backend):
    registry = {}

    def run():
        return collect_job_code(Parallel(n_jobs=2, backend=backend)(
            delayed(recorded_job)(_calculate, value, double=True) for value in [2, 3]
        ))

    assert recording_context(registry, run)() == [4, 6]
    assert set(registry) == {"_calculate", "_double"}


def test_cache_controls_and_recording_on_cache_hit():
    _cached.cache_clear()
    assert _cached(3) == 4
    registry = {}
    assert recording_context(registry, _cached)(3) == 4
    assert set(registry) == {"_cached"}
    assert _cached.cache_info().hits == 1
    assert "@lru_cache" in Module.clean_code(registry["_cached"])


def test_transformer_recording_preserves_sklearn_pickle_and_replay():
    registry = {}
    frame = pd.DataFrame({"flag": pd.Series([True, False, None], dtype="boolean")})
    transformer = LogicalEncodingTransformer(["flag"])
    fitted = recording_context(registry, transformer.fit)(frame)
    assert type(fitted) is LogicalEncodingTransformer
    assert type(clone(fitted)) is LogicalEncodingTransformer
    expected = pickle.loads(pickle.dumps(fitted)).transform(frame)
    assert set(registry) == {"LogicalEncodingTransformer"}
    source = Module.clean_code(registry["LogicalEncodingTransformer"])
    assert "recordable" not in source
    namespace = {"pd": pd, "np": np, "BaseEstimator": BaseEstimator,
                     "TransformerMixin": TransformerMixin, "check_is_fitted": check_is_fitted}
    exec(source, namespace)  # noqa: S102
    replay = namespace["LogicalEncodingTransformer"](["flag"]).fit_transform(frame)
    pd.testing.assert_frame_equal(replay, expected)


def test_cleaning_keeps_guards_and_valid_suites_without_shiny_or_logging():
    source = '''@this.record_code
def calculate(data):
    # A comment mentioning req(data) stays untouched.
    ui.req(data is not None)
    if data.empty:
        this.log.debug(
            "No rows: café ☕"
        )
    return "req(data)"
'''
    cleaned = Module.clean_code(source)
    assert 'assert (data is not None)' in cleaned
    assert 'this.log' not in cleaned
    assert '# A comment mentioning req(data)' in cleaned
    assert 'return "req(data)"' in cleaned
    compile(cleaned, "<recorded>", "exec")


def test_all_recorded_card_definitions_parse_after_cleanup():
    # Check source syntax, including multiline decorators and nested classes,
    # across the actual card inventory, without importing/running all their UIs.
    import ast
    root = Path(__file__).resolve().parents[1] / "app"
    for path in [*root.joinpath("cards").glob("*.py"), *root.glob("*Transformer.py")]:
        source = path.read_text()
        tree = ast.parse(source)
        lines = source.splitlines(keepends=True)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if not any(ast.unparse(d) in {"recordable", "this.record_code", "record_code"}
                       for d in node.decorator_list):
                continue
            start = min(d.lineno for d in node.decorator_list) - 1
            import textwrap
            snippet = textwrap.dedent("".join(lines[start:node.end_lineno]))
            cleaned = Module.clean_code(snippet)
            compile(cleaned, str(path), "exec")
            assert "@this.record_code" not in cleaned


def test_bookmark_serialization_records_only_executed_storage_helpers(tmp_path):
    from cards.sys_bookmark import load_local_bookmark, save_local_bookmark

    registry = {}
    name = "demo--20260921T120000+0000.pythagoras.json"
    configuration = {"bookmark": {"filename": name}, "layout": []}
    path = recording_context(registry, save_local_bookmark)(configuration, directory=tmp_path)
    assert path.name == name
    assert "save_local_bookmark" in registry
    assert "load_local_bookmark" not in registry
    assert recording_context(registry, load_local_bookmark)(name, directory=tmp_path) == configuration
    assert "load_local_bookmark" in registry


def test_cleaned_worker_calculation_runs_without_recording_transport():
    source = """@this.record_code
def calculate(values):
    return collect_job_code(Parallel(n_jobs=1)(
        delayed(recorded_job)(_double, value) for value in values
    ))
"""
    cleaned = Module.clean_code(source)
    assert "recorded_job" not in cleaned
    assert "collect_job_code" not in cleaned
    namespace = {"Parallel": Parallel, "delayed": delayed, "_double": _double}
    exec(cleaned, namespace)  # noqa: S102
    assert namespace["calculate"]([2, 3]) == [4, 6]
