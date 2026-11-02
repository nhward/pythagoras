import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parent.parent / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from cleaning_record import CleaningRecord


@pytest.mark.unit
def test_cleaning_record_captures_descriptive_provenance():
    record = CleaningRecord(
        card="var_modify",
        operation="Convert variable types",
        parameters={"AGE": "numeric", "DATE": "datetime:%Y-%m-%d"},
        input_shape=(1000, 12),
        output_shape=(1000, 12),
    )

    assert record.card == "var_modify"
    assert record.operation == "Convert variable types"
    assert dict(record.parameters) == {
        "AGE": "numeric",
        "DATE": "datetime:%Y-%m-%d",
    }
    assert record.input_shape == (1000, 12)
    assert record.output_shape == (1000, 12)


@pytest.mark.unit
def test_cleaning_record_is_immutable_and_copies_parameters():
    parameters = {"AGE": "numeric"}
    record = CleaningRecord("var_modify", "Convert variable types", parameters)
    parameters["AGE"] = "string"

    assert record.parameters["AGE"] == "numeric"
    with pytest.raises(TypeError):
        record.parameters["AGE"] = "string"
    with pytest.raises(FrozenInstanceError):
        record.operation = "Something else"


@pytest.mark.unit
@pytest.mark.parametrize("field", ["card", "operation"])
def test_cleaning_record_requires_descriptive_names(field):
    values = {"card": "var_modify", "operation": "Convert variable types"}
    values[field] = "  "

    with pytest.raises(ValueError, match=field):
        CleaningRecord(**values)


@pytest.mark.unit
@pytest.mark.parametrize("shape", [(1,), (1, -1), (1, 2, 3), [1, 2], (True, 2)])
def test_cleaning_record_rejects_invalid_shapes(shape):
    with pytest.raises(ValueError, match="input_shape"):
        CleaningRecord("var_modify", "Convert variable types", input_shape=shape)


@pytest.mark.parametrize('record_type', ['cleaning', 'processing'])
def test_records_are_hashable_and_pickleable_without_losing_readonly_parameters(record_type):
    import pickle
    from joblib import hash as job_hash
    from processing_record import ProcessingRecord
    if record_type == 'cleaning':
        record = CleaningRecord('example', 'Clean data', {'nested': {'columns': ['x']}}, (3, 2), (3, 1))
    else:
        record = ProcessingRecord('Learning', 'example', 'Learn transform', True,
            {'nested': {'columns': ['x']}}, (3, 2), (3, 1), 'Example', ('x',))
    restored = pickle.loads(pickle.dumps(record))
    assert restored == record
    assert job_hash(record) == job_hash(restored)
    with pytest.raises(TypeError):
        restored.parameters['new'] = True
    restored.parameters['nested']['columns'].append('y')
    assert record.parameters['nested']['columns'] == ['x']
