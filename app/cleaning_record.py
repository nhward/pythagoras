"""Descriptive records of materialised, non-pipeline data cleaning."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType

Shape = tuple[int, int]


def _validate_shape(name: str, shape: Shape) -> Shape:
    """Return a normalised two-dimensional, non-negative data shape."""
    if (
        not isinstance(shape, tuple)
        or len(shape) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in shape)
        or any(value < 0 for value in shape)
    ):
        raise ValueError(f"{name} must be a pair of non-negative integers")
    return shape


@dataclass(frozen=True, slots=True)
class CleaningRecord:
    """Describe one materialised, non-learning data-cleaning operation.

    The record is provenance rather than executable code. ``parameters`` stores
    the user-visible choices that describe the operation; the card remains
    responsible for performing it.
    """

    card: str
    operation: str
    parameters: Mapping[str, object] = field(default_factory=dict)
    input_shape: Shape = (0, 0)
    output_shape: Shape = (0, 0)

    def __post_init__(self) -> None:
        card = self.card.strip() if isinstance(self.card, str) else ""
        operation = self.operation.strip() if isinstance(self.operation, str) else ""
        if not card:
            raise ValueError("card must be a non-empty string")
        if not operation:
            raise ValueError("operation must be a non-empty string")
        if not isinstance(self.parameters, Mapping):
            raise TypeError("parameters must be a mapping")

        object.__setattr__(self, "card", card)
        object.__setattr__(self, "operation", operation)
        object.__setattr__(
            self,
            "parameters",
            MappingProxyType(deepcopy(dict(self.parameters))),
        )
        object.__setattr__(self, "input_shape", _validate_shape("input_shape", self.input_shape))
        object.__setattr__(self, "output_shape", _validate_shape("output_shape", self.output_shape))
