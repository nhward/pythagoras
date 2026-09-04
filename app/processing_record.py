"""Descriptive records of active and inactive data-processing steps."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal


@dataclass(frozen=True, slots=True)
class ProcessingRecord:
    """Describe whether a cleaning or learning operation was attempted."""

    stage: Literal["Cleaning", "Learning"]
    card: str
    operation: str
    attempted: bool
    parameters: Mapping[str, object] = field(default_factory=dict)
    input_shape: tuple[int, int] = (0, 0)
    output_shape: tuple[int, int] = (0, 0)
    method: str = ""
    variables: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.stage not in {"Cleaning", "Learning"}:
            raise ValueError("stage must be 'Cleaning' or 'Learning'")
        if not isinstance(self.card, str) or not self.card.strip():
            raise ValueError("card must be a non-empty string")
        if not isinstance(self.operation, str) or not self.operation.strip():
            raise ValueError("operation must be a non-empty string")
        if not isinstance(self.attempted, bool):
            raise TypeError("attempted must be boolean")
        if not isinstance(self.parameters, Mapping):
            raise TypeError("parameters must be a mapping")
        for name, shape in (
            ("input_shape", self.input_shape),
            ("output_shape", self.output_shape),
        ):
            if (
                not isinstance(shape, tuple)
                or len(shape) != 2
                or any(not isinstance(value, int) or value < 0 for value in shape)
            ):
                raise ValueError(f"{name} must be a pair of non-negative integers")

        object.__setattr__(self, "card", self.card.strip())
        object.__setattr__(self, "operation", self.operation.strip())
        object.__setattr__(
            self, "parameters", MappingProxyType(deepcopy(dict(self.parameters))),
        )
        object.__setattr__(self, "variables", tuple(map(str, self.variables)))
