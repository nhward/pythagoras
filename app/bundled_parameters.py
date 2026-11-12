"""Unfitted method-specific configuration owned by composite transformers."""
from typing import ClassVar

from code_recording import recordable
from sklearn.base import BaseEstimator


@recordable
class ParameterGroup(BaseEstimator):
    """Cloneable configuration for algorithms implemented by a wrapper itself."""

    def __init__(self, **parameters):
        self.__dict__.update(parameters)

    def get_params(self, deep=True):
        return dict(self.__dict__)

    def set_params(self, **parameters):
        unknown = parameters.keys() - self.__dict__.keys()
        if unknown:
            raise ValueError(f'Unknown grouped parameters: {sorted(unknown)}')
        self.__dict__.update(parameters)
        return self


@recordable
class BundledParametersMixin:
    """Declare nested ownership and activation without search-specific logic.

    Each group declares a selector, its applicable values, and a mapping from
    legacy flat names to child parameter names. Constructors keep legacy names
    for saved configurations; only nested names are offered for new searches.
    """

    parameter_groups: ClassVar[dict[str, dict[str, object]]] = {}
    parameter_aliases: ClassVar[dict[str, str]] = {}

    def _init_parameter_groups(self, **groups):
        for name, definition in self.parameter_groups.items():
            child = groups.get(name)
            if child is None:
                values = {**definition.get('defaults', {}),
                          **{nested: getattr(self, old) for old, nested in definition['parameters'].items()}}
                child = definition.get('factory', ParameterGroup)(**values)
            setattr(self, name, child)

    def parameter_value(self, name):
        name = self.parameter_aliases.get(name, name)
        for group, definition in self.parameter_groups.items():
            if name in definition['parameters']:
                return getattr(self, group).get_params()[definition['parameters'][name]]
        return getattr(self, name)

    def set_params(self, **parameters):
        # Translate old saved paths while giving explicit nested paths priority.
        parameters = dict(parameters)
        for old, new in self.parameter_aliases.items():
            if old in parameters:
                parameters.setdefault(new, parameters.pop(old))
        for group, definition in self.parameter_groups.items():
            for old, nested in definition['parameters'].items():
                if old in parameters:
                    parameters.setdefault(f'{group}__{nested}', parameters[old])
        return super().set_params(**parameters)
