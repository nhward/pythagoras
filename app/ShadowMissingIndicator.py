"""Feature-engine missing indicators with Pythagoras shadow names."""
from feature_engine.imputation import AddMissingIndicator


class ShadowMissingIndicator(AddMissingIndicator):
    """Preserve original columns and name appended indicators shadow__<source>."""

    def fit(self, X, y=None):
        super().fit(X, y)
        collisions = {f'shadow__{column}' for column in self.variables_} & set(X.columns)
        if collisions:
            raise ValueError(f'Shadow columns already exist: {sorted(collisions)}')
        return self

    def transform(self, X):
        result = super().transform(X)
        result.columns = self.get_feature_names_out()
        return result

    def get_feature_names_out(self, input_features=None):
        names = list(super().get_feature_names_out(input_features))
        if self.variables_:
            names[-len(self.variables_):] = [f'shadow__{column}' for column in self.variables_]
        return names
