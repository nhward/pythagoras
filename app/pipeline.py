"""Pipelines combining resampling with joint feature/target transformations."""

from feature_engine.pipeline import Pipeline as FeatureEnginePipeline
from imblearn.pipeline import (
    Pipeline as ImbalancedPipeline,
)
from imblearn.pipeline import (
    _fit_resample_one,
    _fit_transform_one,
    _print_elapsed_time,
)
from sklearn.base import clone
from sklearn.utils.validation import check_memory


def _fit_transform_xy(transformer, X, y, *, params, **logging):
    with _print_elapsed_time(logging["message_clsname"], logging["message"]):
        transformer.fit(X, y, **params.get("fit", {}))
        X, y = transformer.transform_x_y(
            X, y, **params.get("transform", {})
        )
    return X, y, transformer


class Pipeline(ImbalancedPipeline, FeatureEnginePipeline):
    """Combine imbalanced-learn samplers and Feature-engine row transformers.

    Intermediate ``transform_x_y`` steps keep targets aligned when rows are
    removed. Samplers run only during fitting. Constructor options, cloning,
    caching and prediction follow imbalanced-learn; ``transform_x_y`` and
    target-aware scoring follow Feature-engine.

    As with the parent pipelines, row-wise metadata such as sample weights
    must already match the rows consumed by the receiving estimator.
    """

    score = FeatureEnginePipeline.score

    def _fit(self, X, y=None, routed_params=None, raw_params=None):
        self.steps = list(self.steps)
        self._validate_steps()
        memory = check_memory(self.memory)
        transform_one = memory.cache(_fit_transform_one)
        transform_xy = memory.cache(_fit_transform_xy)
        resample_one = memory.cache(_fit_resample_one)

        for index, name, transformer in self._iter(
            with_final=False, filter_passthrough=False, filter_resample=False
        ):
            logging = {"message_clsname": "Pipeline", "message": self._log_message(index)}
            if transformer is None or transformer == "passthrough":
                with _print_elapsed_time("Pipeline", logging["message"]):
                    continue
            fitted = transformer if memory.location is None else clone(transformer)
            params = self._get_metadata_for_step(
                step_idx=index,
                step_params=routed_params[name],
                all_params=raw_params,
            )
            if hasattr(fitted, "transform_x_y"):
                X, y, fitted = transform_xy(fitted, X, y, params=params, **logging)
            elif hasattr(fitted, "fit_resample"):
                X, y, fitted = resample_one(fitted, X, y, params=params, **logging)
            else:
                X, fitted = transform_one(
                    fitted, X, y, weight=None, params=params, **logging
                )
            self.steps[index] = (name, fitted)
        return X, y
