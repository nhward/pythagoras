"""Original-variable shadow selection with native categorical LightGBM trees.

Boruta uses the pinned BorutaPy testing loop. GrootCV follows ARFS's repeated
fold, normalized mean-absolute-SHAP / shadow-cutoff procedure, with fold-local
category dictionaries, seeded shadows and LightGBM's native TreeSHAP output.
"""
from code_recording import recordable


@recordable
def encode_selection_frame(frame, reference=None):
    """One numeric column per source variable; codes remain explicitly nominal."""
    import numpy as np
    import pandas as pd
    from pandas.api import types as t

    encoded, categories = {}, {}
    for column in frame:
        s = frame[column]
        if t.is_complex_dtype(s.dtype) or t.is_datetime64_any_dtype(s.dtype) or t.is_timedelta64_dtype(s.dtype):
            raise ValueError(f"Unsupported selection type for {column!r}: {s.dtype}")
        if t.is_numeric_dtype(s.dtype) and not t.is_bool_dtype(s.dtype):
            values = s.to_numpy(dtype=float, na_value=np.nan)
            if np.isinf(values).any():
                raise ValueError(f"Infinite predictor values in {column!r}; resolve them before selection")
            encoded[column] = values
        else:
            if not (t.is_bool_dtype(s.dtype) or isinstance(s.dtype, pd.CategoricalDtype)
                    or t.is_string_dtype(s.dtype) or t.is_object_dtype(s.dtype)):
                raise ValueError(f"Unsupported selection type for {column!r}: {s.dtype}")
            observed = s.dropna()
            if not observed.map(lambda x: isinstance(x, (str, bool, int, float, np.number))).all():
                raise ValueError(f"Unsupported nested values in {column!r}")
            # Preserve scalar identity; never conflate numeric 1 with string '1'.
            vocabulary = list(pd.unique(observed)) if reference is None else reference[column]
            categories[column] = vocabulary
            codes = pd.Categorical(s.where(s.isin(vocabulary)), categories=vocabulary).codes.astype(float)
            codes[codes < 0] = np.nan
            encoded[column] = codes
    return pd.DataFrame(encoded, index=frame.index), categories


@recordable
def fit_boruta(frame, y, *, weights, task, trees, depth, iterations, alpha,
               percentile, two_step, seed, jobs, progress=None):
    import numpy as np
    from boruta import BorutaPy
    from lightgbm import LGBMClassifier, LGBMRegressor
    from sklearn.utils.validation import check_X_y
    from feature_selection import encode_selection_frame

    class NativeBoruta(BorutaPy):
        """Adapt input validation and keep categorical masks aligned with shadows."""
        def _check_params(self, X, y):
            # Parent checks alpha/percentile; our finite surrogate only bypasses
            # its rejection of predictor NaNs, never outcome validation.
            check_X_y(X, y, ensure_all_finite='allow-nan')
            super()._check_params(np.nan_to_num(X), y)

        def _add_shadows_get_imps(self, X, y, decisions):
            active = np.flatnonzero(decisions >= 0)
            shadow_sources = active.copy()
            while len(shadow_sources) < 5:
                shadow_sources = np.tile(shadow_sources, 2)
            real = X[:, active]
            shadows = X[:, shadow_sources].copy()
            for j in range(shadows.shape[1]):
                self.random_state.shuffle(shadows[:, j])
            mask = [i for i, original in enumerate(np.r_[active, shadow_sources]) if original in cat_indices]
            self.estimator.fit(np.column_stack([real, shadows]), y,
                               sample_weight=weights, categorical_feature=mask)
            importance = self.estimator.feature_importances_.astype(float)
            # Compare normalized importances, preserving the testing decisions.
            total = importance.sum()
            if total:
                importance /= total
            values = np.full(X.shape[1], np.nan)
            values[active] = importance[:len(active)]
            self.real_history.append(values.copy())
            self.shadow_history.append(importance[len(active):])
            if progress:
                progress(f"Boruta iteration {len(self.real_history)} / {iterations - 1}")
            return values, importance[len(active):]

        def _do_tests(self, decisions, hits, iteration):
            result = super()._do_tests(decisions, hits, iteration)
            self.decisions = result.copy()
            self.hits = hits.copy()
            return result

    encoded, categories = encode_selection_frame(frame)
    cat_indices = {i for i, c in enumerate(frame) if c in categories}
    model_type = LGBMClassifier if task == 'classification' else LGBMRegressor
    forest = model_type(boosting_type='rf', n_estimators=trees, max_depth=depth,
        num_leaves=min(2**depth, 63), subsample=.7, subsample_freq=1,
        colsample_bytree=.7, min_child_samples=5, importance_type='gain',
        n_jobs=jobs, verbosity=-1, reg_alpha=0., reg_lambda=0.)
    selector = NativeBoruta(forest, n_estimators=trees, max_iter=iterations,
        perc=percentile, alpha=alpha, two_step=two_step, random_state=seed)
    selector.real_history, selector.shadow_history = [], []
    selector.fit(encoded.to_numpy(), y)
    return dict(decisions=selector.decisions, hits=selector.hits,
        ranking=selector.ranking_, real=np.asarray(selector.real_history),
        shadows=selector.shadow_history,
        thresholds=np.asarray([np.percentile(s, percentile) for s in selector.shadow_history]),
        categories=categories, engine='BorutaPy 0.4.3 + LightGBM random forest (native categorical adapter)')


@recordable
def fit_groot(frame, y, *, weights, task, trees, depth, folds, repeats, cutoff,
              seed, jobs, progress=None, splits=None):
    """GrootCV adaptation; folds and categorical vocabularies are training-local.

    As in ARFS 3.0.0, normalize each fold's SHAP vector and compare mean real
    importance against the mean of per-fold shadow maxima (including the
    maximum shadow mean) divided by cutoff. Use training-fold TreeSHAP values.
    No ARFS global monkeypatches or external SHAP runtime are required.
    """
    import numpy as np
    import lightgbm as lgb
    from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
    from feature_selection import encode_selection_frame

    if splits is None:
        splitter_type = RepeatedStratifiedKFold if task == 'classification' else RepeatedKFold
        splitter = splitter_type(n_splits=folds, n_repeats=repeats, random_state=seed)
        splits = list(splitter.split(frame, y))
    rng = np.random.RandomState(seed)
    history, shadow_history = [], []
    for index, (train, valid) in enumerate(splits):
        if not len(train) or not len(valid) or np.intersect1d(train, valid).size:
            raise ValueError('Inner resampling has empty or overlapping subsets')
        if task == 'classification' and len(np.unique(y[train])) != len(np.unique(y)):
            raise ValueError(f'Inner split {index + 1} training is missing an outcome class')
        xtr, cats = encode_selection_frame(frame.iloc[train])
        xva, _ = encode_selection_frame(frame.iloc[valid], cats)
        a, b = xtr.to_numpy(), xva.to_numpy()
        # Generate shadows separately inside each fold partition.
        sa, sb = a.copy(), b.copy()
        for j in range(a.shape[1]):
            rng.shuffle(sa[:, j]); rng.shuffle(sb[:, j])
        mask = [i for i, c in enumerate(frame) if c in cats]
        mask += [i + frame.shape[1] for i in mask]
        params = dict(objective='regression' if task == 'regression' else
                      ('binary' if len(np.unique(y)) == 2 else 'multiclass'),
                      verbosity=-1, num_threads=jobs, max_depth=depth,
                      num_leaves=min(2**depth, 63), min_data_in_leaf=5,
                      seed=seed + index, learning_rate=.05, feature_pre_filter=False)
        if task == 'classification' and len(np.unique(y)) > 2:
            params['num_class'] = len(np.unique(y))
        dtrain = lgb.Dataset(np.c_[a, sa], label=y[train],
            weight=None if weights is None else weights[train], categorical_feature=mask)
        dvalid = lgb.Dataset(np.c_[b, sb], label=y[valid], reference=dtrain,
            weight=None if weights is None else weights[valid], categorical_feature=mask)
        booster = lgb.train(params, dtrain, num_boost_round=trees, valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(20, verbose=False)])
        contributions = np.asarray(booster.predict(np.c_[a, sa], pred_contrib=True))
        # LightGBM returns one contiguous (features + bias) block per class.
        contributions = contributions.reshape(len(a), -1, 2 * frame.shape[1] + 1)
        importance = np.abs(contributions[:, :, :-1]).sum(axis=1).mean(axis=0)
        total = importance.sum()
        if total:
            importance /= total
        history.append(importance[:frame.shape[1]])
        shadow_history.append(importance[frame.shape[1]:])
        if progress:
            progress(f"GrootCV fold/repeat {index + 1} / {len(splits)}")
    real, shadows = np.asarray(history), np.asarray(shadow_history)
    threshold = np.r_[shadows.max(axis=1), shadows.mean(axis=0).max()].mean() / cutoff
    # Zero-information fits have no evidence of relevance even at zero cutoff.
    selected = (real.mean(axis=0) >= threshold) & (real.mean(axis=0) > 0)
    return dict(decisions=np.where(selected, 1, -1), hits=None,
        ranking=np.where(selected, 1, 2), real=real, shadows=list(shadows),
        thresholds=np.full(len(real), threshold), categories=None,
        splits=splits, engine='GrootCV adaptation (ARFS 3.0.0 rule; native LightGBM TreeSHAP, training-local inner folds)')
