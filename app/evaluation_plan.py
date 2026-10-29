"""Reproducible evaluation membership, with source-aligned stratifier metadata.

Membership is compact: one fold/time code per source row, rather than K copies
of training indices. Public split access checks the source before yielding.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from numbers import Integral

import numpy as np
import pandas as pd
import sklearn
from code_recording import recordable
from roles import Role
from sklearn.model_selection import KFold, StratifiedKFold
from var_types import key_from_dtype

VERSION = 2
SPLITTER_VERSION = f"5/sklearn-{sklearn.__version__}"
STRATEGIES = {"none": "None", "predefined": "Predefined", "kfold": "K-fold",
              "stratified": "Stratified K-fold", "grouped": "Grouped K-fold",
              "logo": "Leave-one-group-out", "temporal": "Forward temporal"}
OBJECTIVES = {"independent": "Independent observations", "unseen": "Unseen entities", "unseen_strata": "Unseen strata",
              "future": "Future observations", "known": "Future observations of known entities",
              "future_unseen": "Future observations of unseen entities (not yet supported)"}
GOALS = {"none": "Nothing", "observations": "New observations", "strata": "New strata",
         "entities": "New entities", "future": "The future"}
PURPOSES = {"": "Unassigned", "development": "Development", "validation": "Validation",
            "holdout": "Final holdout", "prediction": "Prediction"}
STATES = ("Unused", "Training", "Validation", "Final holdout", "Prediction", "Mixed")


def _generated_stratifiers(source):
    return [c for c in source.frame.columns if c not in source.clean_frame.columns
            and Role.STRATIFIER in source.role_map.roles_for(c)]


def _recipe_columns(recipe):
    return tuple(dict.fromkeys(recipe[key] for key in
                 ("stratifier", "stratify", "group", "sequence", "entity") if recipe.get(key)))


def evaluation_frame(source, columns=()):
    """Attach only requested generated strata, preserving stable source positions."""
    generated = set(_generated_stratifiers(source))
    columns = [column for column in dict.fromkeys(columns) if column in generated]
    if not columns:
        return source.clean_frame
    label = ", ".join(repr(column) for column in columns)
    prefix = f"Generated Stratifier {label} requires a row-preserving preview: "
    remedy = " Choose a stable-source Stratifier or None, or define evaluation before the row-changing step."
    if len(source.frame) != len(source.clean_frame):
        raise ValueError(prefix + f"preview has {len(source.frame)} rows but the stable source has "
                         f"{len(source.clean_frame)}." + remedy)
    if not source.frame.index.equals(source.clean_frame.index):
        raise ValueError(prefix + "preview index labels or row order differ from the stable source." + remedy)
    samplers = [name for name, step in source.pipeline.steps
                if callable(getattr(step, "fit_resample", None))
                and getattr(step, "preserves_row_order", False) is not True] if source.has_pipeline else []
    if samplers:
        raise ValueError(prefix + f"sampling step(s) {', '.join(map(repr, samplers))} do not guarantee "
                         "source row identity, even when row counts and index labels match." + remedy)
    frame = source.clean_frame.copy()
    for column in columns:
        # Assign positionally: source indices need not be unique.
        frame[column] = source.frame[column].array
    return frame


def role_columns(source, role):
    columns = list(source.clean_frame.columns)
    if role == Role.STRATIFIER:
        columns.extend(_generated_stratifiers(source))
    return [c for c in columns if role in source.role_map.roles_for(c)]


_AUTO_STRATIFIER = object()


def stratifier_choices(source):
    """Keep source order, with generated cluster memberships last."""
    return sorted(role_columns(source, Role.STRATIFIER), key=lambda name: name.startswith("cluster__"))


def categorical(series):
    return key_from_dtype(series.dtype) in {"nom", "ord", "log", "cde", "obj"}


def level_key(value):
    """Typed tokens avoid collisions between the labels 1 and '1' in bookmarks."""
    if isinstance(value, np.generic):
        value = value.item()
    return f"{type(value).__name__}:{value}"


def partition_levels(series):
    return {level_key(v): str(v) for v in pd.unique(series.dropna())}


def source_fingerprint(source, columns=()):
    frame = source.clean_frame
    digest = hashlib.sha256()
    digest.update(json.dumps(source.modelling_context.to_primitive(), sort_keys=True).encode())
    digest.update(repr([(c, repr(values.dtype), sorted(r.value for r in source.role_map.roles_for(c)))
                        for c, values in frame.items()]).encode())
    digest.update(pd.util.hash_pandas_object(frame.index, index=True).values.tobytes())
    for _, values in frame.items():  # noqa: PERF102
        try:
            hashes = pd.util.hash_pandas_object(values, index=False)
        except TypeError:
            hashes = pd.util.hash_pandas_object(values.map(repr), index=False)
        digest.update(hashes.values.tobytes())
    columns = [c for c in dict.fromkeys(columns) if c in _generated_stratifiers(source)]
    if columns:
        preview = evaluation_frame(source, columns)[columns]
        digest.update(repr([(c, repr(preview[c].dtype)) for c in columns]).encode())
        digest.update(pd.util.hash_pandas_object(preview, index=True).values.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class EvaluationPlan:
    recipe_json: str
    fingerprint: str
    size: int
    development: tuple[int, ...] = ()
    reserved: tuple[int, ...] = ()
    validation: tuple[int, ...] = ()
    holdout: tuple[int, ...] = ()
    prediction: tuple[int, ...] = ()
    codes: tuple[int, ...] = ()
    count: int = 0
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    diagnostics_json: str = "[]"
    supervised: bool = False
    holdout_used: bool = False

    @property
    def recipe(self):
        return json.loads(self.recipe_json)

    @property
    def ready(self):
        return not self.errors and self.count > 0

    @property
    def diagnostics(self):
        return pd.DataFrame(json.loads(self.diagnostics_json))

    def invalidate(self, reason="Source data or roles changed; regenerate the evaluation plan."):
        return replace(self, errors=tuple(dict.fromkeys((*self.errors, reason))))

    def compatible(self, source):
        try:
            return self.fingerprint == source_fingerprint(source, _recipe_columns(self.recipe))
        except (ValueError, TypeError):
            return False

    def require(self, source, *, supervised=False):
        if not self.compatible(source):
            raise ValueError("Source data or roles changed; regenerate the evaluation plan.")
        if not self.ready:
            raise ValueError("; ".join(self.errors) or "Select an evaluation strategy before training.")
        if supervised and not self.supervised:
            raise ValueError("Assign one Target and regenerate the plan before supervised training.")

    def _splits(self):
        """Internal generator; callers must first validate source compatibility."""
        r = self.recipe
        dev = np.asarray(self.development, dtype=int)
        codes = np.asarray(self.codes, dtype=int)
        for j in range(self.count):
            if r["strategy"] == "predefined":
                yield dev.copy(), np.asarray(self.validation, dtype=int)
            elif r["strategy"] == "temporal":
                b = r["window"] + j * r["step"]
                lower = 0 if r["expanding"] else b - r["window"]
                yield (dev[(codes[dev] >= lower) & (codes[dev] < b)],
                       dev[(codes[dev] >= b + r["gap"]) & (codes[dev] < b + r["gap"] + r["horizon"])])
            else:
                yield dev[codes[dev] != j], dev[codes[dev] == j]

    def splits(self, source, *, development_only=False):
        self.require(source)
        mapping = np.full(self.size, -1, dtype=int)
        mapping[list(self.development)] = np.arange(len(self.development))
        for train, validation in self._splits():
            if development_only:
                if (mapping[validation] < 0).any():
                    raise ValueError("Predefined validation requires the complete source frame.")
                yield mapping[train], mapping[validation]
            else:
                yield train, validation

    def membership(self, train, validation):
        states = np.zeros(self.size, dtype=np.uint8)
        states[list(self.holdout)] = 3
        states[list(self.prediction)] = 4
        states[train] = 1
        states[validation] = 2
        return states

    @property
    def refit_positions(self):
        """All labelled model-selection data, excluding final/prediction rows."""
        return tuple(sorted((*self.development, *self.validation)))

    def bookmark(self):
        return {"recipe": self.recipe, "fingerprint": self.fingerprint, "holdout_used": self.holdout_used}

    def membership_frames(self, source):
        """Stream one split at a time for downloads; never build a wide table."""
        self.require(source)
        r = self.recipe
        for j, (train, validation) in enumerate(self._splits(), 1):
            result = pd.DataFrame({"split_id": f"CV{j}", "source_position": np.arange(self.size),
                "membership": np.asarray(STATES)[self.membership(train, validation)],
                "original_index": source.clean_frame.index.map(str)})
            for name in ("group", "sequence", "stratifier"):
                if r.get(name):
                    result[name] = evaluation_frame(source, (r[name],))[r[name]].to_numpy()
            yield result


def _integer(value, label, minimum):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{label} must be an integer of at least {minimum}.")
    return int(value)


def _variable(source, name, label):
    frame = evaluation_frame(source, (name,))
    if not name or name not in frame:
        raise ValueError(f"Select a {label} available in the stable source.")
    values = frame[name]
    if values.isna().any():
        raise ValueError(f"{label.title()} '{name}' contains missing values; resolve them upstream.")
    return values


@recordable
def _sequence(source, name):
    values = _variable(source, name, "Sequence variable")
    if name not in role_columns(source, Role.SEQUENCE):
        raise ValueError("The Sequence variable must have the Sequence role.")
    if key_from_dtype(values.dtype) not in {"int", "dec", "dte"}:
        raise ValueError("Sequence must be numeric or datetime, using a common clock across entities.")
    if pd.api.types.is_numeric_dtype(values) and not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Sequence contains nonfinite values.")
    return values


@recordable
def _boundary(source, recipe, train, validation):
    objective = recipe["objective"]
    if objective in {"unseen", "unseen_strata"}:
        groups = _variable(source, recipe.get("group"), "Grouping variable")
        if set(groups.iloc[train]) & set(groups.iloc[validation]):
            raise ValueError("Unseen-group evaluation requires zero group overlap across every boundary.")
    if objective in {"future", "known"}:
        times = _sequence(source, recipe.get("sequence"))
        if times.iloc[train].max() >= times.iloc[validation].min():
            raise ValueError("Future evaluation requires all training/development times to precede validation/reserved times.")
        if objective == "known":
            entity = recipe.get("entity")
            if entity not in role_columns(source, Role.IDENTIFIER):
                raise ValueError("Select an Identifier for future-known-entity evaluation.")
            ids = _variable(source, entity, "Identifier")
            if not set(ids.iloc[validation]).issubset(set(ids.iloc[train])):
                raise ValueError("Future-known-entity evaluation contains validation/reserved identifiers absent from training.")


def structure(source, stratifier=_AUTO_STRATIFIER):
    """Resolve assigned roles only; repeated Identifier means longitudinal data."""
    result = {}
    for role in (Role.PARTITION, Role.IDENTIFIER, Role.SEQUENCE):
        names = list(source.role_map.columns_with_role(role))
        if len(names) > 1:
            raise ValueError(f"Assign at most one {role.value.title()} variable.")
        result[role.value] = names[0] if names else None
        if names:
            _variable(source, names[0], role.value.title())
    choices = stratifier_choices(source)
    if stratifier is _AUTO_STRATIFIER:
        stratifier = choices[0] if choices else None
    if stratifier is not None:
        if stratifier not in choices:
            raise ValueError("Choose an available Stratifier-role variable.")
        _variable(source, stratifier, "Stratifier")
    result["stratifier"] = stratifier
    identifier, sequence = result["identifier"], result["sequence"]
    result["longitudinal"] = bool(identifier and not source.clean_frame[identifier].is_unique)
    if result["longitudinal"]:
        if not sequence:
            raise ValueError("Repeated Identifier requires an assigned Sequence: longitudinal data need a primary key.")
        if source.clean_frame.duplicated([identifier, sequence]).any():
            raise ValueError("Identifier and Sequence must together form a unique primary key.")
    return result


def goal_options(source, goal, partition_map=None, stratifier=_AUTO_STRATIFIER):
    roles = structure(source, stratifier)
    if goal not in GOALS or goal == "none":
        raise ValueError("Choose a prediction goal in the footer.")
    options = {"goal": goal, "partition_map": partition_map or {}}
    if roles["stratifier"]:
        options["stratifier"] = roles["stratifier"]
    if goal == "observations":
        options.update(strategy="stratified" if roles["stratifier"] else "kfold", objective="independent")
        if roles["stratifier"]:
            options["stratify"] = roles["stratifier"]
    elif goal in {"strata", "entities"}:
        group = roles["stratifier"] if goal == "strata" else roles["identifier"]
        if not group or (goal == "entities" and not roles["longitudinal"]):
            raise ValueError("Select a Stratifier in settings." if goal == "strata" else
                             "Requires a repeated Identifier and its Sequence primary key.")
        options.update(strategy="grouped", objective="unseen_strata" if goal == "strata" else "unseen", group=group)
    else:
        if not roles["sequence"]:
            raise ValueError("Future observations requires an assigned Sequence.")
        _sequence(source, roles["sequence"])
        options.update(strategy="temporal", sequence=roles["sequence"],
                       objective="known" if roles["longitudinal"] else "future")
        if roles["longitudinal"]:
            options["entity"] = roles["identifier"]
    if "validation" in options["partition_map"].values():
        options["strategy"] = "predefined"
    return options


@recordable
def build_plan(source, **options):
    """Derive a method from the goal and roles; never fall back on invalid data."""
    recipe = {"version": VERSION, "splitter_version": SPLITTER_VERSION,
              "strategy": options.get("strategy", "none"), "objective": options.get("objective", "")}
    if "goal" in options:
        recipe["goal"] = options["goal"]
    base = EvaluationPlan(json.dumps(recipe), source_fingerprint(source), len(source.clean_frame))
    try:
        if "goal" in options:
            options = options | goal_options(source, options["goal"], options.get("partition_map"),
                                              options.get("stratifier", _AUTO_STRATIFIER))
            recipe.update({k: options[k] for k in ("goal", "strategy", "objective")})
        return _build(source, recipe, options, base)
    except (ValueError, TypeError) as error:
        return replace(base, recipe_json=json.dumps(recipe), errors=(str(error),))


@recordable
def _build(source, r, o, base):
    frame = source.clean_frame
    n = len(frame)
    strategy = r["strategy"]
    if not frame.columns.is_unique:
        raise ValueError("Source variable names must be unique.")
    if not n:
        raise ValueError("No observations in the stable source.")
    context = source.modelling_context
    survival = context.task == "survival"
    r["modelling_context"] = context.to_primitive()
    r["task"] = context.resolved_task(source)
    if survival:
        if r["objective"] in {"future", "known"}:
            raise ValueError("Survival temporal evaluation requires fold-specific administrative censoring; not yet supported.")
        identifiers = role_columns(source, Role.IDENTIFIER)
        if any(not frame[c].is_unique for c in identifiers):
            raise ValueError("Survival currently requires one baseline record per subject; repeated records are not yet supported.")
    roles = structure(source, o.get("stratifier", _AUTO_STRATIFIER))
    r["stratifier"] = roles["stratifier"]
    if strategy not in STRATEGIES or strategy == "none":
        raise ValueError("Choose a prediction goal in the footer.")
    if r["objective"] not in OBJECTIVES:
        raise ValueError("The evaluation recipe needs a supported prediction objective.")
    if r["objective"] == "future_unseen":
        raise ValueError("Combined future-time and unseen-entity separation is not yet supported.")
    if r["objective"] in {"unseen", "unseen_strata"} and strategy not in {"grouped", "logo", "predefined"}:
        raise ValueError("Unseen strata/entities require a grouped or predefined strategy.")
    if r["objective"] in {"future", "known"} and strategy not in {"temporal", "predefined"}:
        raise ValueError("Future observations require forward temporal or predefined validation.")
    if strategy == "temporal" and r["objective"] not in {"future", "known"}:
        raise ValueError("Forward temporal splitting requires a future prediction objective.")

    for key, needed in (("group", strategy in {"grouped", "logo"} or r["objective"] in {"unseen", "unseen_strata"}),
                        ("sequence", strategy == "temporal" or r["objective"] in {"future", "known"}),
                        ("entity", r["objective"] == "known"), ("stratify", strategy == "stratified")):
        if needed:
            r[key] = o.get(key)
            _variable(source, r[key], key)
    frame = evaluation_frame(source, _recipe_columns(r))
    dev, reserved = np.arange(n), np.array([], dtype=int)
    validation = holdout = prediction = np.array([], dtype=int)
    if r.get("group"):
        group = frame[r["group"]]
        if not categorical(group) and not source.role_map.roles_for(r["group"]) & {Role.IDENTIFIER, Role.STRATIFIER}:
            raise ValueError("Choose a categorical group or an assigned Identifier/Stratifier.")
    partitions = list(source.role_map.columns_with_role(Role.PARTITION))
    if len(partitions) > 1:
        raise ValueError("Only one Partition variable is supported.")
    if partitions:
        r["partition"] = partitions[0]
        values = _variable(source, partitions[0], "Partition variable")
        levels = partition_levels(values)
        if len(levels) < 2:
            raise ValueError("Partition requires at least two observed levels.")
        mapping = o.get("partition_map")
        if mapping is None:
            # Existing programmatic callers can still express a two-way split.
            mapping = {k: "development" for k in o.get("development_levels", ())}
            for k in o.get("reserved_levels", ()):
                if k in mapping:
                    raise ValueError("Partition assignments overlap.")
                mapping[k] = o.get("purpose", "")
        r["partition_map"] = dict(mapping)
        if set(mapping) != set(levels) or any(v not in PURPOSES or not v for v in mapping.values()):
            raise ValueError("Assign every Partition level to a purpose; remove stale assignments by regenerating for the current source.")
        tokens = values.map(level_key)
        assigned = tokens.map(mapping)
        dev = np.flatnonzero(assigned == "development")
        validation = np.flatnonzero(assigned == "validation")
        holdout = np.flatnonzero(assigned == "holdout")
        prediction = np.flatnonzero(assigned == "prediction")
        reserved = np.flatnonzero(assigned != "development")
        if not len(dev):
            raise ValueError("Assign at least one Partition level to Development.")
        if len(validation):
            strategy = r["strategy"] = "predefined"
            _boundary(source, r, dev, validation)
        elif strategy == "predefined":
            raise ValueError("Predefined validation requires levels mapped to Validation.")
        selection = np.sort(np.concatenate([dev, validation]))
        for subset in (holdout, prediction):
            if len(subset):
                _boundary(source, r, selection, subset)
    elif strategy == "predefined":
        raise ValueError("Predefined validation requires an assigned Partition variable.")

    codes = np.full(n, -1, dtype=int)
    if strategy == "predefined":
        count = 1
    elif strategy == "temporal":
        times = _sequence(source, r.get("sequence"))
        r.update(window=_integer(o.get("window"), "Initial training window", 1),
                 horizon=_integer(o.get("horizon"), "Validation window", 1),
                 gap=_integer(o.get("gap", 0), "Gap", 0),
                 step=_integer(o.get("step", o.get("horizon")), "Step", 1),
                 expanding=bool(o.get("expanding", True)))
        if r["step"] < r["horizon"]:
            raise ValueError("Step must be at least the validation window to avoid overlapping validation windows.")
        codes[dev], unique = pd.factorize(times.iloc[dev], sort=True)
        count = max(0, 1 + (len(unique) - r["window"] - r["gap"] - r["horizon"]) // r["step"])
        if not count:
            raise ValueError("The temporal windows do not produce a complete split.")
    else:
        if strategy == "logo":
            groups = _variable(source, r["group"], "Grouping variable")
            codes[dev], unique = pd.factorize(groups.iloc[dev], sort=False)
            count = len(unique)
            if count < 2:
                raise ValueError("Leave-one-group-out requires at least two groups.")
        else:
            count = r["folds"] = _integer(o.get("requested_folds", o.get("folds", 5)), "Fold count", 2)
            if strategy == "grouped":
                groups = _variable(source, r["group"], "Grouping variable")
                labels, unique = pd.factorize(groups.iloc[dev], sort=False)
                r["requested_folds"] = count
                count = r["folds"] = min(count, len(unique))
                if count < 2:
                    raise ValueError("Grouped resampling requires at least two development groups.")
            if count > len(dev):
                raise ValueError("Fold count exceeds the number of development observations.")
            if strategy in {"kfold", "stratified"}:
                r["shuffle"] = bool(o.get("shuffle", True))
                if r["shuffle"]:
                    r["seed"] = _integer(o.get("seed", 42), "Random seed", 0)
                    if r["seed"] > 2**32 - 1:
                        raise ValueError("Random seed must be at most 4294967295.")
                splitter = KFold(count, shuffle=r["shuffle"], random_state=r.get("seed"))
                if strategy == "stratified":
                    values = _variable(source, r["stratify"], "Stratification variable")
                    if not categorical(values) and r["stratify"] not in role_columns(source, Role.STRATIFIER):
                        raise ValueError("Stratification requires a categorical variable; numeric values are not binned.")
                    labels, _ = pd.factorize(values.iloc[dev], sort=False)
                    if np.bincount(labels).min() < count:
                        raise ValueError("Every development stratum must contain at least K observations.")
                    splitter = StratifiedKFold(count, shuffle=r["shuffle"], random_state=r.get("seed"))
                    splits = splitter.split(dev, labels)
                else:
                    splits = splitter.split(dev)
            else:
                r["seed"] = _integer(o.get("seed", 42), "Random seed", 0)
                if r["seed"] > 2**32 - 1:
                    raise ValueError("Random seed must be at most 4294967295.")
                # Randomize whole levels, then distribute them evenly across folds.
                shuffled = np.random.default_rng(r["seed"]).permutation(len(unique))
                group_folds = np.empty(len(unique), dtype=int)
                for j, members in enumerate(np.array_split(shuffled, count)):
                    group_folds[members] = j
                splits = ((None, np.flatnonzero(group_folds[labels] == j)) for j in range(count))
            for j, (_, fold_validation) in enumerate(splits):
                codes[dev[fold_validation]] = j

    targets = list(source.role_map.columns_with_role(Role.TARGET))
    if len(targets) > 1 and not survival:
        raise ValueError("Assign at most one Target for supervised checks.")
    target = targets[0] if targets and not survival else None
    if target:
        if target not in frame:
            raise ValueError("The Target must be available in the stable source, not only in a fitted preview.")
        r["target"] = target
        required = np.setdiff1d(np.arange(n), prediction)
        if frame[target].iloc[required].isna().any():
            raise ValueError("Target contains missing values outside a prediction set; resolve them upstream.")
    if survival:
        required = np.setdiff1d(np.arange(n), prediction)
        context.survival_outcome(source, required)
    notes = []
    if r.get("requested_folds", count) > count:
        notes.append(f"Using K={count}: requested K={r['requested_folds']}, but only {count} development groups are available.")
    if not target and not survival:
        notes.append("Structurally checked; assign a Target before supervised training.")
    if strategy in {"kfold", "stratified"} and (role_columns(source, Role.SEQUENCE) or
            any(not frame[c].is_unique for c in role_columns(source, Role.IDENTIFIER))):
        notes.append("Assigned roles suggest dependence; ordinary/stratified folds do not separate entities or time.")
    if role_columns(source, Role.GEOMETRY):
        # TODO: spatial blocks, distance-aware buffers, CRS/units and missing geometries.
        notes.append("Spatial dependence is not controlled automatically; region grouping does not remove boundary dependence.")
    if strategy == "temporal":
        notes.append("Windows count distinct Sequence values, not elapsed time; verify a common clock and label availability.")
    plan = replace(base, recipe_json=json.dumps(r, sort_keys=True),
                   fingerprint=source_fingerprint(source, _recipe_columns(r)), development=tuple(map(int, dev)),
                   reserved=tuple(map(int, reserved)), validation=tuple(map(int, validation)),
                   holdout=tuple(map(int, holdout)), prediction=tuple(map(int, prediction)), codes=tuple(map(int, codes)), count=count, supervised=bool(target) or survival)
    diagnostics = []
    classes = set(frame[target].iloc[dev]) if target and r["task"] in {"classification", "ordinal"} else None
    coverage = np.zeros(n, dtype=bool)
    for j, (train, validation) in enumerate(plan._splits(), 1):
        if not len(train) or not len(validation) or np.intersect1d(train, validation).size:
            raise ValueError(f"CV{j} has empty or overlapping training/validation subsets.")
        _boundary(source, r, train, validation)
        coverage[validation] = True
        row = {"Split": f"CV{j}", "Training": len(train), "Validation": len(validation),
               "Unused": n - len(train) - len(validation) - (len(holdout) + len(prediction))}
        warnings = []
        if survival:
            for label, indices in (("Train", train), ("Validation", validation)):
                outcome = context.survival_outcome(source, indices)
                events = int(outcome['event'].sum())
                row[f"{label} events"] = events
                row[f"{label} censored"] = len(indices) - events
                row[f"{label} follow-up"] = f"{outcome['time'].min()} – {outcome['time'].max()}"
                if not events:
                    raise ValueError(f"CV{j} {label.lower()} has no observed events; survival evaluation is not ready.")
            warnings.append("Metric-specific comparable pairs and censoring support require evaluation checks")
        if r.get("group"):
            groups = frame[r["group"]]
            a, b = set(groups.iloc[train]), set(groups.iloc[validation])
            row.update({"Train groups": len(a), "Validation groups": len(b), "Group overlap": len(a & b)})
            if strategy in {"grouped", "logo"} and a & b:
                raise ValueError(f"CV{j} has overlapping groups.")
        if r.get("sequence"):
            seq = frame[r["sequence"]]
            for label, indices in (("Train", train), ("Validation", validation)):
                row[f"{label} range"] = f"{seq.iloc[indices].min()} – {seq.iloc[indices].max()}"
        if classes is not None:
            if not classes.issubset(set(frame[target].iloc[train])):
                raise ValueError(f"CV{j} training is missing a development target class.")
            if not classes.issubset(set(frame[target].iloc[validation])):
                warnings.append("Validation missing target classes; some metrics may be undefined")
            for label, indices in (("Train", train), ("Validation", validation)):
                row[f"{label} classes"] = "; ".join(f"{k}: {v}" for k, v in frame[target].iloc[indices].value_counts().items() if v)
        row["Status"] = "; ".join(warnings) or "Valid"
        diagnostics.append(row)
    if any(row["Status"] != "Valid" for row in diagnostics):
        notes.append("Some folds need metric checks; see the flip-side before choosing metrics.")
    notes.append(f"Validation covers {coverage.sum():,} of {len(dev) if strategy != 'predefined' else len(plan.validation):,} {'development' if strategy != 'predefined' else 'reserved validation'} observations.")
    return replace(plan, warnings=tuple(notes), diagnostics_json=json.dumps(diagnostics))


def restore_plan(source, state):
    recipe = state.get("recipe", {})
    plan = build_plan(source, **recipe)
    if recipe.get("version") != VERSION or recipe.get("splitter_version") != SPLITTER_VERSION:
        return plan.invalidate("Unsupported saved evaluation recipe/splitter version; explicitly configure a new plan.")
    if state.get("fingerprint") != plan.fingerprint:
        return plan.invalidate("Saved evaluation source differs from restored data; explicitly configure a new plan.")
    return replace(plan, holdout_used=bool(state.get("holdout_used", False)))
