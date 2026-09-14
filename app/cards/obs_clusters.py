"""Visualize fixed-K cluster memberships and optionally export a nominal predictor."""
from __future__ import annotations

import asyncio
import os
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from html import escape
from pathlib import Path

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shinywidgets
from card import Card
from ClusterMembershipTransformer import ClusterMembershipTransformer
from module import Module
from plotly.colors import qualitative
from proxy_data import proxy_data
from roles import Role, RoleMap
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist, squareform
from shiny import reactive, render, req, ui
from shinywidgets import render_widget
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture

from cards.obs_k_clusters import _diana, _importance_weights, _pam

METHODS = ("Partition", "Agglomerative", "Divisive", "Mixture", "Density", "Spectral")


@dataclass
class ClusterViews:
    source: proxy_data
    options: dict
    positions: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    coordinates: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    labels: dict = field(default_factory=dict)
    transformers: dict = field(default_factory=dict)
    notes: dict = field(default_factory=dict)
    predictors: list = field(default_factory=list)
    error: str = ""


def _prepare(data, *, limit, standardize, use_weights):
    predictors = data.role_map.columns_with_role(Role.PREDICTOR)
    weight_columns = data.role_map.columns_with_role(Role.WEIGHTING)
    columns = [c for c in data.columns if c in predictors and c not in weight_columns
               and not str(c).startswith(Card.SHADOW_PREFIX)
               and pd.api.types.is_numeric_dtype(data.frame[c].dtype)
               and not pd.api.types.is_bool_dtype(data.frame[c].dtype)
               and not pd.api.types.is_complex_dtype(data.frame[c].dtype)]
    raw_weights = None
    if use_weights and weight_columns:
        if len(weight_columns) != 1:
            raise ValueError("Assign exactly one observation-importance column.")
        series = data.frame[next(iter(weight_columns))]
        if (not pd.api.types.is_numeric_dtype(series.dtype)
                or pd.api.types.is_bool_dtype(series.dtype)
                or pd.api.types.is_complex_dtype(series.dtype)):
            raise ValueError("Observation importance must be numeric.")
        raw_weights = series.to_numpy(dtype=float, na_value=np.nan)
        _importance_weights(raw_weights)
    positive = np.ones(len(data), dtype=bool) if raw_weights is None else raw_weights > 0
    frame = data.frame[columns].astype(float).replace([np.inf, -np.inf], np.nan)
    frame = frame.loc[:, frame.iloc[np.flatnonzero(positive)].nunique() > 1]
    positions = np.flatnonzero(positive & frame.notna().all(axis=1).to_numpy())
    if len(positions) > limit:
        positions = np.sort(np.random.default_rng(2025).choice(positions, int(limit), replace=False))
    x = frame.iloc[positions].to_numpy()
    if len(x) < 2 or x.shape[1] == 0:
        raise ValueError("At least two complete rows and one varying numeric predictor are needed.")
    weights = None if raw_weights is None else _importance_weights(raw_weights[positions])
    if weights is not None and np.all(weights == weights[0]):
        weights = None
    scale = np.max(np.abs(x), axis=0)
    safe = x / np.where(scale > 0, scale, 1)
    mean = np.average(safe, axis=0, weights=weights)
    sd = np.sqrt(np.average((safe - mean) ** 2, axis=0, weights=weights))
    keep = sd > 0
    x = (safe[:, keep] - mean[keep]) / sd[keep] if standardize else x[:, keep]
    if x.shape[1] == 0:
        raise ValueError("The analysis sample has no varying numeric predictors.")
    return x, positions, weights, list(frame.columns[keep])


def _density(distance, k, min_points, weights, border, *, return_model=False):
    # Cluster count is not monotonic in epsilon: noise can first become clusters,
    # then clusters merge. Search a bounded grid rather than binary-searching K.
    positive = distance[np.triu_indices(len(distance), 1)]
    positive = positive[positive > 0]
    if not len(positive):
        raise ValueError("Density clustering requires positive pairwise distances.")
    radii = np.unique(np.quantile(positive, np.linspace(0, 1, 60)))
    best = None
    for radius in radii:
        model = DBSCAN(eps=float(radius), min_samples=min_points, metric="precomputed").fit(distance, sample_weight=weights)
        labels = model.labels_.copy()
        if not border:
            noncore = np.ones(len(labels), dtype=bool)
            noncore[model.core_sample_indices_] = False
            labels[noncore] = -1
        count = len(set(labels) - {-1})
        noise = float(np.average(labels == -1, weights=weights))
        rank = (abs(count - k), noise, radius)
        if best is None or rank < best[0]:
            best = rank, labels, count, model
    rank, labels, count, model = best
    note = (f"DBSCAN found {count} groups for requested K={k}; unallocated={rank[1]:.1%}, "
                    f"radius={rank[2]:.4g}. Best of {len(radii)} candidate radii; exact K is not guaranteed.")
    return (labels, note, model) if return_model else (labels, note)


def _fit(method, x, distance, k, weights, options):
    if weights is not None and method not in ("Partition", "Density"):
        raise ValueError("Unavailable with unequal observation importance. Disable weighting to use this method.")
    if method == "Density":
        return _density(distance, k, options["min_points"], weights, options["border"])
    if k == 1:
        return np.zeros(len(x), dtype=int), "K=1: all analyzed observations form one group."
    note = ""
    if method == "Partition":
        if options["centre"] == "medoids":
            labels = _pam(distance, k, weights)
        else:
            labels = KMeans(n_clusters=k, n_init=10, random_state=2025).fit_predict(x, sample_weight=weights)
    elif method == "Agglomerative":
        metric = pdist(x) if options["linkage"] == "ward" else squareform(distance)
        labels = cut_tree(linkage(metric, method=options["linkage"]), n_clusters=[k]).ravel()
        if options["linkage"] == "ward":
            note = "Ward linkage uses Euclidean distances."
    elif method == "Divisive":
        labels = _diana(distance, k)[k]
    elif method == "Mixture":
        models = []
        for covariance in ("spherical", "diag", "tied", "full"):
            try:
                model = GaussianMixture(n_components=k, covariance_type=covariance, random_state=2025,
                                        n_init=2, max_iter=300).fit(x)
                if model.converged_ and np.isfinite(model.bic(x)):
                    models.append(model)
            except (ValueError, np.linalg.LinAlgError):
                continue
        if not models:
            raise ValueError("No Gaussian mixture converged.")
        model = min(models, key=lambda model: model.bic(x))
        labels = model.predict(x)
        note = f"Lowest BIC at fixed K: {model.covariance_type} covariance."
    else:
        neighbours = min(options["neighbours"], len(x) - 1)
        ranked = distance.copy()
        np.fill_diagonal(ranked, np.inf)
        nearest = np.argsort(ranked, axis=1, kind="stable")[:, :neighbours]
        graph = np.zeros_like(distance)
        graph[np.arange(len(x))[:, None], nearest] = 1
        graph = np.maximum(graph, graph.T)
        components = connected_components(graph, directed=False, return_labels=False)
        if components > k:
            raise ValueError(f"Graph has {components} disconnected components, more than K={k}; increase nearest neighbors.")
        labels = SpectralClustering(n_clusters=k, affinity="precomputed", assign_labels="cluster_qr",
                                    random_state=2025, n_jobs=1).fit_predict(graph)
        note = f"Union graph of {neighbours} nearest neighbors; {components} connected components."
    if len(np.unique(labels)) != k:
        raise ValueError(f"The fitted model did not produce {k} occupied groups.")
    return labels, note


def _analyze(data, *, limit=1000, standardize=True, use_weights=True, metric="euclidean",
             centre="centroids", linkage="average", min_points=5, border=True,
             neighbours=10, projection="tsne", perplexity=30):
    options = {"limit": limit, "standardize": standardize, "use_weights": use_weights, "metric": metric,
                   "centre": centre, "linkage": linkage, "min_points": min_points, "border": border,
                   "neighbours": neighbours, "projection": projection, "perplexity": perplexity}
    result = ClusterViews(data, options)
    try:
        x, result.positions, weights, result.predictors = _prepare(
            data, limit=limit, standardize=standardize, use_weights=use_weights)
        k = data.cluster_count or 1
        if k > min(len(x), len(np.unique(x, axis=0))):
            raise ValueError(f"Incoming K={k} exceeds the available distinct analysis rows. Increase the sample or revise K upstream.")
        distance = squareform(pdist(x, metric="cityblock" if metric == "manhattan" else "euclidean"))
        if not np.isfinite(distance).all():
            raise ValueError("Distances overflowed; enable standardization or rescale predictors.")
        if projection == "tsne":
            result.coordinates = TSNE(n_components=2, metric="precomputed", init="random", random_state=2025,
                                      perplexity=min(float(perplexity), (len(x) - 1) / 3),
                                      learning_rate="auto", n_jobs=1).fit_transform(distance)
        else:
            centered = x - np.average(x, axis=0, weights=weights)
            basis = centered if weights is None else centered * np.sqrt(weights[:, None])
            _, _, axes = np.linalg.svd(basis, full_matrices=False)
            coords = centered @ axes[:2].T
            result.coordinates = np.pad(coords, ((0, 0), (0, 2 - coords.shape[1])))
        for method in METHODS:
            try:
                if method in ("Partition", "Mixture", "Density"):
                    weighting = data.role_map.columns_with_role(Role.WEIGHTING) if use_weights else set()
                    transformer = ClusterMembershipTransformer(
                        columns=tuple(c for c in data.columns if c in data.role_map.columns_with_role(Role.PREDICTOR)),
                        n_clusters=k, method=method, centre=centre, metric=metric, min_points=min_points, border=border,
                        standardize=standardize, weighting=next(iter(weighting), None), limit=limit,
                        output_column=_membership_name(data.columns),
                    ).fit(data.frame)
                    preview = transformer.transform(data.frame)
                    result.labels[method] = preview[transformer.output_column].iloc[result.positions].astype(str).to_numpy()
                    result.transformers[method] = transformer
                    result.notes[method] = (f"Lowest BIC at fixed K: {transformer.model_.covariance_type} covariance."
                        if method == "Mixture" and transformer.model_ is not None else "")
                    if method == "Density":
                        result.notes[method] = transformer.density_note_
                    continue
                labels, note = _fit(method, x, distance, k, weights, options)
                # Stable numbering by first occurrence, not estimator-specific IDs.
                mapping = {label: f"c{i + 1}" for i, label in enumerate(dict.fromkeys(labels[labels >= 0]))}
                result.labels[method] = np.array(["Noise" if label < 0 else mapping[label] for label in labels])
                result.notes[method] = note
            except (ValueError, np.linalg.LinAlgError) as error:
                result.notes[method] = str(error)
    except (ValueError, np.linalg.LinAlgError) as error:
        result.error = str(error)
    return result


def _membership_name(columns, base="cluster"):
    name, suffix = base, 2
    while name in columns:
        name = f"{base}_{suffix}"
        suffix += 1
    return name


def _export(result, method, name):
    name = name.strip()
    if not name or name.startswith(Card.SHADOW_PREFIX):
        raise ValueError("Enter a nonempty variable name without the reserved shadow prefix.")
    if name in result.source.columns:
        raise ValueError("That variable already exists; choose a new name.")
    if result.error or method not in result.labels:
        raise ValueError("No membership labels are available for this method.")
    if method not in result.transformers:
        raise ValueError("Learned membership is available only for Partition, Mixture and DBSCAN")
    transformer = deepcopy(result.transformers[method])
    transformer.set_params(output_column=name)
    preview = transformer.transform(result.source.frame)
    roles = RoleMap()
    roles.set_roles(name, [Role.PREDICTOR])
    return result.source.with_pipeline_step(transformer, name="obs_clusters",
        operation="Add cluster membership", preview_frame=preview, added_roles=roles)


def _export_memberships(source, transformers):
    """Rebuild only this card's selected steps, preserving earlier pipeline steps."""
    result = source
    for method in ("Partition", "Mixture", "Density"):
        if method not in transformers:
            continue
        transformer = transformers[method]
        preview = transformer.transform(result.frame)
        roles = RoleMap()
        roles.set_roles(transformer.output_column, [Role.PREDICTOR])
        result = result.with_pipeline_step(transformer, name="obs_clusters",
            operation="Add cluster membership", preview_frame=preview, added_roles=roles)
    return result


def _figure(result, method, fullscreen=False):
    figure = go.Figure()
    labels = result.labels.get(method)
    if labels is None or result.error:
        figure.add_annotation(text=result.error or result.notes.get(method, "No membership labels available."), showarrow=False)
    else:
        identifiers = [column for column in result.source.columns
                       if column in result.source.role_map.columns_with_role(Role.IDENTIFIER)]
        hover = []
        for position in result.positions:
            parts = []
            for column in identifiers:
                value = result.source.frame[column].iloc[position]
                if pd.api.types.is_scalar(value) and not pd.isna(value):
                    parts.append(f"{escape(str(column))}: {escape(str(value))}")
            hover.append("<br>".join(parts) if parts else f"Row {position + 1}")
        hover = np.asarray(hover, dtype=object)
        for label in dict.fromkeys(labels):
            mask = labels == label
            color = "#888888" if label in ("Noise", "unallocated") else qualitative.Plotly[(int(label.removeprefix("c")) - 1) % len(qualitative.Plotly)]
            figure.add_scatter(x=result.coordinates[mask, 0], y=result.coordinates[mask, 1], mode="markers",
                name=label if label in ("Noise", "unallocated") else f"Cluster {label}",
                customdata=hover[mask, None],
                marker={"color": color, "size": 7, "opacity": .8},
                hovertemplate="%{customdata[0]}<br>x=%{x:.3g}<br>y=%{y:.3g}<extra>%{fullData.name}</extra>")
    axis = "t-SNE" if result.options["projection"] == "tsne" else "Principal component"
    figure.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#e5ecf6",
                         margin={"l": 15, "r": 15, "t": 15, "b": 15}, xaxis_title=f"{axis} 1", yaxis_title=f"{axis} 2",
                         legend={"orientation": "h"})
    if len(result.coordinates):
        for i, key in enumerate(("xaxis", "yaxis")):
            low, high = np.min(result.coordinates[:, i]), np.max(result.coordinates[:, i])
            pad = max((high-low)*.05, 1e-6)
            figure.update_layout(**{key: {"range": [low-pad, high+pad]}})
    return figure


def instance():
    this = Card(file=__file__, mutable=True)
    this.long_name = "Cluster membership"
    this.description = "Compare cluster memberships at the incoming K and optionally add Partition, Mixture and DBSCAN labels as nominal predictors."

    def front():
        return ui.navset_bar(*(ui.nav_panel(method, shinywidgets.output_widget(f"Chart_{method}", fill=True,
            guide=this, title=f"{method} membership", position="left",
            text="Clusters are fitted in numeric predictor space and colored on the same 2D coordinates in every tab. Colors identify labels within each method, not equivalent groups across methods. Hover shows Identifier-role values when available, otherwise the original row position. Projection separation does not establish cluster validity."),
            value=method) for method in METHODS), id="ClusterType", selected="Partition", title=None, padding=0, fillable=True)
    this.front = front
    
    def back():
        return ui.div(
            ui.output_text("TableTitle"), 
            ui.output_data_frame("Membership"), 
            class_="card-scroll-content"
        )

    this.back = back

    def footer():
        return ui.TagList(ui.output_ui("Busy"), ui.output_text("Status"),
            ui.output_ui("MembershipControl"))
    this.footer = footer

    def settings():
        def select(id, label, choices, text):
            return ui.input_select(id, label=label, choices=choices, guide=this, position="left", text=text)
        def check(id, label, text):
            return ui.input_checkbox(id, label=label, value=True, guide=this, position="left", text=text)
        def slider(id, label, low, high, value, text):
            return ui.input_slider(id, label=label, min=low, max=high, value=value, guide=this, position="left", text=text)
        return ui.TagList(
            select("Projection", "2D projection", {"tsne":"t-SNE", "pca":"PCA"}, "One shared embedding for all methods. t-SNE emphasizes local neighborhoods and ignores weights in the embedding; PCA is faster and uses importance-weighted axes when enabled. Clustering is always fitted before projection, in predictor space."),
            slider("Perplexity", "t-SNE perplexity", 2, 50, 30, "Neighborhood scale of the t-SNE display. Capped at (analyzed rows minus one) / 3. Changes the visualization, not the clustering inputs."),
            check("Standardize", "Standardize numeric predictors", "Center and scale predictors to unit spread, using observation importance when enabled. Analysis only; outgoing predictor values are unchanged."),
            check("UseWeights", "Use assigned observation weighting", "Use finite nonnegative numeric importance weights, normalized to mean one; omit zero-weight rows. Unequal weights are supported by Partition and Density. Other tabs explain why they are unavailable. Disable for equal importance; the weight column never becomes a predictor."),
            select("Metric", "Distance metric", ["euclidean", "manhattan"], "Used by hierarchies except Ward, PAM, Density, Spectral and t-SNE. K-means, Gaussian mixtures, Ward and PCA use Euclidean geometry."),
            select("Centre", "Partition method", {"centroids":"K-means", "medoids":"PAM (medoids)"}, "K-means uses fitted centers; PAM selects actual observations as medoids and can be slower. Both use incoming K and support importance weights."),
            select("Linkage", "Agglomerative linkage", ["average", "single", "complete", "ward"], "Merge clusters using average, nearest or farthest pair distances, or Ward's increase in squared dispersion. Ward always uses Euclidean distances."),
            slider("MinPoints", "DBSCAN minimum points (including self)", 2, 100, 5, "Minimum neighborhood size for a core point, including self. With weighting, this is mean-one importance mass. A bounded radius search favors counts closest to incoming K, then less noise. Exact K may be unattainable."),
            check("Border", "Assign DBSCAN border points", "Include non-core points reachable from a core point. Disable to require minimum training-neighborhood importance mass as well. Unassigned rows are labeled unallocated; this level is not counted as a cluster. New rows are matched to fixed training core points within the learned radius."),
            slider("Neighbours", "Spectral nearest neighbors", 1, 50, 10, "Binary undirected union of nearest-neighbor connections, excluding self. Increase if the graph has more components than K. Capped at analyzed rows minus one."),
            slider("Limit", "Maximum observations to analyze", 50, 2000, 1000, "Reproducible uniform sampling without replacement above this limit. Distances use quadratic memory; PAM and t-SNE can be expensive. Partition, Mixture and DBSCAN assign all rows with complete fitted predictors on export, including unsampled rows. Incomplete rows retain missing labels. No outgoing rows are removed."))
    this.settings = settings

    def server(input, output, session):
        busy = this.busy()
        committed = reactive.Value(None)
        OutputData = reactive.Value()
        message = reactive.Value("")
        reset_pending = False

        @this.settle(seconds=2)
        @this.suspendable(calc=True)
        def Options():
            return {"limit": int(input.Limit()), "standardize": bool(input.Standardize()), "use_weights": bool(input.UseWeights()),
                        "metric": input.Metric(), "centre": input.Centre(), "linkage": input.Linkage(), "min_points": int(input.MinPoints()),
                        "border": bool(input.Border()), "neighbours": int(input.Neighbours()), "projection": input.Projection(),
                        "perplexity": int(input.Perplexity())}

        @busy.track("Calculating cluster memberships…")
        @this.extended_task
        async def Calculate(source, options):
            if Module.IS_SHINYLIVE:
                return _analyze(source, **options)
            return await asyncio.to_thread(_analyze, source, **options)

        @this.settle(seconds=1)
        @this.suspendable()
        def StartAnalysis():
            Calculate.invoke(this.input_data().clone(), Options())

        @this.suspendable(calc=True)
        def Analysis():
            result = Calculate.result()
            req(result.source.equals(this.input_data()) and result.options == Options(), cancel_output=True)
            return result

        @this.suspendable()
        def SourceChanged():
            nonlocal reset_pending
            source = this.input_data()
            with reactive.isolate():
                saved = committed.get()
                if saved is not None and not saved[0].equals(source):
                    reset_pending = True
                    committed.set(None)
                    message.set("Incoming data changed; membership addition reset.")
                    ui.update_checkbox_group("IncludeMembership", selected=[])
                saved = committed.get()
                OutputData.set(_export_memberships(source, saved[1]) if saved is not None else source)

        @output
        @render.ui
        def MembershipControl():
            saved = committed.get()
            disabled = (this.input_data().cluster_count or 1) == 1
            control = ui.input_checkbox_group("IncludeMembership", label=None,
                choices={"Partition": "Add Partition members", "Mixture": "Add Mixture members", "Density": "Add DBSCAN members"},
                selected=list(saved[1]) if saved is not None else [], inline=True,
                guide=this, title="Learned membership predictors", position="top",
                text="""
                All membership controls are disabled when incoming K is 1, including DBSCAN, to avoid constant predictors.
                Select any combination of methods, independently of the active chart tab. 
                Membership levels are c1, c2, and so on, with unallocated also available for DBSCAN.
                Each adds a learned nominal Predictor named cluster_partition, cluster_mixture or cluster_density; 
                a numeric suffix avoids existing names. Uncheck a method to remove its column and pipeline step. 
                Each selected recipe stays fixed until unchecked and checked again; incoming data changes clear all selections. 
                Training fits learn their own preprocessing and clusters, then assign all complete rows. 
                Mixture is unavailable with unequal importance weights. DBSCAN assigns the nearest learned core cluster within its 
                fitted radius, otherwise unallocated; missing predictors stay missing. With border assignment disabled, a row must 
                also meet minimum training-neighborhood importance mass. New rows never create or merge clusters.
                """
            )
            return ui.tags.fieldset(control, disabled=disabled, class_="d-flex justify-content-center")

        @this.suspendable()
        def MembershipToggle():
            nonlocal reset_pending
            requested = set(input.IncludeMembership() or []) & {"Partition", "Mixture", "Density"}
            if (this.input_data().cluster_count or 1) == 1:
                with reactive.isolate():
                    committed.set(None)
                    OutputData.set(this.input_data())
                    if requested:
                        ui.update_checkbox_group("IncludeMembership", selected=[])
                reset_pending = False
                return
            saved = committed.get()
            plans = {} if saved is None else dict(saved[1])
            if not requested:
                reset_pending = False
                with reactive.isolate():
                    if saved is not None:
                        committed.set(None)
                        OutputData.set(this.input_data())
                        message.set("")
                return
            if reset_pending or requested == set(plans):
                return
            additions = requested - set(plans)
            result = Analysis() if additions else None
            with reactive.isolate():
                source = this.input_data()
                plans = {method: model for method, model in plans.items() if method in requested}
                errors = []
                for method in ("Partition", "Mixture", "Density"):
                    if method not in additions:
                        continue
                    if result.error or method not in result.transformers:
                        errors.append(f"{method}: {result.error or result.notes.get(method, 'Membership unavailable.')}")
                        continue
                    model = deepcopy(result.transformers[method])
                    used = [*source.columns, *(t.output_column for t in plans.values())]
                    model.set_params(output_column=_membership_name(used, f"cluster_{method.lower()}"))
                    plans[method] = model
                if set(plans) == (set(saved[1]) if saved else set()):
                    # A rejected addition must not rebuild the unchanged control:
                    # rerendering races with the checkbox reset and status message.
                    message.set(" ".join(errors))
                    ui.update_checkbox_group("IncludeMembership", selected=list(plans))
                    return
                try:
                    exported = _export_memberships(source, plans)
                except (ValueError, RuntimeError) as error:
                    message.set(str(error))
                    ui.update_checkbox_group("IncludeMembership", selected=list(saved[1]) if saved else [])
                    return
                committed.set((source.clone(), plans) if plans else None)
                OutputData.set(exported)
                message.set(" ".join(errors))
                if set(plans) != requested:
                    ui.update_checkbox_group("IncludeMembership", selected=list(plans))

        @output
        @render.ui
        def Busy():
            return busy.ui()

        @output
        @render.text
        def Status():
            result = Analysis()
            method = input.ClusterType()
            prefix = f"Incoming K={result.source.cluster_count or 1}. {len(result.positions)} of {len(result.source)} rows analyzed; {len(result.predictors)} numeric predictors. "
            return prefix + (result.error or result.notes.get(method, "")) + (" " + message.get() if message.get() else "")

        def register_chart(method):
            @output(id=f"Chart_{method}")
            @render_widget
            def chart():
                widget = go.FigureWidget(_figure(Analysis(), method, bool(this.isFullScreen())))
                widget._config = {"displayModeBar": bool(this.isFullScreen()), "displaylogo": False}
                return widget
        for method in METHODS:
            register_chart(method)

        @output
        @render.text
        def TableTitle():
            return f"{input.ClusterType()} membership — incoming K={this.input_data().cluster_count or 1}"

        @output
        @render.data_frame
        def Membership():
            result = Analysis()
            labels = result.labels.get(input.ClusterType())
            if labels is None or result.error:
                return render.DataTable(pd.DataFrame({"Status": [result.error or result.notes.get(input.ClusterType(), "Unavailable")]}), height="auto")
            frame = result.source.frame.iloc[result.positions].copy().reset_index(drop=True)
            # Prefix metadata to avoid colliding with any incoming column name.
            frame.columns = [f"Data: {col}" for col in frame.columns]
            frame.insert(0, "Membership", labels)
            frame.insert(0, "Row", result.positions + 1)
            return render.DataTable(frame, width="100%", height="auto")

        return OutputData
    this.server = server
    return this


if Module.running_directly(name=__name__):
    this = instance()
    rng = np.random.default_rng(2025)
    frame = pd.DataFrame(np.vstack([rng.normal(-3, .4, (30, 2)), rng.normal(3, .4, (30, 2))]), columns=["x", "y"])
    this._imports.set(proxy_data(_df=frame, _name="Two groups", _cluster_count=2))
    this.run()
