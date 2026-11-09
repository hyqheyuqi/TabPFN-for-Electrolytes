"""Step 2 new-domain context-sensitivity utilities."""
from __future__ import annotations
from .data_sources import recorded_source_hash, project_directory, ampere_data_directory

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .ampere_conductivity import apply_style
apply_style()
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from analysis.ampere_conductivity import API_MODEL_PATH, N_ESTIMATORS, api_predict, load_release, metrics, write_json, write_table

SEEDS = (100, 200, 300, 400, 500, 600)
CONTEXT_SIZES = (0, 2, 5, 10, 15)
METHODS = ("Random", "Diversity")
ATOL = 1e-12
COLORS = {"Random": "#456990", "Diversity": "#48C0AA", "zero": "#8A8F93", "accent": "#EF767A", "text": "#2F3A40"}
BLUE_CMAP = LinearSegmentedColormap.from_list("step02_blue", ("#EDF5FA", "#CEDFEF", "#92C2DD", "#4995C6", "#1663A9", "#0B3C6F"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def project_paths(root: Path, output: Path | None = None) -> dict[str, Path]:
    project = project_directory()
    out = output or project / "results" / "domain_adaptation"
    return {
        "root": root, "project": project, "data": ampere_data_directory(),
        "step01_data": ampere_data_directory(), "results": out,
        "tables": out / "tables", "figures": out / "figures",
        "source_data": out / "figure_source_data", "configs": out / "configs",
        "logs": out / "logs", "snapshots": out / "snapshots",
    }


def ensure_output(paths: dict[str, Path]) -> None:
    for key in ("tables", "figures", "source_data", "configs", "logs", "snapshots"):
        paths[key].mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, paths: dict[str, Path], name: str, source: pd.DataFrame) -> None:
    fig.savefig(paths["figures"].joinpath(name).with_suffix(".png"), dpi=350, bbox_inches="tight", facecolor="white")
    fig.savefig(paths["figures"].joinpath(name).with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    source.to_csv(paths["source_data"] / f"{name}_source_data.csv", index=False)
    plt.close(fig)


def load_new(paths: dict[str, Path]) -> tuple[pd.DataFrame, np.ndarray]:
    x = pd.read_excel(paths["data"] / "GitHub/Data/Property_Prediction/New_Property_Datasets/Feature_Sigma_e_New.xlsx", sheet_name="Sheet1")
    yf = pd.read_excel(paths["data"] / "GitHub/Data/Property_Prediction/New_Property_Datasets/Label_Sigma_e_New.xlsx", sheet_name="Sheet1")
    if x.shape != (30, 15) or yf.shape != (30, 1):
        raise ValueError(f"Expected new-domain shapes (30,15)/(30,1), found {x.shape}/{yf.shape}")
    y = yf.iloc[:, 0].to_numpy(float)
    if not np.isfinite(x.to_numpy(float)).all() or not np.isfinite(y).all() or (y <= 0).any():
        raise ValueError("New-domain data contain invalid values")
    return x, y


def audit_data(paths: dict[str, Path]) -> None:
    root, data, tables = paths["root"], paths["data"], paths["tables"]
    specs = [
        ("github_new_feature_sigma_e", "github_release", root / "github_ampere/Data/Property_Prediction/New_Property_Datasets/Feature_Sigma_e_New.xlsx", data / "GitHub/Data/Property_Prediction/New_Property_Datasets/Feature_Sigma_e_New.xlsx", "model_features"),
        ("github_new_label_sigma_e", "github_release", root / "github_ampere/Data/Property_Prediction/New_Property_Datasets/Label_Sigma_e_New.xlsx", data / "GitHub/Data/Property_Prediction/New_Property_Datasets/Label_Sigma_e_New.xlsx", "model_labels"),
        ("github_new_data_sheet", "github_release", root / "github_ampere/Data/Property_Prediction/New_Property_Datasets/Data sheet.txt", data / "GitHub/Data/Property_Prediction/New_Property_Datasets/Data sheet.txt", "documentation"),
        ("zenodo_new_sigma_measurements", "zenodo_release", root / "dataset/dataset_zenodo/Measurements_of_Ionic_Conductivity_in_New_Property_Datasets.xlsx", data / "Zenodo/Measurements_of_Ionic_Conductivity_in_New_Property_Datasets.xlsx", "measurement_evidence"),
    ]
    now = datetime.now(timezone.utc).isoformat()
    inventory, copied = [], []
    for dataset_id, source_type, src, dst, role in specs:
        if not dst.is_file():
            raise FileNotFoundError(f"Missing required source/copy: {src} -> {dst}")
        sh, dh = recorded_source_hash(root, src, dst), sha256(dst)
        sheets = rows = cols = None
        if dst.suffix == ".xlsx":
            book = pd.ExcelFile(dst); sheets = len(book.sheet_names)
            frame = pd.read_excel(dst, sheet_name="Sheet1"); rows, cols = frame.shape
        inventory.append({"dataset_id": dataset_id, "source_type": source_type, "source_path": str(src.relative_to(root)), "source_filename": src.name, "destination_path": str(dst.relative_to(root)), "file_role": role, "file_size_bytes": dst.stat().st_size, "source_sha256": sh, "destination_sha256": dh, "hashes_match": sh == dh, "n_sheets": sheets, "n_rows": rows, "n_columns": cols, "notes": "retained copy verified against pre-cleanup source SHA256" if sh == dh else "HASH MISMATCH"})
        copied.append({"copied_at": now, "source_path": str(src.relative_to(root)), "destination_path": str(dst.relative_to(root)), "copy_status": "present", "source_sha256": sh, "destination_sha256": dh, "validation_status": "passed" if sh == dh else "failed"})
    inv = pd.DataFrame(inventory)
    write_table(inv, tables / "data_inventory")
    write_table(pd.DataFrame(copied), tables / "copy_manifest")
    if not inv.hashes_match.all():
        raise ValueError("Step 2 copied source hash mismatch")

    x, y = load_new(paths)
    original_x, _, _, _ = load_release({"data": paths["step01_data"]})
    zen = pd.read_excel(data / "Zenodo/Measurements_of_Ionic_Conductivity_in_New_Property_Datasets.xlsx", sheet_name="Sheet1")
    if zen.shape != (30, 8):
        raise ValueError(f"Unexpected Zenodo new-domain shape: {zen.shape}")
    label_diff = np.max(np.abs(y - zen["Average Value (S/m)"].to_numpy(float)))
    temp_diff = np.max(np.abs(x.iloc[:, 0].to_numpy(float) * 100 - zen["Temperature (K)"].to_numpy(float)))
    salt_diff = np.max(np.abs(x.iloc[:, 1].to_numpy(float) * 1000 - zen["Li+ concentration (mol/m3)"].to_numpy(float)))
    checks = pd.DataFrame([
        {"check": "new_feature_shape", "observed": str(x.shape), "expected": "(30, 15)", "passed": x.shape == (30, 15)},
        {"check": "new_label_shape", "observed": str(y.shape), "expected": "(30,)", "passed": y.shape == (30,)},
        {"check": "feature_names_and_order_match_original", "observed": json.dumps(list(x.columns)), "expected": json.dumps(list(original_x.columns)), "passed": list(x.columns) == list(original_x.columns)},
        {"check": "github_zenodo_label_match", "observed": label_diff, "expected": f"<= {ATOL}", "passed": label_diff <= ATOL},
        {"check": "github_zenodo_temperature_match", "observed": temp_diff, "expected": f"<= {ATOL}", "passed": temp_diff <= ATOL},
        {"check": "github_zenodo_salt_match", "observed": salt_diff, "expected": f"<= {ATOL}", "passed": salt_diff <= ATOL},
    ])
    write_table(checks, tables / "data_audit")
    relationships = pd.DataFrame([
        {"relationship_id": "feature_to_measurement", "github_file": "Feature_Sigma_e_New.xlsx", "zenodo_file": "Measurements_of_Ionic_Conductivity_in_New_Property_Datasets.xlsx", "relationship": "same 30 rows/order; model temperature = K/100; salt = mol/m3/1000", "modeling_role": "GitHub feature workbook is the 15-column model input", "evidence": f"max temperature residual={temp_diff:.3g}; max salt residual={salt_diff:.3g}"},
        {"relationship_id": "label_to_measurement", "github_file": "Label_Sigma_e_New.xlsx", "zenodo_file": "Measurements_of_Ionic_Conductivity_in_New_Property_Datasets.xlsx", "relationship": "GitHub label equals Zenodo Average Value (S/m) in the same row order", "modeling_role": "GitHub label is the target; Zenodo is measurement evidence", "evidence": f"max absolute residual={label_diff:.3g}"},
        {"relationship_id": "original_to_new_domain", "github_file": "ampere_conductivity Feature/Label release copies", "zenodo_file": "new-domain measurement workbook", "relationship": "the original published 40-row training split is always retained and new-domain labels are added as context", "modeling_role": "domain adaptation context experiment", "evidence": "15 feature names and order match exactly"},
    ])
    write_table(relationships, tables / "source_relationships")
    if not checks.passed.all():
        raise ValueError("Step 2 data audit failed")


def kcenter_order(features: pd.DataFrame) -> np.ndarray:
    z = StandardScaler().fit_transform(features.to_numpy(float))
    chosen = [int(np.argmin(np.linalg.norm(z - z.mean(axis=0), axis=1)))]
    while len(chosen) < len(z):
        remaining = np.setdiff1d(np.arange(len(z)), np.asarray(chosen))
        distances = np.linalg.norm(z[remaining, None, :] - z[np.asarray(chosen)][None, :, :], axis=2)
        chosen.append(int(remaining[np.argmax(distances.min(axis=1))]))
    return np.asarray(chosen, int)


def build_design(paths: dict[str, Path], repeats: int) -> pd.DataFrame:
    x, _ = load_new(paths)
    split_rows, selection_rows = [], []
    for repeat in range(repeats):
        split_seed = 20261020 + repeat
        order = np.random.default_rng(split_seed).permutation(len(x))
        pool, test = order[:15], order[15:]
        for idx in range(len(x)):
            split_rows.append({"repeat": repeat, "split_seed": split_seed, "row_index": idx, "split": "context_pool" if idx in set(pool) else "test"})
        random_local = np.random.default_rng(20262020 + repeat).permutation(15)
        diversity_local = kcenter_order(x.iloc[pool])
        for method, local_order in (("Random", random_local), ("Diversity", diversity_local)):
            for rank, local in enumerate(local_order, start=1):
                selection_rows.append({"repeat": repeat, "method": method, "selection_rank": rank, "row_index": int(pool[local]), "split_seed": split_seed, "selection_seed": 20262020 + repeat if method == "Random" else None, "uses_labels": False})
    split = pd.DataFrame(split_rows)
    selection = pd.DataFrame(selection_rows)
    write_table(split, paths["tables"] / "split_manifest")
    write_table(selection, paths["tables"] / "selection_manifest")
    return selection


def _selected(design: pd.DataFrame, repeat: int, method: str, size: int) -> np.ndarray:
    if size == 0:
        return np.array([], dtype=int)
    return design[(design.repeat == repeat) & (design.method == method) & (design.selection_rank <= size)].sort_values("selection_rank").row_index.to_numpy(int)


def run_experiment(paths: dict[str, Path], design: pd.DataFrame, seeds: tuple[int, ...], resume: bool = False) -> None:
    original_x, original_y, original_train, _ = load_release({"data": paths["step01_data"]})
    new_x, new_y = load_new(paths)
    split = pd.read_csv(paths["tables"] / "split_manifest.csv")
    pred_path = paths["tables"] / "predictions.csv"
    existing = pd.read_csv(pred_path) if resume and pred_path.exists() else pd.DataFrame()
    rows = existing.to_dict("records") if not existing.empty else []
    completed = set()
    if not existing.empty:
        completed = set(zip(existing.repeat.astype(int), existing.method, existing.context_size.astype(int), existing.seed.astype(int)))
    cache: dict[tuple[int, tuple[int, ...], int], np.ndarray] = {}
    # Rebuild the de-duplication cache from incrementally saved predictions so
    # resume never re-calls an already completed equivalent endpoint/config.
    if not existing.empty:
        for (repeat, method, size, seed), group in existing.groupby(["repeat", "method", "context_size", "seed"]):
            selected = _selected(design, int(repeat), str(method), int(size))
            canonical = tuple(sorted(selected.tolist()))
            cache[(int(repeat), canonical, int(seed))] = group.sort_values("row_index").Sigma_e_pred.to_numpy(float)
    for repeat in sorted(split.repeat.unique()):
        test = split[(split.repeat == repeat) & (split.split == "test")].row_index.to_numpy(int)
        for size in CONTEXT_SIZES:
            for method in METHODS:
                selected = _selected(design, repeat, method, size)
                canonical = tuple(sorted(selected.tolist()))
                train_x = pd.concat([original_x.iloc[original_train], new_x.iloc[list(canonical)]], ignore_index=True)
                train_y = np.r_[original_y[original_train], new_y[list(canonical)]]
                for seed in seeds:
                    group_key = (int(repeat), method, size, int(seed))
                    if group_key in completed:
                        continue
                    key = (int(repeat), canonical, int(seed))
                    ordered_test = np.sort(test)
                    if key not in cache:
                        cache[key] = api_predict(train_x, train_y, new_x.iloc[ordered_test], int(seed), paths["root"])
                    # All stored/reused vectors are aligned to sorted test row index.
                    if key in cache and len(cache[key]) == len(ordered_test):
                        values = cache[key]
                    else:
                        raise ValueError("Cached prediction length does not match test split")
                    for idx, value in zip(ordered_test, values):
                        true = float(new_y[idx]); err = float(value - true)
                        rows.append({"repeat": int(repeat), "method": method, "context_size": size, "seed": int(seed), "row_index": int(idx), "Sigma_e_true": true, "Sigma_e_pred": float(value), "signed_error": err, "absolute_error": abs(err), "relative_error": err / true, "absolute_percentage_error": abs(err / true) * 100, "squared_error": err * err})
                    pd.DataFrame(rows).to_csv(pred_path, index=False)
    predictions = pd.DataFrame(rows).sort_values(["repeat", "context_size", "method", "seed", "row_index"]).reset_index(drop=True)
    write_table(predictions, paths["tables"] / "predictions")

    metric_rows = []
    for keys, g in predictions.groupby(["repeat", "method", "context_size", "seed"], sort=True):
        metric_rows.append(dict(zip(["repeat", "method", "context_size", "seed"], keys)) | metrics(g.Sigma_e_true.to_numpy(), g.Sigma_e_pred.to_numpy()))
    metric_frame = pd.DataFrame(metric_rows)
    write_table(metric_frame, paths["tables"] / "metrics")

    mean_pred = predictions.groupby(["repeat", "method", "context_size", "row_index"], as_index=False).agg(Sigma_e_true=("Sigma_e_true", "first"), Sigma_e_pred=("Sigma_e_pred", "mean"), prediction_seed_std=("Sigma_e_pred", "std"))
    repeat_metrics = []
    for keys, g in mean_pred.groupby(["repeat", "method", "context_size"], sort=True):
        repeat_metrics.append(dict(zip(["repeat", "method", "context_size"], keys)) | metrics(g.Sigma_e_true.to_numpy(), g.Sigma_e_pred.to_numpy()))
    repeat_metrics = pd.DataFrame(repeat_metrics)
    summary = repeat_metrics.groupby(["context_size", "method"])[["MAPE", "MAE", "RMSE", "R2", "median_absolute_error", "median_APE", "APE_q75", "APE_q90", "APE_q95"]].agg(["mean", "std", "median", "min", "max"]).reset_index()
    summary.columns = ["_".join(c).rstrip("_") for c in summary.columns]
    write_table(summary, paths["tables"] / "summary")

    paired = []
    for metric in ("MAPE", "MAE", "RMSE", "R2"):
        pivot = repeat_metrics.pivot(index=["repeat", "context_size"], columns="method", values=metric)
        for (repeat, size), row in pivot.iterrows():
            diff = float(row["Diversity"] - row["Random"])
            better = diff < 0 if metric != "R2" else diff > 0
            paired.append({"repeat": repeat, "context_size": size, "metric": metric, "random_value": row["Random"], "diversity_value": row["Diversity"], "diversity_minus_random": diff, "outcome": "diversity_win" if better else ("tie" if abs(diff) <= 1e-12 else "random_win")})
    write_table(pd.DataFrame(paired), paths["tables"] / "random_vs_diversity")

    freq_rows = []
    for method in METHODS:
        for size in CONTEXT_SIZES[1:]:
            for idx in range(30):
                in_pool = 0; selected_count = 0
                for repeat in sorted(split.repeat.unique()):
                    pool = set(split[(split.repeat == repeat) & (split.split == "context_pool")].row_index)
                    in_pool += int(idx in pool)
                    selected_count += int(idx in set(_selected(design, repeat, method, size)))
                freq_rows.append({"method": method, "context_size": size, "row_index": idx, "times_in_context_pool": in_pool, "times_selected": selected_count, "conditional_selection_rate": selected_count / in_pool if in_pool else np.nan})
    write_table(pd.DataFrame(freq_rows), paths["tables"] / "selection_frequency")

    variance = []
    for (method, size), g in metric_frame.groupby(["method", "context_size"]):
        for metric in ("MAPE", "MAE", "RMSE", "R2"):
            repeat_means = g.groupby("repeat")[metric].mean()
            within = g.groupby("repeat")[metric].var(ddof=1).mean()
            variance.append({"method": method, "context_size": size, "metric": metric, "outer_split_variance": float(repeat_means.var(ddof=1)), "mean_within_split_seed_variance": float(within), "variance_ratio_outer_to_seed": float(repeat_means.var(ddof=1) / within) if within > 0 else np.inf})
    write_table(pd.DataFrame(variance), paths["tables"] / "variance_decomposition")

    endpoints = []
    for size in (0, 15):
        a = predictions[(predictions.method == "Random") & (predictions.context_size == size)].sort_values(["repeat", "seed", "row_index"])
        b = predictions[(predictions.method == "Diversity") & (predictions.context_size == size)].sort_values(["repeat", "seed", "row_index"])
        endpoints.append({"context_size": size, "same_keys": a[["repeat", "seed", "row_index"]].reset_index(drop=True).equals(b[["repeat", "seed", "row_index"]].reset_index(drop=True)), "max_prediction_difference": float(np.max(np.abs(a.Sigma_e_pred.to_numpy() - b.Sigma_e_pred.to_numpy()))), "passed": bool(np.allclose(a.Sigma_e_pred, b.Sigma_e_pred, atol=0, rtol=0))})
    write_json(paths["tables"] / "endpoint_equivalence_audit.json", {"status": "passed" if all(r["passed"] for r in endpoints) else "failed", "checks": endpoints})
    if not all(r["passed"] for r in endpoints):
        raise RuntimeError("Context-size 0/15 equivalence check failed")


def plot_results(paths: dict[str, Path]) -> None:
    plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.spines.top": True, "axes.spines.right": True, "axes.labelcolor": COLORS["text"], "text.color": COLORS["text"], "xtick.color": COLORS["text"], "ytick.color": COLORS["text"]})
    pred = pd.read_csv(paths["tables"] / "predictions.csv")
    metrics_seed = pd.read_csv(paths["tables"] / "metrics.csv")
    paired = pd.read_csv(paths["tables"] / "random_vs_diversity.csv")
    freq = pd.read_csv(paths["tables"] / "selection_frequency.csv")
    variance = pd.read_csv(paths["tables"] / "variance_decomposition.csv")
    design = pd.read_csv(paths["tables"] / "selection_manifest.csv")
    split = pd.read_csv(paths["tables"] / "split_manifest.csv")
    x, _ = load_new(paths)
    mean_pred = pred.groupby(["repeat", "method", "context_size", "row_index"], as_index=False).agg(Sigma_e_true=("Sigma_e_true", "first"), Sigma_e_pred=("Sigma_e_pred", "mean"))
    repeat_metrics = []
    for keys, g in mean_pred.groupby(["repeat", "method", "context_size"]):
        repeat_metrics.append(dict(zip(["repeat", "method", "context_size"], keys)) | metrics(g.Sigma_e_true, g.Sigma_e_pred))
    rm = pd.DataFrame(repeat_metrics)

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for method, g in rm.groupby("method"):
        s = g.groupby("context_size").MAPE.agg(["mean", "std"]).reset_index(); ax.plot(s.context_size, s["mean"], "o-", color=COLORS[method], label=method, lw=1.5, ms=4); ax.fill_between(s.context_size, s["mean"] - s["std"], s["mean"] + s["std"], color=COLORS[method], alpha=.16)
    ax.set(xlabel="New-domain context labels", ylabel="MAPE [%]", xticks=CONTEXT_SIZES); ax.legend(frameon=False)
    save_figure(fig, paths, "Fig01_mape_context_curve", rm[["repeat", "method", "context_size", "MAPE"]])

    fig, axs = plt.subplots(1, 3, figsize=(8.6, 2.7))
    for ax, metric in zip(axs, ("MAE", "RMSE", "R2")):
        for method, g in rm.groupby("method"):
            s = g.groupby("context_size")[metric].agg(["mean", "std"]).reset_index(); ax.plot(s.context_size, s["mean"], "o-", color=COLORS[method], label=method, lw=1.3, ms=3); ax.fill_between(s.context_size, s["mean"] - s["std"], s["mean"] + s["std"], color=COLORS[method], alpha=.14)
        ax.set(xlabel="Context labels", ylabel=metric, xticks=CONTEXT_SIZES)
    axs[-1].legend(frameon=False, fontsize=7)
    save_figure(fig, paths, "Fig02_other_metrics_context_curves", rm[["repeat", "method", "context_size", "MAE", "RMSE", "R2"]])

    fig, ax = plt.subplots(figsize=(6.6, 3.0)); positions=[]; data=[]; box_colors=[]; labels=[]
    pos=0
    for size in CONTEXT_SIZES:
        for method in METHODS:
            data.append(rm[(rm.context_size == size) & (rm.method == method)].MAPE); positions.append(pos); box_colors.append(COLORS[method]); labels.append(f"{size}\n{method[0]}"); pos += 1
        pos += .5
    boxes=ax.boxplot(data,positions=positions,widths=.65,patch_artist=True,showfliers=False)
    for patch,color in zip(boxes["boxes"],box_colors): patch.set(facecolor=color,alpha=.55,edgecolor=color)
    ax.set(xticks=positions,xticklabels=labels,xlabel="Context size and method (R/D)",ylabel="MAPE [%]")
    save_figure(fig, paths, "Fig03_method_distributions", rm[["repeat", "method", "context_size", "MAPE"]])

    p=paired[paired.metric=="MAPE"]; fig,ax=plt.subplots(figsize=(4.3,2.8));
    for size,g in p.groupby("context_size"):
        ax.scatter(np.full(len(g),size)+(np.arange(len(g))-len(g)/2)*.025,g.diversity_minus_random,s=22,color=COLORS["Diversity"],edgecolors="white",linewidths=.4,alpha=.75)
    means=p.groupby("context_size").diversity_minus_random.mean(); ax.plot(means.index,means.values,"o-",color="#1663A9",lw=1.4,label="Mean paired difference"); ax.axhline(0,color="#8A8F93",ls="--",lw=1); ax.set(xlabel="Context labels",ylabel="Diversity − Random MAPE",xticks=CONTEXT_SIZES); ax.legend(frameon=False,fontsize=7)
    save_figure(fig, paths, "Fig04_paired_method_differences", p)

    z=StandardScaler().fit_transform(x); pc=PCA(n_components=2).fit_transform(z); base=pd.DataFrame({"row_index":np.arange(30),"PC1":pc[:,0],"PC2":pc[:,1]}); rep=0; pool=set(split[(split.repeat==rep)&(split.split=="context_pool")].row_index); test=set(range(30))-pool
    fig,axs=plt.subplots(1,2,figsize=(7.0,3.0),sharex=True,sharey=True); src=[]
    for ax,method in zip(axs,METHODS):
        selected=set(_selected(design,rep,method,10)); q=base.copy(); q["repeat"]=rep; q["method"]=method; q["role"]=["selected" if i in selected else ("pool_not_selected" if i in pool else "test") for i in q.row_index]; src.append(q)
        for role,color,marker,size in (("test","#BFC5C9","o",22),("pool_not_selected","#CEDFEF","o",25),("selected",COLORS[method],"o",40)):
            g=q[q.role==role]; ax.scatter(g.PC1,g.PC2,s=size,color=color,marker=marker,edgecolors="white",linewidths=.5,label=role,alpha=.9)
        ax.set(title=f"{method}, k=10",xlabel="PCA 1",ylabel="PCA 2")
    axs[-1].legend(frameon=False,fontsize=6)
    save_figure(fig, paths, "Fig05_feature_space_coverage", pd.concat(src,ignore_index=True))

    q=base.merge(design[(design.repeat==0)&(design.method=="Diversity")][["row_index","selection_rank"]],on="row_index",how="left"); fig,ax=plt.subplots(figsize=(4.2,3.2)); sc=ax.scatter(q.PC1,q.PC2,c=q.selection_rank,cmap=BLUE_CMAP,s=42,edgecolors="white",linewidths=.5); fig.colorbar(sc,ax=ax,label="Diversity selection order");
    for r in q.dropna().itertuples(): ax.text(r.PC1,r.PC2,str(int(r.selection_rank)),fontsize=6,ha="center",va="center")
    ax.set(xlabel="PCA 1",ylabel="PCA 2"); save_figure(fig, paths, "Fig06_diversity_selection_order", q)

    div=mean_pred[mean_pred.method=="Diversity"]; fig,axs=plt.subplots(1,5,figsize=(10,2.3),sharex=True,sharey=True); lim=[min(div.Sigma_e_true.min(),div.Sigma_e_pred.min()),max(div.Sigma_e_true.max(),div.Sigma_e_pred.max())]; palette=("#CEDFEF","#92C2DD","#4995C6","#1663A9","#614099")
    for ax,(size,g),color in zip(axs,div.groupby("context_size"),palette): ax.plot(lim,lim,"--",color="#A5ADB2",lw=.8); ax.scatter(g.Sigma_e_true,g.Sigma_e_pred,s=13,color=color,edgecolors="white",linewidths=.3,alpha=.62); ax.set_title(f"k={size}",color=color,fontweight="bold"); ax.set_xlabel("True")
    axs[0].set_ylabel("Predicted"); save_figure(fig, paths, "Fig07_context_parity_panels", div)

    ape=mean_pred.assign(APE=lambda d:abs(d.Sigma_e_pred-d.Sigma_e_true)/d.Sigma_e_true*100); zero=ape[ape.context_size==0][["repeat","method","row_index","APE"]].rename(columns={"APE":"zero_APE"}); imp=ape.merge(zero,on=["repeat","method","row_index"]); imp["APE_improvement_vs_zero"]=imp.zero_APE-imp.APE; heat=imp[imp.method=="Diversity"].groupby(["row_index","context_size"]).APE_improvement_vs_zero.mean().unstack(); source=heat.reset_index().melt(id_vars="row_index",var_name="context_size",value_name="mean_APE_improvement_vs_zero"); fig,ax=plt.subplots(figsize=(5.4,3.5)); vmax=np.nanquantile(np.abs(source.mean_APE_improvement_vs_zero),.95); im=ax.imshow(heat,aspect="auto",cmap="RdBu",vmin=-vmax,vmax=vmax); fig.colorbar(im,ax=ax,label="APE improvement vs k=0 [% points]"); ax.set(xlabel="Context size",ylabel="New-domain row index",xticks=range(len(heat.columns)),xticklabels=heat.columns); save_figure(fig, paths, "Fig08_sample_improvement_heatmap", source)

    f=freq[freq.context_size==10]; fig,ax=plt.subplots(figsize=(6.8,2.8)); xpos=np.arange(30); width=.38
    for j,method in enumerate(METHODS): g=f[f.method==method].set_index("row_index").reindex(range(30)); ax.bar(xpos+(j-.5)*width,g.times_selected,width,color=COLORS[method],alpha=.82,label=method,edgecolor="white",linewidth=.4)
    ax.set(xlabel="New-domain row index",ylabel="Times selected at k=10"); ax.legend(frameon=False); save_figure(fig, paths, "Fig09_selection_frequency", f)

    v=variance[variance.metric=="MAPE"]; fig,axs=plt.subplots(1,2,figsize=(7.2,2.8),sharey=True)
    for ax,method in zip(axs,METHODS): g=v[v.method==method].sort_values("context_size"); ax.bar(g.context_size-.35/2,g.outer_split_variance,.35,color="#4995C6",label="Outer split",edgecolor="white"); ax.bar(g.context_size+.35/2,g.mean_within_split_seed_variance,.35,color="#BAB4D5",label="Inference seed",edgecolor="white"); ax.set(title=method,xlabel="Context labels",ylabel="MAPE variance",xticks=CONTEXT_SIZES)
    axs[-1].legend(frameon=False,fontsize=7); save_figure(fig, paths, "Fig10_variance_decomposition", v)


def rebuild_figures(root: Path, output: Path | None = None) -> None:
    """Redraw Step 2 using saved results; no inference."""
    plot_results(project_paths(root, output))
