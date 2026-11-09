"""Step 1 data audit, fixed-split reproduction, and learning-curve utilities."""
from __future__ import annotations
from .data_sources import recorded_source_hash, project_directory, ampere_data_directory

import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def apply_style():
    """Apply the shared publication font, embedding, and four-sided axes."""
    from matplotlib import font_manager
    # Register macOS supplemental fonts explicitly, including on cached installs.
    for path in Path('/System/Library/Fonts/Supplemental').glob('Arial*.ttf'):
        font_manager.fontManager.addfont(str(path))
    matplotlib.rcParams.update({
        'font.family': 'Arial', 'mathtext.fontset': 'custom',
        'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic',
        'mathtext.bf': 'Arial:bold', 'mathtext.cal': 'Arial',
        'mathtext.sf': 'Arial', 'mathtext.tt': 'Arial', 'axes.spines.top': True,
        'axes.spines.right': True, 'axes.spines.bottom': True,
        'axes.spines.left': True, 'axes.linewidth': 0.8,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })


apply_style()

SEEDS = (100, 200, 300, 400, 500, 600)
CONTEXT_SIZES = (10, 15, 20, 30, 40)
MODEL_NAMES = ("TabPFN", "XGBoost", "Random Forest", "GPR", "Elastic Net")
API_MODEL_PATH = "tabpfn-v3-regressor-v3_default.ckpt"
N_ESTIMATORS = 8
ATOL = 1e-12
RTOL = 1e-10
REPRO_METRIC_ATOL = 0.02

COLORS = {
    "TabPFN": "#527C8E", "XGBoost": "#7FA6B7", "Random Forest": "#7E9B76",
    "GPR": "#C49A6C", "Elastic Net": "#9583A6", "train": "#527C8E",
    "test": "#B86F6F", "neutral": "#8A8F93", "text": "#2F3A40",
}
USER_TRIAD = ("#EF767A", "#456990", "#48C0AA")
BLUE_PURPLE = ("#CEDFEF", "#92C2DD", "#4995C6", "#1663A9", "#BAB4D5", "#8481BA", "#614099")
BLUE_CMAP = LinearSegmentedColormap.from_list(
    "step01_blue", ("#EDF5FA", "#CEDFEF", "#92C2DD", "#4995C6", "#1663A9", "#0B3C6F")
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def write_table(frame: pd.DataFrame, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(stem.with_suffix(".csv"), index=False)
    frame.to_excel(stem.with_suffix(".xlsx"), index=False)


def save_figure(fig: plt.Figure, stem: Path, source: pd.DataFrame) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=350, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    source.to_csv(stem.parent.parent / "figure_source_data" / f"{stem.name}_source_data.csv", index=False)
    plt.close(fig)


def project_paths(root: Path, output: Path | None = None) -> dict[str, Path]:
    project = project_directory()
    out = output or project / "results" / "ampere_conductivity"
    return {
        "root": root, "project": project, "data": ampere_data_directory(),
        "results": out, "tables": out / "tables", "figures": out / "figures",
        "source_data": out / "figure_source_data", "configs": out / "configs",
        "logs": out / "logs", "snapshots": out / "snapshots",
        "legacy": project / "reference_runs" / "20261001_Step01-07",
    }


def ensure_output(paths: dict[str, Path]) -> None:
    for key in ("tables", "figures", "source_data", "configs", "logs", "snapshots"):
        paths[key].mkdir(parents=True, exist_ok=True)


def load_release(paths: dict[str, Path]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    data = paths["data"]
    x = pd.read_excel(data / "GitHub/Data/Property_Prediction/Original_Property_Datasets/AMPERE_Data/Feature_Sigma_e.xlsx", sheet_name="Sheet1")
    labels = pd.read_excel(data / "GitHub/Data/Property_Prediction/Original_Property_Datasets/AMPERE_Data/Label_Sigma_e.xlsx", sheet_name="Sheet1")
    raw_index = pd.read_excel(data / "GitHub/Data/Property_Prediction/Original_Property_Datasets/AMPERE_Data/Training_Index_AL/Sigma_e/Train_index_46.xlsx", sheet_name="Sheet1").iloc[:, 0]
    if x.shape != (100, 15):
        raise ValueError(f"Feature_Sigma_e expected (100, 15), found {x.shape}")
    if labels.shape != (100, 1):
        raise ValueError(f"Label_Sigma_e expected (100, 1), found {labels.shape}")
    vals = raw_index.to_numpy(dtype=float)
    if len(vals) != 40 or not np.isfinite(vals).all() or not np.equal(vals, np.floor(vals)).all():
        raise ValueError("Training indexes must be exactly 40 finite integers")
    train = vals.astype(int)
    if len(np.unique(train)) != 40 or (train < 0).any() or (train >= 100).any():
        raise ValueError("Training indexes are duplicated or out of range")
    test = np.setdiff1d(np.arange(100), train)
    if len(test) != 60 or np.intersect1d(train, test).size or len(np.union1d(train, test)) != 100:
        raise ValueError("Invalid reconstructed 40/60 split")
    y = labels.iloc[:, 0].to_numpy(dtype=float)
    if not np.isfinite(x.to_numpy(dtype=float)).all() or not np.isfinite(y).all() or (y <= 0).any():
        raise ValueError("Features/labels contain invalid values")
    return x, y, train, test


def _xlsx_shape(path: Path) -> tuple[int, int, int]:
    book = pd.ExcelFile(path)
    nonempty = [pd.read_excel(path, sheet_name=s) for s in book.sheet_names]
    frame = next((f for f in nonempty if f.shape != (0, 0)), pd.DataFrame())
    return len(book.sheet_names), frame.shape[0], frame.shape[1]


def audit_data(paths: dict[str, Path]) -> None:
    root, data, tables = paths["root"], paths["data"], paths["tables"]
    specs = [
        ("github_feature_sigma_e", "github_release", root / "github_ampere/Data/Property_Prediction/Original_Property_Datasets/AMPERE_Data/Feature_Sigma_e.xlsx", data / "GitHub/Data/Property_Prediction/Original_Property_Datasets/AMPERE_Data/Feature_Sigma_e.xlsx", "model_features"),
        ("github_label_sigma_e", "github_release", root / "github_ampere/Data/Property_Prediction/Original_Property_Datasets/AMPERE_Data/Label_Sigma_e.xlsx", data / "GitHub/Data/Property_Prediction/Original_Property_Datasets/AMPERE_Data/Label_Sigma_e.xlsx", "model_labels"),
        ("github_train_index_46", "github_release", root / "github_ampere/Data/Property_Prediction/Original_Property_Datasets/AMPERE_Data/Training_Index_AL/Sigma_e/Train_index_46.xlsx", data / "GitHub/Data/Property_Prediction/Original_Property_Datasets/AMPERE_Data/Training_Index_AL/Sigma_e/Train_index_46.xlsx", "split_source"),
        ("github_ampere_data_sheet", "github_release", root / "github_ampere/Data/Property_Prediction/Original_Property_Datasets/AMPERE_Data/Data sheet.txt", data / "GitHub/Data/Property_Prediction/Original_Property_Datasets/AMPERE_Data/Data sheet.txt", "documentation"),
        ("github_sigma_split_sheet", "github_release", root / "github_ampere/Data/Property_Prediction/Original_Property_Datasets/AMPERE_Data/Training_Index_AL/Sigma_e/Data sheet.txt", data / "GitHub/Data/Property_Prediction/Original_Property_Datasets/AMPERE_Data/Training_Index_AL/Sigma_e/Data sheet.txt", "documentation"),
        ("zenodo_original_sigma_measurements", "zenodo_release", root / "dataset/dataset_zenodo/Measurements_of_Ionic_Conductivity_in_Original_Property_Datasets.xlsx", data / "Zenodo/Measurements_of_Ionic_Conductivity_in_Original_Property_Datasets.xlsx", "measurement_evidence"),
    ]
    now = datetime.now(timezone.utc).isoformat()
    inventory, manifest = [], []
    for dataset_id, source_type, src, dst, role in specs:
        if not dst.is_file():
            raise FileNotFoundError(f"Required source/destination missing: {src} -> {dst}")
        sh, dh = recorded_source_hash(root, src, dst), sha256(dst)
        ns = nr = nc = None
        if dst.suffix.lower() == ".xlsx":
            ns, nr, nc = _xlsx_shape(dst)
        inventory.append({"dataset_id": dataset_id, "source_type": source_type,
            "source_path": str(src.relative_to(root)), "source_filename": src.name,
            "destination_path": str(dst.relative_to(root)), "file_role": role,
            "file_size_bytes": dst.stat().st_size, "source_sha256": sh,
            "destination_sha256": dh, "hashes_match": sh == dh, "n_sheets": ns,
            "n_rows": nr, "n_columns": nc, "notes": "retained copy verified against pre-cleanup source SHA256" if sh == dh else "HASH MISMATCH"})
        manifest.append({"copied_at": now, "source_path": str(src.relative_to(root)),
            "destination_path": str(dst.relative_to(root)), "copy_status": "present",
            "source_sha256": sh, "destination_sha256": dh,
            "validation_status": "passed" if sh == dh else "failed"})
    inv = pd.DataFrame(inventory); man = pd.DataFrame(manifest)
    write_table(inv, tables / "data_inventory")
    write_table(man, tables / "copy_manifest")
    write_table(inv, tables / "data_audit")
    if not inv.hashes_match.all():
        raise ValueError("One or more copied files failed SHA256 validation")

    x, y, train, test = load_release(paths)
    zen = pd.read_excel(data / "Zenodo/Measurements_of_Ionic_Conductivity_in_Original_Property_Datasets.xlsx")
    if zen.shape != (100, 8):
        raise ValueError(f"Unexpected Zenodo measurement shape {zen.shape}")
    label_diff = np.abs(y - zen["Average Value (S/m)"].to_numpy(float))
    temp_diff = np.abs(x.iloc[:, 0].to_numpy(float) * 100 - zen["Temperature (K)"].to_numpy(float))
    salt_diff = np.abs(x.iloc[:, 1].to_numpy(float) * 1000 - zen["Li+ concentration (mol/m3)"].to_numpy(float))
    if label_diff.max() > ATOL or temp_diff.max() > ATOL or salt_diff.max() > ATOL:
        raise ValueError("GitHub and Zenodo values/order/units differ beyond tolerance")
    relationships = pd.DataFrame([
        {"relationship_id": "feature_to_measurement", "github_file": "Feature_Sigma_e.xlsx", "zenodo_file": "Measurements_of_Ionic_Conductivity_in_Original_Property_Datasets.xlsx", "relationship": "same 100 rows/order; GitHub model feature temperature = Zenodo K / 100 and salt concentration = Zenodo mol/m3 / 1000", "modeling_role": "GitHub file is the 15-column model input", "evidence": f"max temperature residual={temp_diff.max():.3g}; max salt residual={salt_diff.max():.3g}"},
        {"relationship_id": "label_to_measurement", "github_file": "Label_Sigma_e.xlsx", "zenodo_file": "Measurements_of_Ionic_Conductivity_in_Original_Property_Datasets.xlsx", "relationship": "GitHub Ionic Conductivity [S/m] equals Zenodo Average Value (S/m), same row order", "modeling_role": "GitHub file is model target; Zenodo workbook is experimental measurement evidence", "evidence": f"max absolute residual={label_diff.max():.3g}"},
        {"relationship_id": "index_to_rows", "github_file": "Train_index_46.xlsx", "zenodo_file": "", "relationship": "40 unique zero-based row indexes select training rows; complement of 0..99 is the fixed 60-row test set", "modeling_role": "published split source", "evidence": f"n_train={len(train)}; n_test={len(test)}; disjoint={np.intersect1d(train,test).size == 0}"},
    ])
    write_table(relationships, tables / "source_relationships")


def reconstruct_and_compare(paths: dict[str, Path]) -> None:
    x, y, train, test = load_release(paths)
    tables, legacy = paths["tables"], paths["legacy"]
    split_manifest = pd.DataFrame({"row_index": np.arange(100), "split": np.where(np.isin(np.arange(100), train), "train", "test"), "published_train_order": pd.Series({int(v): i for i, v in enumerate(train)}).reindex(np.arange(100)).to_numpy()})
    write_table(split_manifest, tables / "fixed_split_manifest")
    derived = {}
    for name, idx in (("train", train), ("test", test)):
        frame = x.iloc[idx].copy(); frame.insert(0, "row_index", idx); frame["Sigma_e"] = y[idx]
        derived[name] = frame
        write_table(frame, tables / f"reconstructed_{name}")
    legacy_train_idx = pd.read_excel(legacy / "Splits/Sigma_e/Train_index_46.xlsx").iloc[:, 0].to_numpy(int)
    legacy_test_idx = pd.read_excel(legacy / "Splits/Sigma_e/Test_index.xlsx").iloc[:, 0].to_numpy(int)
    comparison = []
    comparisons_ok = True
    for name, idx, legacy_idx in (("train", train, legacy_train_idx), ("test", test, legacy_test_idx)):
        lf = pd.read_csv(legacy / f"Splits/Sigma_e/{name}.csv")
        index_equal = np.array_equal(idx, legacy_idx)
        cols_equal = list(x.columns) == list(lf.columns[1:16])
        feature_diff = float(np.max(np.abs(x.iloc[idx].to_numpy(float) - lf[x.columns].to_numpy(float))))
        label_diff = float(np.max(np.abs(y[idx] - lf["y_original"].to_numpy(float))))
        ok = index_equal and cols_equal and feature_diff <= ATOL and label_diff <= ATOL
        comparisons_ok &= ok
        comparison.extend([
            {"split": name, "check": "indexes_exact_in_order", "passed": index_equal, "max_abs_difference": 0 if index_equal else np.nan, "tolerance": 0, "details": f"new={len(idx)}, legacy={len(legacy_idx)}"},
            {"split": name, "check": "feature_names_and_order", "passed": cols_equal, "max_abs_difference": np.nan, "tolerance": 0, "details": "15 published feature columns"},
            {"split": name, "check": "feature_values", "passed": feature_diff <= ATOL, "max_abs_difference": feature_diff, "tolerance": ATOL, "details": "all values compared in row/index order"},
            {"split": name, "check": "labels_and_units_S_per_m", "passed": label_diff <= ATOL, "max_abs_difference": label_diff, "tolerance": ATOL, "details": "Sigma_e / y_original"},
        ])
    write_table(pd.DataFrame(comparison), tables / "split_comparison_with_legacy")
    audit = {"status": "passed" if comparisons_ok else "failed", "atol": ATOL, "rtol": RTOL,
        "train_index_exact_order": bool(np.array_equal(train, legacy_train_idx)),
        "test_index_exact_order": bool(np.array_equal(test, legacy_test_idx)),
        "n_train": len(train), "n_test": len(test), "index_base": "zero-based 0..99",
        "legacy_usage": "read-only comparison; no Legacy file copied or used as model input"}
    write_json(tables / "split_reconstruction_audit.json", audit)
    if not comparisons_ok:
        raise ValueError("Reconstructed split differs from Legacy; see comparison table")


def audit_legacy(paths: dict[str, Path]) -> None:
    legacy, tables = paths["legacy"], paths["tables"]
    config = json.loads((legacy / "run_config.json").read_text())
    metrics = pd.read_csv(legacy / "metrics_all_seeds.csv")
    summary = pd.read_csv(legacy / "summary.csv")
    pred = pd.read_csv(legacy / "test_predictions.csv")
    source = (paths["project"] / "reference_runs/run_step01_07.py").read_text()
    snapshot = (legacy / "source_snapshot.py").read_text()
    source_snapshot_exact = source == snapshot
    path_only_difference = (
        not source_snapshot_exact
        and source.replace("github_ampere/Data/", "github/Data/") == snapshot
    )
    if not source_snapshot_exact and not path_only_difference:
        raise ValueError("Legacy source and saved source snapshot have substantive differences")
    sigma_metrics = metrics[(metrics.task == "Sigma_e") & (metrics.method == "v3_api_raw_e8")].copy()
    sigma_summary = summary[(summary.task == "Sigma_e") & (summary.method == "v3_api_raw_e8")].copy()
    sigma_pred = pred[(pred.task == "Sigma_e") & (pred.method == "v3_api_raw_e8")]
    expected = {
        "backend": "api", "model_path": API_MODEL_PATH, "tabpfn_version": "9.0.0",
        "n_estimators": 8, "n_features": 15, "target": "raw Sigma_e [S/m]",
        "log_transform": False, "target_scaling": False, "feature_selection": "all 15 retained in original order",
        "local_residual_correction": False, "inference_seeds": list(SEEDS),
        "constructor": {"model_path": API_MODEL_PATH, "n_estimators": 8, "random_state": "seed"},
        "prediction_postprocessing": "none",
    }
    checks = [
        ("backend", config["backend"]["Sigma_e"] == "api", config["backend"]["Sigma_e"], "api"),
        ("model_path", config["model_path"]["Sigma_e"] == API_MODEL_PATH, config["model_path"]["Sigma_e"], API_MODEL_PATH),
        ("tabpfn_package_version", config["packages"]["tabpfn"] == "9.0.0", config["packages"]["tabpfn"], "9.0.0"),
        ("n_estimators", config["conductivity_n_estimators"] == 8, config["conductivity_n_estimators"], 8),
        ("n_features", config["n_features"]["Sigma_e"] == 15, config["n_features"]["Sigma_e"], 15),
        ("seeds", config["seeds"] == list(SEEDS), json.dumps(config["seeds"]), json.dumps(SEEDS)),
        ("raw_target", "conductivity raw" in config["target"], config["target"], "conductivity raw"),
        ("six_seed_metric_rows", len(sigma_metrics) == 6, len(sigma_metrics), 6),
        ("prediction_rows", len(sigma_pred) == 360, len(sigma_pred), 360),
        ("historical_snapshot_authority", source_snapshot_exact or path_only_difference,
         "identical" if source_snapshot_exact else "path-only rename: github -> github_ampere",
         "historical source_snapshot.py is authoritative; current source may differ only by data-directory rename"),
        ("no_local_correction", (sigma_pred.delta_log == 0).all(), float(sigma_pred.delta_log.abs().max()), 0),
    ]
    audit = pd.DataFrame([{"item": item, "passed": bool(ok), "observed": observed, "expected": expected_value,
        "evidence": "Legacy run_config + historical source_snapshot + saved results"} for item, ok, observed, expected_value in checks])
    write_table(audit, tables / "legacy_model_audit")
    write_table(sigma_metrics, tables / "legacy_metrics_reference")
    write_json(tables / "legacy_reproduction_target.json", expected)
    if not audit.passed.all() or sigma_summary.empty:
        raise ValueError("Legacy source/config/results conflict; see legacy model audit")


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ae = np.abs(y_pred - y_true); ape = ae / np.abs(y_true) * 100
    return {"MAPE": float(ape.mean()), "MAE": float(ae.mean()),
        "RMSE": float(np.sqrt(np.mean((y_pred-y_true)**2))), "R2": float(r2_score(y_true, y_pred)),
        "median_absolute_error": float(np.median(ae)), "median_APE": float(np.median(ape)),
        "APE_q75": float(np.quantile(ape, .75)), "APE_q90": float(np.quantile(ape, .90)),
        "APE_q95": float(np.quantile(ape, .95))}


def api_predict(x_train: pd.DataFrame, y_train: np.ndarray, x_query: pd.DataFrame, seed: int, root: Path) -> np.ndarray:
    raise RuntimeError("Hosted inference and account access are disabled in this public snapshot.")


def run_fixed(paths: dict[str, Path], seeds: tuple[int, ...]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x, y, train, test = load_release(paths)
    rows, metric_rows = [], []
    for seed in seeds:
        pred = api_predict(x.iloc[train], y[train], x.iloc[test], seed, paths["root"])
        for pos, value in zip(test, pred):
            true = y[pos]; err = value - true
            row = {"seed": seed, "row_index": int(pos), **x.iloc[pos].to_dict(), "Sigma_e_true": true,
                "Sigma_e_pred": value, "signed_error": err, "absolute_error": abs(err),
                "relative_error": err / true, "absolute_percentage_error": abs(err / true) * 100,
                "squared_error": err * err}
            rows.append(row)
        metric_rows.append({"seed": seed, **metrics(y[test], pred)})
    predictions = pd.DataFrame(rows)
    agg = predictions.groupby("row_index", sort=True).agg(
        Sigma_e_true=("Sigma_e_true", "first"), prediction_mean=("Sigma_e_pred", "mean"),
        prediction_std=("Sigma_e_pred", "std"), prediction_min=("Sigma_e_pred", "min"),
        prediction_max=("Sigma_e_pred", "max"), mean_signed_error=("signed_error", "mean"),
        mean_absolute_error=("absolute_error", "mean"), mean_APE=("absolute_percentage_error", "mean"),
        APE_std=("absolute_percentage_error", "std")).reset_index()
    agg["error_rank"] = agg.mean_APE.rank(method="min", ascending=False).astype(int)
    by_seed = pd.DataFrame(metric_rows)
    summary = by_seed.drop(columns="seed").agg(["mean", "std", "min", "max"]).T.reset_index().rename(columns={"index":"metric"})
    write_table(predictions, paths["tables"] / "fixed_predictions")
    write_table(agg, paths["tables"] / "fixed_sample_summary")
    write_table(by_seed, paths["tables"] / "fixed_metrics_by_seed")
    write_table(summary, paths["tables"] / "fixed_metrics_summary")
    return predictions, agg, by_seed


def validate_reproduction(paths: dict[str, Path], new_metrics: pd.DataFrame) -> None:
    legacy = pd.read_csv(paths["tables"] / "legacy_metrics_reference.csv")
    mapping = {"mape_percent":"MAPE", "mae":"MAE", "rmse":"RMSE", "r2_standard":"R2"}
    rows = []
    for old, new in mapping.items():
        merged = legacy[["seed", old]].merge(new_metrics[["seed", new]], on="seed")
        for r in merged.itertuples(index=False):
            delta = float(getattr(r, new) - getattr(r, old))
            rows.append({"seed": r.seed, "metric": new, "legacy_value": getattr(r, old), "new_value": getattr(r, new),
                "absolute_difference": abs(delta), "tolerance": REPRO_METRIC_ATOL, "passed": abs(delta) <= REPRO_METRIC_ATOL})
    frame = pd.DataFrame(rows)
    write_table(frame, paths["tables"] / "fixed_reproduction_comparison")
    if not frame.passed.all():
        raise RuntimeError("Fixed-split reproduction exceeded predeclared absolute metric tolerance 0.02")


def learning_subsets(train: np.ndarray, repeats: int) -> pd.DataFrame:
    rows = []
    for repeat in range(repeats):
        order = np.random.default_rng(20261012 + repeat).permutation(train)
        for budget in CONTEXT_SIZES:
            chosen = train.copy() if budget == 40 else order[:budget]
            for rank, idx in enumerate(chosen):
                rows.append({"repeat": repeat, "budget": budget, "selection_rank": rank, "row_index": int(idx), "split_seed": 20261012 + repeat})
    return pd.DataFrame(rows)


def baseline_spec(name: str, seed: int):
    if name == "XGBoost":
        from xgboost import XGBRegressor
        estimator = XGBRegressor(random_state=seed, n_jobs=1, objective="reg:squarederror")
        grid = {"n_estimators":[50,100], "max_depth":[2,3], "learning_rate":[.03,.1]}
    elif name == "Random Forest":
        estimator = RandomForestRegressor(random_state=seed, n_jobs=1)
        grid = {"n_estimators":[200], "max_depth":[None,4], "min_samples_leaf":[1,2]}
    elif name == "GPR":
        estimator = Pipeline([("scale",StandardScaler()),("model",GaussianProcessRegressor(kernel=ConstantKernel(1.0)*Matern(nu=1.5)+WhiteKernel(.01), normalize_y=True, random_state=seed, n_restarts_optimizer=1))])
        grid = {"model__alpha":[1e-8,1e-5,1e-3]}
    elif name == "Elastic Net":
        estimator = Pipeline([("scale",StandardScaler()),("model",ElasticNet(max_iter=100000, random_state=seed))])
        grid = {"model__alpha":[1e-4,1e-3,.01,.1], "model__l1_ratio":[.1,.5,.9]}
    else:
        raise KeyError(name)
    return estimator, grid


def run_learning(paths: dict[str, Path], repeats: int, models: tuple[str, ...], seeds: tuple[int, ...]) -> None:
    x, y, train, test = load_release(paths)
    subsets = learning_subsets(train, repeats)
    write_table(subsets, paths["tables"] / "learning_split_manifest")
    pred_rows, metric_rows, oof_rows, hp_rows = [], [], [], []
    for repeat in range(repeats):
        for budget in CONTEXT_SIZES:
            selected = subsets[(subsets.repeat == repeat) & (subsets.budget == budget)].sort_values("selection_rank").row_index.to_numpy(int)
            folds = min(5, max(2, budget // 3)); cv = KFold(folds, shuffle=True, random_state=3000 + repeat)
            for name in models:
                if name == "TabPFN":
                    seed_preds = []
                    for seed in seeds:
                        p = api_predict(x.iloc[selected], y[selected], x.iloc[test], seed, paths["root"])
                        seed_preds.append(p)
                        for idx, value in zip(test, p):
                            pred_rows.append({"repeat":repeat,"budget":budget,"model":name,"seed":seed,"row_index":int(idx),"Sigma_e_true":y[idx],"Sigma_e_pred":value})
                    pmean = np.mean(seed_preds, axis=0)
                    metric_rows.append({"repeat":repeat,"budget":budget,"model":name,"seed_aggregation":"mean_of_six_seeds",**metrics(y[test],pmean)})
                    hp_rows.append({"repeat":repeat,"budget":budget,"model":name,"parameters":json.dumps({"model_path":API_MODEL_PATH,"n_estimators":8,"seeds":list(seeds)})})
                    continue
                estimator, grid = baseline_spec(name, 4000 + repeat)
                search = GridSearchCV(estimator, grid, scoring="neg_mean_absolute_percentage_error", cv=cv, n_jobs=1)
                # Use numeric matrices for baseline libraries (notably XGBoost,
                # which rejects otherwise valid released names containing "["/ "]").
                x_selected = x.iloc[selected].to_numpy(dtype=float)
                x_test = x.iloc[test].to_numpy(dtype=float)
                search.fit(x_selected, y[selected])
                p = search.predict(x_test)
                oof = cross_val_predict(clone(search.best_estimator_), x_selected, y[selected], cv=cv, n_jobs=1)
                for idx, true, value in zip(selected, y[selected], oof):
                    oof_rows.append({"repeat":repeat,"budget":budget,"model":name,"row_index":int(idx),"Sigma_e_true":true,"Sigma_e_oof_pred":value})
                for idx, value in zip(test, p):
                    pred_rows.append({"repeat":repeat,"budget":budget,"model":name,"seed":None,"row_index":int(idx),"Sigma_e_true":y[idx],"Sigma_e_pred":value})
                metric_rows.append({"repeat":repeat,"budget":budget,"model":name,"seed_aggregation":"not_applicable",**metrics(y[test],p)})
                hp_rows.append({"repeat":repeat,"budget":budget,"model":name,"parameters":json.dumps(search.best_params_,sort_keys=True)})
    predictions = pd.DataFrame(pred_rows); metric_frame = pd.DataFrame(metric_rows)
    summary = metric_frame.groupby(["budget","model"])[["MAPE","MAE","RMSE","R2","median_absolute_error","median_APE","APE_q75","APE_q90","APE_q95"]].agg(["mean","std","median","min","max"]).reset_index()
    summary.columns = ["_".join(c).rstrip("_") for c in summary.columns]
    pairs=[]
    for metric in ("MAPE","MAE","RMSE","R2"):
        pivot=metric_frame.pivot(index=["repeat","budget"],columns="model",values=metric)
        if "TabPFN" in pivot:
            for model in [c for c in pivot.columns if c != "TabPFN"]:
                for (repeat,budget), row in pivot.iterrows():
                    diff=row["TabPFN"]-row[model]
                    pairs.append({"repeat":repeat,"budget":budget,"metric":metric,"comparator":model,"TabPFN_minus_comparator":diff,"outcome":"win" if (diff<0 if metric!="R2" else diff>0) else ("tie" if diff==0 else "loss")})
    difficulty=predictions.groupby(["model","row_index"]).apply(lambda g: pd.Series({"Sigma_e_true":g.Sigma_e_true.iloc[0],"mean_APE":np.mean(np.abs(g.Sigma_e_pred-g.Sigma_e_true)/g.Sigma_e_true*100),"prediction_count":len(g)}),include_groups=False).reset_index()
    for name, frame in [("model_hyperparameters",pd.DataFrame(hp_rows)),("training_oof_predictions",pd.DataFrame(oof_rows)),("learning_predictions",predictions),("learning_metrics",metric_frame),("learning_summary",summary),("paired_model_comparisons",pd.DataFrame(pairs)),("sample_difficulty",difficulty)]:
        write_table(frame, paths["tables"] / name)


def plot_fixed(paths: dict[str, Path], predictions: pd.DataFrame, sample: pd.DataFrame, by_seed: pd.DataFrame) -> None:
    plt.rcParams.update({"font.family":"Arial","font.size":8,"axes.spines.top":True,"axes.spines.right":True,"axes.labelcolor":COLORS["text"],"text.color":COLORS["text"],"xtick.color":COLORS["text"],"ytick.color":COLORS["text"]})
    x, y, train, test = load_release(paths)
    figs=paths["figures"]
    fig,ax=plt.subplots(figsize=(3.6,2.7)); bins=np.linspace(y.min(),y.max(),12); train_count,_=np.histogram(y[train],bins=bins); test_count,_=np.histogram(y[test],bins=bins); centers=(bins[:-1]+bins[1:])/2; bin_width=np.diff(bins); bar_width=bin_width*.40; ax.bar(centers-bar_width*.52,train_count,width=bar_width,color="#456990",alpha=.88,edgecolor="white",linewidth=.65,label="Train (n=40)"); ax.bar(centers+bar_width*.52,test_count,width=bar_width,color="#48C0AA",alpha=.82,edgecolor="white",linewidth=.65,label="Test (n=60)"); ax.set(xlabel="Sigma_e [S/m]",ylabel="Count"); ax.legend(frameon=False); save_figure(fig,figs/"Fig01_target_distribution",pd.DataFrame({"row_index":np.arange(100),"Sigma_e":y,"split":np.where(np.isin(np.arange(100),train),"train","test")}))
    fig,ax=plt.subplots(figsize=(3.2,3.0)); ax.errorbar(sample.Sigma_e_true,sample.prediction_mean,yerr=sample.prediction_std,fmt="none",ecolor="#92C2DD",alpha=.9,lw=.9,capsize=1.8,zorder=1); ax.scatter(sample.Sigma_e_true,sample.prediction_mean,s=24,color="#1663A9",edgecolors="white",linewidths=.45,alpha=.92,zorder=2); lim=[min(sample.Sigma_e_true.min(),sample.prediction_mean.min()),max(sample.Sigma_e_true.max(),sample.prediction_mean.max())]; ax.plot(lim,lim,"--",color="#8A8F93",lw=1); ax.set(xlabel="Measured Sigma_e [S/m]",ylabel="Predicted Sigma_e [S/m]"); save_figure(fig,figs/"Fig02_parity",sample)
    for num,xcol,xlabel in [(3,"prediction_mean","Predicted Sigma_e [S/m]"),(4,"Sigma_e_true","Measured Sigma_e [S/m]")]:
        fig,ax=plt.subplots(figsize=(3.4,2.6)); residual_colors=np.where(sample.mean_signed_error>=0,USER_TRIAD[0],USER_TRIAD[2]); ax.scatter(sample[xcol],sample.mean_signed_error,s=23,c=residual_colors,edgecolors="white",linewidths=.45,alpha=.9); ax.axhline(0,ls="--",lw=1,color="#8A8F93"); ax.set(xlabel=xlabel,ylabel="Mean signed error [S/m]"); save_figure(fig,figs/f"Fig{num:02d}_{'residual_vs_prediction' if num==3 else 'residual_vs_measurement'}",sample[["row_index",xcol,"mean_signed_error"]])
    fig,ax=plt.subplots(figsize=(3.4,2.6)); ax.hist(predictions.absolute_percentage_error,bins=25,color="#4995C6",alpha=.88,edgecolor="white",linewidth=.75,rwidth=.92); ax.set(xlabel="Absolute percentage error [%]",ylabel="Count"); save_figure(fig,figs/"Fig05_ape_distribution",predictions[["seed","row_index","absolute_percentage_error"]])
    ordered=sample.sort_values("mean_APE",ascending=False); fig,ax=plt.subplots(figsize=(6.8,2.8)); ax.bar(np.arange(len(ordered)),ordered.mean_APE,color=np.where(ordered.error_rank<=10,COLORS["test"],COLORS["TabPFN"]),alpha=.85); ax.set(xlabel="Test samples ranked by error",ylabel="Mean APE [%]"); save_figure(fig,figs/"Fig06_samplewise_error",ordered)
    fig,ax=plt.subplots(figsize=(3.4,2.6)); ax.plot(by_seed.seed,by_seed.MAPE,"-",color="#92C2DD",lw=1.3,zorder=1); ax.scatter(by_seed.seed,by_seed.MAPE,s=36,color="#456990",edgecolors="white",linewidths=.6,zorder=2); ax.axhline(by_seed.MAPE.mean(),color="#48C0AA",ls="--",lw=1,label=f"Mean = {by_seed.MAPE.mean():.2f}%"); ax.set(xlabel="Inference seed",ylabel="MAPE [%]"); ax.legend(frameon=False,fontsize=7); save_figure(fig,figs/"Fig07_seed_stability",by_seed)
    feature_frame=pd.DataFrame({"row_index":test,"Temperature":x.iloc[test,0].to_numpy(),"Lithium_salt_concentration":x.iloc[test,1].to_numpy()}); src=sample[["row_index","mean_APE","APE_std"]].merge(feature_frame,on="row_index"); fig,axs=plt.subplots(1,2,figsize=(6.8,2.6));
    for ax,col,label,color in zip(axs,["Temperature","Lithium_salt_concentration"],["Temperature (model scale)","Li salt concentration (model scale)"],[USER_TRIAD[1],USER_TRIAD[2]]): ax.errorbar(src[col],src.mean_APE,yerr=src.APE_std,fmt="none",ecolor=color,alpha=.24,lw=.7,zorder=1); ax.scatter(src[col],src.mean_APE,s=22,color=color,edgecolors="white",linewidths=.4,alpha=.88,zorder=2); ax.set(xlabel=label,ylabel="Mean APE [%]")
    save_figure(fig,figs/"Fig08_error_vs_features",src)


def plot_learning(paths: dict[str, Path]) -> None:
    table=paths["tables"]; figs=paths["figures"]
    metrics_frame=pd.read_csv(table/"learning_metrics.csv"); paired=pd.read_csv(table/"paired_model_comparisons.csv"); preds=pd.read_csv(table/"learning_predictions.csv")
    for num,metric in [(9,"MAPE"),(10,"MAE"),(11,"RMSE"),(12,"R2")]:
        src=metrics_frame[["repeat","budget","model",metric]]; fig,ax=plt.subplots(figsize=(3.8,2.8))
        for model,g in src.groupby("model"):
            s=g.groupby("budget")[metric].agg(["mean","std"]).reset_index(); ax.plot(s.budget,s["mean"],"o-",label=model,color=COLORS[model],lw=1.2,ms=3); ax.fill_between(s.budget,s["mean"]-s["std"],s["mean"]+s["std"],color=COLORS[model],alpha=.14)
        ax.set(xlabel="Training budget",ylabel=metric); ax.legend(frameon=False,fontsize=6,ncol=2); save_figure(fig,figs/f"Fig{num:02d}_{metric.lower()}_learning_curve",src)
    fig,ax=plt.subplots(figsize=(6.8,3)); models=list(metrics_frame.model.unique()); budgets=list(CONTEXT_SIZES); data=[metrics_frame[(metrics_frame.model==m)&(metrics_frame.budget==b)].MAPE for b in budgets for m in models]; ax.boxplot(data,showfliers=False); ax.set(ylabel="MAPE [%]",xlabel="Budget × model",xticks=[]); save_figure(fig,figs/"Fig13_model_distributions",metrics_frame)
    p=paired[paired.metric=="MAPE"]; fig,ax=plt.subplots(figsize=(4.5,2.8));
    for model,g in p.groupby("comparator"): s=g.groupby("budget").TabPFN_minus_comparator.agg(["mean","std"]).reset_index(); ax.errorbar(s.budget,s["mean"],yerr=s["std"],marker="o",label=model,color=COLORS[model],lw=1)
    ax.axhline(0,ls="--",color=COLORS["neutral"],lw=.8); ax.set(xlabel="Training budget",ylabel="TabPFN − comparator MAPE"); ax.legend(frameon=False,fontsize=6); save_figure(fig,figs/"Fig14_paired_differences",p)
    w=p.groupby(["budget","comparator","outcome"]).size().rename("count").reset_index(); complete=pd.MultiIndex.from_product([sorted(p.budget.unique()),sorted(p.comparator.unique()),["win","tie","loss"]],names=["budget","comparator","outcome"]).to_frame(index=False); w=complete.merge(w,how="left").fillna({"count":0}); fig,ax=plt.subplots(figsize=(7.2,3.4)); comparators=sorted(p.comparator.unique()); budgets=sorted(p.budget.unique()); width=.18; xpos=np.arange(len(budgets));
    comparator_colors={"Elastic Net":"#8481BA","GPR":"#D6A15C","Random Forest":"#48C0AA","XGBoost":"#4995C6"}
    for j,comp in enumerate(comparators):
        q=w[w.comparator==comp].pivot(index="budget",columns="outcome",values="count").reindex(budgets).fillna(0); offset=(j-(len(comparators)-1)/2)*width;
        color=comparator_colors[comp]; ax.bar(xpos+offset,q["win"],width,color=color,alpha=.92,edgecolor="white",linewidth=.7,label=comp if j<len(comparators) else None); ax.bar(xpos+offset,-q["loss"],width,color=color,alpha=.42,edgecolor="white",linewidth=.7); tie_mask=q["tie"].to_numpy()>0; ax.scatter((xpos+offset)[tie_mask],np.zeros(len(budgets))[tie_mask],s=18+q["tie"].to_numpy()[tie_mask]*8,color=color,edgecolors="white",linewidths=.6,zorder=3)
    ax.axhline(0,color="#8A8F93",lw=.9); ax.set(xticks=xpos,xticklabels=budgets,xlabel="Training budget",ylabel="Paired count: wins (+) / losses (−)",ylim=(-10.6,10.6),yticks=[-10,-5,0,5,10],yticklabels=["10 losses","5 losses","0","5 wins","10 wins"]); ax.legend(frameon=False,ncol=2,loc="lower right",fontsize=7,title="Comparator"); save_figure(fig,figs/"Fig15_win_tie_loss",w)
    heat=preds.assign(APE=lambda d:abs(d.Sigma_e_pred-d.Sigma_e_true)/d.Sigma_e_true*100).groupby(["model","row_index"]).APE.mean().unstack(); heat_long=heat.reset_index().melt(id_vars="model",var_name="row_index",value_name="mean_APE"); vmax=float(heat_long.mean_APE.quantile(.95)); fig,ax=plt.subplots(figsize=(7,2.6)); im=ax.imshow(heat,aspect="auto",cmap=BLUE_CMAP,vmin=0,vmax=vmax); fig.colorbar(im,ax=ax,label="Mean APE [%] (clipped at q95)"); ax.set(xlabel="Released test row index",ylabel="Model"); ax.set_yticks(range(len(heat)),labels=heat.index); tickpos=np.arange(0,len(heat.columns),10); ax.set_xticks(tickpos,labels=[str(heat.columns[i]) for i in tickpos]); save_figure(fig,figs/"Fig16_sample_error_heatmap",heat_long)
    meanpred=preds.groupby(["budget","model","row_index"],as_index=False).agg(Sigma_e_true=("Sigma_e_true","first"),Sigma_e_pred=("Sigma_e_pred","mean")); fig,axs=plt.subplots(1,5,figsize=(10,2.2),sharex=True,sharey=True); budget_colors=["#CEDFEF","#92C2DD","#4995C6","#1663A9","#614099"]
    lim=[min(meanpred.Sigma_e_true.min(),meanpred.Sigma_e_pred.min()),max(meanpred.Sigma_e_true.max(),meanpred.Sigma_e_pred.max())]
    for ax,(budget,g),color in zip(axs,meanpred.groupby("budget"),budget_colors): q=g[g.model=="TabPFN"]; ax.plot(lim,lim,"--",color="#A5ADB2",lw=.8,zorder=1); ax.scatter(q.Sigma_e_true,q.Sigma_e_pred,s=15,color=color,edgecolors="white",linewidths=.35,alpha=.88,zorder=2); ax.set_title(f"n={budget}",color=color,fontweight="bold"); ax.set_xlabel("True")
    axs[0].set_ylabel("Predicted"); save_figure(fig,figs/"Fig17_budget_parity_panels",meanpred)
    fig,axs=plt.subplots(1,2,figsize=(7,2.8));
    for model,g in metrics_frame.groupby("model"): s=g.groupby("budget").agg(MAPE=("MAPE","mean"),R2=("R2","mean")).reset_index(); axs[0].plot(s.budget,s.MAPE,"o-",label=model,color=COLORS[model]); axs[1].plot(s.budget,s.R2,"o-",label=model,color=COLORS[model])
    axs[0].set(xlabel="Budget",ylabel="MAPE [%]"); axs[1].set(xlabel="Budget",ylabel="R²"); axs[1].legend(frameon=False,fontsize=6); save_figure(fig,figs/"Fig18_evaluation_overview",metrics_frame)


def software_versions() -> dict[str, str]:
    out={}
    for p in ("numpy","pandas","scikit-learn","xgboost","matplotlib","openpyxl","tabpfn","tabpfn-client"):
        try: out[p]=importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError: out[p]="not installed"
    return out


def call_estimate(repeats: int, seeds: tuple[int, ...], models: tuple[str, ...]) -> dict[str, int]:
    fixed=len(seeds); learning=(repeats*len(CONTEXT_SIZES)*len(seeds)) if "TabPFN" in models else 0
    return {"fixed_split_api_fit_predict_calls":fixed,"learning_curve_api_fit_predict_calls":learning,"total_api_fit_predict_calls":fixed+learning,"note":"One fit+predict operation is counted as one logical API call; service-side request accounting may differ."}


def rebuild_figures(root: Path, output: Path | None = None) -> None:
    """Redraw Step 1 using saved predictions and metrics; no inference."""
    paths = project_paths(root, output)
    def read(name):
        return pd.read_csv(paths["tables"] / (name + ".csv"))
    plot_fixed(paths, read("fixed_predictions"),
               read("fixed_sample_summary"), read("fixed_metrics_by_seed"))
    plot_learning(paths)
