"""flores fixed-split conductivity evaluation: six inference seeds by default.

Examples (from the ME5001 root):
  .venv/bin/python study/prediction_benchmarks/run_benchmark.py flores audit
  .venv/bin/python study/prediction_benchmarks/run_benchmark.py flores run --resume

Use --inference-seeds 100 for a pilot; repeated splits remain explicit opt-in.
No parameter selection uses validation or test labels; validation is reported only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent / 'flores'
PROJECT = HERE.parents[1]
ROOT = PROJECT.parent
sys.path.insert(0, str(PROJECT))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from analysis.ampere_conductivity import API_MODEL_PATH, N_ESTIMATORS, SEEDS, api_predict, metrics, software_versions

FEATURES = ["temperature_K", "pc_ratio", "salt_molality"]
PROTOCOLS = ("flores_60_20_20", "ampere_fixed_40_60")
DATA = PROJECT / "datasets/flores/HTE_dataset_PC_EMC_EC_LiPF6.csv"
PAPER = {
    "doi": "10.1039/D2DD00027J", "raw_measurements": 3626, "aggregated_conditions": 859,
    "split_counts": {"train": 515, "validation": 172, "test": 172},
    "training_MSE": 1.08, "training_R2": 0.92,
    "validation_MSE": 1.22, "validation_R2": 0.90,
    "validation_RMSE": float(np.sqrt(1.22)), "conductivity_unit": "mS/cm",
    "note": "Table 1; RMSE derived from MSE. Published split IDs are not available here.",
}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def prepare_data(path=DATA):
    raw = pd.read_csv(path, sep="\t", index_col=0)
    raw.index.name = "raw_row_id"
    raw = raw.rename(columns={"T [K]": FEATURES[0], "r": FEATURES[1],
                              "c [mol/kg]": FEATURES[2], "k [mS/cm]": "conductivity"})
    if list(raw.columns) != FEATURES + ["conductivity"]:
        raise ValueError("Unexpected flores schema; inspect the original file before running")
    if not np.isfinite(raw.to_numpy(float)).all() or (raw.conductivity <= 0).any():
        raise ValueError("Nonfinite or nonpositive values; no silent exclusion is allowed")
    # Exact condition grouping follows the paper; do not round to manufacture 859 rows.
    grouped = raw.groupby(FEATURES, sort=True, dropna=False)
    data = grouped.agg(conductivity=("conductivity", "mean"),
                       measurement_std=("conductivity", "std"),
                       n_measurements=("conductivity", "size")).reset_index()
    data.insert(0, "condition_id", [f"C{i:04d}" for i in range(len(data))])
    data["formulation_id"] = pd.factorize(pd.MultiIndex.from_frame(data[FEATURES[1:]]), sort=True)[0]
    # Record membership for traceability; condition duplicates cannot cross splits.
    mapping = raw.reset_index().merge(data[["condition_id"] + FEATURES], on=FEATURES,
                                     validate="many_to_one")
    audit = {"source_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
             "raw_rows": len(raw), "aggregated_conditions": len(data),
             "formulations": int(data.formulation_id.nunique()),
             "aggregation": "mean at exact (temperature_K, pc_ratio, salt_molality)",
             "measurement_std": "sample standard deviation; missing for single measurement",
             "imputation": False, "outlier_removal": False,
             "paper_count_matches": len(data) == PAPER["aggregated_conditions"],
             "ranges": {k: {"min": float(raw[k].min()), "max": float(raw[k].max())}
                        for k in raw.columns},
             "limitation": "849 current conditions vs 859 in paper; not an exact reproduction"}
    return data, mapping, audit


def make_split(data, protocol, seed):
    ids = np.arange(len(data))
    assignments = np.full(len(data), "unused", dtype=object)
    if protocol == PROTOCOLS[0]:
        development, test = train_test_split(ids, test_size=.2, random_state=seed)
        train, validation = train_test_split(development, test_size=.25, random_state=seed)
        assignments[validation] = "validation"
    elif protocol == PROTOCOLS[1]:
        # Literal small-context ampere control: only 100 conditions are selected.
        selected = np.random.RandomState(seed).permutation(ids)[:100]
        if len(selected) != 100:
            raise ValueError("The 40/60 control requires at least 100 conditions")
        train, test = selected[:40], selected[40:]
    else:
        raise ValueError(protocol)
    assignments[train], assignments[test] = "train", "test"
    result = data.copy()
    result.insert(0, "split", assignments)
    result.insert(0, "split_seed", seed)
    result.insert(0, "protocol", protocol)
    if result.condition_id.duplicated().any() or set(train) & set(test):
        raise ValueError("Overlapping conditions in split")
    return result


def sr_features(x):
    temperature, ratio, concentration = x[FEATURES].to_numpy(float).T
    return np.column_stack([concentration, temperature,
                            concentration**2.5 * ratio**.5,
                            concentration**.25 * temperature**2])


def local_predictions(train, query):
    a, b = sr_features(train), sr_features(query)
    scale = np.linalg.norm(a, axis=0)
    if (scale == 0).any():
        raise ValueError("Degenerate SR design")
    coefficients = np.linalg.lstsq(a / scale, train.conductivity.to_numpy(float), rcond=None)[0]
    return {"Training mean": np.full(len(query), train.conductivity.mean()),
            "SR4 refit (OLS, no intercept)": (b / scale) @ coefficients}


def prediction_frame(query, values, protocol, split_seed, inference_seed, model):
    values = np.asarray(values, float).reshape(-1)
    if len(values) != len(query) or not np.isfinite(values).all():
        raise ValueError("Invalid predictions; no truncation or clipping is allowed")
    frame = query[["condition_id", "formulation_id", "split"] + FEATURES].copy()
    frame.insert(0, "model", model)
    frame.insert(0, "inference_seed", inference_seed)
    frame.insert(0, "split_seed", split_seed)
    frame.insert(0, "protocol", protocol)
    frame["true_mS_cm"], frame["pred_mS_cm"] = query.conductivity.to_numpy(float), values
    frame["absolute_error"] = np.abs(frame.pred_mS_cm - frame.true_mS_cm)
    frame["APE_percent"] = 100 * frame.absolute_error / frame.true_mS_cm
    return frame


def evaluate(predictions):
    rows = []
    keys = ["protocol", "split_seed", "inference_seed", "model", "split"]
    for key, part in predictions.groupby(keys, dropna=False, sort=False):
        rows.append(dict(zip(keys, key)) | {"n": len(part),
                    "MSE": float(np.mean((part.pred_mS_cm - part.true_mS_cm)**2)),
                    **metrics(part.true_mS_cm.to_numpy(), part.pred_mS_cm.to_numpy())})
    return pd.DataFrame(rows)


def summarize(predictions, scores, out):
    """Match ampere_conductivity fixed-run summaries; keep mean-of-metrics separate from ensemble metrics."""
    identifiers = ["protocol", "split_seed", "model", "split"]
    columns = [c for c in scores.columns if c not in identifiers + ["inference_seed", "n"]]
    summary = scores.groupby(identifiers)[columns].agg(["mean", "std", "min", "max"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    counts = scores.groupby(identifiers).size().rename("n_seeds").reset_index()
    summary = summary.merge(counts, on=identifiers, validate="one_to_one")
    summary.to_csv(out / "tables/metrics_summary.csv", index=False)
    tab = predictions[predictions.model.eq("TabPFN")].copy()
    keys = ["protocol", "split_seed", "split", "condition_id", "formulation_id"] + FEATURES
    sample = tab.groupby(keys).agg(true_mS_cm=("true_mS_cm", "first"),
             prediction_mean=("pred_mS_cm", "mean"), prediction_std=("pred_mS_cm", "std"),
             prediction_min=("pred_mS_cm", "min"), prediction_max=("pred_mS_cm", "max"),
             mean_absolute_error=("absolute_error", "mean"), mean_APE=("APE_percent", "mean"),
             APE_std=("APE_percent", "std"), n_seeds=("inference_seed", "nunique")).reset_index()
    sample["error_rank"] = sample.groupby(["protocol", "split_seed", "split"]).mean_APE.rank(
        ascending=False, method="min").astype(int)
    sample.to_csv(out / "tables/sample_summary.csv", index=False)
    ensemble = sample.rename(columns={"prediction_mean": "pred_mS_cm"})
    ensemble["model"], ensemble["inference_seed"] = "TabPFN mean prediction across requested seeds", -2
    evaluate(ensemble).to_csv(out / "tables/ensemble_metrics.csv", index=False)
    return summary, sample


def verify_outputs(predictions, scores, plans, config, out):
    checks = []
    def check(name, result):
        checks.append({"check": name, "passed": bool(result)})
    tab = predictions[predictions.model.eq("TabPFN")]
    check("unique_prediction_keys", not tab.duplicated(
        ["protocol", "split_seed", "inference_seed", "condition_id"]).any())
    check("finite_predictions", np.isfinite(tab.pred_mS_cm).all())
    for (protocol, split_seed), plan in plans.groupby(["protocol", "split_seed"]):
        query = plan[plan.split.isin(["test", "validation"])]
        check(f"{protocol}:unique_conditions", plan.condition_id.is_unique)
        for seed in config["inference_seeds"]:
            part = tab[(tab.protocol == protocol) & (tab.split_seed == split_seed) & (tab.inference_seed == seed)]
            check(f"{protocol}:{split_seed}:{seed}:coverage", set(part.condition_id) == set(query.condition_id))
            check(f"{protocol}:{split_seed}:{seed}:labels", np.allclose(
                part.set_index("condition_id").true_mS_cm.loc[query.condition_id], query.conductivity,
                rtol=1e-12, atol=1e-12))
    for row in scores.itertuples():
        part = predictions[(predictions.protocol == row.protocol) & (predictions.split_seed == row.split_seed)
                           & (predictions.inference_seed == row.inference_seed)
                           & (predictions.model == row.model) & (predictions.split == row.split)]
        error = part.pred_mS_cm.to_numpy() - part.true_mS_cm.to_numpy()
        expected = [np.abs(error).mean(), np.sqrt(np.mean(error**2)),
                    np.mean(np.abs(error) / part.true_mS_cm.to_numpy()) * 100]
        check(f"metrics:{row.protocol}:{row.model}:{row.split}:{row.inference_seed}",
              np.allclose(expected, [row.MAE, row.RMSE, row.MAPE], atol=1e-12))
    passed = all(c["passed"] for c in checks)
    write_json(out / "tables/result_audit.json", {"passed": passed, "checks": checks,
        "scope": "fixed splits; inference-seed variability only; not split/generalization uncertainty"})
    if not passed:
        raise ValueError("Result audit failed")


def update_readme(out, audit, summary, config):
    readme = HERE.parent / "README.md"
    start, end = "<!-- flores_RESULTS_START -->", "<!-- flores_RESULTS_END -->"
    lines = [start, "## flores — Flores 等，2022", "",
        "论文：[Learning the laws of lithium-ion transport in electrolytes using symbolic regression]"
        "(https://doi.org/10.1039/D2DD00027J)。研究 LiPF₆／EC-PC-EMC 液态电解液，"
        "通过温度、盐浓度和 PC 比例预测离子电导率，并寻找简洁、可解释的符号回归公式。", "",
        f"数据：{audit['raw_rows']} 次测量，按完全相同的三项输入聚合为 {audit['aggregated_conditions']} 条条件，"
        f"涉及 {audit['formulations']} 组配方；不填补、不删异常值。原文聚合为859条，差异尚未解释。", "",
        "### 配置与完整性", "",
        f"- 模型沿用 ampere_conductivity：`{API_MODEL_PATH}`，`n_estimators={N_ESTIMATORS}`。",
        f"- 推理种子：{', '.join(map(str, config['inference_seeds']))}；划分种子：{config['split_seeds']}。",
        "- 两种固定划分：509训练／170验证／170测试；抽取100条作40训练／60测试，其余749条不用。",
        "- 原始三项特征和 mS/cm 标签直接输入；无手工缩放、对数变换、调参或残差修正。",
        "- 先聚合再划分，同一测量条件的重复测量不跨集合；验证标签不用于训练或选模型。",
        "- 已保存逐种子指标、均值／样本标准差（ddof=1）／最小值／最大值、逐样本预测波动、"
        "六种子平均预测的单独指标、数据哈希、代码快照和结果核验。",
        "- 六个推理种子衡量固定划分下的模型随机性，不代表六次独立数据划分；尚未做配方分组留出。",
        "- 对照 ampere_conductivity 的固定划分评估部分；未运行其额外的多模型学习曲线与重复抽样实验。", "",
        "### 指标汇总", "",
        "表中为各种子指标的均值 ± 标准差，不是把种子预测先平均再计算指标。"
        "本地确定性对照只有一组结果，以 — 表示标准差。", "",
        "|划分|集合|模型|种子数|MAPE (%)|MAE (mS/cm)|RMSE (mS/cm)|MSE|R²|",
        "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    def cell(row, metric):
        mean, std = row[f"{metric}_mean"], row[f"{metric}_std"]
        return f"{mean:.4f} ± {std:.4f}" if pd.notna(std) else f"{mean:.4f}（—）"
    for _, row in summary.iterrows():
        label = "509/170/170" if row.protocol == PROTOCOLS[0] else "40/60"
        lines.append(f"|{label}|{row['split']}|{row.model}|{row.n_seeds}|" +
                     "|".join(cell(row, m) for m in ["MAPE", "MAE", "RMSE", "MSE", "R2"]) + "|")
    val = summary[(summary.protocol == PROTOCOLS[0]) & (summary.model == "TabPFN")
                  & (summary['split'] == "validation")].iloc[0]
    lines += ["", "### 与原论文核对", "",
        "原文 Table 1 的所选四项符号公式在515条训练数据上拟合，验证集172条。"
        "比较应使用我们的验证集；原文未在该表报告 MAPE。", "",
        "|验证指标|原文 Table 1／Fig. 3a|本实验 TabPFN（种子均值 ± 标准差）|",
        "|---|---:|---:|",
        f"|MSE|1.22|{cell(val, 'MSE')}|",
        f"|RMSE (mS/cm)|约1.105（由MSE开方）|{cell(val, 'RMSE')}|",
        f"|R²|0.90|{cell(val, 'R2')}|", "",
        "原文训练集 MSE=1.08、R²=0.92，仅作原文背景记录。样本数量和划分不同，"
        "不能作为严格的原文复现或同条件优劣结论。SR4 refit 是原文 Eq.(3) 的四项函数在"
        "当前训练集上重新进行无截距 OLS 拟合，不是作者系数，也不是重跑其符号发现流程。", "",
        "相同配方的不同温度可以跨集合，目前结果支持同体系条件预测，不能据此宣称新配方泛化。", "",
        "### 文件与重跑", "",
        "- [脚本](flores.py)",
        "- [逐种子指标](flores/results/tables/metrics.csv) · "
        "[汇总](flores/results/tables/metrics_summary.csv) · "
        "[逐样本](flores/results/tables/sample_summary.csv)",
        "- [平均预测指标](flores/results/tables/ensemble_metrics.csv) · "
        "[核验](flores/results/tables/result_audit.json) · "
        "[预测图](flores/results/figures/parity.png)",
        "- `Output/data` 保存聚合数据、原始行映射和固定划分；`Output/cache` 保存每个种子的预测及校验值。",
        "- 重建并复用缓存：项目根目录运行 "
        "`.venv/bin/python study/prediction_benchmarks/run_benchmark.py flores run --resume`。",
        f"- 本次执行新增预测调用 {config['api_calls_this_execution']} 次；"
        f"复用缓存 {config['cache_hits_this_execution']} 份；最终应有 {config['planned_prediction_calls']} 份种子预测。",
        "", end]
    section = "\n".join(lines)
    text = readme.read_text() if readme.exists() else "# prediction_benchmarks 实验结果\n\n记录各 Z 编号数据集的实验配置、结果和原文指标对比。\n"
    if start in text:
        a, b = text.index(start), text.index(end) + len(end)
        text = text[:a] + section + text[b:]
    else:
        text += "\n" + section + "\n"
    readme.write_text(text)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["audit", "run"])
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--protocols", nargs="+", choices=PROTOCOLS, default=list(PROTOCOLS))
    parser.add_argument("--inference-seeds", type=int, nargs="+", default=list(SEEDS),
                        help=f"Default six seeds: {' '.join(map(str, SEEDS))}")
    parser.add_argument("--split-seeds", type=int, nargs="+", default=[42],
                        help="Future repeated splits are explicit; pilot uses only 42")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    for values in [args.protocols, args.inference_seeds, args.split_seeds]:
        if len(values) != len(set(values)):
            parser.error("Duplicate protocols or seeds would repeat API calls")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    for folder in ["data", "tables", "cache", "figures"]:
        (out / folder).mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out / "execution.log")])
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    data, mapping, audit = prepare_data()
    plans = pd.concat([make_split(data, protocol, seed)
                       for protocol in args.protocols for seed in args.split_seeds], ignore_index=True)
    contract = {"data_sha256": audit["source_sha256"], "protocols": args.protocols,
                "split_seeds": args.split_seeds, "inference_seeds": args.inference_seeds,
                "model_path": API_MODEL_PATH, "n_estimators": N_ESTIMATORS,
                "features": FEATURES, "target": "raw conductivity [mS/cm]",
                "aggregation": audit["aggregation"], "schema_version": 1}
    signature = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    config = contract | {"signature": signature, "status": "audit_completed",
                         "run_type": "fixed_split_seed_evaluation", "planned_prediction_calls":
                         len(args.protocols)*len(args.split_seeds)*len(args.inference_seeds),
                         "api_calls_this_execution": 0, "api_calls_total": 0,
                         "software_versions": software_versions(),
                         "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                         "validation_policy": "report-only; no hyperparameter/model selection",
                         "six_inference_seeds_requested": set(args.inference_seeds) == set(SEEDS),
                         "repeated_split_experiment": len(args.split_seeds) > 1,
                         "cache_hits_this_execution": 0}
    config_path = out / "run_config.json"
    accepted_signatures = {signature}
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        old_contract = {k: previous[k] for k in contract}
        comparable = {k: v for k, v in contract.items() if k != "inference_seeds"}
        old_comparable = {k: v for k, v in old_contract.items() if k != "inference_seeds"}
        if old_comparable != comparable or not set(previous["inference_seeds"]).issubset(args.inference_seeds):
            raise ValueError("Only inference-seed extension is allowed in this Output")
        if previous["signature"] != signature and not args.resume:
            raise ValueError("Seed extension requires --resume to preserve paid predictions")
        accepted_signatures.update(previous.get("compatible_cache_signatures", []))
        accepted_signatures.add(previous["signature"])
        if args.mode == "audit" and previous["status"] != "audit_completed":
            raise FileExistsError("Use a separate audit output to preserve existing run results")
        if args.mode == "run" and previous["status"] == "completed" and not args.resume:
            raise FileExistsError("Completed run exists; use --resume to reuse cached predictions")
        config["api_calls_total"] = previous.get("api_calls_total", previous["api_calls_this_execution"])
        prior_snapshot = out / "snapshots" / f"config_{previous['signature'][:12]}.json"
        prior_snapshot.parent.mkdir(exist_ok=True)
        if not prior_snapshot.exists():
            write_json(prior_snapshot, previous)
    config["compatible_cache_signatures"] = sorted(accepted_signatures)
    snapshots = out / "snapshots"
    snapshots.mkdir(exist_ok=True)
    snapshot = snapshots / f"run_z11_{config['script_sha256'][:12]}.py"
    if not snapshot.exists():
        snapshot.write_bytes(Path(__file__).read_bytes())
    shared = PROJECT / "analysis/ampere_conductivity.py"
    config["shared_step01_sha256"] = hashlib.sha256(shared.read_bytes()).hexdigest()
    shared_snapshot = snapshots / f"step01_{config['shared_step01_sha256'][:12]}.py"
    if not shared_snapshot.exists():
        shared_snapshot.write_bytes(shared.read_bytes())
    data.to_csv(out / "data/conditions.csv", index=False)
    mapping.to_csv(out / "data/raw_to_condition.csv", index=False)
    plans.to_csv(out / "data/split_manifest.csv", index=False)
    write_json(out / "data/data_audit.json", audit)
    write_json(out / "paper_reference.json", PAPER)
    write_json(config_path, config)
    logging.info("Data audit: %s raw -> %s conditions; protocols=%s; planned calls=%s",
                 audit["raw_rows"], len(data), args.protocols, config["planned_prediction_calls"])
    all_predictions = []
    try:
        for (protocol, split_seed), plan in plans.groupby(["protocol", "split_seed"], sort=False):
            train = plan[plan.split.eq("train")]
            query = plan[plan.split.isin(["validation", "test"])]
            for model, values in local_predictions(train, query).items():
                all_predictions.append(prediction_frame(query, values, protocol, split_seed, -1, model))
            if args.mode == "audit":
                continue
            for seed in args.inference_seeds:
                cache = out / "cache" / f"{protocol}_split{split_seed}_seed{seed}.csv"
                cache_meta = cache.with_suffix(".json")
                if args.resume and cache.exists() and cache_meta.exists():
                    meta = json.loads(cache_meta.read_text())
                    saved = pd.read_csv(cache)
                    if (meta["signature"] not in accepted_signatures or meta["sha256"] !=
                            hashlib.sha256(cache.read_bytes()).hexdigest()):
                        raise ValueError("Cached prediction contract or checksum changed")
                    if not np.array_equal(saved.condition_id, query.condition_id):
                        raise ValueError("Cached prediction row order changed")
                    if (not np.allclose(saved[FEATURES], query[FEATURES], atol=0, rtol=1e-12)
                            or not np.allclose(saved.true_mS_cm, query.conductivity, atol=0, rtol=1e-12)):
                        raise ValueError("Cached inputs or labels changed")
                    config["cache_hits_this_execution"] += 1
                    values = saved.pred_mS_cm.to_numpy(float)
                    logging.info("Reused cache: %s", cache.name)
                else:
                    if cache.exists():
                        raise FileExistsError("Cache exists: pass --resume; do not repeat paid predictions")
                    logging.info("TabPFN %s split=%s seed=%s: train=%s query=%s (one call)",
                                 protocol, split_seed, seed, len(train), len(query))
                    config["api_calls_this_execution"] += 1
                    config["api_calls_total"] += 1
                    write_json(config_path, config | {"status": "running"})
                    started = time.monotonic()
                    values = api_predict(train[FEATURES], train.conductivity.to_numpy(float),
                                         query[FEATURES], seed, ROOT)
                    saved = prediction_frame(query, values, protocol, split_seed, seed, "TabPFN")
                    saved.to_csv(cache, index=False)
                    write_json(cache_meta, {"signature": signature, "elapsed_seconds": time.monotonic()-started,
                                           "sha256": hashlib.sha256(cache.read_bytes()).hexdigest()})
                all_predictions.append(prediction_frame(query, values, protocol, split_seed, seed, "TabPFN"))
        predictions = pd.concat(all_predictions, ignore_index=True)
        scores = evaluate(predictions)
        predictions.to_csv(out / "tables/predictions.csv", index=False)
        scores.to_csv(out / "tables/metrics.csv", index=False)
        config["status"] = "completed" if args.mode == "run" else "audit_completed"
        if args.mode == "run":
            summary, sample = summarize(predictions, scores, out)
            verify_outputs(predictions, scores, plans, config, out)
            make_plots(predictions, out)
            update_readme(out, audit, summary, config)
            (out / "report.md").unlink(missing_ok=True)
            config["six_inference_seeds_completed"] = set(args.inference_seeds) == set(SEEDS)
        logging.info("Completed: %s; actual calls=%s", config["status"], config["api_calls_this_execution"])
    except Exception as exc:
        config["status"], config["error_type"] = "failed", type(exc).__name__
        # Do not persist arbitrary client exception text, tokens, or authentication state.
        logging.error("Stopped (%s); no automatic retry or model substitution", type(exc).__name__)
        raise
    finally:
        write_json(config_path, config)


def make_plots(predictions, out):
    import matplotlib.pyplot as plt
    chosen = pd.read_csv(out / "tables/sample_summary.csv")
    fig, axes = plt.subplots(1, len(chosen.protocol.unique()), figsize=(9, 3.7), squeeze=False)
    for ax, (protocol, rows) in zip(axes[0], chosen.groupby("protocol", sort=False)):
        for split, group in rows.groupby("split"):
            ax.errorbar(group.true_mS_cm, group.prediction_mean, yerr=group.prediction_std.fillna(0),
                        fmt="o", markersize=3, elinewidth=.6, alpha=.65, label=split)
        low = min(rows.true_mS_cm.min(), rows.prediction_mean.min())
        high = max(rows.true_mS_cm.max(), rows.prediction_mean.max())
        ax.plot([low, high], [low, high], "--", color="gray", linewidth=1)
        ax.set(xlabel="Measured conductivity (mS/cm)", ylabel="Predicted conductivity (mS/cm)",
               title=protocol)
        ax.legend(frameon=False)
    fig.tight_layout()
    for suffix in ["png", "pdf"]:
        fig.savefig(out / "figures" / f"parity.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
