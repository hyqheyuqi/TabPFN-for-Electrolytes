#!/usr/bin/env python3
"""Step 2: audit, pilot, or full new-domain context-sensitivity experiment."""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from analysis.ampere_conductivity import software_versions, write_json
from analysis.domain_adaptation import (
    CONTEXT_SIZES,
    METHODS,
    SEEDS,
    audit_data,
    build_design,
    ensure_output,
    plot_results,
    rebuild_figures,
    project_paths,
    run_experiment,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["audit", "pilot", "full", "plots"])
    p.add_argument("--output-dir", type=Path, default=HERE / "results" / "domain_adaptation")
    p.add_argument("--context-sizes", type=int, nargs="+", default=list(CONTEXT_SIZES))
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--inference-seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--methods", nargs="+", choices=list(METHODS), default=list(METHODS))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p


def main() -> None:
    args = parser().parse_args()
    if args.mode == "plots":
        rebuild_figures(HERE.parent, args.output_dir.resolve())
        return
    if tuple(args.context_sizes) != CONTEXT_SIZES:
        raise ValueError(f"Step 2 fixed context sizes are {CONTEXT_SIZES}")
    if tuple(args.methods) != METHODS:
        raise ValueError(f"Step 2 requires both methods in this order: {METHODS}")
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and args.mode != "audit" and not (args.resume or args.overwrite):
        raise FileExistsError("Output exists; pass --resume or --overwrite")

    paths = project_paths(HERE.parent, args.output_dir.resolve())
    ensure_output(paths)
    run_repeats = 1 if args.mode == "pilot" else args.repeats
    run_seeds = (args.inference_seeds[0],) if args.mode == "pilot" else tuple(args.inference_seeds)
    unique_configs_per_repeat = 1 + 2 * 3 + 1
    config = {
        "status": "started",
        "step": "02",
        "mode": args.mode,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "context_sizes": list(CONTEXT_SIZES),
        "methods": list(METHODS),
        "repeats_requested": args.repeats,
        "repeats_executed": run_repeats if args.mode != "audit" else 0,
        "inference_seeds_requested": args.inference_seeds,
        "inference_seeds_executed": list(run_seeds) if args.mode != "audit" else [],
        "outer_split": "30 new-domain rows -> 15 context candidates + 15 fixed test rows per repeat",
        "selection": "nested prefixes; Random uses seeded permutation; Diversity uses label-blind standardized-feature k-center greedy",
        "model_path": "tabpfn-v3-regressor-v3_default.ckpt",
        "n_estimators": 8,
        "software_versions": software_versions(),
        "api_call_estimate": {
            "unique_training_configs_per_repeat": unique_configs_per_repeat,
            "logical_fit_predict_calls": 0 if args.mode == "audit" else run_repeats * unique_configs_per_repeat * len(run_seeds),
            "note": "context 0 and 15 are computed once per repeat/seed and emitted for both methods",
        },
        "credential_policy": "cached TabPFN login only; credentials are never copied, logged, or snapshotted",
    }
    write_json(paths["configs"] / "run_config.json", config)
    shutil.copy2(__file__, paths["snapshots"] / "run_domain_adaptation.py")
    shutil.copy2(HERE / "analysis" / "domain_adaptation.py", paths["snapshots"] / "domain_adaptation_module.py")
    shutil.copy2(HERE / "analysis" / "ampere_conductivity.py", paths["snapshots"] / "ampere_conductivity_dependency.py")
    try:
        audit_data(paths)
        design = build_design(paths, run_repeats if args.mode != "audit" else args.repeats)
        if args.mode in ("pilot", "full"):
            run_experiment(paths, design, run_seeds, resume=args.resume)
            plot_results(paths)
        config["status"] = "completed"
    except Exception as exc:
        config.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        config["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(paths["configs"] / "run_config.json", config)


if __name__ == "__main__":
    main()
