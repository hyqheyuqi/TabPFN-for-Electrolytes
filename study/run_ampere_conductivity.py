#!/usr/bin/env python3
"""Step 1: audit/reconstruct, pilot fixed split, or full Sigma_e evaluation."""
from __future__ import annotations
import argparse, json, shutil, sys
from datetime import datetime, timezone
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from analysis.ampere_conductivity import (CONTEXT_SIZES, MODEL_NAMES, SEEDS, audit_data, audit_legacy,
    rebuild_figures, call_estimate, ensure_output, load_release, plot_fixed, plot_learning, project_paths,
    reconstruct_and_compare, run_fixed, run_learning, software_versions,
    validate_reproduction, write_json)

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=["audit","pilot","full","plots"])
    p.add_argument("--output-dir",type=Path,default=HERE/"results"/"ampere_conductivity")
    p.add_argument("--context-sizes",type=int,nargs="+",default=list(CONTEXT_SIZES))
    p.add_argument("--repeats",type=int,default=10)
    p.add_argument("--inference-seeds",type=int,nargs="+",default=list(SEEDS))
    p.add_argument("--models",nargs="+",default=list(MODEL_NAMES),choices=list(MODEL_NAMES))
    p.add_argument("--resume",action="store_true")
    p.add_argument("--overwrite",action="store_true")
    return p

def main():
    args=parser().parse_args(); root=HERE.parent; paths=project_paths(root,args.output_dir.resolve())
    if args.mode == "plots":
        rebuild_figures(root, args.output_dir.resolve())
        return
    if tuple(args.context_sizes)!=CONTEXT_SIZES: raise ValueError(f"Step 1 fixed budgets are {CONTEXT_SIZES}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and args.mode!="audit" and not (args.resume or args.overwrite):
        raise FileExistsError("Output exists; pass --resume or --overwrite")
    ensure_output(paths)
    config={"status":"started","mode":args.mode,"created_utc":datetime.now(timezone.utc).isoformat(),"output_dir":str(args.output_dir.resolve()),"context_sizes":args.context_sizes,"repeats":args.repeats,"inference_seeds":args.inference_seeds,"models":args.models,"model_path":"tabpfn-v3-regressor-v3_default.ckpt","n_estimators":8,"software_versions":software_versions(),"api_call_estimate":call_estimate(args.repeats,tuple(args.inference_seeds),tuple(args.models)),"credential_policy":"Hosted inference disabled in this public snapshot"}
    write_json(paths["configs"]/"run_config.json",config)
    shutil.copy2(__file__,paths["snapshots"]/"run_ampere_conductivity.py")
    shutil.copy2(HERE/"analysis"/"ampere_conductivity.py",paths["snapshots"]/"ampere_conductivity_module.py")
    try:
        audit_data(paths); reconstruct_and_compare(paths); audit_legacy(paths)
        if args.mode in ("pilot","full"):
            fixed_seeds=(args.inference_seeds[0],) if args.mode=="pilot" else tuple(args.inference_seeds)
            pred,sample,by_seed=run_fixed(paths,fixed_seeds); plot_fixed(paths,pred,sample,by_seed)
            if args.mode=="full":
                validate_reproduction(paths,by_seed)
                run_learning(paths,args.repeats,tuple(args.models),tuple(args.inference_seeds)); plot_learning(paths)
        config["status"]="completed"
    except Exception as exc:
        config.update(status="failed",error_type=type(exc).__name__,error=str(exc)); raise
    finally:
        config["finished_utc"]=datetime.now(timezone.utc).isoformat(); write_json(paths["configs"]/"run_config.json",config)

if __name__=="__main__": main()
