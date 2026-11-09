"""Single command-line entry point. Importing or --help never runs an experiment."""
from pathlib import Path
import argparse
import importlib.util
import sys

ROOT = Path(__file__).resolve().parent
STUDIES = ("flores", "clio", "calisol23", "ionic_liquids", "chemarr", "electrolytomics", "aqueous")


def load_experiment(study):
    study = study.lower()
    path = ROOT / f"{study.lower()}.py"
    spec = importlib.util.spec_from_file_location(study.lower(), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Select a published conductivity benchmark.", epilog="Example: python study/prediction_benchmarks/run_benchmark.py flores --help")
    parser.add_argument("study", choices=STUDIES)
    # Parse the study only, leaving its existing experiment options untouched.
    selected = parser.parse_args(args[:1])
    forwarded = args[1:]
    if not forwarded:
        forwarded = ["--help"]
    try:
        experiment = load_experiment(selected.study)
        if selected.study in {"ionic_liquids", "electrolytomics", "aqueous"}:
            if "--help" in forwarded or "-h" in forwarded:
                parser.exit(message="Study options: {audit,run} [--resume] [--output-dir PATH] [--seeds N ...]\n")
            from benchmark_utils import run
            data, features, audit = experiment.prepare()
            files = (experiment.FILES if selected.study == "ionic_liquids" else
                     [experiment.RAW, experiment.FEATURE_FILE] if selected.study == "electrolytomics" else [experiment.SOURCE])
            run(experiment.HERE, data, features, audit, files, experiment.SPEC, forwarded)
        else:
            experiment.main(forwarded)
    except FileNotFoundError as exc:
        parser.exit(2, f"Missing required file: {exc.filename or exc}. "
                    "See study/datasets/README.md for the expected input paths.\n")



if __name__ == "__main__":
    main()
