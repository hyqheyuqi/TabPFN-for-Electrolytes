# TabPFN for Electrolytes

[English](README.md) · [简体中文](README.zh-CN.md)

Analysis code and selected derived results for *From Small-Data Conductivity Prediction to Bayesian Optimization of Electrolyte Formulations with a Tabular Foundation Model*.

## Start here

From the repository root, install pandas and verify key reported numbers using the included CSVs:

```sh
python -m pip install pandas
python study/check_saved_results.py
```

The check recomputes AMPERE fixed-split MAPE and seed SD from individual absolute percentage errors, calculates the two-temperature mean search cost and comparator reductions, and reads the frozen-policy Clio validation cost. It prints to stdout and changes no result files.

`requirements.txt` lists dependencies for inspecting and running the analysis modules and tests. The published CSVs can also be opened directly in a spreadsheet application. [MANIFEST.sha256](MANIFEST.sha256) identifies the files in this snapshot.

## Put downloaded inputs in the expected locations

The scripts resolve paths relative to their own files. From the repository root, place each downloaded source under `study/datasets/<dataset>/` using the exact names in the [dataset inventory](study/datasets/README.md). For example, the Flores CSV belongs at `study/datasets/flores/HTE_dataset_PC_EMC_EC_LiPF6.csv`. Use a separate output directory to keep audit runs apart from the published CSVs:

```sh
python -m pip install -r requirements.txt
python study/prediction_benchmarks/run_benchmark.py flores audit --output-dir recomputed/flores
```

This audit checks and aggregates the source table and writes the split and audit records under `recomputed/flores/`. It needs the Flores source file. The saved-result check above can run immediately after cloning, without downloading any dataset. End-to-end research modes that require unavailable historical inputs or hosted inference remain outside the verified public workflow.

## Paper sections and analysis scripts

The links below point to code. The output column distinguishes shipped CSVs from files the research scripts expect or generate in a working run.

| Paper section | Script | Function and outputs |
| --- | --- | --- |
| §2.1.1 / §3.1.1 | [run_benchmark.py](study/prediction_benchmarks/run_benchmark.py) | Dispatches by dataset name. Use `python study/prediction_benchmarks/run_benchmark.py flores --help` to inspect dataset options. A benchmark work run uses `study/prediction_benchmarks/<dataset>/results/` by default. Saved metric summaries are grouped under [study/results/benchmarks/](study/results/benchmarks/). |
| §2.1.1 / §3.1.1 | [benchmark_utils.py](study/prediction_benchmarks/benchmark_utils.py) | Shared preparation, split audit, metric aggregation and cache handling for ionic liquids, Electrolytomics and aqueous. Working outputs include split manifests and predictions, metric CSVs under `tables/`, and result audits. |
| §2.1.2 / §3.1.2 | [run_ampere_conductivity.py](study/run_ampere_conductivity.py), [ampere_conductivity.py](study/analysis/ampere_conductivity.py) | CLI and implementation for the fixed 40/60 conductivity split, conventional regressors and nested label-budget comparisons. [Saved results](study/results/ampere_conductivity/) include fixed split indices, learning-subset indices, fixed-split sample/seed summaries and learning-curve statistics. |
| §2.1.3 / §3.1.3 | [run_domain_adaptation.py](study/run_domain_adaptation.py), [domain_adaptation.py](study/analysis/domain_adaptation.py) | Constructs repeated target-domain splits, compares random versus diversity selection, and aggregates performance by context size. [Saved results](study/results/domain_adaptation/) include split indices, per-repeat predictions, paired method differences and aggregate metrics. |
| §2.2.1 / §3.2.1 | [run_conductivity_guided_search.py](study/run_conductivity_guided_search.py), [conductivity_guided_search.py](study/analysis/conductivity_guided_search.py) | Constructs conductivity/salt coordinates and runs matched finite-pool GP batch replays. [Saved results](study/results/conductivity_guided_search/) include reference-concentration predictions, candidate coordinates, acquisition rankings, hit rates, and UCB batch recommendations. |
| §2.2.2 / §3.2.2 | [run_representation_ablation.py](study/run_representation_ablation.py), [representation_ablation.py](study/analysis/representation_ablation.py) | Compares composition, one-coordinate, two-coordinate and random routes and assesses conductivity-seed sensitivity. [Saved results](study/results/representation_ablation/) include cost/AUC summaries, paired contrasts, hit rates and seed summaries. |
| §2.1–2.2 | [data_sources.py](study/analysis/data_sources.py), [`__init__.py`](study/analysis/__init__.py) | Resolve study paths, check historical input provenance and initialize the analysis package. These modules do not produce result tables. |
| Results check | [check_saved_results.py](study/check_saved_results.py) | Checks selected numerical results offline and prints a short summary to stdout. |

### Dataset implementations (§2.1.1)

Each implementation prepares the indicated dataset and defines its targets, grouping or splits. The linked CSV is the released metric output. Prediction execution uses the recorded experiment protocol; hosted inference is disabled in this public snapshot.

| Script | What it does | Saved output |
| --- | --- | --- |
| [calisol23.py](study/prediction_benchmarks/calisol23.py) | Cleans and aggregates CALiSol-23, applies the conductivity inclusion threshold, separates mol/L from mol/kg, and computes split and error-tail diagnostics. | [CALiSol-23 metrics](study/results/benchmarks/calisol23.csv) |
| [flores.py](study/prediction_benchmarks/flores.py) | Aggregates repeated liquid-electrolyte conditions and builds conductivity splits and symbolic-formula baselines. | [Flores metrics](study/results/benchmarks/flores.csv) |
| [clio.py](study/prediction_benchmarks/clio.py) | Aggregates measured replicates by formulation, audits the AEM table separately, and builds the static conductivity benchmark and matched GP. | [Clio prediction metrics](study/results/benchmarks/clio.csv) |
| [ionic_liquids.py](study/prediction_benchmarks/ionic_liquids.py) | Expands the nh and ILThermo species/temperature data into independent log-conductivity benchmarks. | [Ionic-liquid metrics](study/results/benchmarks/ionic_liquids.csv) |
| [chemarr.py](study/prediction_benchmarks/chemarr.py) | Computes polymer and salt descriptors, groups normalized polymer structures, and evaluates fold-level log-conductivity errors and ranks. | [ChemArr metrics](study/results/benchmarks/chemarr.csv) |
| [electrolytomics.py](study/prediction_benchmarks/electrolytomics.py) | Matches measured conditions to QSAR features and forms the log-conductivity prediction split. | [Electrolytomics metrics](study/results/benchmarks/electrolytomics.csv) |
| [aqueous.py](study/prediction_benchmarks/aqueous.py) | Reconstructs aqueous recipe features from stock-solution volumes, densities and molalities and groups observations by recipe. | [Aqueous metrics](study/results/benchmarks/aqueous.csv) |

### Objective-surrogate cases (§2.2.3–2.2.4)

| Script | What it does | Outputs and inputs |
| --- | --- | --- |
| [prepare_polymer_search.py](study/objective_search/ionic_polymer/prepare_polymer_search.py) | Validates the original workbooks and GP reference CSVs, standardizes the composite objective from observed labels and defines the candidate grid. | A working audit writes data and validation manifests under the case's `results/`. Raw inputs come from the Song et al. repository. |
| [evaluate_polymer_rankings.py](study/objective_search/ionic_polymer/evaluate_polymer_rankings.py) | Evaluates GP and saved predictive distributions in the restricted measured pool and summarizes full-grid ranks and normalization sensitivity. | Uses the original measured tables and prediction caches. The selected outputs are in the [ionic-polymer result directory](study/objective_search/ionic_polymer/results/). |
| [compare_polymer_acquisitions.py](study/objective_search/ionic_polymer/compare_polymer_acquisitions.py) | Compares acquisition rules, performs rank fusion, and evaluates coverage and known-score bounds. | A cached analysis writes ranking tables and associated figure-source CSVs under the case's `results/`. The release includes the ranking profile, target ranks and aggregate coverage. |
| [replay_clio_search.py](study/objective_search/clio/replay_clio_search.py) | Implements sequential feedback, GP acquisition, exact conditional random reference and trajectory audits for the measured Clio pool. | A working replay writes initial sets, traces and metrics under the case's `results/`. TabPFN trajectories require previously saved predictive distributions. |
| [compare_clio_acquisition_rules.py](study/objective_search/clio/compare_clio_acquisition_rules.py) | Compares acquisition rules on development starts and evaluates frozen rules on new starts. | [Saved validation outputs](study/objective_search/clio/acquisition_results/) include per-rule cost and budget summaries, paired differences, initial predictive diagnostics and the cost figure source. |
| [summarize_clio_validation_progress.py](study/objective_search/clio/summarize_clio_validation_progress.py) | Aggregates recorded new-start traces into mean best conductivity and hit rate by budget. | The working script requires trace and initialization tables. Its saved validation-progress CSV is under `acquisition_results/figure_source_data/`. |

The scientific analysis scripts retain experiment-level diagnostics. Manuscript layout, figure-assembly scripts and intermediate work directories are outside this snapshot. Core results use `tables/` for analysis CSVs and `figure_source_data/` for the values plotted in selected supplementary figures.

## Verification and inference scope

The retained test files check synthetic batch-search semantics ([core](study/tests/test_conductivity_guided_search.py)), [Clio sequential replay](study/objective_search/clio/test_clio_replay.py), [Clio acquisition calculations](study/objective_search/clio/test_clio_acquisition_rules.py), and [polymer acquisition math](study/objective_search/ionic_polymer/test_polymer_acquisitions.py). From the root:

```sh
PYTHONPATH=study python -m unittest discover -s study/tests
python -m unittest discover -s study/objective_search/clio
python -m unittest discover -s study/objective_search/ionic_polymer
```

[Upstream dataset links and input paths](study/datasets/README.md) identify the original sources. Original datasets, account configuration, prediction caches and historical local audit inputs are excluded. The legacy experiment runners require the corresponding input files or caches and are retained to document how the analyses were performed. They are not a turnkey full-replay pipeline from this snapshot.

The published AMPERE conductivity benchmark used the hosted TabPFN V3 regression checkpoint `tabpfn-v3-regressor-v3_default.ckpt`, `tabpfn-client` 0.5.3, eight estimators, and six inference seeds. The hosted inference and account-access implementations have been removed from the release. The model and client names here document the original run.
