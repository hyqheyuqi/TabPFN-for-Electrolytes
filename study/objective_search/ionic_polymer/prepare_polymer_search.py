"""ionic_polymer data audit and bounded first-batch BO replay. Never downloads author code."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import logging
import os
from pathlib import Path
import re
import sys
import time
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT))
os.environ.setdefault('MPLCONFIGDIR', str(PROJECT / '.matplotlib'))

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from analysis.ampere_conductivity import API_MODEL_PATH, N_ESTIMATORS, apply_style

DATA = PROJECT / 'datasets/ionic_polymer'
COMMIT = 'd2c5476e3a9bd54bc139ed59830aac461ee50770'
FEATURES = ['P1', 'P2', 'P3', 'P4']
OUTCOMES = ['conductivity_mS_cm', 'capacity_mAh_g', 'high_voltage_pass']
SEEDS = [100, 200, 300, 400, 500, 600]
KAPPA = 1.96
MAX_CREDITS = 100000
MODEL_CONFIG = dict(model_path=API_MODEL_PATH, n_estimators=N_ESTIMATORS,
                    categorical_features_indices=[1, 2])
SOURCES = {
    'bo_inputs_round1.xlsx': ('input/data/bo_inputs_round1.xlsx', '4d99e07a4892dd6e1a46cc8bf9e8de6e548c7b76'),
    'bo_inputs_round2.xlsx': ('input/data/bo_inputs_round2.xlsx', '3bb365ef0f05f796f92a0ce6bc997d2064fd3023'),
    'bo_Final_parameters.xlsx': ('bo_Final_parameters.xlsx', 'cdf4a1922397ab797fa1a59cf87183aaa24be89b'),
    'af_results_bo_inputs_round1.csv': ('results/data/af_results_bo_inputs_round1.csv', '94e822bd9683bdd189ad7ac529d097b3d9eeebac'),
    'af_results_bo_inputs_round2.csv': ('results/data/af_results_bo_inputs_round2.csv', '7975915e95b257bbc1a45cf2164b8736fbaf3363'),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def zscore_objective(observed_outcomes):
    """Only revealed labels may be passed here. Matches author ddof=0 scaling."""
    a = np.asarray(observed_outcomes, dtype=float)
    mu, sd = a.mean(axis=0), a.std(axis=0, ddof=0)
    if a.ndim != 2 or a.shape[1] != 3 or not np.isfinite(a).all() or (sd <= 0).any():
        raise ValueError('Observed outcome normalization is undefined')
    return ((a - mu) / sd).sum(axis=1), mu, sd


def prepare_data():
    manifest = []
    for name, (remote, expected_blob) in SOURCES.items():
        path = DATA / name
        payload = path.read_bytes()
        blob = hashlib.sha1(f'blob {len(payload)}\0'.encode() + payload).hexdigest()
        if blob != expected_blob:
            raise ValueError(f'Source Git blob mismatch: {name}')
        manifest.append(dict(file=name, url=f'https://raw.githubusercontent.com/WangGroupFDU/Bayes_optimization/{COMMIT}/{remote}',
                             commit=COMMIT, bytes=len(payload), git_blob=blob, sha256=sha(path),
                             role='measured_data' if path.suffix == '.xlsx' else 'author_model_predictions_not_labels'))
    books = {name: pd.read_excel(DATA / name) for name in list(SOURCES)[:3]}
    final = books['bo_Final_parameters.xlsx']
    if final.shape != (15, 7) or not np.isfinite(final.to_numpy(float)).all():
        raise ValueError('Unexpected final experiment table')
    for name, n in [('bo_inputs_round1.xlsx', 7), ('bo_inputs_round2.xlsx', 11)]:
        if len(books[name]) != n or not np.array_equal(books[name].iloc[:, :7], final.iloc[:n, :7]):
            raise ValueError('Cumulative input/final row alignment changed')
    d = final.copy()
    d.columns = FEATURES + OUTCOMES
    d['row_id'] = np.arange(len(d))
    d['candidate_id'] = d[FEATURES].astype(int).astype(str).agg(''.join, axis=1)
    if not d.candidate_id.is_unique or not d.high_voltage_pass.isin([0, 1]).all():
        raise ValueError('Duplicate encoded formulation or invalid binary outcome')
    d['paper_cohort'] = d.row_id.lt(14)
    d['historical_batch'] = np.select([d.row_id.lt(7), d.row_id.lt(11), d.row_id.lt(14)], [0, 1, 2], default=3)
    d['polymer_percent'] = d.P1.map({1: 5, 2: 10, 3: 15})
    d['ionic_liquid'] = d.P2.map({1: 'C2mimBF4', 2: 'C2mimTFO', 3: 'C2mimFSI', 4: 'C2mimTFSI'})
    d['salt_pair'] = d.P3.map({1: 'FSI:TFSI', 2: 'FSI:BF4', 3: 'FSI:TFO', 4: 'TFSI:BF4', 5: 'TFSI:TFO', 6: 'BF4:TFO'})
    d['salt_ratio'] = d.P4.map(dict(enumerate([0, .2, .4, .5, .6, .8, 1], 1)))
    if d[['polymer_percent', 'ionic_liquid', 'salt_pair', 'salt_ratio']].isna().any().any():
        raise ValueError('Unknown encoded level')
    reference_y, reference_mu, reference_sd = zscore_objective(d.loc[d.paper_cohort, OUTCOMES])
    # Evaluator-only columns. Never supplied to model fitting or candidate scoring.
    d['evaluation_score'] = ((d[OUTCOMES] - reference_mu) / reference_sd).sum(axis=1)
    if d.loc[d.evaluation_score.idxmax(), 'candidate_id'] != '1415':
        raise ValueError('Published-reference winner changed')
    means = d[d.paper_cohort].groupby('historical_batch').evaluation_score.mean().to_numpy()
    if not np.allclose(means, [-1.14, .81, 1.57], atol=.011):
        raise ValueError('Cannot reproduce published batch averages within rounding')
    audit = dict(passed=True, measured_rows=15, paper_figure_rows=14, cumulative_rows=[7, 11, 15],
                 paper_batches=[7, 4, 3], initial_rows=7, unique_encoded_formulations=15,
                 extra_row='1321; all three outcomes zero; reason for omission from paper figure unresolved',
                 extra_column_ignored=list(books['bo_inputs_round2.xlsx'].columns[7:].astype(str)),
                 paper_batch_means_recalculated=means.tolist(), paper_batch_means_reported=[-1.14, .81, 1.57],
                 retrospective_evaluation_mean=reference_mu.tolist(), retrospective_evaluation_std=reference_sd.tolist(),
                 reference_scale_role='evaluation_only_never_in_prediction_contract', source_manifest=manifest)
    return d, audit


def gp_predict(x_train, y_train, x_query, optimize=False):
    """Independent RBF implementation. GPy export has 1e-8 jitter + 1e-10 noise."""
    kernel = RBF(length_scale=1., length_scale_bounds=(.05, 20.) if optimize else 'fixed')
    model = GaussianProcessRegressor(kernel=kernel, alpha=1.01e-8, optimizer='fmin_l_bfgs_b' if optimize else None,
                                    n_restarts_optimizer=0, normalize_y=False, random_state=0)
    model.fit(np.asarray(x_train, float), np.asarray(y_train, float))
    mean, std = model.predict(np.asarray(x_query, float), return_std=True)
    std = np.sqrt(std**2 + 1e-10)
    return dict(mean=mean, std=std, ucb=mean + KAPPA * std,
                q025=mean + norm.ppf(.025) * std, q975=mean + norm.ppf(.975) * std,
                q05=mean + norm.ppf(.05) * std, q95=mean + norm.ppf(.95) * std,
                fitted_kernel=str(model.kernel_))


def audit_gp_reference(d):
    checks = []
    for round_number, n in [(1, 7), (2, 11)]:
        r = pd.read_csv(DATA / f'af_results_bo_inputs_round{round_number}.csv')
        y, _, _ = zscore_objective(d.iloc[:n][OUTCOMES])
        pred = gp_predict(d.iloc[:n][FEATURES], y, r[FEATURES])
        errors = {'mean': float(np.max(np.abs(pred['mean'] - r['mean']))),
                  'variance': float(np.max(np.abs(pred['std']**2 - r.variance))),
                  'ucb': float(np.max(np.abs(pred['ucb'] - r.ucb)))}
        checks.append(dict(round=round_number, queried_rows=len(r), max_absolute_errors=errors, passed=max(errors.values()) < 1e-9))
    if not all(c['passed'] for c in checks):
        raise ValueError('Independent GPR failed exported author-reference check')
    return checks


def usage():
    raise RuntimeError("Hosted inference and account access are disabled in this public snapshot.")


def full_candidate_grid():
    grid = pd.DataFrame(itertools.product(range(1, 4), range(1, 5), range(1, 7), range(1, 8)), columns=FEATURES)
    grid['candidate_id'] = grid.astype(str).agg(''.join, axis=1)
    return grid


def initial_prediction_contract(d, seed, scope='measured'):
    observed = d.iloc[:7]
    candidates = (full_candidate_grid().loc[lambda g: ~g.candidate_id.isin(observed.candidate_id)]
                  if scope == 'full_grid' else d.iloc[7:])
    y, mu, sd = zscore_objective(observed[OUTCOMES])
    return dict(model=MODEL_CONFIG, seed=int(seed), output_type='full', kappa=KAPPA,
                training_features=observed[FEATURES].to_numpy().tolist(), training_target=y.tolist(),
                observed_mean=mu.tolist(), observed_std=sd.tolist(),
                query_features=candidates[FEATURES].to_numpy().tolist(), query_ids=candidates.candidate_id.tolist())


def live_distribution(contract):
    raise RuntimeError("Hosted inference and account access are disabled in this public snapshot.")


def validate_prediction(result, n):
    for name in ['mean', 'std', 'ucb', 'q025', 'q05', 'q95', 'q975']:
        if np.asarray(result[name]).shape != (n,) or not np.isfinite(result[name]).all():
            raise ValueError(f'Invalid prediction {name}')
    if (result['std'] < 0).any() or np.any(result['q025'] > result['q975']):
        raise ValueError('Invalid predictive uncertainty')


def cached_prediction(d, seed, out, resume, scope='measured'):
    contract = initial_prediction_contract(d, seed, scope)
    signature = fingerprint(contract)
    prefix = 'full_grid_' if scope == 'full_grid' else ''
    stem = out / 'cache' / f'{prefix}initial7_seed{seed}'
    meta_path = stem.with_suffix('.json')
    csv_path = stem.with_suffix('.csv')
    raw_path = stem.with_suffix('.npz')
    if meta_path.exists():
        if not resume:
            raise FileExistsError('Existing prediction cache; use --resume')
        md = json.loads(meta_path.read_text())
        if md['signature'] != signature or md['csv_sha256'] != sha(csv_path) or md['raw_sha256'] != sha(raw_path):
            raise ValueError('Prediction cache hash/contract mismatch')
        frame = pd.read_csv(csv_path)
        if frame.candidate_id.astype(str).tolist() != contract['query_ids']:
            raise ValueError('Cached query alignment changed')
        validate_prediction({k: frame[k].to_numpy() for k in ['mean', 'std', 'ucb', 'q025', 'q05', 'q95', 'q975']}, len(frame))
        return frame, False, md
    if csv_path.exists() or raw_path.exists() or stem.with_suffix('.attempt.json').exists():
        raise FileExistsError('Partial/previously attempted paid request; no automatic retry')
    write_json(stem.with_suffix('.attempt.json'), dict(signature=signature, contract=contract, started_utc=datetime.now(timezone.utc).isoformat()))
    started = time.monotonic()
    result, raw, metadata = live_distribution(contract)
    elapsed = time.monotonic() - started
    validate_prediction(result, len(contract['query_ids']))
    frame = pd.DataFrame({k: v for k, v in result.items()})
    frame.insert(0, 'candidate_id', contract['query_ids'])
    frame.to_csv(csv_path, index=False)
    np.savez_compressed(raw_path, **raw)
    md = dict(signature=signature, contract=contract, csv_sha256=sha(csv_path), raw_sha256=sha(raw_path),
              wall_seconds=elapsed, source_sha256=sha(__file__), metadata=metadata)
    write_json(meta_path, md)
    return frame, True, md


def save_audit(d, audit, out):
    for folder in ['data', 'tables', 'figures', 'figure_source_data', 'cache', 'snapshots']:
        (out / folder).mkdir(parents=True, exist_ok=True)
    d.to_csv(out / 'data/model_table_evaluator_only.csv', index=False)
    audit['gpr_reference_reproduction'] = audit_gp_reference(d)
    write_json(out / 'data/data_audit.json', audit)
    write_json(out / 'data/source_manifest.json', audit['source_manifest'])
    source = out / 'snapshots' / f'prepare_polymer_search_{sha(__file__)[:12]}.py'
    if not source.exists():
        source.write_bytes(Path(__file__).read_bytes())
    protocol = HERE / 'pilot_protocol.md'
    (out / 'snapshots' / f'pilot_protocol_{sha(protocol)[:12]}.md').write_bytes(protocol.read_bytes())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['audit', 'run', 'report'])
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--scope', choices=['measured', 'full_grid'], default='full_grid')
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    parser.add_argument('--output-dir', type=Path, default=HERE / 'results')
    args = parser.parse_args(argv)
    if len(args.seeds) != len(set(args.seeds)) or not set(args.seeds).issubset(SEEDS):
        raise ValueError('Only the six frozen, unique inference seeds are supported')
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    logging.getLogger('fontTools').setLevel(logging.WARNING)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', force=True,
                        handlers=[logging.StreamHandler(), logging.FileHandler(out / 'execution.log')])
    d, audit = prepare_data()
    save_audit(d, audit, out)
    if args.mode == 'audit':
        logging.info('Audit passed; independent GPR matches two author CSVs on 1008 rows')
        return
    if args.mode == 'report':
        generate_report(d, audit, out)
        return
    before = usage()
    history_path = out / 'quota_history.json'
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    initial_used = history[0]['used'] if history else before['used']
    history.append(before)
    write_json(history_path, history)
    state = dict(run_type='feasibility_pilot', status='running', planned_seeds=SEEDS,
                 requested_seeds=args.seeds, max_new_prediction_calls=7, max_credits=MAX_CREDITS, scope=args.scope,
                 initial_usage=initial_used, new_calls=0, cache_hits=0,
                 model=MODEL_CONFIG, client_version=importlib.metadata.version('tabpfn-client'),
                 paper_reference='14 observed formulations; retrospective common normalization',
                 primary_pool=504 if args.scope == 'full_grid' else 15,
                 query_rows=len(initial_prediction_contract(d, args.seeds[0], args.scope)['query_ids']),
                 source_sha256=sha(__file__))
    write_json(out / 'run_config.json', state)
    try:
        prefix = 'full_grid_' if args.scope == 'full_grid' else ''
        for seed in args.seeds:
            cached = (out / 'cache' / f'{prefix}initial7_seed{seed}.json').exists()
            attempted = len(list((out / 'cache').glob('*.attempt.json')))
            if not cached and attempted >= 7:
                raise RuntimeError('Frozen maximum of seven prediction attempts reached')
            if not cached and (history[-1]['used'] - initial_used >= MAX_CREDITS or history[-1]['remaining'] < 10000):
                raise RuntimeError('Pilot credit budget exhausted')
            logging.info('TabPFN seed=%s: %s', seed, 'reuse verified cache' if cached else f"one prediction request, train=7 query={state['query_rows']}")
            frame, new, md = cached_prediction(d, seed, out, args.resume, args.scope)
            state['new_calls'] += int(new)
            state['cache_hits'] += int(not new)
            if new:
                current = usage()
                current.update(seed=seed, delta=current['used'] - history[-1]['used'])
                history.append(current)
                write_json(history_path, history)
                logging.info('seed=%s complete: %.2fs, credits delta=%s', seed, md['wall_seconds'], current['delta'])
                if current['used'] - initial_used > MAX_CREDITS:
                    raise RuntimeError('Pilot credit cap exceeded; no further calls')
            write_json(out / 'run_config.json', state)
        completed = sorted(int(p.stem.split('seed')[-1]) for p in (out / 'cache').glob(f'{prefix}initial7_seed*.json') if '.attempt' not in p.name)
        state.update(status='completed' if completed == SEEDS else 'partial', completed_seeds=completed,
                     credits_consumed=history[-1]['used'] - initial_used)
    except Exception as exc:
        state.update(status='stopped', error_type=type(exc).__name__, reason=str(exc))
        raise
    finally:
        write_json(out / 'run_config.json', state)
    logging.info('Predictive distribution pilot status=%s', state['status'])
    if 'generate_report' in globals():
        generate_report(d, audit, out)


from evaluate_polymer_rankings import generate_report


if __name__ == '__main__':
    main()
