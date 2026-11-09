"""clio measured-conductivity evaluation, six seeds on one fixed formulation split.

From project root:
  .venv/bin/python study/prediction_benchmarks/run_benchmark.py clio audit
  .venv/bin/python study/prediction_benchmarks/run_benchmark.py clio run --resume

22/8/8 is a project-defined 60/20/20 split, NOT a clio original protocol.
AEM simulation data is audited but never used as training/test labels here.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent / 'clio'
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT))
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from analysis.ampere_conductivity import API_MODEL_PATH, N_ESTIMATORS, SEEDS, api_predict, metrics, software_versions

SOURCE = PROJECT / 'datasets/clio/ECDMCEMC_optimization_march2021.csv'
AEM = SOURCE.with_name('AEM_training_data.csv')
FEATURES = ['salt_molality', 'EC_mass_fraction', 'DMC_cosolvent_ratio']
RENAMES = {'LiPF6_molalitt': FEATURES[0], 'Temp (C) 2': 'temperature_C'}
PROTOCOL = 'project_formulation_60_20_20'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def prepare_data(source=SOURCE, aem=AEM):
    raw = pd.read_csv(source, index_col=0).rename(columns=RENAMES)
    raw.index.name = 'raw_row_id'
    required = FEATURES + ['temperature_C', 'conductivity', 'GUID', 'run_order']
    if not set(required).issubset(raw.columns):
        raise ValueError('Unexpected source schema')
    if not np.isfinite(raw[FEATURES + ['temperature_C', 'conductivity']].to_numpy(float)).all():
        raise ValueError('Nonfinite input/target; no silent deletion')
    if raw.GUID.isna().any() or (raw.conductivity <= 0).any() or not raw.run_order.isin([1, 2, 3]).all():
        raise ValueError('Invalid GUID, target or run order')
    # A GUID denotes one physical record: do not treat duplicated exported versions
    # as independent experiments, and do not arbitrarily pick a conflicting value.
    for col in FEATURES + ['run_order', 'temperature_C']:
        if raw.groupby('GUID')[col].nunique().max() != 1:
            raise ValueError('GUID has conflicting composition/run/temperature')
    guid = raw.groupby('GUID', sort=True).agg(
        **{c: (c, 'first') for c in FEATURES}, temperature_C=('temperature_C', 'first'),
        run_order=('run_order', 'first'), conductivity=('conductivity', 'mean'),
        exported_rows=('conductivity', 'size'), unique_export_values=('conductivity', 'nunique'),
        exported_min=('conductivity', 'min'), exported_max=('conductivity', 'max')).reset_index()
    selected = guid[guid.run_order.isin([2, 3])].copy()
    # The paper's metrology excludes the first pass after the rinse. Combine all
    # eligible GUIDs at exactly the same composition before any data splitting.
    data = selected.groupby(FEATURES, sort=True).agg(
        conductivity=('conductivity', 'mean'), measurement_std=('conductivity', 'std'),
        n_measurements=('conductivity', 'size'), temperature_mean_C=('temperature_C', 'mean'),
        temperature_min_C=('temperature_C', 'min'), temperature_max_C=('temperature_C', 'max'),
        run_order_count=('run_order', 'nunique')).reset_index()
    data.insert(0, 'formulation_id', [f'F{i:03d}' for i in range(len(data))])
    data['condition_id'] = data.formulation_id
    mapping = raw.reset_index().merge(data[['formulation_id'] + FEATURES], on=FEATURES,
                                      validate='many_to_one')
    mapping['included_in_target'] = mapping.run_order.isin([2, 3])
    mapping = mapping.merge(guid[['GUID', 'exported_rows', 'unique_export_values']], on='GUID', validate='many_to_one')
    mapping['target_weight'] = np.where(mapping.included_in_target,
        1 / mapping.exported_rows / mapping.formulation_id.map(data.set_index('formulation_id').n_measurements), 0)
    conflicts = guid[guid.exported_rows > 1].copy()
    simulated = pd.read_csv(aem, index_col=0)
    audit = {
        'source_sha256': sha(source), 'raw_rows': len(raw), 'unique_GUIDs': len(guid),
        'run_order_counts': {str(k): int(v) for k, v in raw.run_order.value_counts().items()},
        'eligible_exported_rows': int(raw.run_order.isin([2, 3]).sum()),
        'eligible_unique_GUIDs': len(selected), 'formulations': len(data),
        'repeated_GUIDs': int((guid.exported_rows > 1).sum()),
        'conflicting_GUIDs': int((guid.unique_export_values > 1).sum()),
        'GUID_policy': 'same GUID counts once; conflicting exported target versions averaged, not resolved as ground truth',
        'aggregation': 'exclude run_order=1 by paper method; mean within GUID, then mean by exact three composition fields',
        'singleton_formulations': data.loc[data.n_measurements == 1, 'formulation_id'].tolist(),
        'incomplete_run23_formulations': data.loc[data.run_order_count < 2, 'formulation_id'].tolist(),
        'temperature_range_C': [float(selected.temperature_C.min()), float(selected.temperature_C.max())],
        'temperature_role': 'metadata only; near-ambient target, no temperature-response claim',
        'no_outlier_filter': True, 'no_imputation': True,
        'AEM': {'sha256': sha(aem), 'rows': len(simulated), 'used_for_training_or_evaluation': False},
        'paper': {'doi': '10.1038/s41467-022-32938-1',
                  'method_location': 'Methods: Clio hardware and metrology',
                  'original_static_split': None,
                  'prediction_metrics': 'No directly comparable held-out MAPE/MSE/RMSE/R2 table identified in inspected main text',
                  'supplementary_status': 'Full supplement not retrieved; no numerical prediction benchmark asserted'},
    }
    return data, mapping, guid, conflicts, audit


def make_split(data, seed=42):
    dev, test = train_test_split(np.arange(len(data)), test_size=.2, random_state=seed)
    train, validation = train_test_split(dev, test_size=.25, random_state=seed)
    result = data.copy()
    result['split'] = 'train'
    result.loc[validation, 'split'] = 'validation'
    result.loc[test, 'split'] = 'test'
    if not result.formulation_id.is_unique or set(train) & set(test):
        raise ValueError('Overlapping formulation groups')
    return result


def frame(query, values, seed, model):
    values = np.asarray(values, float).reshape(-1)
    if len(values) != len(query) or not np.isfinite(values).all():
        raise ValueError('Invalid prediction values')
    result = query[['formulation_id', 'split'] + FEATURES].copy()
    result['inference_seed'], result['model'] = seed, model
    result['true_mS_cm'], result['pred_mS_cm'] = query.conductivity.to_numpy(), values
    result['absolute_error'] = np.abs(result.pred_mS_cm - result.true_mS_cm)
    result['APE_percent'] = 100 * result.absolute_error / result.true_mS_cm
    return result


def evaluate(predictions):
    rows = []
    for (model, seed, split), p in predictions.groupby(['model', 'inference_seed', 'split'], sort=False):
        rows.append({'model': model, 'inference_seed': seed, 'split': split, 'n': len(p),
            'MSE': float(np.mean((p.pred_mS_cm - p.true_mS_cm)**2)),
            **metrics(p.true_mS_cm.to_numpy(), p.pred_mS_cm.to_numpy())})
    return pd.DataFrame(rows)


def baseline_predictions(train, query):
    # Local controls only. This GPR is not the paper's AEM-trained optimization surrogate.
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    kernel = ConstantKernel(1., (1e-3, 1e3)) * Matern(np.ones(3), (1e-2, 1e2), nu=2.5) + WhiteKernel(.01, (1e-6, 1))
    gpr = make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                n_restarts_optimizer=0, random_state=42))
    gpr.fit(train[FEATURES], train.conductivity)
    return {'Training mean': np.full(len(query), train.conductivity.mean()),
            'Local GPR (Matern 2.5)': gpr.predict(query[FEATURES])}


def summarize(predictions, scores, out):
    cols = ['MAPE', 'MAE', 'MSE', 'RMSE', 'R2', 'median_absolute_error', 'median_APE', 'APE_q75', 'APE_q90', 'APE_q95']
    summary = scores.groupby(['model', 'split'])[cols].agg(['mean', 'std', 'min', 'max'])
    summary.columns = [f'{a}_{b}' for a, b in summary.columns]
    summary = summary.reset_index().merge(scores.groupby(['model', 'split']).size().rename('n_seeds').reset_index())
    summary.to_csv(out/'tables/metrics_summary.csv', index=False)
    tab = predictions[predictions.model.eq('TabPFN')]
    sample = tab.groupby(['formulation_id', 'split'] + FEATURES).agg(
        true_mS_cm=('true_mS_cm', 'first'), prediction_mean=('pred_mS_cm', 'mean'),
        prediction_std=('pred_mS_cm', 'std'), prediction_min=('pred_mS_cm', 'min'),
        prediction_max=('pred_mS_cm', 'max'), mean_APE=('APE_percent', 'mean'),
        APE_std=('APE_percent', 'std'), n_seeds=('inference_seed', 'nunique')).reset_index()
    sample['error_rank'] = sample.groupby('split').mean_APE.rank(ascending=False, method='min').astype(int)
    sample.to_csv(out/'tables/sample_summary.csv', index=False)
    ensemble = frame(sample.rename(columns={'true_mS_cm':'conductivity'}), sample.prediction_mean, -2,
                     'TabPFN mean prediction across requested seeds')
    evaluate(ensemble).to_csv(out/'tables/ensemble_metrics.csv', index=False)
    return summary, sample


def verify(predictions, scores, plan, mapping, seeds, out):
    checks = []
    def check(name, ok): checks.append({'check': name, 'passed': bool(ok)})
    check('unique_formulations', plan.formulation_id.is_unique)
    check('raw_rows_map_to_one_split', mapping.merge(plan[['formulation_id','split']], on='formulation_id')
          .groupby('GUID')['split'].nunique().max() == 1)
    weights = mapping.groupby('formulation_id').target_weight.sum().sort_index()
    check('aggregation_weights_sum_to_one', np.allclose(weights, 1))
    weighted = (mapping.conductivity * mapping.target_weight).groupby(mapping.formulation_id).sum()
    check('aggregation_reconstructed', np.allclose(weighted.loc[plan.formulation_id], plan.conductivity))
    tab = predictions[predictions.model.eq('TabPFN')]
    check('unique_prediction_keys', not tab.duplicated(['inference_seed','formulation_id']).any())
    query = plan[plan.split.ne('train')]
    for seed in seeds:
        p = tab[tab.inference_seed.eq(seed)].set_index('formulation_id')
        check(f'seed_{seed}_coverage', set(p.index) == set(query.formulation_id))
        check(f'seed_{seed}_labels', np.allclose(p.loc[query.formulation_id].true_mS_cm, query.conductivity))
    for row in scores.itertuples():
        p = predictions[(predictions.model == row.model) & (predictions.inference_seed == row.inference_seed)
                        & (predictions.split == row.split)]
        e = p.pred_mS_cm.to_numpy() - p.true_mS_cm.to_numpy()
        check(f'metrics_{row.model}_{row.inference_seed}_{row.split}', np.allclose(
            [np.abs(e).mean(), np.mean(e**2), np.sqrt(np.mean(e**2)),
             100*np.mean(np.abs(e)/p.true_mS_cm.to_numpy())], [row.MAE,row.MSE,row.RMSE,row.MAPE], atol=1e-12))
    passed = all(c['passed'] for c in checks)
    write_json(out/'tables/result_audit.json', {'passed':passed, 'checks':checks})
    if not passed: raise ValueError('Result verification failed')


def plots(sample, scores, out):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.5,4))
    for split, g in sample.groupby('split'):
        ax.errorbar(g.true_mS_cm, g.prediction_mean, yerr=g.prediction_std.fillna(0), fmt='o',
                    ms=4, elinewidth=.8, label=f'{split} (n={len(g)})', alpha=.8)
    lo=min(sample.true_mS_cm.min(),sample.prediction_mean.min())
    hi=max(sample.true_mS_cm.max(),sample.prediction_mean.max())
    ax.plot([lo,hi],[lo,hi],'--',color='gray',lw=1)
    ax.set(xlabel='Measured conductivity (mS/cm)',ylabel='Predicted conductivity (mS/cm)',title='clio: seed mean and SD')
    ax.legend(frameon=False); fig.tight_layout()
    for ext in ['png','pdf']: fig.savefig(out/f'figures/parity.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    fig, axes=plt.subplots(1,2,figsize=(8,3.2))
    tab=scores[scores.model.eq('TabPFN')]
    for split,g in tab.groupby('split'):
        for ax,metric in zip(axes,['MAPE','RMSE']):
            ax.plot(g.inference_seed,g[metric],'o-',label=split,ms=4,lw=1)
            ax.set(xlabel='Inference seed',ylabel='MAPE (%)' if metric=='MAPE' else 'RMSE (mS/cm)')
            ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ['png','pdf']: fig.savefig(out/f'figures/seed_metrics.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)


def update_readme(audit, summary, config, plan):
    path=HERE.parent/'README.md'
    start,end='<!-- clio_RESULTS_START -->','<!-- clio_RESULTS_END -->'
    lines=[start,'## clio — Dave 等，2022','',
        '论文：[Autonomous optimization of non-aqueous Li-ion battery electrolytes via robotic experimentation and machine learning coupling]'
        '(https://doi.org/10.1038/s41467-022-32938-1)。Clio 机器人配合 Dragonfly 贝叶斯优化，'
        '在 LiPF₆／EC-DMC-EMC 液态体系中寻找高电导率配方，并做快充电池验证。后续可作为闭环配方优化参考。','',
        '### 本轮任务与数据处理','',
        f"- 仅评估真实实验：原表 {audit['raw_rows']} 行、{audit['unique_GUIDs']} 个 GUID、38组配方。197条 AEM 模拟数据不参与本轮训练或评价。",
        '- 原文 Methods 的测量规则是排除每组三遍中的第一遍，平均第二、三遍，原因是清洗残留；本项目按 `run_order` 实施。',
        f"- 保留第二、三遍的 {audit['eligible_exported_rows']} 行导出记录，对应 {audit['eligible_unique_GUIDs']} 个唯一 GUID，再按三项组成精确聚合为38条配方标签。",
        '- 原表3个 GUID 重复，其中2个有不同电导率版本。每个 GUID 先平均版本值、只计一次，再对配方平均；'
        '不挑选其中某个版本，冲突仍未从来源上解决。原始文件不改动，重复与冲突值留在 Output/data/guid_issues.csv。',
        f"- 有 {len(audit['singleton_formulations'])} 组配方仅有1个有效独立 GUID，无法计算测量标准差，保留并标记，不补值。",
        '- 输入为 LiPF₆ 浓度（mol/kg溶剂）、EC质量分数、DMC在非EC溶剂中的质量占比；'
        '实际温度只作元数据，目标是约26–28°C环境下的电导率（mS/cm），不研究温度响应。',
        '','### 划分与配置','',
        '- **本项目设置：22训练／8验证／8测试，划分种子42。** 沿用 flores 的60/20/20比例并取整；不是 ampere 的40/60划分，也不是 clio原文划分。',
        '- clio原文主要是顺序贝叶斯优化，先用5个初始样本，再逐轮测量、更新；未在正文规定本轮这样的静态预测划分。',
        '- 配方先聚合后拆分，同配方的全部原始测量归入同一集合。验证仅作报告，不参与训练、调参或模型选择。',
        f"- TabPFN 配置沿用 ampere_conductivity：`{API_MODEL_PATH}`、`n_estimators={N_ESTIMATORS}`；推理种子 {config['inference_seeds']}；原始标签，无手工缩放或对数变换。",
        '- 验证和测试合并查询，每个种子只调用一次预测，共6份缓存；默认复用缓存，不自动重试。',
        '- 免费本地对照为训练集均值及 GPR（Matern 2.5，训练集特征标准化，训练内MLE拟合核参数，无验证集调参）。'
        '本地GPR训练于实测训练集，不是原文用AEM数据拟合的优化代理。',
        '','### 常见指标','',
        '表中为每个种子指标的均值 ± 样本标准差（ddof=1），不是先平均预测再算分数。确定性本地对照仅算一次，以 — 表示标准差。','',
        '|集合|模型|种子数|MAPE (%)|MAE (mS/cm)|MSE|RMSE (mS/cm)|R²|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    def cell(r,m):
        mean,std=r[f'{m}_mean'],r[f'{m}_std']
        return f'{mean:.4f} ± {std:.4f}' if pd.notna(std) else f'{mean:.4f}（—）'
    for _,r in summary.iterrows():
        lines.append(f"|{r['split']}|{r.model}|{r.n_seeds}|"+'|'.join(cell(r,m) for m in ['MAPE','MAE','MSE','RMSE','R2'])+'|')
    lines += ['', '### 解释与完整性','',
        '- 本轮只看常见预测指标，不用原文优化加速因子、最优电导率或测量重复性误差充当预测误差基线。'
        '已检查正文，未找到可直接对照的静态留出误差表；完整补充资料未取回，因此不声称其一定没有相关指标。',
        '- 六种子衡量同一划分下的推理波动，不代表六次独立数据划分；每个验证/测试集只有8组配方，不能据小种子标准差推断总体泛化误差很小。',
        '- 数据来自优化过程，DMC富集、高电导率区域较密集，不是整个设计空间均匀采样；本轮未重现闭环优化。',
        '- 已保存逐种子指标、指标均值/标准差/范围、逐样本波动、种子平均预测的独立指标、数据哈希、代码快照和结果核验。',
        '','### 文件与重跑','',
        '- [脚本](clio.py) · [本地测试](test_experiments.py)',
        '- [逐种子指标](clio/results/tables/metrics.csv) · '
        '[汇总](clio/results/tables/metrics_summary.csv) · '
        '[样本波动](clio/results/tables/sample_summary.csv) · '
        '[结果核验](clio/results/tables/result_audit.json)',
        '- [预测图](clio/results/figures/parity.png) · '
        '[种子指标图](clio/results/figures/seed_metrics.png)',
        '- `Output/data` 包含原始行权重映射、GUID合并结果和冲突记录、聚合配方及划分；`Output/cache` 保存预测及校验值。',
        '- 项目根目录重跑：`.venv/bin/python study/prediction_benchmarks/run_benchmark.py clio run --resume`。',
        f"- 本次新增预测调用 {config['api_calls_this_execution']} 次，复用 {config['cache_hits_this_execution']} 份缓存。",'',end]
    section='\n'.join(lines)
    text=path.read_text() if path.exists() else '# prediction_benchmarks 实验结果\n'
    if start in text:
        a,b=text.index(start),text.index(end)+len(end);text=text[:a]+section+text[b:]
    else: text+='\n'+section+'\n'
    path.write_text(text)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['audit','run'])
    parser.add_argument('--output-dir',type=Path,default=HERE/'results')
    parser.add_argument('--inference-seeds',type=int,nargs='+',default=list(SEEDS))
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv)
    if len(args.inference_seeds)!=len(set(args.inference_seeds)): parser.error('Duplicate seeds')
    out=args.output_dir.resolve();out.mkdir(exist_ok=True,parents=True)
    for folder in ['data','tables','cache','figures','snapshots']: (out/folder).mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',force=True,
                        handlers=[logging.StreamHandler(),logging.FileHandler(out/'execution.log')])
    logging.getLogger('fontTools').setLevel(logging.WARNING)
    data,mapping,guid,issues,audit=prepare_data();plan=make_split(data)
    contract={'source_sha256':audit['source_sha256'],'model_path':API_MODEL_PATH,'n_estimators':N_ESTIMATORS,
              'features':FEATURES,'target':'conductivity mS/cm, near ambient','aggregation':audit['aggregation'],
              'GUID_policy':audit['GUID_policy'],'protocol':PROTOCOL,'split_seed':42,'schema_version':1,
              'split_hash':hashlib.sha256(plan.to_csv(index=False).encode()).hexdigest()}
    signature=digest(contract)
    config=contract|{'signature':signature,'inference_seeds':args.inference_seeds,'status':'audit_completed',
             'run_type':'fixed_split_seed_evaluation','planned_prediction_calls':len(args.inference_seeds),
             'api_calls_this_execution':0,'cache_hits_this_execution':0,'api_calls_total':0,
             'started_utc':datetime.now(timezone.utc).isoformat(),'software_versions':software_versions(),
             'script_sha256':sha(__file__),'shared_step01_sha256':sha(PROJECT/'analysis/ampere_conductivity.py')}
    cp=out/'run_config.json'
    if cp.exists():
        previous=json.loads(cp.read_text())
        if previous['signature']!=signature: raise ValueError('Changed prediction contract; use a different Output')
        if args.mode=='audit' and previous['status']!='audit_completed': raise FileExistsError('Audit cannot overwrite run results')
        if args.mode=='run' and previous['status']=='completed' and not args.resume: raise FileExistsError('Use --resume to avoid paid duplicate calls')
        config['api_calls_total']=previous['api_calls_total']
    for file in [Path(__file__),PROJECT/'analysis/ampere_conductivity.py']:
        dest=out/'snapshots'/f'{file.stem}_{sha(file)[:12]}.py'
        if not dest.exists():dest.write_bytes(file.read_bytes())
    for name,df in [('formulations',data),('raw_to_formulation',mapping),('guid_collapsed',guid),('guid_issues',issues),('split_manifest',plan)]:
        df.to_csv(out/f'data/{name}.csv',index=False)
    write_json(out/'data/data_audit.json',audit);write_json(cp,config)
    train=plan[plan.split.eq('train')];query=plan[plan.split.ne('train')]
    logging.info('clio: %s raw -> %s eligible GUIDs -> %s formulations; split=%s',len(mapping),audit['eligible_unique_GUIDs'],len(data),plan.split.value_counts().to_dict())
    try:
        parts=[frame(query,values,-1,model) for model,values in baseline_predictions(train,query).items()]
        if args.mode=='run':
            for seed in args.inference_seeds:
                cache=out/f'cache/seed{seed}.csv';meta_path=cache.with_suffix('.json')
                call_signature=digest(contract|{'inference_seed':seed})
                if args.resume and cache.exists() and meta_path.exists():
                    meta=json.loads(meta_path.read_text());saved=pd.read_csv(cache)
                    if meta['signature']!=call_signature or meta['sha256']!=sha(cache): raise ValueError('Invalid cache signature/checksum')
                    if (not np.array_equal(saved.formulation_id,query.formulation_id)
                        or not np.allclose(saved[FEATURES],query[FEATURES],atol=0,rtol=1e-12)
                        or not np.allclose(saved.true_mS_cm,query.conductivity,atol=0,rtol=1e-12)):
                        raise ValueError('Cached inputs or labels changed')
                    values=saved.pred_mS_cm.to_numpy();config['cache_hits_this_execution']+=1
                    logging.info('Reused seed=%s',seed)
                else:
                    if cache.exists() or meta_path.exists(): raise FileExistsError('Partial/existing cache; inspect before any paid retry')
                    config['api_calls_this_execution']+=1;config['api_calls_total']+=1
                    write_json(cp,config|{'status':'running'})
                    logging.info('TabPFN seed=%s: train=%s query=%s (one prediction call)',seed,len(train),len(query))
                    started=time.monotonic()
                    values=api_predict(train[FEATURES],train.conductivity.to_numpy(),query[FEATURES],seed,PROJECT.parent)
                    frame(query,values,seed,'TabPFN').to_csv(cache,index=False)
                    write_json(meta_path,{'signature':call_signature,'sha256':sha(cache),'elapsed_seconds':time.monotonic()-started})
                parts.append(frame(query,values,seed,'TabPFN'))
        predictions=pd.concat(parts,ignore_index=True);scores=evaluate(predictions)
        predictions.to_csv(out/'tables/predictions.csv',index=False);scores.to_csv(out/'tables/metrics.csv',index=False)
        if args.mode=='run':
            summary,sample=summarize(predictions,scores,out)
            verify(predictions,scores,plan,mapping,args.inference_seeds,out)
            plots(sample,scores,out)
            update_readme(audit,summary,config,plan)
            config['status']='completed';config['six_inference_seeds_completed']=set(args.inference_seeds)==set(SEEDS)
        logging.info('Completed %s; new API calls=%s, cache hits=%s',config['status'],config['api_calls_this_execution'],config['cache_hits_this_execution'])
    except Exception as exc:
        config['status']='failed';config['error_type']=type(exc).__name__
        logging.error('Stopped (%s); no automatic retry or model substitution',type(exc).__name__)
        raise
    finally:
        config['finished_utc']=datetime.now(timezone.utc).isoformat();write_json(cp,config)


if __name__=='__main__':main()
