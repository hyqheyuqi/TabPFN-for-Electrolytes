"""CALiSol-23: separate mol/kg and mol/L models; fixed 60/20/20, six seeds.

Run from project root:
  .venv/bin/python study/prediction_benchmarks/run_benchmark.py calisol23 audit
  .venv/bin/python study/prediction_benchmarks/run_benchmark.py calisol23 run --resume

Drop missing/zero/nonpositive labels per user instruction. No concentration
conversion, manual target scaling, log transform, or test-driven selection.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent / 'calisol23'
PROJECT=HERE.parents[1]
sys.path.insert(0,str(PROJECT))
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from analysis.ampere_conductivity import API_MODEL_PATH, N_ESTIMATORS, SEEDS, api_predict, metrics, software_versions

SOURCE=PROJECT/'datasets/calisol23/CALiSol-23/calisol23_DOI_10.11583DTU.c.6929599.csv'
# Fixed source dictionary, not selected using targets or evaluation scores.
SOLVENTS = ['EC', 'PC', 'DMC', 'EMC', 'DEC', 'DME', 'DMSO', 'AN', 'MOEMC', 'TFP', 'EA', 'MA', 'FEC', 'DOL', '2-MeTHF', 'DMM', 'Freon 11', 'Methylene chloride', 'THF', 'Toluene', 'Sulfolane', '2-Glyme', '3-Glyme', '4-Glyme', '3-Me-2-Oxazolidinone', '3-MeSulfolane', 'Ethyldiglyme', 'DMF', 'Ethylbenzene', 'Ethylmonoglyme', 'Benzene', 'g-Butyrolactone', 'Cumene', 'Propylsulfone', 'Pseudocumeme', 'TEOS', 'm-Xylene', 'o-Xylene']
NUMERIC=['T','c']+SOLVENTS
LEVELS={'salt':['LiAsF6','LiBF4','LiBMB','LiBOB','LiBPFPB','LiCF3SO3','LiClO4','LiFSI',
                'LiN(CF3SO2)2','LiPDI','LiPF6','LiTDI','LiTFSI'],
        'solvent ratio type':['mol','v','w']}
FEATURES=NUMERIC+[f'{key}={value}' for key,values in LEVELS.items() for value in values]
KEYS=['c units','salt','solvent ratio type']+NUMERIC
GROUPS={'mol/kg':'mol_kg','mol/l':'mol_L'}
MIN_CONDUCTIVITY=0.1  # explicit study-range boundary, not a tuned MAPE denominator


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()
def write_json(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def encode(data):
    x=data[NUMERIC].copy().astype(float)
    for key,values in LEVELS.items():
        if not data[key].isin(values).all():raise ValueError(f'Unknown category: {key}')
        for value in values:x[f'{key}={value}']=data[key].eq(value).astype(float)
    return x[FEATURES]


def prepare_data(source=SOURCE,minimum=MIN_CONDUCTIVITY):
    if not np.isfinite(minimum) or minimum<0:raise ValueError("Minimum conductivity must be finite and nonnegative")
    raw=pd.read_csv(source,index_col=0);raw.index.name='raw_row_id'
    if len(SOLVENTS)!=38 or not set(KEYS+['doi','k']).issubset(raw.columns):raise ValueError('Unexpected schema')
    cleaned=raw.copy()
    for col in ['salt','c units','solvent ratio type','doi']:
        if cleaned[col].isna().any():raise ValueError(f'Missing metadata: {col}')
        cleaned[col]=cleaned[col].str.strip()
    if not cleaned['c units'].isin(GROUPS).all():raise ValueError('Unknown concentration unit')
    if not np.isfinite(cleaned[NUMERIC].to_numpy(float)).all():raise ValueError('Nonfinite features')
    if (cleaned[SOLVENTS]<0).any().any() or (cleaned[SOLVENTS].sum(axis=1)<=0).any():raise ValueError('Invalid solvent fractions')
    reason=np.select([cleaned.k.isna(),~np.isfinite(cleaned.k),cleaned.k.eq(0),cleaned.k.lt(0),
                      cleaned.c.lt(0),cleaned['T'].le(0),cleaned.k.lt(minimum)],
                     ['missing_conductivity','nonfinite_conductivity','zero_conductivity','negative_conductivity',
                      'negative_concentration','nonpositive_kelvin_temperature','below_conductivity_range'],default='included')
    cleaned['exclusion_reason']=reason
    retained=cleaned[cleaned.exclusion_reason.eq('included')].copy()
    # Aggregate exact identical model inputs (including unit and ratio basis)
    # before the random condition split, irrespective of DOI. No rounding.
    data=retained.groupby(KEYS,sort=True,dropna=False).agg(
        conductivity=('k','mean'),measurement_std=('k','std'),n_measurements=('k','size'),
        sources=('doi',lambda s:' | '.join(sorted(set(s))))).reset_index()
    data.insert(0,'condition_id',[f'C{i:05d}' for i in range(len(data))])
    data['group']=data['c units'].map(GROUPS)
    formulation_keys=[key for key in KEYS if key!='T']
    data['formulation_id']=pd.factorize(pd.MultiIndex.from_frame(data[formulation_keys]),sort=True)[0]
    x=encode(data)
    for col in x.columns:
        if col not in data.columns:data[col]=x[col].to_numpy()
    mapping=cleaned.reset_index().merge(data[KEYS+['condition_id','formulation_id','group']],on=KEYS,
                                      how='left',validate='many_to_one')
    # Dropped measurements may share a condition with retained ones; mapping does
    # not grant them target membership.
    audit={'source_sha256':sha(source),'raw_rows':len(raw),'retained_rows':len(retained),
        'excluded_rows':int((cleaned.exclusion_reason!='included').sum()),
        'exclusion_counts':{str(k):int(v) for k,v in cleaned.exclusion_reason.value_counts().items()},
        'conditions':len(data),'formulations':int(data.formulation_id.nunique()),
        'group_conditions':{str(k):int(v) for k,v in data.group.value_counts().items()},
        'group_raw_retained_rows':{GROUPS[k]:int(v) for k,v in retained['c units'].value_counts().items()},
        'whitespace_salt_rows':int((raw.salt!=cleaned.salt).sum()),
        'salt_policy':'strip whitespace only; no unverified chemical synonym merge',
        'solvent_fraction_sum_range':[float(retained[SOLVENTS].sum(axis=1).min()),float(retained[SOLVENTS].sum(axis=1).max())],
        'solvent_policy':'original fractions retained; no renormalization or ratio-basis conversion',
        'aggregation':'mean at exact same unit, salt, ratio basis, temperature, concentration, 38 solvent fractions',
        'target':f'raw conductivity mS/cm, at least {minimum}, positive after exclusion',
        'minimum_conductivity_mS_cm':minimum,
        'tiny_positive_records_below_1e_12_mS_cm':data.loc[data.conductivity.lt(1e-12),
            ['condition_id','group','conductivity','T','c','salt','sources']].to_dict('records'),
        'tiny_value_policy':f'exclude raw conductivity below {minimum} mS/cm; no denominator floor; no filtering by prediction error',
        'features':FEATURES,'feature_count':len(FEATURES),'doi_used_as_feature':False,
        'concentration_policy':'mol/kg and mol/L fitted and evaluated independently; no density-based conversion',
        'paper':{'doi':'10.1038/s41597-024-03575-8','kind':'dataset descriptor, not a prediction benchmark'},
        'limitation':'random condition interpolation; same formulation at other temperatures and same DOI can cross splits'}
    return data,mapping,audit


def make_split(data,seed=42):
    pieces=[]
    for group,part in data.groupby('group',sort=False):
        part=part.copy()
        dev,test=train_test_split(np.arange(len(part)),test_size=.2,random_state=seed)
        train,val=train_test_split(dev,test_size=.25,random_state=seed)
        split=np.full(len(part),'train',dtype=object);split[val]='validation';split[test]='test'
        part['split']=split;part['split_seed']=seed;pieces.append(part)
    plan=pd.concat(pieces,ignore_index=True)
    if not plan.condition_id.is_unique:raise ValueError('Duplicate condition assignments')
    return plan


def frame(query,values,seed,model):
    values=np.asarray(values,float).reshape(-1)
    if len(values)!=len(query) or not np.isfinite(values).all():raise ValueError('Invalid predictions')
    result=query[['condition_id','formulation_id','group','split','sources','T','c','salt','solvent ratio type']].copy()
    result['model'],result['inference_seed']=model,seed
    result['true_mS_cm'],result['pred_mS_cm']=query.conductivity.to_numpy(),values
    result['absolute_error']=np.abs(result.pred_mS_cm-result.true_mS_cm)
    result['APE_percent']=100*result.absolute_error/result.true_mS_cm
    return result


def evaluate(predictions,extra=None):
    keys=['group','model','inference_seed','split']+([] if extra is None else [extra])
    rows=[]
    for key,p in predictions.groupby(keys,sort=False,observed=True):
        rows.append(dict(zip(keys,key))|{'n':len(p),'MSE':float(np.mean((p.pred_mS_cm-p.true_mS_cm)**2)),
                    **metrics(p.true_mS_cm.to_numpy(),p.pred_mS_cm.to_numpy())})
    return pd.DataFrame(rows)


def summarize(predictions,scores,out):
    keys=['group','model','split'];cols=[c for c in scores if c not in keys+['n','inference_seed']]
    summary=scores.groupby(keys)[cols].agg(['mean','std','min','max'])
    summary.columns=[f'{a}_{b}' for a,b in summary.columns]
    summary=summary.reset_index().merge(scores.groupby(keys).size().rename('n_seeds').reset_index())
    summary.to_csv(out/'tables/metrics_summary.csv',index=False)
    tab=predictions[predictions.model.eq('TabPFN')]
    sample=tab.groupby(['group','split','condition_id','formulation_id']).agg(
        true_mS_cm=('true_mS_cm','first'),prediction_mean=('pred_mS_cm','mean'),
        prediction_std=('pred_mS_cm','std'),prediction_min=('pred_mS_cm','min'),
        prediction_max=('pred_mS_cm','max'),mean_APE=('APE_percent','mean'),APE_std=('APE_percent','std'),
        n_seeds=('inference_seed','nunique')).reset_index()
    sample.to_csv(out/'tables/sample_summary.csv',index=False)
    ensemble=sample.rename(columns={'prediction_mean':'pred_mS_cm'}).copy()
    ensemble['model'],ensemble['inference_seed']='TabPFN mean prediction across requested seeds',-2
    evaluate(ensemble).to_csv(out/'tables/ensemble_metrics.csv',index=False)
    # Predefined conductivity ranges expose near-zero MAPE sensitivity; no floor,
    # clipping, exclusions, or model choice based on these post-evaluation views.
    banded=tab.copy()
    banded['conductivity_band']=pd.cut(banded.true_mS_cm,[0,.01,.1,1,10,np.inf],right=False,
                     labels=['0<k<0.01','0.01<=k<0.1','0.1<=k<1','1<=k<10','k>=10'])
    evaluate(banded,'conductivity_band').to_csv(out/'tables/metrics_by_conductivity.csv',index=False)
    return summary,sample


def verify(predictions,scores,plan,mapping,seeds,out):
    checks=[]
    def check(name,ok):checks.append({'check':name,'passed':bool(ok)})
    check('unique_condition_ids',plan.condition_id.is_unique)
    retained=mapping[mapping.exclusion_reason.eq('included')]
    recovered=retained.groupby('condition_id').k.mean()
    check('aggregation_reconstructed',np.allclose(recovered.loc[plan.condition_id],plan.conductivity))
    check('no_missing_or_zero_targets',(plan.conductivity>0).all() and np.isfinite(plan.conductivity).all())
    check('excluded_rows_never_used',len(retained)==int(plan.n_measurements.sum()))
    check('unit_group_consistency',plan['c units'].map(GROUPS).equals(plan.group))
    tab=predictions[predictions.model.eq('TabPFN')]
    check('unique_prediction_keys',not tab.duplicated(['group','inference_seed','condition_id']).any())
    for group,part in plan.groupby('group'):
        query=part[part.split.ne('train')]
        for seed in seeds:
            p=tab[(tab.group==group)&(tab.inference_seed==seed)].set_index('condition_id')
            check(f'{group}_{seed}_coverage',set(p.index)==set(query.condition_id))
            check(f'{group}_{seed}_labels',np.allclose(p.loc[query.condition_id].true_mS_cm,query.conductivity))
    for row in scores.itertuples():
        p=predictions[(predictions.group==row.group)&(predictions.model==row.model)
                      &(predictions.inference_seed==row.inference_seed)&(predictions.split==row.split)]
        e=p.pred_mS_cm.to_numpy()-p.true_mS_cm.to_numpy()
        check(f'metrics_{row.group}_{row.model}_{row.inference_seed}_{row.split}',np.allclose(
            [np.abs(e).mean(),np.mean(e**2),np.sqrt(np.mean(e**2)),100*np.mean(np.abs(e)/p.true_mS_cm.to_numpy())],
            [row.MAE,row.MSE,row.RMSE,row.MAPE],atol=1e-12))
    passed=all(c['passed'] for c in checks)
    write_json(out/'tables/result_audit.json',{'passed':passed,'checks':checks,
        'negative_predictions_not_clipped':int((tab.pred_mS_cm<0).sum())})
    if not passed:raise ValueError('Result audit failed')


def plots(sample,out):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(9,4))
    for ax,(group,g) in zip(axes,sample.groupby('group',sort=False)):
        for split,p in g.groupby('split'):
            ax.scatter(p.true_mS_cm,p.prediction_mean,s=8,alpha=.35,label=f'{split} (n={len(p)})')
        lo=min(g.true_mS_cm.min(),g.prediction_mean.min());hi=max(g.true_mS_cm.max(),g.prediction_mean.max())
        ax.plot([lo,hi],[lo,hi],'--',color='gray',lw=1)
        ax.set(title=group,xlabel='Measured conductivity (mS/cm)',ylabel='Predicted conductivity (mS/cm)')
        ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/f'figures/parity.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)


def update_readme(audit,summary,plan,config,out):
    path=HERE.parent/'README.md';start,end='<!-- calisol23_RESULTS_START -->','<!-- calisol23_RESULTS_END -->'
    minimum=audit['minimum_conductivity_mS_cm']
    counts=audit['exclusion_counts']
    lines=[start,'## calisol23 — CALiSol-23，De Blasio 等，2024','',
        '论文：[CALiSol-23: Experimental electrolyte conductivity data for various Li-salts and solvent combinations]'
        '(https://doi.org/10.1038/s41597-024-03575-8)。文献汇编数据论文，本轮不设原文预测分数对照。','',
        '### 适用范围与清洗','',
        f'**本轮仅研究 k≥{minimum:g} mS/cm 的电解液，结果不代表完整 CALiSol-23 的全电导率范围。**',
        '此范围在前几轮观察到低值导致很高MAPE后确定，属于探索性范围限定，不能当作独立确认。'
        '低于范围的测量未必错误，因此称为范围排除，不统一称为异常或坏数据。'
        '10%是期望误差，不是数据筛选条件；不按单条预测误差删样本，不反复改变阈值以保证达标。',
        f"- 原表 {audit['raw_rows']} 行；电导率缺失 {counts.get('missing_conductivity',0)} 行、为0的 {counts.get('zero_conductivity',0)} 行、浓度为负的 {counts.get('negative_concentration',0)} 行，另有 {counts.get('below_conductivity_range',0)} 行低于本轮范围。",
        f"- 共排除 {audit['excluded_rows']} 行，保留 {audit['retained_rows']} 行；相同输入在拆分前聚合为 {audit['conditions']} 条条件。未填补缺失值或按预测误差删除记录。",
        '- mol/kg 与 mol/L 完全分开训练和评价，不做浓度换算。质量/体积/摩尔溶剂比例口径保留为独热标识。',
        '- 输入为温度K、盐浓度、38种溶剂比例、13个盐名独热位和3个配比口径独热位，共56维。DOI不输入模型。'
        '盐名只去首尾空格；原始溶剂比例保持原样，不强制归一化。',
        '- 标签保留原始 mS/cm，不取对数、不裁剪预测；原始下载文件保持不变。','',
        '### 固定6:2:2划分','',
        '|浓度组|范围内原始记录|聚合条件|训练|验证|测试|','|---|---:|---:|---:|---:|---:|']
    for group,p in plan.groupby('group',sort=False):
        count=p.split.value_counts()
        lines.append(f"|{group}|{audit['group_raw_retained_rows'][group]}|{len(p)}|{count['train']}|{count['validation']}|{count['test']}|")
    lines += ['',f"划分种子42，推理种子 {config['inference_seeds']}；模型 `{API_MODEL_PATH}`，`n_estimators={N_ESTIMATORS}`。",
        '两组范围限定后重新划分。相同条件的重复记录不跨集合；同一配方的其他温度、同一DOI可以跨集合。'
        '验证只作报告，不参与训练或模型选择。删除记录后划分改变，前后分数不能直接作为清洗效果的配对比较。','',
        '### 常见指标','',
        '逐种子指标的均值 ± 样本标准差（ddof=1）；训练均值对照只算一次。MAPE分母使用原始标签，不设下限。','',
        '|浓度组|集合|模型|种子数|MAPE (%)|MAE (mS/cm)|MSE|RMSE (mS/cm)|R²|',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    def cell(r,m):
        mean,std=r[f'{m}_mean'],r[f'{m}_std'];fmt='.3e' if abs(mean)>=1e5 else '.4f'
        return f'{mean:{fmt}} ± {std:{fmt}}' if pd.notna(std) else f'{mean:{fmt}}（—）'
    for _,r in summary.iterrows():
        lines.append(f"|{r.group}|{r['split']}|{r.model}|{r.n_seeds}|"+'|'.join(cell(r,m) for m in ['MAPE','MAE','MSE','RMSE','R2'])+'|')
    lines += ['', '### 旧结果与解释','',
        '先前完整正值范围的 mol/L 测试 MAPE 为17.01%，该组没有10⁻⁶级标签（最小约0.00049 mS/cm）。'
        '较大误差集中在约0.001～0.01 mS/cm的标签，另有中等电导率样本预测偏差较大；仅删10⁻⁶不会改变该组。',
        '原始正值范围结果保留在 `Output_before_near_zero_cleanup`；仅删除≤10⁻¹²的结果保留在 `Output_before_range_cleanup`。'
        '本轮不应与它们作为相同测试集或相同适用范围的性能优劣比较。','',
        '### 完整性与文件','',
        '- 两组各六种子，验证和测试合并查询；六种子只衡量固定划分下推理波动，不代表六次独立数据划分。',
        f"- 本轮输出累计新增预测调用 {config['api_calls_total']} 次；本次执行新增 {config['api_calls_this_execution']} 次，复用 {config['cache_hits_this_execution']} 份缓存。",
        '- [脚本](calisol23.py) · [本地测试](test_experiments.py)',
        '- [逐种子指标](calisol23/results/tables/metrics.csv) · '
        '[汇总](calisol23/results/tables/metrics_summary.csv) · '
        '[分段指标](calisol23/results/tables/metrics_by_conductivity.csv)',
        '- [样本波动](calisol23/results/tables/sample_summary.csv) · '
        '[平均预测指标](calisol23/results/tables/ensemble_metrics.csv) · '
        '[结果核验](calisol23/results/tables/result_audit.json) · '
        '[预测图](calisol23/results/figures/parity.png)',
        '- `Output/data` 保存排除记录、行映射、聚合条件、模型输入和固定划分；`Output/cache` 保存预测及校验值。',
        f'- 重建并复用缓存：`.venv/bin/python study/prediction_benchmarks/run_benchmark.py calisol23 run --min-conductivity {minimum:g} --resume`。','',end]
    section='\n'.join(lines);text=path.read_text() if path.exists() else '# prediction_benchmarks 实验结果\n'
    if start in text:
        a,b=text.index(start),text.index(end)+len(end);text=text[:a]+section+text[b:]
    else:text+='\n'+section+'\n'
    path.write_text(text)


def import_unchanged_cache(oldout,contract,group,seed,train,query,cache,meta_path):
    old=json.loads((oldout/'run_config.json').read_text())
    if old['status']!='completed':raise ValueError('Archived run is incomplete')
    for key in ['source_sha256','model_path','n_estimators','features','shared_step01_sha256']:
        current=contract.get(key,sha(PROJECT/'analysis/ampere_conductivity.py'))
        if old[key]!=current:raise ValueError('Archived source/model configuration differs')
    previous=pd.read_csv(oldout/'data/split_manifest.csv')
    previous=previous[previous.group.eq(group)]
    oldtrain=previous[previous.split.eq('train')];oldquery=previous[previous.split.ne('train')]
    for before,after in [(oldtrain,train),(oldquery,query)]:
        if (len(before)!=len(after) or not np.array_equal(before.split,after.split)
                or not np.allclose(before[FEATURES+['conductivity']],after[FEATURES+['conductivity']],rtol=1e-12,atol=0)):
            return False  # changed unit group must be rerun
    src=oldout/f'cache/{group}_seed{seed}.csv';meta=json.loads(src.with_suffix('.json').read_text())
    old_contract={key:old[key] for key in contract}
    if meta['sha256']!=sha(src) or meta['signature']!=digest(old_contract|{'group':group,'inference_seed':seed}):
        raise ValueError('Archived cache signature/checksum invalid')
    saved=pd.read_csv(src)
    if not np.array_equal(saved.condition_id,oldquery.condition_id) or not np.allclose(saved.true_mS_cm,query.conductivity):
        raise ValueError('Archived prediction alignment invalid')
    frame(query,saved.pred_mS_cm.to_numpy(),seed,'TabPFN').to_csv(cache,index=False)
    write_json(meta_path,{'signature':digest(contract|{'group':group,'inference_seed':seed}),
        'sha256':sha(cache),'imported_from':str(src),'original_sha256':sha(src),
        'verification':'same ordered training features/labels, query features/labels and split assignments; atol=0 rtol=1e-12'})
    logging.info('Verified unchanged %s seed=%s; imported archived prediction',group,seed)
    return True


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['audit','run']);parser.add_argument('--output-dir',type=Path,default=HERE/'results')
    parser.add_argument('--min-conductivity',type=float,default=MIN_CONDUCTIVITY,help='Inclusive lower boundary in mS/cm')
    parser.add_argument('--reuse-from',type=Path,help='Verify and reuse only unchanged unit groups from an archived run')
    parser.add_argument('--inference-seeds',type=int,nargs='+',default=list(SEEDS));parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv)
    if len(args.inference_seeds)!=len(set(args.inference_seeds)):parser.error('Duplicate seeds')
    out=args.output_dir.resolve();out.mkdir(exist_ok=True,parents=True)
    for folder in ['data','tables','cache','figures','snapshots']:(out/folder).mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',force=True,
                        handlers=[logging.StreamHandler(),logging.FileHandler(out/'execution.log')])
    logging.getLogger('fontTools').setLevel(logging.WARNING)
    data,mapping,audit=prepare_data(minimum=args.min_conductivity);plan=make_split(data)
    contract={'source_sha256':audit['source_sha256'],'model_path':API_MODEL_PATH,'n_estimators':N_ESTIMATORS,
        'features':FEATURES,'target':audit['target'],'aggregation':audit['aggregation'],'split_seed':42,
        'unit_groups':GROUPS,'schema_version':3,'split_hash':hashlib.sha256(plan.to_csv(index=False).encode()).hexdigest()}
    signature=digest(contract)
    config=contract|{'signature':signature,'inference_seeds':args.inference_seeds,'status':'audit_completed',
        'run_type':'fixed_split_seed_evaluation','planned_prediction_calls':len(GROUPS)*len(args.inference_seeds),
        'api_calls_this_execution':0,'cache_hits_this_execution':0,'api_calls_total':0,
        'started_utc':datetime.now(timezone.utc).isoformat(),'software_versions':software_versions(),
        'script_sha256':sha(__file__),'shared_step01_sha256':sha(PROJECT/'analysis/ampere_conductivity.py')}
    cp=out/'run_config.json'
    if cp.exists():
        old=json.loads(cp.read_text())
        if old['signature']!=signature:raise ValueError('Changed contract; use a new Output')
        if args.mode=='audit' and old['status']!='audit_completed':raise FileExistsError('Audit cannot overwrite run results')
        if args.mode=='run' and old['status']=='completed' and not args.resume:raise FileExistsError('Use --resume to reuse paid predictions')
        config['api_calls_total']=old['api_calls_total']
    for file in [Path(__file__),PROJECT/'analysis/ampere_conductivity.py']:
        dest=out/'snapshots'/f'{file.stem}_{sha(file)[:12]}.py'
        if not dest.exists():dest.write_bytes(file.read_bytes())
    for name,df in [('conditions',data),('raw_to_condition',mapping),('excluded_records',mapping[mapping.exclusion_reason.ne('included')]),('split_manifest',plan)]:
        df.to_csv(out/f'data/{name}.csv',index=False)
    encode(data).to_csv(out/'data/model_inputs.csv',index=False)
    write_json(out/'data/data_audit.json',audit);write_json(cp,config)
    logging.info('Cleaned: %s raw -> %s retained -> %s conditions; split=%s',audit['raw_rows'],audit['retained_rows'],len(data),
                 plan.groupby(['group','split']).size().to_dict())
    try:
        parts=[]
        for group,p in plan.groupby('group',sort=False):
            train=p[p.split.eq('train')];query=p[p.split.ne('train')]
            parts.append(frame(query,np.full(len(query),train.conductivity.mean()),-1,'Training mean'))
            if args.mode=='audit':continue
            for seed in args.inference_seeds:
                cache=out/f'cache/{group}_seed{seed}.csv';mp=cache.with_suffix('.json')
                call_signature=digest(contract|{'group':group,'inference_seed':seed})
                if args.reuse_from and not cache.exists() and not mp.exists():
                    import_unchanged_cache(args.reuse_from.resolve(),contract,group,seed,train,query,cache,mp)
                if args.resume and cache.exists() and mp.exists():
                    meta=json.loads(mp.read_text());saved=pd.read_csv(cache)
                    if meta['signature']!=call_signature or meta['sha256']!=sha(cache):raise ValueError('Cache signature/checksum mismatch')
                    if not np.array_equal(saved.condition_id,query.condition_id) or not np.allclose(saved.true_mS_cm,query.conductivity,rtol=1e-12,atol=0):raise ValueError('Cache condition/target mismatch')
                    values=saved.pred_mS_cm.to_numpy();config['cache_hits_this_execution']+=1
                    logging.info('Reused %s seed=%s',group,seed)
                else:
                    if cache.exists() or mp.exists():raise FileExistsError('Partial/existing cache; inspect before retry')
                    config['api_calls_this_execution']+=1;config['api_calls_total']+=1
                    write_json(cp,config|{'status':'running'})
                    logging.info('TabPFN %s seed=%s train=%s query=%s features=%s (one prediction call)',group,seed,len(train),len(query),len(FEATURES))
                    started=time.monotonic()
                    values=api_predict(train[FEATURES],train.conductivity.to_numpy(),query[FEATURES],seed,PROJECT.parent)
                    frame(query,values,seed,'TabPFN').to_csv(cache,index=False)
                    write_json(mp,{'signature':call_signature,'sha256':sha(cache),'elapsed_seconds':time.monotonic()-started})
                parts.append(frame(query,values,seed,'TabPFN'))
        predictions=pd.concat(parts,ignore_index=True);scores=evaluate(predictions)
        predictions.to_csv(out/'tables/predictions.csv',index=False);scores.to_csv(out/'tables/metrics.csv',index=False)
        if args.mode=='run':
            summary,sample=summarize(predictions,scores,out)
            verify(predictions,scores,plan,mapping,args.inference_seeds,out)
            plots(sample,out);update_readme(audit,summary,plan,config,out)
            config['status']='completed';config['six_inference_seeds_completed']=set(args.inference_seeds)==set(SEEDS)
        logging.info('Completed %s; new calls=%s cache hits=%s',config['status'],config['api_calls_this_execution'],config['cache_hits_this_execution'])
    except Exception as exc:
        config['status']='failed';config['error_type']=type(exc).__name__
        logging.error('Stopped (%s); no automatic retry or model substitution',type(exc).__name__);raise
    finally:
        config['finished_utc']=datetime.now(timezone.utc).isoformat();write_json(cp,config)


if __name__=='__main__':main()
