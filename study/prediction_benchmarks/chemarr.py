"""chemarr: ten polymer-disjoint folds, one inference seed, log10(S/cm) target.
Run: .venv/bin/python study/prediction_benchmarks/run_benchmark.py chemarr run --resume
Paper-protocol adaptation to current released table; not exact original fold IDs.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib,json,logging,math,re,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent / 'chemarr';PROJECT=HERE.parents[1];sys.path.insert(0,str(PROJECT))
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from rdkit import Chem,rdBase
from rdkit.Chem import Descriptors,Lipinski,rdMolDescriptors,rdChemReactions
from analysis.ampere_conductivity import api_predict,API_MODEL_PATH,N_ESTIMATORS,software_versions
SOURCE=PROJECT/'datasets/chemarr/data/clean_train_data.csv'
DESC=['MolWt','MolLogP','TPSA','HBA','HBD','RotatableBonds','Rings','AromaticRings','FractionCSP3','O','N','S','F','P','B','FormalCharge']
FEATURES=[f'{prefix}_{name}' for prefix in ['polymer','salt'] for name in DESC]+['log10_polymer_MW','salt_molality','temperature_K','inverse_temperature_K']
PAPER={'XGBoost':{'MAE_log10_S_cm':1.09,'MAE_reported_pm':.027,'Spearman':.38,'Spearman_reported_pm':.026},
       'Chemprop':{'MAE_log10_S_cm':1.08,'MAE_reported_pm':.012,'Spearman':.45,'Spearman_reported_pm':.006},
       'ChemArr':{'MAE_log10_S_cm':1.,'MAE_reported_pm':.030,'Spearman':.59,'Spearman_reported_pm':.022}}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def write_json(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def polymer_structure(smiles):
    m=Chem.MolFromSmiles(smiles)
    if m is None:raise ValueError('Unparseable polymer SMILES')
    if '[Cu]' in smiles:
        base=m;repeats=max(0,math.ceil(50/(m.GetNumHeavyAtoms()-2))-1)
        reaction=rdChemReactions.ReactionFromSmarts('[Cu][*:1].[*:2][Au]>>[*:1]-[*:2]')
        for _ in range(repeats):
            results=reaction.RunReactants((base,m))
            if len(results)!=1 or len(results[0])!=1:raise ValueError('Ambiguous polymer joins')
            m=results[0][0];Chem.SanitizeMol(m)
        capped=Chem.MolToSmiles(m).replace('[Cu]','C').replace('[Au]','C').replace('[Ca]','C')
        m=Chem.MolFromSmiles(capped)
    if m is None:raise ValueError('Invalid methyl-capped oligomer')
    return Chem.MolToSmiles(m)

def descriptors(smiles):
    m=Chem.MolFromSmiles(smiles)
    if m is None:raise ValueError('Unparseable structure')
    values=[Descriptors.MolWt(m),Descriptors.MolLogP(m),rdMolDescriptors.CalcTPSA(m),
        Lipinski.NumHAcceptors(m),Lipinski.NumHDonors(m),Lipinski.NumRotatableBonds(m),
        rdMolDescriptors.CalcNumRings(m),rdMolDescriptors.CalcNumAromaticRings(m),rdMolDescriptors.CalcFractionCSP3(m)]
    values += [sum(a.GetSymbol()==element for a in m.GetAtoms()) for element in ['O','N','S','F','P','B']]
    values += [sum(a.GetFormalCharge() for a in m.GetAtoms())]
    if not np.isfinite(values).all():raise ValueError('Nonfinite descriptor')
    return values

def prepare_data(source=SOURCE):
    raw=pd.read_csv(source)
    if list(raw)!=['smiles','conductivity','mw','salt smiles','molality','temperature']:raise ValueError('Unexpected schema')
    if not np.isfinite(raw[['conductivity','temperature']].to_numpy()).all():raise ValueError('Nonfinite label/temperature')
    if (raw.temperature<=-273.15).any() or (raw.mw.dropna()<=0).any() or (raw.molality.dropna()<0).any():raise ValueError('Invalid numeric inputs')
    polymers={s:polymer_structure(s) for s in raw.smiles.unique()}
    salt_strings=raw['salt smiles'].fillna('[Li+]')
    salts={s:Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in salt_strings.unique()}
    pc={s:descriptors(s) for s in set(polymers.values())};sc={s:descriptors(s) for s in set(salts.values())}
    data=pd.DataFrame({'row_id':np.arange(len(raw)), 'original_polymer':raw.smiles,
        'polymer_group':raw.smiles.map(polymers),'salt_structure':salt_strings.map(salts),'target_log10_S_cm':raw.conductivity})
    for prefix,structures,lookup in [('polymer',data.polymer_group,pc),('salt',data.salt_structure,sc)]:
        values=np.asarray([lookup[s] for s in structures])
        for i,name in enumerate(DESC):data[f'{prefix}_{name}']=values[:,i]
    data['log10_polymer_MW']=np.log10(raw.mw.fillna(65000))
    data['salt_molality']=raw.molality.fillna(0)
    data['temperature_K']=raw.temperature+273.15
    data['inverse_temperature_K']=1/data.temperature_K
    if not np.isfinite(data[FEATURES].to_numpy()).all():raise ValueError('Invalid model matrix')
    peo=polymer_structure('[Cu]OCC[Au]')
    audit={'source_sha256':sha(source),'rows':len(data),'paper_rows':12383,'raw_polymers':raw.smiles.nunique(),
        'canonical_polymer_groups':data.polymer_group.nunique(),'salt_structures':data.salt_structure.nunique(),
        'MW_missing_filled_65000':int(raw.mw.isna().sum()),'molality_missing_filled_0':int(raw.molality.isna().sum()),
        'salt_missing_filled_Li':int(raw['salt smiles'].isna().sum()),'excluded_rows':0,
        'representation':'16 fixed RDKit descriptors per polymer/salt plus 4 numeric features; not learned MPNN',
        'polymer_policy':'oligomer >=50 backbone heavy atoms before methyl caps, Cu/Au/Ca markers replaced by C; canonical alias groups kept together',
        'target':'released conductivity is log10(S/cm); negative values are valid, not negative physical conductivities',
        'PEO_rows_always_training':int(data.polymer_group.eq(peo).sum()),'PEO_group':peo,
        'no_conductivity_threshold':True,'rdkit_version':rdBase.rdkitVersion,'features':FEATURES,
        'protocol_limitations':'released data has 10114 not 12383 rows; original fold IDs unavailable; paper text rather than differing repository split script',
        'source_code_reference':'https://github.com/learningmatter-mit/Chem-prop-pred/blob/main/chemproppred/make_balanced_train.py'}
    structures=pd.DataFrame([{'kind':'polymer','source_SMILES':s,'capped_canonical_SMILES':c} for s,c in polymers.items()]+
        [{'kind':'salt','source_SMILES':s,'capped_canonical_SMILES':c} for s,c in salts.items()])
    return data,structures,audit

def make_folds(data,peo):
    counts=data.polymer_group.value_counts()
    groups=[g for g in counts.index if g!=peo]
    # Round-robin by frequency, matching the published frequency-ordered idea.
    # Two adjacent validation buckets implement the paper TEXT's 20%, not the
    # public script's one-bucket validation. All non-PEO polymers test exactly once.
    buckets=[groups[i::10] for i in range(10)]
    plans=[]
    for fold in range(10):
        test=set(buckets[fold]);validation=set(buckets[(fold+1)%10]+buckets[(fold+2)%10])
        p=data[['row_id','polymer_group']].copy();p['fold']=fold;p['split']='train'
        p.loc[p.polymer_group.isin(validation),'split']='validation';p.loc[p.polymer_group.isin(test),'split']='test'
        plans.append(p)
    plans=pd.concat(plans,ignore_index=True)
    for _,p in plans.groupby('fold'):
        if p.groupby('polymer_group')['split'].nunique().max()!=1:raise ValueError('Polymer overlap')
        if not p.loc[p.polymer_group.eq(peo),'split'].eq('train').all():raise ValueError('PEO holdout leakage')
    test_counts=plans[plans.split.eq('test')].groupby('row_id').size()
    if set(test_counts)!= {1} or len(test_counts)!=int(data.polymer_group.ne(peo).sum()):raise ValueError('CV test coverage invalid')
    return plans

def prediction_frame(query,pred,fold,seed):
    pred=np.asarray(pred,float).reshape(-1)
    if len(pred)!=len(query) or not np.isfinite(pred).all():raise ValueError('Invalid predictions')
    p=query[['row_id','polymer_group','split','target_log10_S_cm']].copy();p['fold'],p['seed']=fold,seed;p['pred_log10_S_cm']=pred
    p['true_mS_cm']=1000*np.power(10.,p.target_log10_S_cm);p['pred_mS_cm']=1000*np.power(10.,pred)
    if not np.isfinite(p[['true_mS_cm','pred_mS_cm']].to_numpy()).all():raise ValueError('Exponent overflow')
    p['APE_percent']=100*np.abs(p.pred_mS_cm-p.true_mS_cm)/p.true_mS_cm
    return p

def score(p):
    y=p.target_log10_S_cm.to_numpy();v=p.pred_log10_S_cm.to_numpy();e=v-y
    linear=p.pred_mS_cm.to_numpy()-p.true_mS_cm.to_numpy()
    return {'n':len(p),'MAE_log10_S_cm':float(np.abs(e).mean()),'MSE_log10':float(np.mean(e**2)),
        'RMSE_log10':float(np.sqrt(np.mean(e**2))),'R2_log10':float(r2_score(y,v)),
        'Spearman':float(spearmanr(y,v).statistic),'within_half_order_percent':float(100*np.mean(np.abs(e)<=.5)),
        'within_one_order_percent':float(100*np.mean(np.abs(e)<=1)),
        'MAPE_linear_percent':float(p.APE_percent.mean()),'median_APE_percent':float(p.APE_percent.median()),
        'MAE_mS_cm':float(np.abs(linear).mean()),'MSE_mS_cm2':float(np.mean(linear**2)),
        'RMSE_mS_cm':float(np.sqrt(np.mean(linear**2))),'R2_linear':float(r2_score(p.true_mS_cm,p.pred_mS_cm))}

def read_usage():
    raise RuntimeError("Hosted inference and account access are disabled in this public snapshot.")

def save_results(parts,out,audit,config,plans):
    if not parts:return
    predictions=pd.concat(parts,ignore_index=True);predictions.to_csv(out/'tables/predictions.csv',index=False)
    rows=[{'fold':int(f),'split':s,**score(p)} for (f,s),p in predictions.groupby(['fold','split'])]
    byfold=pd.DataFrame(rows);byfold.to_csv(out/'tables/metrics_by_fold.csv',index=False)
    cols=[c for c in byfold if c not in ['fold','split','n']]
    summary=byfold.groupby('split')[cols].agg(['mean','std','min','max']);summary.columns=[f'{a}_{b}' for a,b in summary.columns]
    summary=summary.reset_index();summary.to_csv(out/'tables/metrics_summary.csv',index=False)
    test=predictions[predictions.split.eq('test')]
    write_json(out/'tables/pooled_out_of_fold_metrics.json',score(test))
    completed=sorted(predictions.fold.unique().tolist())
    checks={'finite_predictions':bool(np.isfinite(predictions.pred_log10_S_cm).all()),
        'no_duplicate_prediction_rows':not predictions.duplicated(['fold','row_id']).any(),
        'no_repeated_test_rows':not test.row_id.duplicated().any(),
        'all_requested_queries_present':all(len(predictions[predictions.fold.eq(f)])==int(((plans.fold==f)&plans.split.ne('train')).sum()) for f in completed)}
    write_json(out/'tables/result_audit.json',{'passed':all(checks.values()),'checks':checks,'completed_folds':completed})
    if not all(checks.values()):raise ValueError('Prediction audit failed')
    update_readme(audit,summary,config,plans,completed)
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(4.5,4));ax.scatter(test.target_log10_S_cm,test.pred_log10_S_cm,s=5,alpha=.25)
    lo=min(test.target_log10_S_cm.min(),test.pred_log10_S_cm.min());hi=max(test.target_log10_S_cm.max(),test.pred_log10_S_cm.max())
    ax.plot([lo,hi],[lo,hi],'--',c='gray',lw=1);ax.set(xlabel='Measured log10 conductivity (S/cm)',ylabel='Predicted log10 conductivity (S/cm)')
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/f'figures/parity_log10.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)

def update_readme(audit,summary,config,plans,completed):
    path=HERE.parent/'README.md';start,end='<!-- chemarr_RESULTS_START -->','<!-- chemarr_RESULTS_END -->'
    lines=[start,'## chemarr — Bradford 等，ChemArr，2023','',
        '论文：[Chemistry-Informed Machine Learning for Polymer Electrolyte Discovery](https://doi.org/10.1021/acscentsci.2c01123)。'
        '以聚合物、盐结构、分子量、浓度和温度预测固态聚合物电解质电导率；ChemArr将Arrhenius关系嵌入神经网络。','',
        '### 数据与方法','',
        f"- 当前作者训练表 {audit['rows']} 行，原文报告12,383行；原始聚合物{audit['raw_polymers']}种，甲基封端并规范化后分为{audit['canonical_polymer_groups']}组。没有按电导率大小删除记录，也没有套用Z45阈值。",
        f"- 结构按正文扩展到至少50个重原子后甲基封端；Cu/Au/Ca是连接标记，不作为真实金属特征。每种聚合物和盐各16个固定RDKit描述符，加log10聚合物分子量、盐浓度、温度K及逆温度，共{len(FEATURES)}维；不是原文MPNN，也不是其指纹XGBoost。",
        f"- 按作者处理规则：缺失聚合物分子量{audit['MW_missing_filled_65000']}行填65,000后取log10；缺失盐{audit['salt_missing_filled_Li']}行用[Li+]，缺失浓度{audit['molality_missing_filled_0']}行填0。固定填充值，不从测试标签学习。",
        '- 目标直接使用作者表的log10(S/cm)，不是负的线性电导率。主指标在log10尺度计算；线性MAPE由10的幂还原后计算，绝不对负的log值直接算百分比。','',
        '### 划分与计算','',
        '- 按正文的聚合物分组10折，名义结构数约70%训练／20%验证／10%测试；PEO始终在训练侧。非PEO组按频次排序后轮转成十桶，每折一桶测试、两桶验证，其余训练；每个非PEO组仅测试一次。',
        '- 聚合物各有不同测量数，结构比例不等于行数比例；尤其PEO占比较大，实际训练行比例往往超过70%。逐折行数如下，不能写成精确70/20/10行数划分。',
        '- 公开脚本采用一桶验证、30重原子等设置，与正文20%验证、50重原子不完全一致。本项目以正文为准，记录这一差异；没有使用作者原始划分编号，不能称精确复现。',
        f"- 已完成 {len(completed)}/10 折，每折仅推理种子{config['seed']}；原文每折为5模型平均，本轮没有做该模型集成。TabPFN模型`{API_MODEL_PATH}`，n_estimators={N_ESTIMATORS}。",'',
        '|折|训练行|验证行|测试行|','|---|---:|---:|---:|']
    for f,p in plans.groupby('fold'):
        n=p.split.value_counts();lines.append(f"|{f}|{n['train']}|{n['validation']}|{n['test']}|")
    lines+=['','### 指标与原文核对','','本轮为各折指标的均值 ± 样本标准差；原文±项按Table 2原样记录，其误差条含义不强行等同。数据版本、描述符和模型集成均不同，因此仅作背景核对。','',
        '|模型|测试MAE log10(S/cm)|测试Spearman R|','|---|---:|---:|']
    for name,r in PAPER.items():lines.append(f"|原文{name}|{r['MAE_log10_S_cm']:.2f} ± {r['MAE_reported_pm']:.3f}|{r['Spearman']:.2f} ± {r['Spearman_reported_pm']:.3f}|")
    row=summary[summary['split'].eq('test')].iloc[0]
    def cell(m):
        mean,std=row[f'{m}_mean'],row[f'{m}_std'];return f'{mean:.4g} ± {std:.3g}' if pd.notna(std) else f'{mean:.4g}'
    lines.append(f"|本轮TabPFN（{len(completed)}折）|{cell('MAE_log10_S_cm')}|{cell('Spearman')}|")
    lines+=['','|本轮测试指标|折间均值 ± 标准差|','|---|---:|']
    for metric in ['MSE_log10','RMSE_log10','R2_log10','within_one_order_percent','MAPE_linear_percent','median_APE_percent','MAE_mS_cm','RMSE_mS_cm','R2_linear']:
        lines.append(f'|{metric}|{cell(metric)}|')
    lines+=['','MAE=1个log10单位表示约一个数量级的偏差，不能理解为1%的误差。聚合物电导率跨越很多数量级，线性MAPE可能很大；主对比采用原文的log10 MAE和Spearman。',
        f"本轮执行新增预测调用{config['api_calls_this_execution']}次，复用{config['cache_hits']}份。额度查询与逐折消耗保存在Output/quota_history.json。",'',
        '- [脚本](chemarr.py) · [逐折指标](chemarr/results/tables/metrics_by_fold.csv) · [汇总](chemarr/results/tables/metrics_summary.csv)',
        '- [合并折外指标](chemarr/results/tables/pooled_out_of_fold_metrics.json) · [核验](chemarr/results/tables/result_audit.json) · [图](chemarr/results/figures/parity_log10.png)',
        '- 重跑复用缓存：`.venv/bin/python study/prediction_benchmarks/run_benchmark.py chemarr run --resume`。','',end]
    section='\n'.join(lines);text=path.read_text() if path.exists() else '# prediction_benchmarks 实验结果\n'
    if start in text:
        a,b=text.index(start),text.index(end)+len(end);text=text[:a]+section+text[b:]
    else:text+='\n'+section+'\n'
    path.write_text(text)

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['audit','run'])
    parser.add_argument('--output-dir',type=Path,default=HERE/'results');parser.add_argument('--resume',action='store_true');parser.add_argument('--seed',type=int,default=100)
    args=parser.parse_args(argv);out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    for f in ['data','cache','tables','figures','snapshots']:(out/f).mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',force=True,handlers=[logging.StreamHandler(),logging.FileHandler(out/'execution.log')]);logging.getLogger('fontTools').setLevel(logging.WARNING)
    data,structures,audit=prepare_data();plans=make_folds(data,audit['PEO_group'])
    contract={'data_hash':audit['source_sha256'],'model_path':API_MODEL_PATH,'n_estimators':N_ESTIMATORS,'features':FEATURES,'seed':args.seed,
        'target':'log10(S/cm)','protocol':'ten polymer-group folds; two validation buckets; PEO training only',
        'matrix_hash':hashlib.sha256(data.to_csv(index=False).encode()).hexdigest(),'split_hash':hashlib.sha256(plans.to_csv(index=False).encode()).hexdigest()}
    signature=digest(contract);cp=out/'run_config.json'
    config=contract|{'signature':signature,'status':'audit_completed','planned_prediction_calls':10,'api_calls_this_execution':0,'cache_hits':0,
        'software_versions':software_versions()|{'rdkit':rdBase.rdkitVersion},'script_sha256':sha(__file__),'started_utc':datetime.now(timezone.utc).isoformat()}
    if cp.exists():
        previous=json.loads(cp.read_text())
        if previous['signature']!=signature:raise ValueError('Changed run contract; use a new output')
        if args.mode=='audit' and previous['status']!='audit_completed':raise FileExistsError('Do not overwrite run with audit')
        if args.mode=='run' and not args.resume and previous['status']=='completed':raise FileExistsError('Use --resume for cached predictions')
    data.to_csv(out/'data/model_table.csv',index=False);structures.to_csv(out/'data/structure_conversion.csv',index=False)
    plans.to_csv(out/'data/split_manifest.csv',index=False);write_json(out/'data/data_audit.json',audit);write_json(out/'paper_reference.json',PAPER)
    for source in [Path(__file__),PROJECT/'analysis/ampere_conductivity.py']:
        snap=out/'snapshots'/f'{source.stem}_{sha(source)[:12]}.py'
        if not snap.exists():snap.write_bytes(source.read_bytes())
    write_json(cp,config);logging.info('Data %s rows, %s groups, %s features',len(data),audit['canonical_polymer_groups'],len(FEATURES))
    if args.mode=='audit':return
    parts=[];quota=[]
    try:
        quota.append({'stage':'before',**read_usage()});write_json(out/'quota_history.json',quota)
        for fold,p in plans.groupby('fold'):
            joined=p.merge(data,on=['row_id','polymer_group'],validate='one_to_one')
            train=joined[joined.split.eq('train')];query=joined[joined.split.ne('train')]
            cache=out/f'cache/fold{fold}_seed{args.seed}.csv';mp=cache.with_suffix('.json')
            if args.resume and cache.exists() and mp.exists():
                meta=json.loads(mp.read_text());saved=pd.read_csv(cache)
                if meta['signature']!=signature or meta['sha256']!=sha(cache):raise ValueError('Cache signature/checksum invalid')
                if not np.array_equal(saved.row_id,query.row_id) or not np.allclose(saved.target_log10_S_cm,query.target_log10_S_cm):raise ValueError('Cache alignment invalid')
                parts.append(saved);config['cache_hits']+=1
            else:
                if cache.exists() or mp.exists():raise FileExistsError('Partial cache; inspect before paid retry')
                if quota[-1]['remaining']<=0:config['status']='quota_exhausted';break
                if len(quota)>1:
                    measured=[q['delta'] for q in quota if 'delta' in q and q['delta']>0]
                    if measured and quota[-1]['remaining']<max(measured):config['status']='quota_insufficient_for_next_fold';break
                config['api_calls_this_execution']+=1;write_json(cp,config|{'status':'running'})
                logging.info('Fold %s: train=%s query=%s, seed=%s',fold,len(train),len(query),args.seed)
                started=time.monotonic();pred=api_predict(train[FEATURES],train.target_log10_S_cm.to_numpy(),query[FEATURES],args.seed,PROJECT.parent)
                saved=prediction_frame(query,pred,int(fold),args.seed);saved.to_csv(cache,index=False)
                write_json(mp,{'signature':signature,'sha256':sha(cache),'elapsed_seconds':time.monotonic()-started})
                parts.append(saved)
                usage=read_usage();usage['delta']=max(0,usage['used']-quota[-1]['used']);usage['stage']=f'after_fold_{fold}';quota.append(usage);write_json(out/'quota_history.json',quota)
                logging.info('Fold %s done; cost=%s credits; remaining=%s',fold,usage['delta'],usage['remaining'])
            write_json(cp,config|{'status':'running'})
        if len(parts)==10:config['status']='completed'
        save_results(parts,out,audit,config,plans)
        config['completed_folds']=len(parts)
    except Exception as exc:
        config['status']='failed';config['error_type']=type(exc).__name__;logging.error('Stopped: %s; no automatic retry',type(exc).__name__)
        save_results(parts,out,audit,config,plans)
        raise
    finally:
        config['finished_utc']=datetime.now(timezone.utc).isoformat();write_json(cp,config)
    logging.info('Completed %s folds; status=%s',len(parts),config['status'])

if __name__=='__main__':main()
