"""Shared results/cache/metric plumbing; each dataset retains its own preparation script."""
from __future__ import annotations
import argparse,hashlib,json,logging,re,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
PROJECT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(PROJECT))
from analysis.ampere_conductivity import API_MODEL_PATH,N_ESTIMATORS,SEEDS,api_predict,software_versions

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()
def write_json(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def usage():
    raise RuntimeError("Hosted inference and account access are disabled in this public snapshot.")

def prediction_frame(query,pred,seed,mode):
 pred=np.asarray(pred,float).reshape(-1)
 if len(pred)!=len(query) or not np.isfinite(pred).all():raise ValueError('Invalid prediction')
 p=query[['row_id','group','split','conductivity_mS_cm']].copy();p['seed']=seed;p['target']=query.y.to_numpy();p['prediction']=pred
 p['pred_mS_cm']=np.power(10.,pred) if mode=='log10' else pred
 if not np.isfinite(p.pred_mS_cm).all():raise ValueError('Invalid converted conductivity')
 p['APE_percent']=100*np.abs(p.pred_mS_cm-p.conductivity_mS_cm)/p.conductivity_mS_cm
 return p

def scores(predictions):
 rows=[]
 for (group,seed,split),p in predictions.groupby(['group','seed','split'],sort=False):
  e=p.prediction.to_numpy()-p.target.to_numpy();le=p.pred_mS_cm.to_numpy()-p.conductivity_mS_cm.to_numpy()
  rows.append({'group':group,'seed':seed,'split':split,'n':len(p),
   'MAE_target':float(np.abs(e).mean()),'MSE_target':float(np.mean(e**2)),'RMSE_target':float(np.sqrt(np.mean(e**2))),
   'R2_target':float(r2_score(p.target,p.prediction)),
   'MAPE_percent':float(p.APE_percent.mean()),'median_APE_percent':float(p.APE_percent.median()),
   'MAE_mS_cm':float(np.abs(le).mean()),'MSE_mS_cm2':float(np.mean(le**2)),
   'RMSE_mS_cm':float(np.sqrt(np.mean(le**2))),'R2_linear':float(r2_score(p.conductivity_mS_cm,p.pred_mS_cm)),
   'Spearman':float(spearmanr(p.conductivity_mS_cm,p.pred_mS_cm).statistic)})
 return pd.DataFrame(rows)

def export_results(parts,out,plan,config,spec):
 if not parts:return
 pred=pd.concat(parts,ignore_index=True);byseed=scores(pred)
 pred.to_csv(out/'tables/predictions.csv',index=False);byseed.to_csv(out/'tables/metrics_by_seed.csv',index=False)
 keys=['group','split'];cols=[c for c in byseed if c not in keys+['seed','n']]
 summary=byseed.groupby(keys)[cols].agg(['mean','std','min','max'])
 summary.columns=[f'{a}_{b}' for a,b in summary.columns];summary=summary.reset_index()
 summary=summary.merge(byseed.groupby(keys).size().rename('n_seeds').reset_index())
 summary.to_csv(out/'tables/metrics_summary.csv',index=False)
 sample=pred.groupby(['group','split','row_id']).agg(true_mS_cm=('conductivity_mS_cm','first'),prediction_mean_mS_cm=('pred_mS_cm','mean'),
  prediction_std_mS_cm=('pred_mS_cm','std'),prediction_min_mS_cm=('pred_mS_cm','min'),prediction_max_mS_cm=('pred_mS_cm','max'),
  mean_APE=('APE_percent','mean'),APE_std=('APE_percent','std'),n_seeds=('seed','nunique')).reset_index()
 sample.to_csv(out/'tables/sample_summary.csv',index=False)
 checks={'unique_prediction_keys':not pred.duplicated(['group','seed','row_id']).any(),
  'finite_predictions':bool(np.isfinite(pred[['prediction','pred_mS_cm']].to_numpy()).all())}
 for (group,seed),p in pred.groupby(['group','seed']):
  q=plan[(plan.group==group)&plan.split.ne('train')]
  checks[f'coverage_{group}_{seed}']=set(p.row_id)==set(q.row_id)
  checks[f'labels_{group}_{seed}']=bool(np.allclose(p.set_index('row_id').conductivity_mS_cm.loc[q.row_id],q.conductivity_mS_cm))
 write_json(out/'tables/result_audit.json',{'passed':all(checks.values()),'checks':checks})
 if not all(checks.values()):raise ValueError('Prediction audit failed')
 update_readme(out,spec,summary,config,plan)
 import matplotlib.pyplot as plt
 groups=sample.group.unique();fig,axes=plt.subplots(1,len(groups),figsize=(4.7*len(groups),4),squeeze=False)
 for ax,group in zip(axes[0],groups):
  sub=sample[sample.group.eq(group)]
  for split,p in sub.groupby('split'):
   x=p.true_mS_cm.to_numpy();y=p.prediction_mean_mS_cm.to_numpy()
   if spec['mode']=='log10':x=np.log10(x);y=np.log10(y)
   ax.scatter(x,y,s=9,alpha=.45,label=f'{split} (n={len(p)})')
  true=sub.true_mS_cm;hat=sub.prediction_mean_mS_cm
  lo=min(true.min(),hat.min());hi=max(true.max(),hat.max())
  if spec['mode']=='log10':lo,hi=np.log10(lo),np.log10(hi)
  ax.plot([lo,hi],[lo,hi],'--',color='gray',lw=1)
  scale='log10(mS/cm)' if spec['mode']=='log10' else 'mS/cm'
  ax.set(title=f"{spec['z']} {group}",xlabel=f'Measured ({scale})',ylabel=f'Predicted ({scale})');ax.legend(frameon=False)
 fig.tight_layout()
 for ext in ['png','pdf']:fig.savefig(out/f'figures/parity.{ext}',dpi=300,bbox_inches='tight')
 plt.close(fig)

def update_readme(out,spec,summary,config,plan):
 path=out.parent.parent/'README.md';z=spec['z'];start,end=f'<!-- {z}_RESULTS_START -->',f'<!-- {z}_RESULTS_END -->'
 lines=[start,f"## {z} — {spec['title']}",'',f"论文：[{spec['title']}]({spec['doi']})。",'',*spec['notes'],'',
  '### 数据与划分','','|子集|训练|验证|测试|','|---|---:|---:|---:|']
 for group,p in plan.groupby('group',sort=False):
  n=p.split.value_counts();lines.append(f"|{group}|{n.get('train',0)}|{n.get('validation',0)}|{n.get('test',0)}|")
 lines+=['',f"模型`{API_MODEL_PATH}`，n_estimators={N_ESTIMATORS}；推理种子{config['seeds']}。验证只作报告，不参与训练或调参。",
  '六种子衡量固定划分下的推理波动，不是六次独立数据划分。表内为逐种子指标均值±样本标准差，不是先平均预测再算指标。','',
  f"训练目标：{spec['target_label']}。target指标指训练目标尺度；其他指标在线性mS/cm尺度计算。",'',
  '|子集|集合|种子数|MAPE (%)|MAE (mS/cm)|MSE (mS/cm)²|RMSE (mS/cm)|R²线性|MAE target|RMSE target|R² target|',
  '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 def cell(r,m):
  a,b=r[f'{m}_mean'],r[f'{m}_std'];return f'{a:.5g} ± {b:.3g}' if pd.notna(b) else f'{a:.5g}'
 for _,r in summary.iterrows():lines.append(f"|{r.group}|{r['split']}|{r.n_seeds}|"+'|'.join(cell(r,c) for c in ['MAPE_percent','MAE_mS_cm','MSE_mS_cm2','RMSE_mS_cm','R2_linear','MAE_target','RMSE_target','R2_target'])+'|')
 diagnostics=pd.read_csv(out/'tables/predictions.csv')
 lines+=['','### 误差诊断','']
 for group,g in diagnostics.groupby('group',sort=False):
  test=g[g['split'].eq('test')].groupby('row_id').agg(k=('conductivity_mS_cm','first'),ape=('APE_percent','mean'))
  val=g[g['split'].eq('validation')].groupby('row_id').agg(k=('conductivity_mS_cm','first'),pred=('pred_mS_cm','mean'),ape=('APE_percent','mean'))
  worst=val.loc[val.ape.idxmax()];share=100*worst.ape/val.ape.sum() if val.ape.sum()>0 else 0
  lines.append(f"- {group}：测试集逐条件平均APE的中位数为{test.ape.median():.3f}%。验证集中最高误差样本真实值{worst.k:.6g}、平均预测{worst.pred:.6g} mS/cm，平均APE为{worst.ape:.3f}%，贡献了验证集总APE的{share:.1f}%。")
  if spec['mode']=='log10':
   low=test[test.k.lt(.1)];contribution=100*low.ape.sum()/test.ape.sum() if test.ape.sum()>0 else 0
   lines.append(f"- {group}测试集有{len(low)}/{len(test)}条低于0.1 mS/cm，贡献总APE的{contribution:.1f}%；这里0.1只是诊断区间，不用于删样本或修改MAPE分母。")
 lines+=['高误差只说明当前预测失准，不据此删除记录或认定原始测量有误；需结合来源和配方进一步核查。']
 lines+=['','### 原文指标核对','',*spec['comparison'],'',
  f"本次执行新增预测调用{config['api_calls_this_execution']}次，复用{config['cache_hits']}份；额度记录见Output/quota_history.json。",'',
  f"- [脚本]({z.lower()}.py) · [逐种子指标]({out.parent.name}/results/tables/metrics_by_seed.csv) · [汇总]({out.parent.name}/results/tables/metrics_summary.csv)",
  f"- [样本波动]({out.parent.name}/results/tables/sample_summary.csv) · [核验]({out.parent.name}/results/tables/result_audit.json) · [预测图]({out.parent.name}/results/figures/parity.png)",
  '- Output/data保存数据处理、映射和划分，Output/cache保存每次预测和校验值；原始下载文件不修改。','',end]
 section='\n'.join(lines);text=path.read_text() if path.exists() else '# prediction_benchmarks 实验结果\n'
 if start in text:
  a,b=text.index(start),text.index(end)+len(end);text=text[:a]+section+text[b:]
 else:text+='\n'+section+'\n'
 path.write_text(text)

def run(here,data,features,audit,files,spec,argv=None):
 parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['audit','run']);parser.add_argument('--resume',action='store_true')
 parser.add_argument('--output-dir',type=Path,default=here/'results');parser.add_argument('--seeds',type=int,nargs='+',default=list(SEEDS));args=parser.parse_args(argv)
 if len(args.seeds)!=len(set(args.seeds)):raise ValueError('Duplicate seeds')
 out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
 for folder in ['data','tables','cache','figures','snapshots']:(out/folder).mkdir(exist_ok=True)
 logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',force=True,handlers=[logging.StreamHandler(),logging.FileHandler(out/'execution.log')]);logging.getLogger('fontTools').setLevel(logging.WARNING)
 if not data.row_id.is_unique or not np.isfinite(data[features+['y','conductivity_mS_cm']].to_numpy()).all() or not data.conductivity_mS_cm.gt(0).all():raise ValueError('Invalid model data')
 if not data.split.isin(['train','validation','test']).all():raise ValueError('Unexpected split')
 contract={'files':{str(f.relative_to(PROJECT)):sha(f) for f in files},'matrix_hash':hashlib.sha256(data.to_csv(index=False).encode()).hexdigest(),
  'model_path':API_MODEL_PATH,'n_estimators':N_ESTIMATORS,'features':features,'mode':spec['mode'],'seeds':args.seeds}
 signature=digest(contract);cp=out/'run_config.json'
 if cp.exists():
  old=json.loads(cp.read_text())
  if old['signature']!=signature:raise ValueError('Changed contract; use a new output directory')
  if args.mode=='audit' and old['status']!='audit_completed':raise FileExistsError('Audit cannot overwrite run results')
  if args.mode=='run' and old['status']=='completed' and not args.resume:raise FileExistsError('Use --resume to reuse cached predictions')
 config=contract|{'signature':signature,'status':'audit_completed','seeds':args.seeds,'software_versions':software_versions(),
  'planned_prediction_calls':data.group.nunique()*len(args.seeds),'api_calls_this_execution':0,'cache_hits':0}
 for f in [Path(spec.get('source_path', here/spec['script_name'])),Path(__file__),PROJECT/'analysis/ampere_conductivity.py']:
  dest=out/'snapshots'/f'{f.stem}_{sha(f)[:12]}.py'
  if not dest.exists():dest.write_bytes(f.read_bytes())
 data.to_csv(out/'data/model_table.csv',index=False);data[['row_id','group','split']].to_csv(out/'data/split_manifest.csv',index=False)
 write_json(out/'data/data_audit.json',audit);write_json(cp,config)
 logging.info('%s split sizes %s',spec['z'],data.groupby(['group','split']).size().to_dict())
 if args.mode=='audit':return
 parts=[];quotas=[]
 try:
  quotas.append(usage());write_json(out/'quota_history.json',quotas)
  for group,p in data.groupby('group',sort=False):
   train=p[p.split.eq('train')];query=p[p.split.ne('train')]
   for seed in args.seeds:
    cache=out/f'cache/{group}_seed{seed}.csv';meta=cache.with_suffix('.json')
    if args.resume and cache.exists() and meta.exists():
     md=json.loads(meta.read_text());saved=pd.read_csv(cache)
     if md['signature']!=signature or md['sha256']!=sha(cache):raise ValueError('Cache checksum/contract mismatch')
     if not np.array_equal(saved.row_id,query.row_id) or not np.allclose(saved.target,query.y):raise ValueError('Cache row/target alignment changed')
     parts.append(saved);config['cache_hits']+=1
    else:
     if cache.exists() or meta.exists():raise FileExistsError('Partial cache, no automatic retry')
     if quotas[-1]['remaining']<10000:raise RuntimeError('Insufficient credit budget for next prediction')
     config['api_calls_this_execution']+=1;write_json(cp,config|{'status':'running'})
     logging.info('%s %s seed=%s train=%s query=%s',spec['z'],group,seed,len(train),len(query))
     started=time.monotonic();values=api_predict(train[features],train.y.to_numpy(),query[features],seed,PROJECT.parent)
     saved=prediction_frame(query,values,seed,spec['mode']);saved.to_csv(cache,index=False)
     write_json(meta,{'signature':signature,'sha256':sha(cache),'elapsed_seconds':time.monotonic()-started});parts.append(saved)
     q=usage();q['delta']=q['used']-quotas[-1]['used'];quotas.append(q);write_json(out/'quota_history.json',quotas)
   logging.info('%s %s complete',spec['z'],group)
  export_results(parts,out,data,config,spec);config['status']='completed'
 except Exception as exc:
  config['status']='failed';config['error_type']=type(exc).__name__;logging.error('Stopped %s; no automatic retries',type(exc).__name__)
  export_results(parts,out,data,config,spec);raise
 finally:write_json(cp,config)
 logging.info('Completed %s new calls=%s cached=%s remaining=%s',spec['z'],config['api_calls_this_execution'],config['cache_hits'],quotas[-1]['remaining'])
