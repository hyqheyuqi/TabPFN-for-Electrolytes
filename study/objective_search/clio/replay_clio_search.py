"""Paired, sequential measured-pool replay. Hidden outcomes stay outside predictors."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
import re
import sys
import time
import warnings
from datetime import datetime, timezone

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(PROJECT/'.matplotlib'))
sys.path.insert(0,str(PROJECT))
import numpy as np
import pandas as pd
from scipy.stats import norm, t
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from analysis.ampere_conductivity import API_MODEL_PATH, N_ESTIMATORS, software_versions

SPEC=importlib.util.spec_from_file_location('clio_preparation',PROJECT/'prediction_benchmarks/clio.py')
PREP=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(PREP)
FEATURES=list(PREP.FEATURES)
INIT_SEEDS=list(range(42,52))
INFERENCE_SEED=100
KAPPA=1.96
MAX_CREDITS=3_300_000
CALL_CREDITS=10_000
RATIOS=[.95,.985,1.]
METHODS=['GPR + UCB','TabPFN + UCB']
SETTINGS=dict(schema=1,pool='38 measured formulations',features=FEATURES,initial_count=5,
 initialization_seeds=INIT_SEEDS,initialization='uniform without replacement among DMC ratio 0/1',
 model_path=API_MODEL_PATH,n_estimators=N_ESTIMATORS,inference_seed=INFERENCE_SEED,kappa=KAPPA,
 gpr='training-standardized Constant*Matern(nu=2.5)+WhiteKernel, normalize_y, MLE, no restarts, alpha=1e-8',
 batch_size=1,target='raw conductivity mS/cm',ratios=RATIOS,
 stopping='evaluator-only stop once measured-pool maximum observed; best-so-far then constant')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,allow_nan=False).encode()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def write_json(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix(path.suffix+'.tmp')
 tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def prepare(out):
 d,mapping,guid,issues,audit=PREP.prepare_data();d=d.sort_values('formulation_id').reset_index(drop=True)
 reference=PROJECT/'prediction_benchmarks/clio/results/data/formulations.csv'
 saved=pd.read_csv(reference)
 if len(d)!=38 or not np.allclose(d[FEATURES+['conductivity']],saved[FEATURES+['conductivity']],rtol=1e-12,atol=1e-12):
  raise ValueError('Prepared data do not match audited 38-formulation source')
 if not d.formulation_id.is_unique or not np.isfinite(d[FEATURES+['conductivity']]).all().all():raise ValueError('Invalid pool')
 for sub in ['data','tables','cache','figures','figure_source_data','snapshots']:(out/sub).mkdir(parents=True,exist_ok=True)
 d.to_csv(out/'data/pool_evaluator_only.csv',index=False)
 mapping.to_csv(out/'data/raw_to_formulation.csv',index=False)
 issues.to_csv(out/'data/duplicate_guid_issues.csv',index=False)
 boundary=d.index[d.DMC_cosolvent_ratio.isin([0,1])].to_numpy()
 initial={seed:np.random.default_rng(seed).choice(boundary,5,replace=False).tolist() for seed in INIT_SEEDS}
 initrows=[dict(initialization_seed=seed,initial_order=i,formulation_id=d.iloc[idx].formulation_id)
           for seed,ids in initial.items() for i,idx in enumerate(ids)]
 pd.DataFrame(initrows).to_csv(out/'data/initializations.csv',index=False)
 audit.update(pool_max_mS_cm=float(d.conductivity.max()),pool_argmax=d.loc[d.conductivity.idxmax(),'formulation_id'],
              threshold_counts={str(r):int((d.conductivity>=r*d.conductivity.max()-1e-12).sum()) for r in RATIOS},
              boundary_candidates=len(boundary),source_reference_sha256=sha(reference))
 write_json(out/'data/data_audit.json',audit)
 for f in [Path(__file__),HERE/'pilot_protocol.md',Path(PREP.__file__)]:
  dest=out/'snapshots'/f'{f.stem}_{sha(f)[:12]}{f.suffix}'
  if not dest.exists():dest.write_bytes(f.read_bytes())
 signature=digest(SETTINGS|{'source_sha256':audit['source_sha256'],'pool_sha256':sha(out/'data/pool_evaluator_only.csv')})
 return d,initial,audit,signature

def prediction_contract(observed,query):
 """No unobserved targets, global scales, ranks, or success thresholds permitted."""
 if not set(['formulation_id']+FEATURES+['conductivity']).issubset(observed):raise ValueError('Missing observed fields')
 if set(query.columns)!=set(['formulation_id']+FEATURES):raise ValueError('Query must not contain outcomes or metadata')
 if set(observed.formulation_id)&set(query.formulation_id):raise ValueError('Observed/query overlap')
 return dict(model_path=API_MODEL_PATH,n_estimators=N_ESTIMATORS,inference_seed=INFERENCE_SEED,
             feature_names=FEATURES,observed_ids=observed.formulation_id.tolist(),
             X_train=observed[FEATURES].to_numpy(float).tolist(),y_train=observed.conductivity.to_numpy(float).tolist(),
             query_ids=query.formulation_id.tolist(),X_query=query[FEATURES].to_numpy(float).tolist(),
             output_type='full',kappa=KAPPA)

def gp_distribution(contract):
 x=np.array(contract['X_train']);y=np.array(contract['y_train']);q=np.array(contract['X_query'])
 scaler=StandardScaler().fit(x)
 kernel=ConstantKernel(1.,(1e-3,1e3))*Matern(length_scale=np.ones(3),length_scale_bounds=(1e-2,1e2),nu=2.5)+WhiteKernel(.01,(1e-6,1))
 model=GaussianProcessRegressor(kernel=kernel,alpha=1e-8,normalize_y=True,n_restarts_optimizer=0,random_state=0)
 with warnings.catch_warnings(record=True) as caught:
  warnings.simplefilter('always',ConvergenceWarning)
  model.fit(scaler.transform(x),y)
  mean,std=model.predict(scaler.transform(q),return_std=True)
 pred=pd.DataFrame(dict(formulation_id=contract['query_ids'],mean=mean,std=std,
       q05=mean+norm.ppf(.05)*std,q95=mean+norm.ppf(.95)*std))
 return pred,dict(fitted_kernel=str(model.kernel_),optimizer_warning_count=len(caught))

def live_distribution(contract):
    raise RuntimeError("Hosted inference and account access are disabled in this public snapshot.")

def validate_prediction(pred,contract):
 if pred.formulation_id.tolist()!=contract['query_ids'] or len(pred)!=len(contract['X_query']):raise ValueError('Prediction alignment failure')
 if not np.isfinite(pred[['mean','std','q05','q95']].to_numpy()).all():raise ValueError('Nonfinite prediction')
 if (pred['std']<0).any() or (pred.q05>pred.q95).any():raise ValueError('Invalid predictive distribution')

def usage():
    raise RuntimeError("Hosted inference and account access are disabled in this public snapshot.")

class Quota:
 def __init__(self,out):
  self.path=out/'quota_history.json';self.history=json.loads(self.path.read_text()) if self.path.exists() else []
  q=usage();self.origin=self.history[0]['used'] if self.history else q['used']
  self.history.append(q|{'event':'execution_start'});write_json(self.path,self.history)
 def before(self):
  q=usage()
  if q['remaining']<CALL_CREDITS or q['used']-self.origin+CALL_CREDITS>MAX_CREDITS:raise RuntimeError('Credit budget exhausted before request')
  return q
 def after(self,before,call_id):
  q=usage();delta=q['used']-before['used']
  self.history.append(q|{'event':'prediction_complete','call_id':call_id,'delta':delta});write_json(self.path,self.history)
  if delta<0 or delta>CALL_CREDITS or q['used']-self.origin>MAX_CREDITS:raise RuntimeError('Observed charge exceeds frozen per-call or total bound')
  return delta

def cached_tabpfn(contract,signature,out,quota):
 key=digest(contract|{'replay_signature':signature});stem=out/'cache'/key
 meta=stem.with_suffix('.json');csv=stem.with_suffix('.csv');raw=stem.with_suffix('.npz');attempt=stem.with_suffix('.attempt.json')
 if meta.exists():
  md=json.loads(meta.read_text())
  if md['key']!=key or md['csv_sha256']!=sha(csv) or md['raw_sha256']!=sha(raw):raise ValueError('Cache integrity failure')
  pred=pd.read_csv(csv);validate_prediction(pred,contract)
  return pred,md|{'cache_hit':True}
 if csv.exists() or raw.exists() or attempt.exists():raise FileExistsError('Previously attempted/partial prediction; no automatic paid retry')
 before=quota.before()
 write_json(attempt,dict(key=key,contract=contract,start=now(),before_usage=before,script_sha256=sha(__file__)))
 start=time.monotonic();pred,arrays,md=live_distribution(contract);elapsed=time.monotonic()-start
 validate_prediction(pred,contract)
 pred.to_csv(csv,index=False);np.savez_compressed(raw,**arrays)
 # Persist the successful response before a quota request can fail.
 md.update(key=key,contract=contract,csv_sha256=sha(csv),raw_sha256=sha(raw),seconds=elapsed,
           code_sha256=sha(__file__),finished=now(),cache_hit=False)
 write_json(meta,md)
 delta=quota.after(before,key);md['credits']=delta;write_json(meta,md)
 return pred,md

def choose(pred):
 scored=pred.assign(ucb=pred['mean']+KAPPA*pred['std'])
 return scored.sort_values(['ucb','formulation_id'],ascending=[False,True],kind='stable').iloc[0]

def hit_cost(sequence_values,threshold,initial_count=5):
 hits=np.flatnonzero(np.asarray(sequence_values)>=threshold-1e-12)
 if not len(hits):return None
 return max(initial_count,int(hits[0])+1)

def combin_ratio(a,b,k):
 if k<0 or k>b:return 0.
 return 0. if a<k else math.comb(a,k)/math.comb(b,k)

def random_reference(y,initial_ids):
 """Exact conditional best curves and hitting times for uniform sampling."""
 y=np.asarray(y,float);idx=np.array(initial_ids,int);rest=np.delete(y,idx);n=len(rest);n0=len(idx);best=float(y[idx].max())
 rows=[]
 for budget in range(n0,len(y)+1):
  k=budget-n0
  levels=np.unique(np.r_[best,np.maximum(rest,best)]);previous=0.;expected=0.
  for level in levels:
   cdf=combin_ratio(int((rest<=level+1e-12).sum()),n,k)
   expected+=level*(cdf-previous);previous=cdf
  row=dict(budget=budget,best_mS_cm=expected)
  for ratio in RATIOS:
   threshold=ratio*y.max();m=int((rest>=threshold-1e-12).sum())
   row[f'hit_{ratio}']=1. if best>=threshold-1e-12 else 1-combin_ratio(n-m,n,k)
  rows.append(row)
 costs={r:float(n0 if best>=r*y.max()-1e-12 else n0+(n+1)/(int((rest>=r*y.max()-1e-12).sum())+1)) for r in RATIOS}
 return pd.DataFrame(rows),costs

def replay(d,ids,method,seed,out,signature,predictor=None):
 observed=list(ids);allids=list(range(len(d)));maxy=d.conductivity.max();selected_rows=[];diagnostics=[];query_rows=[]
 for order,i in enumerate(ids,1):
  selected_rows.append(dict(initialization_seed=seed,method=method,evaluation=order,formulation_id=d.iloc[i].formulation_id,
     conductivity=float(d.iloc[i].conductivity),phase='initial',context_size=0,mean=None,std=None,ucb=None,prediction_seconds=0.))
 while d.iloc[observed].conductivity.max()<maxy-1e-12:
  remaining=[i for i in allids if i not in observed]
  training=d.iloc[observed][['formulation_id']+FEATURES+['conductivity']].copy()
  query=d.iloc[remaining][['formulation_id']+FEATURES].copy()
  contract=prediction_contract(training,query)
  start=time.monotonic()
  if method==METHODS[0]:pred,meta=gp_distribution(contract)
  else:pred,meta=predictor(contract)
  validate_prediction(pred,contract)
  seconds=meta.get('seconds',time.monotonic()-start)
  chosen=choose(pred);index=int(d.index[d.formulation_id.eq(chosen.formulation_id)][0])
  # Hidden outcomes are used only after prediction, for scoring/revealing.
  actual=d.iloc[remaining].conductivity.to_numpy();err=pred['mean'].to_numpy()-actual
  diagnostics.append(dict(initialization_seed=seed,method=method,context_size=len(observed),query_n=len(remaining),
    MAE=float(np.abs(err).mean()),RMSE=float(np.sqrt(np.mean(err**2))),MAPE=float(100*np.mean(np.abs(err)/actual)),
    coverage90=float(np.mean((actual>=pred.q05)&(actual<=pred.q95))),
    interval_width90=float(np.mean(pred.q95-pred.q05))))
  for row,truth in zip(pred.to_dict('records'),actual):
   query_rows.append(row|dict(initialization_seed=seed,method=method,context_size=len(observed),actual_mS_cm=float(truth),
     prediction_key=meta.get('key','local_gp'),selected=row['formulation_id']==chosen.formulation_id))
  selected_rows.append(dict(initialization_seed=seed,method=method,evaluation=len(observed)+1,formulation_id=chosen.formulation_id,
    conductivity=float(d.iloc[index].conductivity),phase='selected',context_size=len(observed),mean=float(chosen['mean']),
    std=float(chosen['std']),ucb=float(chosen.ucb),prediction_seconds=float(seconds)))
  observed.append(index)
  trace=pd.DataFrame(selected_rows);trace['best_mS_cm']=trace.conductivity.cummax()
  # Each method/start writes only its own trace; completed paths survive interruptions.
  path=out/'tables'/f'trace_{"tabpfn" if method==METHODS[1] else "gpr"}_seed{seed}.csv'
  trace.to_csv(path,index=False)
  pd.DataFrame(diagnostics).to_csv(path.with_name(path.stem+'_diagnostics.csv'),index=False)
  pd.DataFrame(query_rows).to_csv(path.with_name(path.stem+'_predictions.csv'),index=False)
  logging.info('%s init=%s observed=%s selected=%s best=%.5f cache=%s',method,seed,len(observed),chosen.formulation_id,
               trace.best_mS_cm.iloc[-1],meta.get('cache_hit',False))
 trace=pd.DataFrame(selected_rows);trace['best_mS_cm']=trace.conductivity.cummax()
 path=out/'tables'/f'trace_{"tabpfn" if method==METHODS[1] else "gpr"}_seed{seed}.csv'
 trace.to_csv(path,index=False)
 return trace,pd.DataFrame(diagnostics)

def cost_rows(trace,d):
 values=trace.conductivity.to_numpy();result=dict(initialization_seed=int(trace.initialization_seed.iloc[0]),method=trace.method.iloc[0],
     n_selected=len(trace),initial_best=float(values[:5].max()))
 for ratio in RATIOS:result[f'cost_{ratio}']=hit_cost(values,ratio*d.conductivity.max())
 return result

def audit_trace(trace,d,ids,complete=False):
 checks={'unique':trace.formulation_id.is_unique,'initial_match':trace.formulation_id.iloc[:5].tolist()==d.iloc[ids].formulation_id.tolist(),
  'labels_match':bool(np.allclose(trace.conductivity,d.set_index('formulation_id').loc[trace.formulation_id].conductivity)),
  'consecutive_evaluations':trace.evaluation.tolist()==list(range(1,len(trace)+1)),
  'context_growth':bool((trace.loc[trace.phase.eq('selected'),'context_size'].to_numpy()==trace.loc[trace.phase.eq('selected'),'evaluation'].to_numpy()-1).all())}
 if complete:checks['maximum_observed']=bool(np.isclose(trace.conductivity.max(),d.conductivity.max()))
 if not all(checks.values()):raise ValueError(f'Trace integrity failed: {checks}')
 return checks

def collect(d,initial,out):
 costs=[];curves=[];diags=[];alltraces=[];audits=[]
 for seed,ids in initial.items():
  rand,rc=random_reference(d.conductivity.to_numpy(),ids);rand['initialization_seed']=seed;rand['method']='Random (exact)';curves.append(rand)
  costs.append(dict(initialization_seed=seed,method='Random (exact)',n_selected=38,initial_best=float(d.iloc[ids].conductivity.max()),**{f'cost_{r}':rc[r] for r in RATIOS}))
  for method,short in zip(METHODS,['gpr','tabpfn']):
   path=out/'tables'/f'trace_{short}_seed{seed}.csv'
   if not path.exists():continue
   trace=pd.read_csv(path);complete=bool(np.isclose(trace.conductivity.max(),d.conductivity.max()))
   audits.append(dict(initialization_seed=seed,method=method,complete=complete,checks=audit_trace(trace,d,ids,complete)))
   alltraces.append(trace)
   if not complete:continue
   costs.append(cost_rows(trace,d))
   for budget in range(5,39):
    observed=trace[trace.evaluation.le(budget)];best=observed.conductivity.max()
    curves.append(pd.DataFrame([dict(initialization_seed=seed,method=method,budget=budget,best_mS_cm=best,
         **{f'hit_{r}':float(best>=r*d.conductivity.max()-1e-12) for r in RATIOS})]))
   dp=path.with_name(path.stem+'_diagnostics.csv')
   if dp.exists():diags.append(pd.read_csv(dp))
 costs=pd.DataFrame(costs);curve=pd.concat(curves,ignore_index=True)
 costs.to_csv(out/'tables/discovery_costs.csv',index=False);curve.to_csv(out/'tables/budget_curves.csv',index=False)
 if diags:pd.concat(diags,ignore_index=True).to_csv(out/'tables/prediction_diagnostics.csv',index=False)
 if alltraces:pd.concat(alltraces,ignore_index=True).to_csv(out/'tables/all_traces.csv',index=False)
 write_json(out/'tables/trace_audit.json',dict(passed=True,trajectories=audits))
 return costs,curve

def report(d,initial,audit,out,status):
 costs,curves=collect(d,initial,out)
 # A matched table only uses starts with completed paths for BOTH model arms.
 completed=set(costs[costs.method.eq(METHODS[0])].initialization_seed)&set(costs[costs.method.eq(METHODS[1])].initialization_seed)
 matched=costs[costs.initialization_seed.isin(completed)]
 summary=[]
 for method,g in matched.groupby('method',sort=False):
  row=dict(method=method,n_paired_initializations=len(g))
  for r in RATIOS:
   col=f'cost_{r}';row[f'mean_{col}']=float(g[col].mean());row[f'sd_{col}']=float(g[col].std(ddof=1)) if len(g)>1 else 0.
   rand=matched[matched.method.eq('Random (exact)')][col].mean();row[f'AF_{r}']=float(rand/g[col].mean())
  summary.append(row)
 summary=pd.DataFrame(summary)
 if len(summary):summary.to_csv(out/'tables/summary.csv',index=False)
 c=curves[curves.initialization_seed.isin(completed)]
 if len(c):
  bybudget=c.groupby(['method','budget'],sort=False)[['best_mS_cm']+[f'hit_{r}' for r in RATIOS]].mean().reset_index()
  random_curve=bybudget[bybudget.method.eq('Random (exact)')].set_index('budget').best_mS_cm
  bybudget['EF_percent']=100*(bybudget.best_mS_cm/bybudget.budget.map(random_curve)-1)
  bybudget.to_csv(out/'tables/budget_summary.csv',index=False)
 else:bybudget=pd.DataFrame()
 paired=[]
 for r in RATIOS:
  if len(completed):
   pv=matched.pivot(index='initialization_seed',columns='method',values=f'cost_{r}')
   delta=pv[METHODS[1]]-pv[METHODS[0]];mean=delta.mean();se=delta.std(ddof=1)/np.sqrt(len(delta)) if len(delta)>1 else float('nan')
   margin=t.ppf(.975,len(delta)-1)*se if len(delta)>1 else float('nan')
   paired.append(dict(threshold_ratio=r,n=len(delta),TabPFN_minus_GPR=float(mean),low=float(mean-margin),high=float(mean+margin)))
 if paired:pd.DataFrame(paired).to_csv(out/'tables/paired_differences.csv',index=False)
 make_plots(out,summary,bybudget,costs,completed)
 heading=['# clio：TabPFN替换GP代理的逐轮贝叶斯优化回放','',
  f'运行状态：**{status}**；已完成两模型配对起点 **{len(completed)}/10**。冻结协议见[pilot_protocol.md](pilot_protocol.md)。',
  '', '## 先看结论','']
 if len(summary):
  rows=summary.set_index('method');tb=rows.loc[METHODS[1]];gp=rows.loc[METHODS[0]];ra=rows.loc['Random (exact)']
  heading += [f"在当前{len(completed)}组配对起点中，达到池内最高值95%的平均评价数为：TabPFN {tb['mean_cost_0.95']:.2f}，GP {gp['mean_cost_0.95']:.2f}，随机 {ra['mean_cost_0.95']:.2f}。",
   f"精确命中池内最优的平均评价数为：TabPFN {tb['mean_cost_1.0']:.2f}，GP {gp['mean_cost_1.0']:.2f}，随机 {ra['mean_cost_1.0']:.2f}。98.5%与最优命中在此池中是同一事件。",
   f"TabPFN相对随机的平均最优发现成本减少{100*(1-tb['mean_cost_1.0']/ra['mean_cost_1.0']):.1f}%；相对GP增加{100*(tb['mean_cost_1.0']/gp['mean_cost_1.0']-1):.1f}%。当前固定UCB协议不支持替换GP后搜索更快。",
   '95%阈值在9/10组初始化已满足，TabPFN 5.1次与GP 5.5次的差异不能作为稳定优化优势；该配对区间包含零。',
   '不根据单一有利起点得出结论；下面保留全部配对结果和不利结果。','',
   '## 直观指标总表','', '|方法|配对起点数|达到95%平均次数|达到98.5%/最优平均次数|AF 95%|AF 最优|',
   '|---|---:|---:|---:|---:|---:|']
  for row in summary.to_dict('records'):
   heading.append(f"|{row['method']}|{row['n_paired_initializations']}|{row['mean_cost_0.95']:.3f}|{row['mean_cost_1.0']:.3f}|{row['AF_0.95']:.3f}|{row['AF_1.0']:.3f}|")
  heading+=['','次数包含初始5条；初始已达标记5。AF为随机条件精确平均成本除以方法平均成本。不是平均逐起点比值。','',
   '|阈值|TabPFN−GP平均次数差|配对t型95%区间|','|---|---:|---|']
  for row in paired:heading.append(f"|{row['threshold_ratio']*100:g}%|{row['TabPFN_minus_GPR']:.3f}|[{row['low']:.3f}, {row['high']:.3f}]|")
  heading += ['','区间仅描述固定38池、这些初始化的变化；样本较小、未校正多重比较，不是新实验误差或跨体系泛化区间。','',
   '## 相同预算能找到多好的配方','', '|方法|累计评价数|平均最好电导率 mS/cm|95%成功率|最优命中率|相对随机提升 EF|','|---|---:|---:|---:|---:|---:|']
  for row in bybudget[bybudget.budget.isin([5,10,15,20,30,38])].to_dict('records'):
   heading.append(f"|{row['method']}|{row['budget']}|{row['best_mS_cm']:.4f}|{100*row['hit_0.95']:.1f}%|{100*row['hit_1.0']:.1f}%|{row['EF_percent']:.2f}%|")
  heading+=['','![最好电导率和命中曲线](Output/figures/Fig01_progress.png)','',
   '![逐起点发现成本](Output/figures/Fig02_costs.png)','',
   '## 全部配对起点','', '|初始化种子|TabPFN 95%|GP 95%|随机期望95%|TabPFN 最优|GP 最优|随机期望最优|',
   '|---|---:|---:|---:|---:|---:|---:|']
  for seed in sorted(completed):
   g=matched[matched.initialization_seed.eq(seed)].set_index('method')
   heading.append('|'+str(seed)+'|'+'|'.join(f"{g.loc[m,col]:.3f}" for col in ['cost_0.95','cost_1.0'] for m in [METHODS[1],METHODS[0],'Random (exact)'])+'|')
 heading += ['','## 本轮做了什么','',
  '- 38个实测配方，输入仅盐浓度、EC质量分数和DMC共溶剂比例；原始mS/cm标签。197条AEM数据不参与。',
  '- 原始125行按测量规则和GUID审计后聚合为38组；2个冲突GUID仍未从来源解决，2组测量不完整，未按结果删点。',
  '- 初始化种子42–51，每组从30条边界配方中抽5条。两模型共享起点，之后独立选择、揭示单条真实标签、重新预测。',
  '- 主分析TabPFN推理种子100、V3 default、8 estimators；不把推理种子当优化重复。',
  '- GP为当前训练集标准化的Constant×Matérn 2.5＋WhiteKernel，每轮MLE。共同UCB=mean＋1.96×std；TabPFN std来自返回分布。',
  '- 随机参照逐起点用组合公式精确计算。主报告只使用两模型都已完成的起点，缺失轨迹不当作成功。',
  '- 命中池内最大值后停止额外付费推理，随后best-so-far保持该最大值；这是回顾性评价节省计算，不是现实中预知最优的停止法。','',
  '## 与原论文的对应与差距','',
  '|维度|原论文|本轮|能否直接说超过原文|','|---|---|---|---|',
  '|搜索空间|离散化的三变量空间，1000多个候选|仅38个历史实测配方|不能；候选池更小且来自原优化轨迹|',
  '|反馈|Clio重新配液、清洗、测量|揭示既有配方聚合实测标签|只能称实测池回放|',
  '|初始化|5条边界初始样本|同思路、10组新抽取起点|不是原始轨迹重演|',
  '|代理|Dragonfly Matérn GP，每轮MLE|明确配置的sklearn GP或TabPFN|本地GP不是严格逐参数复现|',
  '|采集|自适应EI/UCB/TTEI，并穿插随机探索|两模型固定同一UCB|隔离代理贡献，不是原策略完整复现|',
  '|高电导率|约13.7 mS/cm，原文EC/DMC约40:60、0.9m|池内最大13.69289339 mS/cm，F015，发布表对应0.93m、EC 0.39、DMC比1|是同一历史高值的回放，不是新发现|',
  '|近最优AF|AEM衍生GP模拟环境，正文98.5%约6倍|38实测池，相同阈值本轮重新计算AF|不能直接以AF大小判断超过6倍|',
  '|实验预算/时间|42次评价、两个工作日|最多38条历史配方；另记API计算时间|不能宣称节省了真实机器人天数|',
  '|电池验证|筛选后进行软包快充测试|没有新电池测试|不能推断快充改善|','',
  '原文Fig.5图注的近最优差值表述与正文98.5%略有差异；本轮明确使用98.5%，不声称已复现其SI全部统计。原文1.3%为测量重复性，不是模型MAPE。',
  '', '## 指标边界与诊断','',
  f"池内最大值为{audit['pool_max_mS_cm']:.8f} mS/cm；95%阈值{.95*audit['pool_max_mS_cm']:.8f}（{audit['threshold_counts']['0.95']}组达标），98.5%阈值{.985*audit['pool_max_mS_cm']:.8f}（1组达标）。",
  '95%若起点已经达标，模型无须选点即可成功。须结合起点命中比例解释饱和效应，不将它包装成模型提升。',
  '每轮未测候选的MAE/MAPE/区间覆盖仅在预测保存之后计算；这些标签未用于选点或调参。不同策略访问的训练历史不同，后续诊断也不是完全相同上下文的静态预测比较。',
  '', '## 文件与重跑','',
  '- [冻结协议](pilot_protocol.md) · [运行脚本](replay_clio_search.py) · [离线检查](test_clio_replay.py)',
  '- [指标总表](Output/tables/summary.csv) · [预算表](Output/tables/budget_summary.csv) · [逐起点成本](Output/tables/discovery_costs.csv)',
  '- [全部轨迹](Output/tables/all_traces.csv) · [逐轮预测诊断](Output/tables/prediction_diagnostics.csv) · [轨迹核验](Output/tables/trace_audit.json)',
  '- `Output/cache`保留逐轮合同、预测分布和attempt标记；`snapshots`保留代码、数据处理和协议快照。',
  '- `Output/run_config.json`记录状态、版本、代码哈希与资源；`Output/quota_history.json`记录真实服务额度。',
  '- 离线检查：`.venv/bin/python -m unittest discover -s study/objective_search/clio -p test_clio_replay.py`',
  '- 离线重新报告：`.venv/bin/python study/objective_search/clio/replay_clio_search.py report`',
  '- 断点继续：同脚本`run --resume`；完整缓存按合同校验复用，部分/失败付费请求不自动重试。',
  '', '原文：[Nature Communications 2022](https://doi.org/10.1038/s41467-022-32938-1)；[作者数据仓库](https://github.com/BattModels/Clio-NatCommData)，数据固定提交`2ba755d87abf60ac00bc55e874ddd9c7f3d64bcb`。','']
 (HERE/'README.md').write_text('\n'.join(heading))
 return len(completed)

def make_plots(out,summary,bybudget,costs,completed):
 if not len(bybudget):return
 import matplotlib.pyplot as plt
 colors={METHODS[0]:'#c46a69',METHODS[1]:'#476f97','Random (exact)':'#7d858d'}
 fig,axes=plt.subplots(1,3,figsize=(13,3.7))
 for method,g in bybudget.groupby('method',sort=False):
  axes[0].plot(g.budget,g.best_mS_cm,label=method,color=colors[method])
  axes[1].plot(g.budget,100*g['hit_0.95'],color=colors[method])
  axes[2].plot(g.budget,100*g['hit_1.0'],color=colors[method])
 axes[0].set(ylabel='Mean best measured conductivity (mS/cm)',xlabel='Total evaluations (including initial 5)')
 axes[1].set(ylabel='Reached 95% of pool maximum (%)',xlabel='Total evaluations')
 axes[2].set(ylabel='Hit measured-pool maximum (%)',xlabel='Total evaluations')
 axes[0].legend(frameon=False,fontsize=8)
 axes[1].set_ylim(0,100);axes[2].set_ylim(0,100)
 for ax in axes:ax.grid(alpha=.2)
 fig.suptitle(f'clio: sequential measured-pool replay, {len(completed)} paired starts');fig.tight_layout()
 for ext in ['png','pdf']:fig.savefig(out/f'figures/Fig01_progress.{ext}',dpi=240,bbox_inches='tight')
 plt.close(fig);bybudget.to_csv(out/'figure_source_data/Fig01_progress.csv',index=False)
 fig,axes=plt.subplots(1,2,figsize=(10,3.7))
 matched=costs[costs.initialization_seed.isin(completed)];xs=np.arange(len(completed))
 for ax,col,title in zip(axes,['cost_0.95','cost_1.0'],['95% threshold','98.5% / exact pool maximum']):
  for j,method in enumerate([METHODS[0],METHODS[1],'Random (exact)']):
   g=matched[matched.method.eq(method)].sort_values('initialization_seed');ax.plot(xs,g[col],'o-',color=colors[method],label=method,alpha=.85)
  ax.set(xticks=xs,xticklabels=sorted(completed),xlabel='Initialization seed',ylabel='Total evaluations to target',title=title);ax.grid(alpha=.2)
 axes[0].legend(frameon=False,fontsize=8);fig.tight_layout()
 for ext in ['png','pdf']:fig.savefig(out/f'figures/Fig02_costs.{ext}',dpi=240,bbox_inches='tight')
 plt.close(fig);matched.to_csv(out/'figure_source_data/Fig02_costs.csv',index=False)


def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['audit','run','report'])
 parser.add_argument('--resume',action='store_true');parser.add_argument('--pilot-only',action='store_true')
 parser.add_argument('--output-dir',type=Path,default=HERE/'results');args=parser.parse_args()
 out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
 logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',force=True,
     handlers=[logging.StreamHandler(),logging.FileHandler(out/'execution.log')]);logging.getLogger('fontTools').setLevel(logging.WARNING)
 d,initial,audit,signature=prepare(out);cp=out/'run_config.json'
 previous=json.loads(cp.read_text()) if cp.exists() else {}
 if previous and previous['signature']!=signature:raise ValueError('Changed frozen protocol/data; use a new output directory')
 config=SETTINGS|dict(signature=signature,status='audit_completed',run_type='feasibility_pilot_then_paired_replay',
      execution_authorization='user: implement and run the proposed measured-pool comparison',
      started=previous.get('started',now()),software_versions=software_versions(),code_sha256=sha(__file__),
      max_credits=MAX_CREDITS,maximum_paid_calls=330,data_audit=audit,
      current_gate=previous.get('current_gate','FEASIBILITY_PILOT_ONLY'))
 if args.mode=='report':
  report(d,initial,audit,out,previous.get('status','no_run'));return
 if args.mode=='audit':write_json(cp,config);logging.info('Data/protocol audit passed; no API predictions');return
 if previous.get('status')=='completed' and not args.resume:raise FileExistsError('Completed run exists; use --resume')
 quota=None
 try:
  write_json(cp,config|{'status':'running_baselines'})
  for seed,ids in initial.items():replay(d,ids,METHODS[0],seed,out,signature)
  quota=Quota(out)
  for i,(seed,ids) in enumerate(initial.items()):
   if args.pilot_only and i>0:break
   write_json(cp,config|dict(status='running_tabpfn',active_initialization=seed))
   trace,diag=replay(d,ids,METHODS[1],seed,out,signature,
       predictor=lambda contract:cached_tabpfn(contract,signature,out,quota))
   audit_trace(trace,d,ids,complete=True)
   if i==0:
    config['current_gate']='FULL_RUN_READY'
    config['pilot_evidence']=dict(initialization_seed=seed,trajectory_rows=len(trace),
      valid_distribution_and_feedback=True,result='valid evaluable trajectory; performance not a gate criterion',
      user_authorized_remaining_paired_runs=True)
   report(d,initial,audit,out,'pilot_completed' if args.pilot_only else 'running')
  status='pilot_completed' if args.pilot_only else 'completed';config['status']=status
  n=report(d,initial,audit,out,status);config['paired_initializations_completed']=n
  if not args.pilot_only and n!=len(initial):raise ValueError('Incomplete matched result set')
 except Exception as exc:
  config.update(status='stopped',error_type=type(exc).__name__)
  logging.error('Stopped (%s); no automatic paid retries',type(exc).__name__)
  report(d,initial,audit,out,'stopped_partial')
  raise
 finally:
  config['finished']=now();config['code_sha256']=sha(__file__)
  if quota is not None:
   config['credits_observed']=quota.history[-1]['used']-quota.origin
   config['paid_calls_recorded']=sum(q.get('event')=='prediction_complete' for q in quota.history)
  write_json(cp,config)

if __name__=='__main__':main()
