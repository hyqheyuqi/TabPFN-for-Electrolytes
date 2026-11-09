"""Sequential acquisition development and frozen-policy new-start evaluation."""
from __future__ import annotations
import argparse
import json
import logging
import math
from pathlib import Path
import time
import warnings
import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import norm,t
import replay_clio_search as bo

HERE=Path(__file__).resolve().parent
OUT=HERE/'acquisition_results'
BASE=HERE/'results'
RULES=['Mean','UCB0.5','EI']
DEV=list(range(42,52));VALIDATION=list(range(52,62))
MAX_CREDITS=6_600_000
CONFIG=dict(schema=1,development_seeds=DEV,validation_seeds=VALIDATION,rules=RULES,
  kappa=.5,ei_xi=0.,base_settings=bo.SETTINGS,validation_selection='cost100 mean, then best6 descending, then Mean/UCB0.5/EI',
  prediction_contract='exact ordered context; excludes legacy acquisition kappa',max_credits=MAX_CREDITS)


def canonical(c):return {k:v for k,v in c.items() if k!='kappa'}
def key(c):return bo.digest(canonical(c))

def bar_ei(logits,borders,threshold):
 """Exact positive improvement for uniform interior buckets and half-normal tails."""
 p=softmax(np.asarray(logits,float),axis=-1);b=np.asarray(borders,float);width=np.diff(b)
 left=b[:-1];right=b[1:];z=np.maximum(left,threshold)
 contrib=np.where(right>threshold,((right-threshold)**2-(z-threshold)**2)/(2*width),0.)
 s0=width[0]/norm.ppf(.75);s1=width[-1]/norm.ppf(.75)
 d0=b[1]-threshold
 if d0>0:
  contrib[0]=d0*(2*norm.cdf(d0/s0)-1)-s0*np.sqrt(2/np.pi)*(-np.expm1(-.5*(d0/s0)**2))
 else:contrib[0]=0.
 d1=threshold-b[-2]
 if d1<=0:contrib[-1]=-d1+s1*np.sqrt(2/np.pi)
 else:contrib[-1]=s1*np.sqrt(2/np.pi)*np.exp(-.5*(d1/s1)**2)-d1*2*norm.sf(d1/s1)
 return np.maximum(p@contrib,0.)

def acquisition(pred,rule,best,arrays=None):
 if rule=='Mean':values=pred['mean'].to_numpy()
 elif rule=='UCB0.5':values=(pred['mean']+.5*pred['std']).to_numpy()
 elif rule=='UCB1.96':values=(pred['mean']+1.96*pred['std']).to_numpy()
 elif rule=='EI':
  if arrays is not None:values=bar_ei(arrays['logits'],arrays['borders'],best)
  else:
   mu=pred['mean'].to_numpy();sd=pred['std'].to_numpy();safe=np.maximum(sd,1e-14);z=(mu-best)/safe
   values=np.where(sd>1e-14,(mu-best)*norm.cdf(z)+sd*norm.pdf(z),np.maximum(mu-best,0.))
 else:raise ValueError('Unfrozen rule')
 if not np.isfinite(values).all():raise ValueError('Nonfinite acquisition')
 return values

class Cache:
 def __init__(self,out,online=False):
  self.out=out;self.index={};self.hits=0;self.new=0;self.quota=None;self.online=online
  self.history_path=out/'quota_history.json'
  self.history=json.loads(self.history_path.read_text()) if self.history_path.exists() else []
  for folder in [BASE/'cache',out/'cache']:
   for path in sorted(folder.glob('*.json')):
    if path.name.endswith('.attempt.json'):continue
    md=json.loads(path.read_text())
    if not {'contract','csv_sha256','raw_sha256'}.issubset(md):continue
    k=key(md['contract'])
    if k not in self.index:self.index[k]=path
 def load(self,c):
  k=key(c)
  if k in self.index:
   path=self.index[k];md=json.loads(path.read_text());csv=path.with_suffix('.csv');npz=path.with_suffix('.npz')
   if canonical(md['contract'])!=canonical(c) or bo.sha(csv)!=md['csv_sha256'] or bo.sha(npz)!=md['raw_sha256']:raise ValueError('Cache hash/contract mismatch')
   pred=pd.read_csv(csv);bo.validate_prediction(pred,c)
   arrays=dict(np.load(npz));self.hits+=1
   return pred,arrays,dict(key=k,metadata_path=str(path.resolve()),cache_hit=True,seconds=md.get('seconds',0.))
  if not self.online:raise FileNotFoundError('No exact prediction cache')
  stem=self.out/'cache'/k;attempt=stem.with_suffix('.attempt.json')
  if any(stem.with_suffix(s).exists() for s in ['.attempt.json','.csv','.npz','.json']):raise FileExistsError('Partial/attempted paid request; no automatic retry')
  before=bo.usage()
  if not self.history:
   self.history.append(before|{'event':'execution_start'});bo.write_json(self.history_path,self.history)
  origin=self.history[0]['used']
  if before['remaining']<10_000 or before['used']-origin+10_000>MAX_CREDITS:raise RuntimeError('Frozen credit bound reached')
  bo.write_json(attempt,dict(contract=c,key=k,before_usage=before,start=bo.now(),code_sha256=bo.sha(__file__)))
  start=time.monotonic();pred,arrays,metadata=bo.live_distribution(c);elapsed=time.monotonic()-start
  bo.validate_prediction(pred,c)
  pred.to_csv(stem.with_suffix('.csv'),index=False);np.savez_compressed(stem.with_suffix('.npz'),**arrays)
  metadata.update(contract=c,key=k,csv_sha256=bo.sha(stem.with_suffix('.csv')),raw_sha256=bo.sha(stem.with_suffix('.npz')),
    seconds=elapsed,code_sha256=bo.sha(__file__),base_code_sha256=bo.sha(bo.__file__),finished=bo.now())
  bo.write_json(stem.with_suffix('.json'),metadata);self.index[k]=stem.with_suffix('.json');self.new+=1
  after=bo.usage();delta=after['used']-before['used']
  self.history.append(after|dict(event='prediction_complete',delta=delta,key=k));bo.write_json(self.history_path,self.history)
  if delta<0 or delta>10_000 or after['used']-origin>MAX_CREDITS:raise RuntimeError('Charge exceeds bound')
  return pred,arrays,dict(key=k,metadata_path=str(stem.with_suffix('.json').resolve()),cache_hit=False,seconds=elapsed)


def prepare(out):
 for folder in ['data','tables','traces','cache','figures','figure_source_data','snapshots']:(out/folder).mkdir(parents=True,exist_ok=True)
 d,mapping,guid,issues,audit=bo.PREP.prepare_data();d=d.sort_values('formulation_id').reset_index(drop=True)
 source=pd.read_csv(BASE/'data/pool_evaluator_only.csv')
 assert d.formulation_id.tolist()==source.formulation_id.tolist()
 assert np.allclose(d[bo.FEATURES+['conductivity']],source[bo.FEATURES+['conductivity']],rtol=1e-12,atol=1e-12)
 public=json.loads((BASE/'data/public_source_verification.json').read_text())
 assert public['exact_normalized_text_match'] and public['local_sha256']==audit['source_sha256']
 d.to_csv(out/'data/pool_evaluator_only.csv',index=False);bo.write_json(out/'data/data_audit.json',audit|{'public_source':public})
 boundary=d.index[d.DMC_cosolvent_ratio.isin([0,1])].to_numpy()
 initial={seed:np.random.default_rng(seed).choice(boundary,5,replace=False).tolist() for seed in DEV+VALIDATION}
 pd.DataFrame([dict(seed=seed,phase='development' if seed in DEV else 'validation',order=i,formulation_id=d.iloc[idx].formulation_id)
  for seed,ids in initial.items() for i,idx in enumerate(ids)]).to_csv(out/'data/initializations.csv',index=False)
 for path in [Path(__file__),Path(bo.__file__),HERE/'acquisition_optimization_protocol.md']:
  dest=out/'snapshots'/f'{path.stem}_{bo.sha(path)[:12]}{path.suffix}'
  if not dest.exists():dest.write_bytes(path.read_bytes())
 sig=bo.digest(CONFIG|{'source':audit['source_sha256'],'pool_sha256':bo.sha(out/'data/pool_evaluator_only.csv')})
 return d,initial,sig


def run_id(phase,model,rule,seed):return f'{phase}_{model}_{rule}_{seed}'

def replay(d,ids,phase,model,rule,seed,out,cache):
 rid=run_id(phase,model,rule,seed);path=out/'traces'/f'{rid}.csv'
 # Rebuild from verified prefix caches; no mutation of the original experiment.
 observed=list(ids);rows=[];diagnostics=[];queries=[]
 for order,idx in enumerate(ids,1):
  rows.append(dict(run_id=rid,phase=phase,model=model,rule=rule,seed=seed,evaluation=order,formulation_id=d.iloc[idx].formulation_id,
    conductivity=float(d.iloc[idx].conductivity),source='initial',context_size=0,prediction_key='',score=np.nan))
 while float(d.iloc[observed].conductivity.max())<float(d.conductivity.max())-1e-12:
  remaining=[i for i in d.index if i not in observed]
  training=d.iloc[observed][['formulation_id']+bo.FEATURES+['conductivity']]
  query=d.iloc[remaining][['formulation_id']+bo.FEATURES]
  contract=bo.prediction_contract(training,query);start=time.monotonic()
  if model=='TabPFN':pred,arrays,meta=cache.load(contract)
  else:
   pred,meta=bo.gp_distribution(contract);arrays=None;meta.update(key=key(contract),cache_hit=False,metadata_path='local_GP')
  bo.validate_prediction(pred,contract);best=float(training.conductivity.max());score=acquisition(pred,rule,best,arrays)
  ranked=pred.assign(score=score).sort_values(['score','formulation_id'],ascending=[False,True],kind='stable');choice=ranked.iloc[0]
  idx=int(d.index[d.formulation_id.eq(choice.formulation_id)][0]);n=len(observed)
  # No hidden labels exist in the scoring arguments above.
  true=d.iloc[remaining].conductivity.to_numpy();error=pred['mean'].to_numpy()-true
  diagnostics.append(dict(run_id=rid,phase=phase,model=model,rule=rule,seed=seed,context_size=n,n_queries=len(pred),
      MAE=float(np.abs(error).mean()),RMSE=float(np.sqrt(np.mean(error**2))),MAPE=float(100*np.mean(np.abs(error)/true)),
      coverage90=float(np.mean((true>=pred.q05)&(true<=pred.q95))),width90=float(np.mean(pred.q95-pred.q05))))
  for r,truth,sc in zip(pred.to_dict('records'),true,score):
   queries.append(r|dict(run_id=rid,phase=phase,model=model,rule=rule,seed=seed,context_size=n,actual_mS_cm=float(truth),
      score=float(sc),prediction_key=meta['key'],metadata_path=meta['metadata_path'],selected=r['formulation_id']==choice.formulation_id))
  rows.append(dict(run_id=rid,phase=phase,model=model,rule=rule,seed=seed,evaluation=n+1,formulation_id=choice.formulation_id,
      conductivity=float(d.iloc[idx].conductivity),source='selected',context_size=n,prediction_key=meta['key'],score=float(choice.score)))
  observed.append(idx)
  trace=pd.DataFrame(rows);trace['best_mS_cm']=trace.conductivity.cummax();trace.to_csv(path,index=False)
  pd.DataFrame(diagnostics).to_csv(path.with_name(rid+'_diagnostics.csv'),index=False)
  pd.DataFrame(queries).to_csv(path.with_name(rid+'_predictions.csv'),index=False)
  logging.info('%s n=%s selected=%s best=%.5f reused=%s',rid,len(observed),choice.formulation_id,trace.best_mS_cm.iloc[-1],meta['cache_hit'])
 trace=pd.DataFrame(rows);trace['best_mS_cm']=trace.conductivity.cummax();trace.to_csv(path,index=False)
 return trace


def trace_metrics(trace,poolmax):
 r=dict(run_id=trace.run_id.iloc[0],phase=trace.phase.iloc[0],model=trace.model.iloc[0],rule=trace.rule.iloc[0],seed=int(trace.seed.iloc[0]),
      completed=bool(trace.conductivity.max()>=poolmax-1e-12),n_observed=len(trace))
 for ratio in bo.RATIOS:r[f'cost_{ratio}']=bo.hit_cost(trace.conductivity.to_numpy(),ratio*poolmax)
 for b in [6,10,15,20,30,38]:
  best=float(trace.loc[trace.evaluation.le(b),'conductivity'].max());r[f'best_{b}']=best;r[f'hit_{b}']=float(best>=poolmax-1e-12)
 return r


def collect(d,out):
 metrics=[];diag=[];traces=[]
 for path in sorted((out/'traces').glob('*.csv')):
  if path.stem.endswith(('_diagnostics','_predictions')):continue
  trace=pd.read_csv(path);metrics.append(trace_metrics(trace,d.conductivity.max()));traces.append(trace)
  dp=path.with_name(path.stem+'_diagnostics.csv')
  if dp.exists():diag.append(pd.read_csv(dp))
 m=pd.DataFrame(metrics)
 if len(m):m.to_csv(out/'tables/run_metrics.csv',index=False)
 if traces:pd.concat(traces,ignore_index=True).to_csv(out/'tables/all_traces.csv',index=False)
 if diag:pd.concat(diag,ignore_index=True).to_csv(out/'tables/prediction_diagnostics.csv',index=False)
 return m


def select_rule(metrics,model):
 sub=metrics[(metrics.phase=='development')&(metrics.model==model)&metrics.completed]
 if any(len(sub[sub.rule.eq(rule)])!=len(DEV) for rule in RULES):raise ValueError('Selection requires all frozen development runs')
 scores=sub.groupby('rule').agg(cost=('cost_1.0','mean'),best6=('best_6','mean')).reset_index()
 scores['priority']=scores.rule.map(dict(zip(RULES,range(len(RULES)))))
 return scores.sort_values(['cost','best6','priority'],ascending=[True,False,True]).iloc[0].rule


def summarize(d,initial,out,status):
 m=collect(d,out)
 if not len(m):return
 complete=m[m.completed].copy();cols=[f'cost_{r}' for r in bo.RATIOS]+[f'best_{b}' for b in [6,10,15,20]]+[f'hit_{b}' for b in [6,10,15,20]]
 summary=complete.groupby(['phase','model','rule'],sort=False)[cols].mean().reset_index()
 counts=complete.groupby(['phase','model','rule']).size().rename('n').reset_index();summary=summary.merge(counts)
 for phase,seeds in [('development',DEV),('validation',VALIDATION)]:
  # Random is reference for the FULL planned phase, not mixed with a partial model mean.
  rr=[]
  for seed in seeds:
   curve,costs=bo.random_reference(d.conductivity.to_numpy(),initial[seed]);c=curve.set_index('budget')
   rr.append({**{f'cost_{r}':costs[r] for r in bo.RATIOS},**{f'best_{b}':c.loc[b,'best_mS_cm'] for b in [6,10,15,20]},
               **{f'hit_{b}':c.loc[b,'hit_1.0'] for b in [6,10,15,20]}})
  summary=pd.concat([summary,pd.DataFrame([dict(phase=phase,model='Random',rule='exact',n=len(seeds),**pd.DataFrame(rr).mean().to_dict())])],ignore_index=True)
 summary.to_csv(out/'tables/summary.csv',index=False)
 # Immutable legacy reference: same development starts, not a validation result.
 old=pd.read_csv(BASE/'tables/summary.csv');old.to_csv(out/'tables/original_UCB1.96_reference.csv',index=False)
 diags_path=out/'tables/prediction_diagnostics.csv'
 initial_diag=pd.DataFrame()
 if diags_path.exists():
  all_diag=pd.read_csv(diags_path);first=all_diag[all_diag.context_size.eq(5)].copy()
  first=first.drop_duplicates(['phase','model','seed'])
  initial_diag=first.groupby(['phase','model'])[['MAE','RMSE','MAPE','coverage90','width90']].mean().reset_index()
  initial_diag['n']=first.groupby(['phase','model']).size().to_numpy()
  initial_diag.to_csv(out/'tables/initial_prediction_summary.csv',index=False)
 pairrows=[]
 selected=json.loads((out/'selected_policy.json').read_text()) if (out/'selected_policy.json').exists() else None
 if selected:
  tab=complete[(complete.phase=='validation')&(complete.model=='TabPFN')&(complete.rule==selected['TabPFN'])].set_index('seed')
  for rule in sorted(set([selected['TabPFN'],selected['GP'],'UCB1.96'])):
   gp=complete[(complete.phase=='validation')&(complete.model=='GP')&(complete.rule==rule)].set_index('seed');shared=tab.index.intersection(gp.index)
   for metric in ['cost_1.0','best_6','best_15']:
    if not len(shared):continue
    delta=tab.loc[shared,metric]-gp.loc[shared,metric];mean=float(delta.mean());margin=float(t.ppf(.975,len(delta)-1)*delta.std(ddof=1)/np.sqrt(len(delta))) if len(delta)>1 else float('nan')
    pairrows.append(dict(TabPFN_rule=selected['TabPFN'],GP_rule=rule,metric=metric,n=len(shared),TabPFN_minus_GP=mean,
      low=mean-margin,high=mean+margin,TabPFN_lower=int((delta<0).sum()),equal=int((delta==0).sum()),TabPFN_higher=int((delta>0).sum())))
 if pairrows:pd.DataFrame(pairrows).to_csv(out/'tables/validation_paired_comparisons.csv',index=False)
 figures(out,summary)
 lines=['# clio：采集策略优化与新起点检验','',f'状态：**{status}**。协议见[固定协议](../acquisition_optimization_protocol.md)。原UCB主结果见[原报告](../README.md)。','',
  '本轮只改选点规则，不改TabPFN、输入、标签或GP超参数。每个方法都按自己所选的新标签逐轮重新预测，非静态排名。','',
  '## 规则是什么意思','',
  '- Mean：优先试预测均值最高的配方。',
  '- UCB0.5：均值加0.5倍标准差，较原1.96降低探索奖励。',
  '- EI：综合超过目前已测最好电导率的概率和幅度，完整考虑TabPFN分布尾部。','',
  '## 阶段和选择','',
  '开发：原起点42–51，三规则×两模型。结果是在看过旧UCB结果后开展的方法开发，不是独立确认。',
  '检验：开发结果按平均精确最优发现次数、预算6最好值、固定规则顺序选出一个TabPFN方案及GP自身最好方案，固定后在起点52–61运行。新起点仍共享同一38配方池，不代表新化学体系。']
 if selected:lines += ['',f"冻结所选方案：TabPFN **{selected['TabPFN']}**；GP开发最佳 **{selected['GP']}**。检验同时保留GP相同采集规则、GP自身开发最佳和GP原UCB1.96。"]
 for phase,title in [('development','开发对比：原10起点'),('validation','新起点检验：52–61')]:
  lines+=['',f'## {title}','','|模型|规则|已完成起点|最优平均发现次数|达到95%平均次数|预算6最好电导率|预算15最好电导率|预算15最优命中率|',
          '|---|---|---:|---:|---:|---:|---:|---:|']
  for r in summary[summary.phase.eq(phase)].to_dict('records'):
   lines.append(f"|{r['model']}|{r['rule']}|{int(r['n'])}|{r['cost_1.0']:.3f}|{r['cost_0.95']:.3f}|{r['best_6']:.4f}|{r['best_15']:.4f}|{100*r['hit_15']:.1f}%|")
  lines+=['','未完成的模型行仅作进度，不与满10起点的随机行直接比较；完整检验必须所有预定方案均完成10起点。']
 lines+=['','## 初始预测指标','','采集函数尚未选点时，相同5条上下文的预测完全相同。因此MAE和区间覆盖率不随采集规则变化；不能将复用的同一预测算成三次独立胜出。初始已含最优而无需预测的起点不计入此诊断。',
  '', '|阶段|模型|有预测的起点数|MAE mS/cm|MAPE|90%区间覆盖率|区间宽度 mS/cm|','|---|---|---:|---:|---:|---:|---:|']
 for r in initial_diag.to_dict('records'):
  lines.append(f"|{r['phase']}|{r['model']}|{int(r['n'])}|{r['MAE']:.4f}|{r['MAPE']:.2f}%|{100*r['coverage90']:.1f}%|{r['width90']:.4f}|")
 lines+=['','覆盖率越接近90%且区间较窄才更有用，不是区间越宽、覆盖率越高就必然更好。后续上下文由不同路径决定，误差不再是完全匹配上下文的模型比较。',
  '', '## 配对检验','','成本差为TabPFN−GP，负值有利于TabPFN；最好电导率差则正值有利。区间仅描述10个初始化在本池内的变化，未作多重比较校正。',
  '', '|GP对照规则|指标|配对起点|TabPFN−GP|95%区间|','|---|---|---:|---:|---|']
 for r in pairrows:lines.append(f"|{r['GP_rule']}|{r['metric']}|{r['n']}|{r['TabPFN_minus_GP']:.4f}|[{r['low']:.4f}, {r['high']:.4f}]|")
 lines+=['','![开发与新起点的发现成本](figures/Fig01_costs.png)','',
  '## 证据边界','','38组历史实测池回放、同一近室温目标；不等于原文1000多个候选空间或新机器人闭环。95%初始命中仍可能饱和；98.5%与精确最优在本池是同一事件。',
  '原文约6倍AF来自AEM衍生模拟环境，不能与本轮直接排名。开发方案是在看过旧结果后提出，不把参数选择收益冒充独立验证。',
  '', '## 文件','','[逐起点指标](tables/run_metrics.csv) · [总表](tables/summary.csv) · [初始预测指标](tables/initial_prediction_summary.csv) · [新起点配对比较](tables/validation_paired_comparisons.csv)',
  '', '缓存和原141份预测按完整有序上下文匹配复用；`cache_index.json`记录实际分布来源与哈希。额度记录见`quota_history.json`；失败或不完整请求不会自动重复计费。',
  '', '离线重建：项目根目录运行 `.venv/bin/python study/objective_search/clio/compare_clio_acquisition_rules.py report`。','']
 (out/'README.md').write_text('\n'.join(lines))


def figures(out,summary):
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(11,4))
 for ax,phase,title in zip(axes,['development','validation'],['Development: original starts','New-start evaluation']):
  g=summary[(summary.phase==phase)&(summary.model!='Random')]
  labels=[f"{r.model}\n{r.rule}\nn={r.n}" for r in g.itertuples()]
  colors=['#476f97' if m=='TabPFN' else '#c46a69' for m in g.model]
  ax.bar(np.arange(len(g)),g['cost_1.0'],color=colors);ax.set(xticks=np.arange(len(g)),xticklabels=labels,
       ylabel='Mean evaluations to measured-pool maximum',title=title,ylim=(0,38))
  random=summary[(summary.phase==phase)&(summary.model=='Random')]['cost_1.0'].iloc[0]
  ax.axhline(random,color='gray',ls='--',label='Random, full phase');ax.grid(axis='y',alpha=.2);ax.tick_params(axis='x',labelsize=7)
 fig.tight_layout()
 for ext in ['png','pdf']:fig.savefig(out/f'figures/Fig01_costs.{ext}',dpi=240,bbox_inches='tight')
 plt.close(fig);summary.to_csv(out/'figure_source_data/Fig01_costs.csv',index=False)


def audit(d,initial,out,cache):
 rows=[]
 for path in sorted((out/'traces').glob('*.csv')):
  if path.stem.endswith(('_predictions','_diagnostics')):continue
  tr=pd.read_csv(path);rid=tr.run_id.iloc[0];model=tr.model.iloc[0];rule=tr.rule.iloc[0];seed=int(tr.seed.iloc[0])
  assert tr.formulation_id.is_unique and tr.evaluation.tolist()==list(range(1,len(tr)+1))
  assert tr.formulation_id.iloc[:5].tolist()==d.iloc[initial[seed]].formulation_id.tolist()
  assert np.allclose(tr.conductivity,d.set_index('formulation_id').loc[tr.formulation_id].conductivity)
  assert tr.conductivity.max()>=d.conductivity.max()-1e-12
  if len(tr)>5:
   qs=pd.read_csv(path.with_name(rid+'_predictions.csv'))
   for n,g in qs.groupby('context_size'):
    n=int(n);observed=tr.iloc[:n].formulation_id.tolist();remain=d[~d.formulation_id.isin(observed)]
    contract=bo.prediction_contract(d.set_index('formulation_id').loc[observed].reset_index()[['formulation_id']+bo.FEATURES+['conductivity']],remain[['formulation_id']+bo.FEATURES])
    arrays=None
    if model=='TabPFN':
     meta_path=Path(g.metadata_path.iloc[0]);md=json.loads(meta_path.read_text())
     assert canonical(md['contract'])==canonical(contract)
     assert bo.sha(meta_path.with_suffix('.csv'))==md['csv_sha256'] and bo.sha(meta_path.with_suffix('.npz'))==md['raw_sha256']
     arrays=dict(np.load(meta_path.with_suffix('.npz')))
    score=acquisition(g,rule,float(tr.conductivity.iloc[:n].max()),arrays)
    assert np.allclose(score,g.score,atol=1e-8)
    chosen=g.assign(recomputed=score).sort_values(['recomputed','formulation_id'],ascending=[False,True],kind='stable').iloc[0].formulation_id
    assert chosen==tr.formulation_id.iloc[n]
    assert g.formulation_id.tolist()==remain.formulation_id.tolist()
  rows.append(dict(run_id=rid,passed=True,decisions=len(tr)-5))
 bo.write_json(out/'tables/result_audit.json',dict(passed=True,checked_runs=len(rows),checked_decisions=sum(r['decisions'] for r in rows),runs=rows))


def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['audit','run','report']);parser.add_argument('--resume',action='store_true');parser.add_argument('--development-only',action='store_true');args=parser.parse_args()
 out=OUT;out.mkdir(exist_ok=True)
 logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',force=True,handlers=[logging.StreamHandler(),logging.FileHandler(out/'execution.log')]);logging.getLogger('fontTools').setLevel(logging.WARNING)
 d,initial,sig=prepare(out);cp=out/'run_config.json';prev=json.loads(cp.read_text()) if cp.exists() else {}
 if prev and prev['signature']!=sig:raise ValueError('Changed frozen protocol/data')
 config=CONFIG|dict(signature=sig,status='audited',started=prev.get('started',bo.now()),code_sha256=bo.sha(__file__),
   software_versions=bo.software_versions(),execution_authorization='user requested sequential Mean, weak-UCB and EI optimization with metric comparisons')
 if args.mode=='audit':bo.write_json(cp,config);return
 if args.mode=='report':summarize(d,initial,out,prev.get('status','unknown'));return
 if prev.get('status')=='completed' and not args.resume:raise FileExistsError('Use --resume for existing completed experiment')
 cache=Cache(out,online=True)
 try:
  # Free GP paths first; TabPFN then reuses prediction prefixes wherever exact context matches.
  for model in ['GP','TabPFN']:
   for rule in RULES:
    for seed in DEV:
     config.update(status='development_running',active_run=run_id('development',model,rule,seed));bo.write_json(cp,config)
     replay(d,initial[seed],'development',model,rule,seed,out,cache)
     if model=='TabPFN':summarize(d,initial,out,'development_running')
  metrics=collect(d,out);selection=dict(TabPFN=select_rule(metrics,'TabPFN'),GP=select_rule(metrics,'GP'),selection_time=bo.now(),development_metrics_sha256=bo.sha(out/'tables/run_metrics.csv'))
  if (out/'selected_policy.json').exists():
   old=json.loads((out/'selected_policy.json').read_text())
   if any(old[m]!=selection[m] for m in ['TabPFN','GP']):raise ValueError('Frozen selection changed')
  else:bo.write_json(out/'selected_policy.json',selection)
  config['selected_policy']=selection
  if not args.development_only:
   for model,rules in [('GP',list(dict.fromkeys([selection['TabPFN'],selection['GP'],'UCB1.96']))),('TabPFN',[selection['TabPFN']])]:
    for rule in rules:
     for seed in VALIDATION:
      config.update(status='validation_running',active_run=run_id('validation',model,rule,seed));bo.write_json(cp,config)
      replay(d,initial[seed],'validation',model,rule,seed,out,cache)
      if model=='TabPFN':summarize(d,initial,out,'validation_running')
  config['status']='development_completed' if args.development_only else 'completed'
  audit(d,initial,out,cache);summarize(d,initial,out,config['status'])
 except Exception as exc:
  config.update(status='stopped',error_type=type(exc).__name__);logging.error('Stopped (%s), no automatic paid retries',type(exc).__name__)
  summarize(d,initial,out,'stopped_partial');raise
 finally:
  config.update(finished=bo.now(),cache_hits_this_execution=cache.hits,new_predictions_this_execution=cache.new)
  if cache.history:config['credits_observed']=cache.history[-1]['used']-cache.history[0]['used']
  bo.write_json(cp,config)
  bo.write_json(out/'cache_index.json',{k:str(v.resolve()) for k,v in cache.index.items()})

if __name__=='__main__':main()
