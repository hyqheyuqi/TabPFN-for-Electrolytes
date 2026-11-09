"""Final, offline new-start curves for the frozen acquisition comparison."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import replay_clio_search as bo


def main():
 out=Path(__file__).resolve().parent/'acquisition_results'
 assert json.loads((out/'run_config.json').read_text())['status']=='completed'
 tr=pd.read_csv(out/'tables/all_traces.csv');tr=tr[tr.phase.eq('validation')]
 rows=[]
 for (model,rule,seed),g in tr.groupby(['model','rule','seed']):
  for budget in range(5,39):
   best=float(g[g.evaluation.le(budget)].conductivity.max())
   rows.append(dict(model=model,rule=rule,seed=seed,budget=budget,best=best,hit=float(best>=13.69289339-1e-12)))
 pool=pd.read_csv(out/'data/pool_evaluator_only.csv');initial=pd.read_csv(out/'data/initializations.csv')
 for seed,g in initial[initial.phase.eq('validation')].groupby('seed'):
  indices=[int(pool.index[pool.formulation_id.eq(fid)][0]) for fid in g.sort_values('order').formulation_id]
  curve,_=bo.random_reference(pool.conductivity.to_numpy(),indices)
  for r in curve.to_dict('records'):rows.append(dict(model='Random',rule='exact',seed=seed,budget=r['budget'],best=r['best_mS_cm'],hit=r['hit_1.0']))
 allrows=pd.DataFrame(rows);summary=allrows.groupby(['model','rule','budget'])[['best','hit']].mean().reset_index()
 allrows.to_csv(out/'tables/validation_budget_curves.csv',index=False);summary.to_csv(out/'tables/validation_budget_summary.csv',index=False)
 fig,axes=plt.subplots(1,2,figsize=(11,4))
 colors={'Mean':'#b8605f','UCB0.5':'#cf9680','EI':'#8f5262','UCB1.96':'#a0a0a0'}
 for (model,rule),g in summary.groupby(['model','rule']):
  if model=='TabPFN':color='#416f99';label=f'TabPFN / {rule}';style='-'
  elif model=='Random':color='#555555';label='Random (exact)';style='--'
  else:color=colors[rule];label=f'GP / {rule}';style=':' if rule=='UCB1.96' else '-'
  axes[0].plot(g.budget,g.best,style,color=color,label=label,lw=2)
  axes[1].plot(g.budget,100*g.hit,style,color=color,lw=2)
 axes[0].set(xlabel='Total evaluations (including initial 5)',ylabel='Mean best measured conductivity (mS/cm)')
 axes[1].set(xlabel='Total evaluations',ylabel='Measured-pool optimum hit rate (%)',ylim=(0,100))
 axes[0].legend(frameon=False,fontsize=8)
 for ax in axes:ax.grid(alpha=.2)
 fig.suptitle('Frozen policy, 10 new initialization seeds (52–61)');fig.tight_layout()
 for ext in ['png','pdf']:fig.savefig(out/f'figures/Fig02_validation_progress.{ext}',dpi=240,bbox_inches='tight')
 plt.close(fig);summary.to_csv(out/'figure_source_data/Fig02_validation_progress.csv',index=False)

if __name__=='__main__':main()
