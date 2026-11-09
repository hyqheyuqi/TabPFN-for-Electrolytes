"""Step 3: audited conductivity cards and matched offline acquisitions."""
from __future__ import annotations

import itertools
import importlib.metadata
import json
import random
import shutil
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .data_sources import project_directory, ampere_data_directory
from .ampere_conductivity import SEEDS, API_MODEL_PATH, api_predict, load_release, write_table, write_json, sha256, save_figure
import matplotlib.pyplot as plt
from .ampere_conductivity import apply_style
apply_style()

METHODS = ('EI', 'PI', 'UCB', 'Thompson')
CONCENTRATIONS = (.1, .3, 1., 2.)
COLORS = dict(zip(METHODS, ('#7E9B76', '#C49A6C', '#527C8E', '#9583A6')))


def paths_for(root, output=None):
    out = output or project_directory()/'results/conductivity_guided_search'
    paths = {'root': root, 'results': out, 'legacy': project_directory()/'reference_runs/20261001_Step10-11',
             'data': ampere_data_directory(), 'design': ampere_data_directory()/'GitHub/Data/Electrolyte_Design'}
    for name in ('tables', 'figures', 'figure_source_data', 'configs', 'logs', 'snapshots'):
        paths[name] = out/name
        paths[name].mkdir(parents=True, exist_ok=True)
    return paths


def build_queries(compositions, features):
    if compositions.shape != (44, 14) or features.shape != (100, 15):
        raise ValueError('Invalid released input dimensions')
    cuts = np.cumsum([0]+[8]*9+[14]*2)
    rows, audits = [], []
    descriptor_positions = [3,4,5,7,8,9,11,12,13]
    for idx, candidate in compositions.iterrows():
        group = idx//4
        block = features.iloc[cuts[group]:cuts[group+1]]
        block = block[np.isclose(block.iloc[:,0], candidate.iloc[0])]
        np.testing.assert_allclose(block.iloc[0,descriptor_positions], candidate.iloc[descriptor_positions], atol=1e-12)
        salt = block.iloc[:,1].to_numpy(float)
        coefficients = {}
        for pos in [2,6,10]:
            coefficients[pos] = np.polyfit(salt, block.iloc[:,pos].to_numpy(float), 1)
            residual = float(np.abs(np.polyval(coefficients[pos], salt)-block.iloc[:,pos]).max())
            audits.append({'formulation_id': group+1, 'feature': features.columns[pos],
                           'max_linear_residual': residual, 'within_rounding_1e_4': residual<=1e-4,
                           'policy':'exact source rows; missing queries use bracket interpolation'})
        for c in CONCENTRATIONS:
            query = block.iloc[0].copy()
            query.iloc[0:2] = [candidate.iloc[0], c]
            exact=block[np.isclose(salt,c)]
            if not exact.empty:
                query=exact.iloc[0].copy()
            else:
                if not salt.min()<c<salt.max(): raise ValueError('Query requires extrapolation')
                sort=np.argsort(salt)
                for pos in [2,6,10]:
                    query.iloc[pos]=np.interp(c,salt[sort],block.iloc[:,pos].to_numpy(float)[sort])
            rows.append({'candidate_id': idx+1, 'formulation_id': group+1,
                         'ce_0': float(candidate.iloc[1]), 'query_concentration_mol_L': c,
                         **query.to_dict()})
    return pd.DataFrame(rows), pd.DataFrame(audits).drop_duplicates()


def calculate_rcs(workbook, temperature):
    rows = []
    book = pd.ExcelFile(workbook)
    if len(book.sheet_names) != 11:
        raise ValueError('Expected 11 formulation sheets')
    for group, sheet in enumerate(book.sheet_names):
        raw = pd.read_excel(workbook, sheet_name=sheet, header=None)
        rates = raw.iloc[3:,0].astype(str).str.extract(r'([0-9.]+)')[0].astype(float).to_numpy()
        np.testing.assert_array_equal(rates, [2,4,6] if temperature==25 else [2,4,6,8,10])
        for j in range(4):
            values = raw.iloc[3:,1+2*j:3+2*j].to_numpy(float)
            rows.append({'candidate_id': group*4+j+1, 'formulation': sheet,
                         'calculated_RCS': float((rates*values.mean(axis=1)).sum())})
    return pd.DataFrame(rows)


def audit_inputs(paths):
    features, _, _, _ = load_release(paths)
    queries, candidates, audits, source_paths = [], [], [], []
    for t in (25,45):
        comp_path = paths['design']/f'Composition-based_Representations_{t}.xlsx'
        label_path = paths['design']/f'Label_RCS_{t}.xlsx'
        raw_path = paths['data']/f'Zenodo/{t} Degrees Celsius.xlsx'
        prop_path = paths['design']/f'Property_Representations_{t}.xlsx'
        source_paths.extend([comp_path,label_path,raw_path,prop_path])
        comp, prop = pd.read_excel(comp_path), pd.read_excel(prop_path)
        y = pd.read_excel(label_path).iloc[:,0].to_numpy(float)
        q, audit = build_queries(comp, features)
        q.insert(0,'temperature_c',t); audit.insert(0,'temperature_c',t)
        q.insert(3,'formulation',np.repeat(pd.ExcelFile(raw_path).sheet_names,16))
        rcs = calculate_rcs(raw_path,t)
        rcs['temperature_c']=t; rcs['RCS']=y
        rcs['recalculation_difference']=rcs.calculated_RCS-y
        rcs['ce_0']=comp.iloc[:,1].to_numpy(float)
        rcs['AMPERE_Infor_Sigma_e']=prop.Sigma_e
        np.testing.assert_allclose(prop.ce_0,rcs.ce_0,atol=1e-12)
        if y.shape!=(44,) or not np.isfinite(y).all():
            raise ValueError('RCS input audit failed')
        rcs['within_rounding_0_01']=rcs.recalculation_difference.abs().le(.011)
        rcs['bo_label_source']='released Label_RCS; raw reconstruction is audit only'
        queries.append(q); candidates.append(rcs); audits.append(audit)
    queries=pd.concat(queries,ignore_index=True); candidates=pd.concat(candidates,ignore_index=True)
    write_table(queries,paths['tables']/'query_features')
    write_table(candidates,paths['tables']/'candidate_manifest')
    write_table(pd.concat(audits),paths['tables']/'concentration_relation_audit')
    source_paths.extend([paths['data']/'GitHub/Data/Property_Prediction/Original_Property_Datasets/AMPERE_Data/Feature_Sigma_e.xlsx',
                         paths['data']/'GitHub/Data/Property_Prediction/Original_Property_Datasets/AMPERE_Data/Label_Sigma_e.xlsx',
                         paths['data']/'GitHub/Data/Property_Prediction/Original_Property_Datasets/AMPERE_Data/Training_Index_AL/Sigma_e/Train_index_46.xlsx'])
    write_table(pd.DataFrame([{'path':str(p.relative_to(paths['root'])),'sha256':sha256(p)} for p in source_paths]),
                paths['tables']/'input_inventory')
    # Legacy is a comparison source; only a verified seed=100 prediction can be reused.
    old_q=pd.read_csv(paths['legacy']/'query_features_352x15.csv')
    np.testing.assert_array_equal(queries[['temperature_c','candidate_id','query_concentration_mol_L']],
                                  old_q[['temperature_c','candidate_id','query_concentration_mol_L']])
    maxdiff=float(np.max(np.abs(queries[features.columns].to_numpy()-old_q[features.columns].to_numpy())))
    legacy_config=json.loads((paths['legacy']/'run_config.json').read_text())
    model=legacy_config['models']['Sigma_e']
    assert model['checkpoint']==API_MODEL_PATH and model['n_estimators']==8 and model['target']=='raw' and not model['local_correction']
    for p in source_paths[-3:]:
        name=str(p.name)
        matched=[v for k,v in legacy_config['input_sha256'].items() if k.endswith('/'+name)]
        assert len(matched)==1 and matched[0]==sha256(p),f'Legacy input hash mismatch: {name}'
    delta=queries[features.columns].to_numpy()-old_q[features.columns].to_numpy()
    diff_table=queries[['temperature_c','candidate_id','query_concentration_mol_L']].copy()
    diff_table['max_abs_feature_difference']=np.abs(delta).max(axis=1)
    write_table(diff_table,paths['tables']/'legacy_query_comparison')
    write_json(paths['configs']/'legacy_input_audit.json',{'passed':True,'query_max_abs_difference':maxdiff,
               'seed_100_reuse':False,'reason':'source exact rows preserved; Legacy smooths all source concentrations',
               'rcs_max_recalculation_difference':float(candidates.recalculation_difference.abs().max())})
    return queries,candidates


def run_properties(paths, queries, resume):
    x,y,train,_=load_release(paths)
    file=paths['tables']/'sigma_predictions.csv'
    existing=pd.read_csv(file) if resume and file.exists() else pd.DataFrame()
    for seed in SEEDS:
        if not existing.empty and len(existing[existing.seed.eq(seed)])==352: continue
        print(f'property seed={seed}: API 352 queries',flush=True)
        values=api_predict(x.iloc[train],y[train],queries[x.columns],seed,paths['root'])
        origin='conductivity_guided_search TabPFN API'
        if values.shape!=(352,) or not np.isfinite(values).all(): raise ValueError('Invalid predictions')
        frame=queries[['temperature_c','candidate_id','formulation_id','ce_0','query_concentration_mol_L']].copy()
        frame['seed']=seed; frame['Sigma_e_pred']=values; frame['origin']=origin
        existing=pd.concat([existing,frame],ignore_index=True)
        existing.to_csv(file,index=False)
    write_table(existing,paths['tables']/'sigma_predictions')
    counts=existing.groupby(['temperature_c','candidate_id','seed']).size()
    assert counts.eq(4).all() and len(existing)==2112
    by_seed=existing.groupby(['temperature_c','candidate_id','seed'],as_index=False).agg(
        Infor_Sigma_e=('Sigma_e_pred','sum'),ce_0=('ce_0','first'))
    summary=by_seed.groupby(['temperature_c','candidate_id'],as_index=False).agg(
        Infor_Sigma_e=('Infor_Sigma_e','mean'),seed_std=('Infor_Sigma_e','std'),
        seed_min=('Infor_Sigma_e','min'),seed_max=('Infor_Sigma_e','max'),ce_0=('ce_0','first'))
    write_table(by_seed,paths['tables']/'infor_sigma_by_seed')
    write_table(summary,paths['tables']/'infor_sigma_summary')
    return existing,by_seed,summary


def replay(x,y,seed,method,temperature):
    observed=np.random.RandomState(seed).choice(len(y),size=6,replace=False).tolist()
    rounds=[0]*6
    while len(observed)<len(y):
        remaining=np.array(sorted(set(range(len(y)))-set(observed)))
        gp=GaussianProcessRegressor(kernel=RBF(.6,(1e-3,1e3)),alpha=.25,normalize_y=True,
                                    random_state=seed,n_restarts_optimizer=0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',ConvergenceWarning)
            gp.fit(x[observed],y[observed])
        if method=='Thompson':
            mean,cov=gp.predict(x[remaining],return_cov=True)
            cov=(cov+cov.T)/2
            jitter=max(1e-10,float(np.diag(cov).max())*1e-10)
            rng=np.random.default_rng(seed*1000+max(rounds)+1)
            score=rng.multivariate_normal(mean,cov+np.eye(len(cov))*jitter,check_valid='raise')
        else:
            mean,std=gp.predict(x[remaining],return_std=True)
            improvement=mean-y[observed].max()-.01
            z=improvement/np.maximum(std,1e-12)
            if method=='EI': score=improvement*norm.cdf(z)+std*norm.pdf(z)
            elif method=='PI': score=norm.cdf(z)
            elif method=='UCB': score=mean+1.96*std
            else: raise ValueError(method)
        chosen=remaining[np.lexsort((remaining,-score))[:6]].tolist()
        rounds.extend([max(rounds)+1]*len(chosen)); observed.extend(chosen)
    return measure_replay(observed,rounds,y,seed,method,temperature)


def measure_replay(order,rounds,y,seed,method,temperature):
    if len(order)!=len(y) or len(set(order))!=len(y): raise ValueError('Repeated/missing candidate')
    best=np.maximum.accumulate(y[order]); optimum=float(y.max())
    first=int(np.flatnonzero(np.isclose(best,optimum,rtol=0,atol=1e-12))[0])+1
    found_round=rounds[first-1]
    cost=min(len(y),6+found_round*6)
    normalized=(best-y.min())/(y.max()-y.min())
    common={'temperature_c':temperature,'method':method,'replay_seed':seed}
    summary={**common,'initial_candidates':' '.join(str(i+1) for i in order[:6]),
             'evaluations_to_optimum':first,'batch_cost_to_optimum':cost,'round_to_optimum':found_round,
             'normalized_auc':float(normalized.mean()),
             'normalized_auc_batch':float(normalized[np.unique(np.append(np.arange(6,len(y)+1,6),len(y)))-1].mean())}
    trace=pd.DataFrame({**common,'evaluation':np.arange(1,len(y)+1),'candidate_id':np.array(order)+1,
                        'batch_round':rounds,'selected_RCS':y[order],'best_so_far_RCS':best,
                        'normalized_best_so_far':normalized,'hit':np.isclose(best,optimum,rtol=0,atol=1e-12)})
    return summary,trace


def run_bo(paths,cards,candidates,n_replays,resume):
    seeds=random.Random(0).sample(range(1001),n_replays)
    file=paths['tables']/'replay_summary.csv'
    tf=paths['tables']/'replay_trajectories.csv'
    summaries=pd.read_csv(file).to_dict('records') if resume and file.exists() else []
    done={(int(r['temperature_c']),r['method'],int(r['replay_seed'])) for r in summaries}
    if not resume and (file.exists() or tf.exists()): raise FileExistsError('Use --resume')
    started=time.monotonic()
    with threadpool_limits(limits=1):
        for number,seed in enumerate(seeds,1):
            for t in (25,45):
                c=cards[cards.temperature_c.eq(t)].sort_values('candidate_id')
                x=StandardScaler().fit_transform(c[['Infor_Sigma_e','ce_0']])
                y=candidates[candidates.temperature_c.eq(t)].sort_values('candidate_id').RCS.to_numpy(float)
                for method in METHODS:
                    if (t,method,seed) in done: continue
                    row,trace=replay(x,y,seed,method,t)
                    trace.to_csv(tf,index=False,mode='a' if tf.exists() else 'w',header=not tf.exists())
                    summaries.append(row)
                pd.DataFrame(summaries).to_csv(file,index=False)
            if number%10==0 or number==1:
                print(f'replay {number}/{n_replays}; elapsed={time.monotonic()-started:.1f}s',flush=True)
    summary=pd.DataFrame(summaries)
    write_table(summary,paths['tables']/'replay_summary')
    return summary,pd.read_csv(tf)


def summarize(paths,summary,trace,cards,candidates):
    aggregate=summary.groupby(['temperature_c','method'],as_index=False).agg(
        replays=('replay_seed','size'),evaluations_mean=('evaluations_to_optimum','mean'),
        evaluations_median=('evaluations_to_optimum','median'),batch_cost_mean=('batch_cost_to_optimum','mean'),
        auc_mean=('normalized_auc','mean'),auc_std=('normalized_auc','std'))
    hits=[]
    for (t,m),g in summary.groupby(['temperature_c','method']):
        for budget in [6,12,18,24,30,36,42,44]:
            p=float(g.batch_cost_to_optimum.le(budget).mean());n=len(g)
            denom=1+1.96**2/n; center=(p+1.96**2/(2*n))/denom
            half=1.96*np.sqrt(p*(1-p)/n+1.96**2/(4*n*n))/denom
            hits.append({'temperature_c':t,'method':m,'budget':budget,'hit_rate':p,'n':n,
                         'ci_low':center-half,'ci_high':center+half})
    hits=pd.DataFrame(hits)
    aggregate=aggregate.merge(hits[hits.budget.eq(18)][['temperature_c','method','hit_rate']],on=['temperature_c','method'])
    aggregate=aggregate.sort_values(['temperature_c','batch_cost_mean','hit_rate','auc_mean'],ascending=[True,True,False,False])
    aggregate['rank']=aggregate.groupby('temperature_c').cumcount()+1
    write_table(aggregate,paths['tables']/'acquisition_ranking')
    write_table(hits,paths['tables']/'budget_hit_rates')
    pairs=[]
    for t in (25,45):
        for metric in ['batch_cost_to_optimum','evaluations_to_optimum','normalized_auc']:
            wide=summary[summary.temperature_c.eq(t)].pivot(index='replay_seed',columns='method',values=metric)
            for a,b in itertools.combinations(METHODS,2):
                diff=wide[a]-wide[b];se=float(diff.std(ddof=1)/np.sqrt(len(diff)))
                pairs.append({'temperature_c':t,'metric':metric,'method_a':a,'method_b':b,
                              'a_minus_b':float(diff.mean()),'ci_low':float(diff.mean()-1.96*se),
                              'ci_high':float(diff.mean()+1.96*se),'n':len(diff),
                              'a_better':int((diff>1e-12 if metric=='normalized_auc' else diff< -1e-12).sum()),
                              'ties':int(np.isclose(diff,0,atol=1e-12).sum())})
    pairs=pd.DataFrame(pairs);write_table(pairs,paths['tables']/'pairwise_comparisons')
    overall=aggregate.groupby('method',as_index=False).agg(mean_temperature_rank=('rank','mean'),
                    batch_cost_mean=('batch_cost_mean','mean'),hit18_mean=('hit_rate','mean'),auc_mean=('auc_mean','mean'))
    overall=overall.sort_values(['mean_temperature_rank','batch_cost_mean','hit18_mean','auc_mean'],ascending=[True,True,False,False])
    write_table(overall,paths['tables']/'overall_ranking')
    associations=[]
    full=cards.merge(candidates,on=['temperature_c','candidate_id','ce_0'])
    for t,g in full.groupby('temperature_c'):
        associations.append({'temperature_c':t,'spearman_sigma_rcs':float(spearmanr(g.Infor_Sigma_e,g.RCS).statistic),
                             'spearman_ce0_rcs':float(spearmanr(g.ce_0,g.RCS).statistic)})
    write_table(pd.DataFrame(associations),paths['tables']/'property_rcs_associations')
    legacy=pd.read_csv(project_directory()/'reference_runs/20261006_Step12-14_MultiAcquisition/strategy_summary.csv')
    legacy['method']=legacy.strategy.map({'property_2d_ei':'EI','property_2d_pi':'PI','property_2d_ucb':'UCB','property_2d_ts':'Thompson'})
    legacy=legacy.dropna(subset=['method'])
    comparison=aggregate.merge(legacy,on=['temperature_c','method'])
    write_table(comparison,paths['tables']/'legacy_results_comparison')
    write_json(paths['configs']/'selection.json',{'selected_method':str(overall.iloc[0]['method']),
               'rule':'equal-temperature average rank; batch cost, hit@18, AUC tie breaks',
               'hypothesis_UCB_best':bool(overall.iloc[0]['method']=='UCB'),
               'claim_scope':'two finite offline candidate spaces; replay uncertainty is initialization variation, not new experimental replication'})
    return aggregate,hits,pairs,overall,full


def figures(paths,predictions,by_seed,cards,candidates,summary,trace,aggregate,hits,pairs,full,*,saved_rounds=False):
    plt.rcParams.update({'font.size':9,'axes.spines.top':True,'axes.spines.right':True,
                         'pdf.fonttype':42,'axes.prop_cycle':plt.cycler(color=list(COLORS.values()))})
    def save(number,name,source,draw):
        fig,axs=plt.subplots(1,2,figsize=(10,3.7))
        for ax,t in zip(axs,(25,45)):
            g=source[source.temperature_c.eq(t)]
            draw(ax,g,t);ax.set_title(f'{t}°C');ax.grid(alpha=.15)
        fig.tight_layout()
        save_figure(fig,paths['figures']/f'Fig{number:02d}_{name}',source)
    def concentration(ax,g,t):
        v=g.groupby('query_concentration_mol_L').Sigma_e_pred.agg(['mean','std'])
        ax.errorbar(v.index*1000,v['mean'],yerr=v['std'],marker='o',color='#527C8E')
        ax.set(xlabel='Query concentration (mol/m³)',ylabel='Predicted conductivity (S/m)')
    save(1,'sigma_by_query_concentration',predictions,concentration)
    def curves(ax,g,t):
        for candidate in [1,9,21,37,41]:
            v=g[g.candidate_id.eq(candidate)].groupby('query_concentration_mol_L').Sigma_e_pred.mean()
            ax.plot(v.index*1000,v,marker='o',label=f'Candidate {candidate}')
        ax.set(xlabel='Query concentration (mol/m³)',ylabel='Conductivity (S/m)');ax.legend(fontsize=7)
    save(2,'representative_concentration_curves',predictions,curves)
    def ranking(ax,g,t):
        g=g.sort_values('Infor_Sigma_e');ax.errorbar(np.arange(len(g)),g.Infor_Sigma_e,yerr=g.seed_std,fmt='.',color='#527C8E')
        ax.set(xlabel='Candidate rank by mean conductivity',ylabel='Infor. Sigma_e (S/m)')
    save(3,'candidate_infor_sigma_ranking',cards,ranking)
    def space(ax,g,t):
        pts=ax.scatter(g.Infor_Sigma_e,g.ce_0,c=g.RCS,cmap='viridis',s=27)
        ax.figure.colorbar(pts,ax=ax,label='RCS');ax.set(xlabel='Infor. Sigma_e (S/m)',ylabel='ce_0 (mol/L)')
    save(4,'sigma_ce0_candidate_space',full,space)
    def association(ax,g,t):
        ax.scatter(g.Infor_Sigma_e,g.RCS,alpha=.6,color='#527C8E');ax.set(xlabel='Infor. Sigma_e (S/m)',ylabel='RCS')
    save(5,'sigma_rcs_association',full,association)
    def uncertainty(ax,g,t):
        ax.bar(g.candidate_id,g.seed_std,color='#527C8E');ax.set(xlabel='Candidate ID',ylabel='Conductivity seed SD (S/m)')
    save(6,'property_seed_uncertainty',cards,uncertainty)
    curve=trace.groupby(['temperature_c','method','evaluation'],as_index=False).agg(
        mean=('normalized_best_so_far','mean'),std=('normalized_best_so_far','std'),n=('replay_seed','size'),hit=('hit','mean'))
    def bestcurve(ax,g,t):
        for m,v in g.groupby('method'):
            ax.plot(v.evaluation,v['mean'],label=m,color=COLORS[m]);half=1.96*v['std']/np.sqrt(v.n)
            ax.fill_between(v.evaluation,v['mean']-half,v['mean']+half,color=COLORS[m],alpha=.13)
        ax.set(xlabel='Evaluations (within-batch order)',ylabel='Normalized best-so-far');ax.legend(fontsize=7)
    save(7,'best_so_far_curves',curve,bestcurve)
    def probability(ax,g,t):
        for m,v in g.groupby('method'):
            ax.plot(v.budget,100*v.hit_rate,marker='o',label=m,color=COLORS[m])
            ax.fill_between(v.budget,100*v.ci_low,100*v.ci_high,color=COLORS[m],alpha=.10)
        ax.set(xlabel='Actual evaluations at batch end',ylabel='Optimum hit rate (%)');ax.legend(fontsize=7)
    save(8,'optimum_hit_probability',hits,probability)
    def box(metric,label):
        def draw(ax,g,t):
            ax.boxplot([g[g.method.eq(m)][metric] for m in METHODS],tick_labels=METHODS,showfliers=False)
            ax.set(ylabel=label)
        return draw
    save(9,'evaluations_to_optimum',summary,box('batch_cost_to_optimum','Batch cost to optimum'))
    save(10,'normalized_auc',summary,box('normalized_auc','Normalized best-so-far AUC'))
    def paired(ax,g,t):
        g=g[g.metric.eq('batch_cost_to_optimum')].reset_index(drop=True)
        ax.errorbar(g.a_minus_b,np.arange(len(g)),xerr=np.vstack([g.a_minus_b-g.ci_low,g.ci_high-g.a_minus_b]),fmt='o',color='#527C8E')
        ax.set_yticks(np.arange(len(g)),g.method_a+' − '+g.method_b);ax.axvline(0,color='gray',ls='--')
        ax.set(xlabel='Paired mean batch cost difference (95% CI)')
    save(11,'acquisition_paired_differences',pairs,paired)
    save(12,'fixed_budget_hit_rates',hits[hits.budget.isin([12,18,24])],probability)
    stability=[]
    for t in (25,45):
        for n in [30,60,100,150,200,300]:
            seed_order=summary.replay_seed.drop_duplicates().tolist()[:n]
            a=summary[summary.temperature_c.eq(t)&summary.replay_seed.isin(seed_order)].groupby('method').batch_cost_to_optimum.mean()
            for m,r in a.rank(method='min').items():stability.append({'temperature_c':t,'replays':len(seed_order),'method':m,'rank':r})
    stability=pd.DataFrame(stability).drop_duplicates()
    def ranks(ax,g,t):
        for m,v in g.groupby('method'):ax.plot(v.replays,v['rank'],marker='o',label=m,color=COLORS[m])
        ax.set(xlabel='Matched replays',ylabel='Rank by mean batch cost');ax.invert_yaxis();ax.legend(fontsize=7)
    save(13,'ranking_stability',stability,ranks)
    firstseed=int(summary.replay_seed.iloc[0])
    representative=trace[trace.replay_seed.eq(firstseed)]
    def rep(ax,g,t):
        for m,v in g.groupby('method'):ax.step(v.evaluation,v.normalized_best_so_far,where='post',label=m,color=COLORS[m])
        ax.set(xlabel=f'Evaluations; seed={firstseed}',ylabel='Normalized best-so-far');ax.legend(fontsize=7)
    save(14,'representative_replays',representative,rep)
    round_figures(paths,cards,candidates,trace,saved=saved_rounds)
    recommendation_figure(paths,trace,candidates)


def round_figures(paths,cards,candidates,trace,*,saved=False):
    """Explain UCB recommendations and observed winners in fixed Legacy seeds."""
    if saved:
        highlight={25:287,45:478}
        rec=pd.read_csv(paths['tables']/'ucb_round_recommendations.csv')
        win=pd.read_csv(paths['tables']/'ucb_round_best_formulations.csv')
    else:
        recommendations,winners=[],[]
        highlight={25:287,45:478}
        for temperature,seed in highlight.items():
            part=trace[trace.temperature_c.eq(temperature)&trace.method.eq('UCB')&trace.replay_seed.eq(seed)]
            if len(part)!=44: raise ValueError('Missing fixed highlighted replay')
            part=part.sort_values('evaluation')
            card=cards[cards.temperature_c.eq(temperature)].sort_values('candidate_id')
            meta=candidates[candidates.temperature_c.eq(temperature)].set_index('candidate_id')
            x=StandardScaler().fit_transform(card[['Infor_Sigma_e','ce_0']])
            y=meta.reindex(range(1,45)).RCS.to_numpy(float)
            for round_id,batch in part.groupby('batch_round',sort=True):
                observed=part[part.batch_round.lt(round_id)].candidate_id.to_numpy(int)-1
                if round_id>0:
                    gp=GaussianProcessRegressor(kernel=RBF(.6,(1e-3,1e3)),alpha=.25,normalize_y=True,
                                                random_state=seed,n_restarts_optimizer=0)
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore',ConvergenceWarning)
                        gp.fit(x[observed],y[observed])
                    remaining=np.array(sorted(set(range(44))-set(observed)))
                    mean,std=gp.predict(x[remaining],return_std=True)
                    score=mean+1.96*std
                    ids=remaining[np.lexsort((remaining,-score))[:6]]+1
                    np.testing.assert_array_equal(ids,batch.candidate_id.to_numpy(int))
                    for rank,cid in enumerate(ids,1):
                        pos=int(np.flatnonzero(remaining==cid-1)[0])
                        recommendations.append({'temperature_c':temperature,'replay_seed':seed,'batch_round':int(round_id),
                            'rank':rank,'candidate_id':int(cid),'formulation':meta.loc[cid,'formulation'],
                            'ce_0':float(meta.loc[cid,'ce_0']),'posterior_mean':float(mean[pos]),
                            'exploration_bonus':float(1.96*std[pos]),'UCB':float(score[pos]),
                            'observed_RCS_after_batch':float(y[cid-1])})
                all_seen=part[part.batch_round.le(round_id)]
                best=all_seen.loc[all_seen.selected_RCS.idxmax()]
                batch_best=batch.loc[batch.selected_RCS.idxmax()]
                cid=int(best.candidate_id)
                winners.append({'temperature_c':temperature,'replay_seed':seed,'batch_round':int(round_id),
                    'evaluations':len(all_seen),'best_candidate_id':cid,'formulation':meta.loc[cid,'formulation'],
                    'ce_0':float(meta.loc[cid,'ce_0']),'best_RCS':float(best.selected_RCS),
                    'batch_best_candidate_id':int(batch_best.candidate_id),'batch_best_RCS':float(batch_best.selected_RCS),
                    'global_optimum_RCS':float(y.max()),'found_global_optimum':bool(np.isclose(best.selected_RCS,y.max()))})
        rec=pd.DataFrame(recommendations);win=pd.DataFrame(winners)
        write_table(rec,paths['tables']/'ucb_round_recommendations')
        write_table(win,paths['tables']/'ucb_round_best_formulations')
    fig,axes=plt.subplots(7,2,figsize=(11,19))
    for col,temperature in enumerate((25,45)):
        for round_id in range(1,8):
            ax=axes[round_id-1,col]
            g=rec[rec.temperature_c.eq(temperature)&rec.batch_round.eq(round_id)].sort_values('rank')
            positions=np.arange(len(g))
            ax.barh(positions,g.posterior_mean,color='#39A99D',label='Posterior mean')
            ax.barh(positions,g.exploration_bonus,left=g.posterior_mean,color='#A494C5',label='1.96 × posterior SD')
            ax.set_yticks(positions,[f'C{cid} | {c:.1f} M' for cid,c in zip(g.candidate_id,g.ce_0)])
            ax.invert_yaxis();ax.grid(axis='x',alpha=.15)
            ax.set_title(f'{temperature}°C · seed {highlight[temperature]} · round {round_id}',fontsize=9)
            ax.set_xlabel('UCB score (RCS units)',fontsize=8)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False)
    fig.tight_layout(rect=(0,.025,1,1))
    save_figure(fig,paths['figures']/'Fig15_ucb_next_six_by_round',rec)
    fig,axes=plt.subplots(1,2,figsize=(11,4.4))
    for ax,temperature in zip(axes,(25,45)):
        g=win[win.temperature_c.eq(temperature)].sort_values('batch_round')
        ax.step(g.batch_round,g.best_RCS,where='post',color='#39A99D',label='Best observed so far')
        ax.scatter(g.batch_round,g.best_RCS,color='#39A99D',s=25)
        ax.plot(g.batch_round,g.batch_best_RCS,'o--',color='#A494C5',alpha=.7,label='Best in this batch')
        for _,row in g.iterrows():
            ax.annotate(f'C{int(row.best_candidate_id)}', (row.batch_round,row.best_RCS),
                        xytext=(0,8),textcoords='offset points',ha='center',fontsize=8)
            if row.batch_best_candidate_id!=row.best_candidate_id:
                ax.annotate(f'C{int(row.batch_best_candidate_id)}',(row.batch_round,row.batch_best_RCS),
                            xytext=(0,-14),textcoords='offset points',ha='center',fontsize=8,color='#A494C5')
        low=float(g.batch_best_RCS.min());high=float(g.global_optimum_RCS.max())
        pad=max((high-low)*.13,1)
        ax.set_ylim(low-pad,high+pad)
        ax.axhline(g.global_optimum_RCS.iloc[0],color='#8A8F93',ls=':',label='Global optimum')
        ax.set(xlabel='BO round (0 = initial six)',ylabel='Measured RCS',xticks=range(8))
        ax.set_title(f'{temperature}°C · seed {highlight[temperature]}');ax.grid(alpha=.15)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,fontsize=8)
    fig.tight_layout(rect=(0,.1,1,1))
    save_figure(fig,paths['figures']/'Fig16_ucb_best_formulation_by_round',win)


def recommendation_counts(trace,meta):
    """Count first-batch recommendations and candidate eligibility per method."""
    rows=[]
    for t in (25,45):
        part=trace[trace.temperature_c.eq(t)]
        assert part.replay_seed.nunique()==300
        assert set(part.method)==set(METHODS)
        assert not part.duplicated(['method','replay_seed','candidate_id']).any()
        first=part[part.batch_round.eq(1)]
        assert first.groupby(['method','replay_seed']).size().eq(6).all()
        assert first.groupby(['method','replay_seed']).ngroups==1200
        counts=first.groupby(['candidate_id','method']).size().unstack('method',fill_value=0)
        counts=counts.reindex(index=np.arange(1,45),columns=METHODS,fill_value=0).reset_index()
        counts['total_recommendations']=counts[list(METHODS)].sum(axis=1)
        highlighted=set()
        for method in METHODS:
            order=counts.sort_values([method,'candidate_id'],ascending=[False,True]).candidate_id.tolist()
            counts[method+'_frequency_rank']=counts.candidate_id.map({cid:i+1 for i,cid in enumerate(order)})
            highlighted.update(order[:6])
        initial=part[part.method.eq('UCB')&part.batch_round.eq(0)].groupby('candidate_id').size()
        counts['eligible_replays']=300-counts.candidate_id.map(initial).fillna(0).astype(int)
        for method in METHODS:
            assert counts[method].le(counts.eligible_replays).all()
            assert counts[method].sum()==1800
            counts[method+'_conditional_rate']=counts[method]/counts.eligible_replays
        counts['displayed']=counts.candidate_id.isin(highlighted)
        counts['temperature_c']=t;counts['replays_per_method']=300
        candidates=meta[meta.temperature_c.eq(t)].set_index('candidate_id')
        counts['formulation']=counts.candidate_id.map(candidates.formulation)
        counts['ce_0']=counts.candidate_id.map(candidates.ce_0)
        rows.append(counts)
    return pd.concat(rows,ignore_index=True)


def recommendation_figure(paths,trace,meta):
    """Fig17: four-method first-batch recommendation frequencies."""
    audit=recommendation_counts(trace,meta)
    shown=audit[audit.displayed].sort_values(['temperature_c','total_recommendations','candidate_id'],ascending=[True,False,True])
    write_table(audit,paths['tables']/'recommendation_frequency')
    write_table(shown,paths['tables']/'recommendation_frequency_displayed')
    write_json(paths['configs']/'recommendation_plot_config.json',{
        'role':'descriptive four-method recommendation comparison; no composite acquisition',
        'data':'saved conductivity_guided_search four-method trajectories','round':1,'replays_per_method':300,
        'display_rule':'union of each method top six candidates by first-batch selection count; ties candidate_id',
        'bar_segments':'number of matched replays selecting the candidate in the first post-initialization batch',
        'row_order':'total selection count descending; candidate_id ascending',
        'eligibility':'counts/rates excluding matched initial-six observations also saved in tables',
        'api_calls':0,'model_refits':0,'ranking_points':False})
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'axes.spines.top':True,'axes.spines.right':True})
    fig,axes=plt.subplots(1,2,figsize=(14,5.8))
    for ax,t,letter in zip(axes,(25,45),('a','b')):
        part=shown[shown.temperature_c.eq(t)];positions=np.arange(len(part));left=np.zeros(len(part))
        for method in METHODS:
            values=part[method].to_numpy()
            ax.barh(positions,values,left=left,color=COLORS[method],alpha=.85,edgecolor='white',height=.72,label=method)
            for i,(base,value) in enumerate(zip(left,values)):
                if value>=20:ax.text(base+value/2,i,str(int(value)),ha='center',va='center',fontsize=8)
            left+=values
        labels=[f'C{int(r.candidate_id)} | {r.formulation.replace("_",":")} | {r.ce_0:.1f} M' for _,r in part.iterrows()]
        ax.set_yticks(positions,labels,fontsize=8);ax.invert_yaxis();ax.grid(axis='x',alpha=.15)
        for i,value in enumerate(left):ax.text(value+5,i,str(int(value)),va='center',fontsize=8)
        ax.set_xlim(0,max(left)*1.12)
        ax.set_xlabel('First-batch recommendation count (300 replays per method)')
        ax.set_title(f'({letter}) {t}°C',loc='left')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.5,.035))
    fig.text(.5,.012,'Displayed candidates: union of each method’s six most frequently recommended formulations.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.12,1,1))
    save_figure(fig,paths['figures']/'Fig17_method_recommendation_frequency',shown)
    return shown


def rebuild_supplementary_figures(root,output=None,plots='all'):
    """Rebuild Fig15–17 from saved Step 3 results using this module alone."""
    paths=paths_for(root,output)
    trace=pd.read_csv(paths['tables']/'replay_trajectories.csv')
    candidates=pd.read_csv(paths['tables']/'candidate_manifest.csv')
    if plots in ('all','rounds'):
        cards=pd.read_csv(paths['tables']/'infor_sigma_summary.csv')
        round_figures(paths,cards,candidates,trace)
    if plots in ('all','recommendations'):
        recommendation_figure(paths,trace,candidates)
    shutil.copy2(Path(__file__),paths['snapshots']/'conductivity_guided_search.py')


def rebuild_figures(root, output=None):
    """Redraw all Step 3 figures from saved tables, without model calls."""
    paths = paths_for(root, output)
    def read(name):
        return pd.read_csv(paths['tables'] / (name + '.csv'))
    cards = read('infor_sigma_summary')
    candidates = read('candidate_manifest')
    figures(paths, read('sigma_predictions'), read('infor_sigma_by_seed'),
            cards, candidates, read('replay_summary'),
            read('replay_trajectories'), read('acquisition_ranking'),
            read('budget_hit_rates'), read('pairwise_comparisons'),
            cards.merge(candidates, on=['temperature_c', 'candidate_id', 'ce_0']),
            saved_rounds=True)


def run(root,output=None,mode='full',resume=False):
    if mode == 'plots':
        rebuild_figures(root, output)
        return
    paths=paths_for(root,output);config_path=paths['configs']/'run_config.json'
    config={'status':'running','step':3,'mode':mode,'created_utc':datetime.now(timezone.utc).isoformat(),
            'methods':list(METHODS),'property_seeds':list(SEEDS),'model_path':API_MODEL_PATH,'n_estimators':8,
            'training_rows':40,'candidate_rows_per_temperature':44,'n_replays':300 if mode=='full' else 3,
            'batch_size':6,'initial_points':6,'xi':.01,'kappa':1.96,'alpha':.25,
            'gpr':'RBF optimized on observed labels; bounds=(1e-3,1e3); normalize_y=True',
            'standardization':'all candidate X only, no RCS', 'thompson':'one joint posterior function sample per batch',
            'query_policy':'exact released rows; bracket interpolation only for missing 0.1 and 0.3 M',
            'software_versions':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','tabpfn-client','matplotlib']},
            'replay_seeds':random.Random(0).sample(range(1001),300 if mode=='full' else 3)}
    write_json(config_path,config)
    try:
        q,candidates=audit_inputs(paths)
        if mode=='audit':
            config['status']='audit_passed';return
        predictions,by_seed,cards=run_properties(paths,q,resume)
        summary,trace=run_bo(paths,cards,candidates,config['n_replays'],resume)
        aggregate,hits,pairs,overall,full=summarize(paths,summary,trace,cards,candidates)
        figures(paths,predictions,by_seed,cards,candidates,summary,trace,aggregate,hits,pairs,full)
        assert len(summary)==config['n_replays']*8 and len(trace)==len(summary)*44
        assert not trace.duplicated(['temperature_c','method','replay_seed','candidate_id']).any()
        assert summary.groupby(['temperature_c','replay_seed']).initial_candidates.nunique().eq(1).all()
        write_table(pd.DataFrame([{k:v for k,v in config.items() if isinstance(v,(int,float,str))}]),
                    paths['tables']/'acquisition_config')
        config['selected_method']=str(overall.iloc[0]['method']);config['status']='completed'
        config['validation']={'query_rows':len(q),'property_prediction_rows':len(predictions),
                              'replay_rows':len(summary),'trajectory_rows':len(trace),
                              'matched_initializations':True,'no_duplicate_candidates':True}
        print(aggregate.to_string(index=False),flush=True);print('Overall:',overall.to_string(index=False),flush=True)
    except Exception as exc:
        config['status']='interrupted';config['error_type']=type(exc).__name__;raise
    finally:
        config['finished_utc']=datetime.now(timezone.utc).isoformat();write_json(config_path,config)
        for p in [Path(__file__),project_directory()/'run_conductivity_guided_search.py',project_directory()/'analysis/ampere_conductivity.py']:
            shutil.copy2(p,paths['snapshots']/p.name)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Rebuild supplementary Step 3 figures from saved results')
    parser.add_argument('--plots',choices=['all','rounds','recommendations'],required=True)
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    rebuild_supplementary_figures(Path(__file__).resolve().parents[2],args.output_dir,args.plots)
