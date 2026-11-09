"""Step 4: local UCB representation ablation using existing property predictions."""
from __future__ import annotations

import importlib.metadata
import itertools
import json
import random
import shutil
import time
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .data_sources import source_records, project_directory, ampere_data_directory
from .ampere_conductivity import SEEDS,write_table,write_json,sha256,save_figure
from .conductivity_guided_search import replay,measure_replay
import matplotlib.pyplot as plt
from .ampere_conductivity import apply_style
apply_style()

REPRESENTATIONS=('Random','Salt only','Conductivity only','Proposed 2D','Composition')
SENSITIVE=('Conductivity only','Proposed 2D')
COLORS=dict(zip(REPRESENTATIONS,('#8A8F93','#C49A6C','#7E9B76','#527C8E','#B86F6F')))
METRICS=('batch_cost_to_optimum','evaluations_to_optimum','normalized_auc','normalized_auc_batch')


def paths_for(root,output=None):
    out=output or project_directory()/'results/representation_ablation'
    paths={'root':root,'results':out,'conductivity_guided_search':project_directory()/'results/conductivity_guided_search',
           'design':ampere_data_directory()/'GitHub/Data/Electrolyte_Design'}
    for name in ('tables','figures','figure_source_data','configs','logs','snapshots'):
        paths[name]=out/name;paths[name].mkdir(parents=True,exist_ok=True)
    return paths


def feature_sets(cards,composition):
    return {
        'Random':pd.DataFrame(index=cards.index),
        'Salt only':cards[['ce_0']],
        'Conductivity only':cards[['Infor_Sigma_e']],
        'Proposed 2D':cards[['Infor_Sigma_e','ce_0']],
        'Composition':composition.copy()}


def load_inputs(paths):
    base=paths['conductivity_guided_search']
    config=json.loads((base/'configs/run_config.json').read_text())
    selection=json.loads((base/'configs/selection.json').read_text())
    if config['status']!='completed' or selection['selected_method']!='UCB':
        raise ValueError('Step 3 must be complete and select UCB')
    assert config['kappa']==1.96 and config['batch_size']==6 and config['initial_points']==6
    cards=pd.read_csv(base/'tables/infor_sigma_summary.csv')
    by_seed=pd.read_csv(base/'tables/infor_sigma_by_seed.csv')
    candidates=pd.read_csv(base/'tables/candidate_manifest.csv')
    if len(cards)!=88 or len(by_seed)!=528 or by_seed.duplicated(['temperature_c','candidate_id','seed']).any():
        raise ValueError('Invalid Step 3 property cards')
    group=by_seed.groupby(['temperature_c','candidate_id']).Infor_Sigma_e
    np.testing.assert_allclose(cards.set_index(['temperature_c','candidate_id']).Infor_Sigma_e.sort_index(),group.mean().sort_index(),atol=1e-12)
    inventory=pd.read_csv(base/'tables/input_inventory.csv').set_index('path').sha256.to_dict()
    # Historical conductivity_guided_search inventories retain their original paths; alias only byte-identical retained inputs.
    for record in source_records(paths['root']):
        aliases = [record['original_path'], *record.get('previous_paths', []), record['path']]
        recorded = [inventory[key] for key in aliases if key in inventory]
        if recorded:
            if any(value != record['sha256'] for value in recorded):
                raise ValueError('Historical conductivity_guided_search input differs from retained source')
            inventory[record['path']] = record['sha256']
    sources=[base/'configs/selection.json',base/'configs/run_config.json',
             base/'tables/infor_sigma_summary.csv',base/'tables/infor_sigma_by_seed.csv',
             base/'tables/candidate_manifest.csv',base/'tables/replay_summary.csv',
             base/'tables/replay_trajectories.csv']
    inputs={};manifest=[];associations=[];values=[]
    for t in (25,45):
        c=cards[cards.temperature_c.eq(t)].sort_values('candidate_id').reset_index(drop=True)
        m=candidates[candidates.temperature_c.eq(t)].sort_values('candidate_id').reset_index(drop=True)
        np.testing.assert_array_equal(c.candidate_id,np.arange(1,45))
        np.testing.assert_array_equal(m.candidate_id,c.candidate_id)
        files=[paths['design']/f'Composition-based_Representations_{t}.xlsx',
               paths['design']/f'Label_RCS_{t}.xlsx']
        for file in files:
            if sha256(file)!=inventory[str(file.relative_to(paths['root']))]:
                raise ValueError(f'Step 3 source hash changed: {file.name}')
        comp=pd.read_excel(files[0]);y=pd.read_excel(files[1]).iloc[:,0].to_numpy(float)
        np.testing.assert_allclose(m.RCS,y,atol=1e-12,rtol=0)
        np.testing.assert_allclose(c.ce_0,comp.iloc[:,1],atol=1e-12,rtol=0)
        assert comp.shape==(44,14)
        frames=feature_sets(c,comp)
        inputs[t]={'cards':c,'composition':comp,'y':y,'frames':frames}
        sources.extend(files)
        for representation,frame in frames.items():
            if not np.isfinite(frame.to_numpy(float)).all():raise ValueError('Nonfinite features')
            unique=len(frame.drop_duplicates()) if frame.shape[1] else 0
            manifest.append({'temperature_c':t,'representation':representation,'dimensions':frame.shape[1],
                'varying_dimensions':int(frame.nunique().gt(1).sum()),'unique_feature_rows':unique,
                'feature_names':json.dumps(list(frame.columns),ensure_ascii=False),
                'scaling':'candidate-X StandardScaler, no RCS' if representation!='Random' else 'no model',
                'uses_RCS_for_features':False,'conductivity_source':'conductivity_guided_search six-seed mean' if representation in SENSITIVE else 'none'})
            for column in frame.columns:
                corr=float(spearmanr(frame[column],y).statistic) if frame[column].nunique()>1 else None
                associations.append({'temperature_c':t,'representation':representation,'feature':column,
                                     'spearman_rcs':corr,'constant_feature':frame[column].nunique()==1})
                values.extend({'temperature_c':t,'representation':representation,'candidate_id':i+1,
                               'feature':column,'value':float(v)} for i,v in enumerate(frame[column]))
    write_table(pd.DataFrame(manifest),paths['tables']/'feature_manifest')
    write_table(pd.DataFrame(values),paths['tables']/'feature_values')
    write_table(pd.DataFrame(associations),paths['tables']/'feature_rcs_associations')
    write_table(pd.DataFrame([{'path':str(p.relative_to(paths['root'])),'sha256':sha256(p)} for p in sources]),
                paths['tables']/'input_inventory')
    return inputs,by_seed,config['replay_seeds']


def run_replay(frame,y,seed,representation,temperature):
    if representation=='Random':
        initial=np.random.RandomState(seed).choice(len(y),6,replace=False).tolist()
        remaining=np.array(sorted(set(range(len(y)))-set(initial)))
        remaining=np.random.default_rng(seed*1000+917).permutation(remaining).tolist()
        order=initial+remaining
        rounds=[0]*6+[(i-6)//6+1 for i in range(6,len(y))]
        row,trace=measure_replay(order,rounds,y,seed,'Random',temperature)
    else:
        x=StandardScaler().fit_transform(frame.to_numpy(float))
        row,trace=replay(x,y,seed,'UCB',temperature)
    row.pop('method');trace=trace.drop(columns='method')
    row['representation']=representation;trace['representation']=representation
    return row,trace


def experiments(paths,inputs,by_seed,seeds):
    rows=[];traces=[];sensitivity=[]
    sensitivity_file=paths['tables']/'property_seed_trajectories.csv'
    if sensitivity_file.exists():raise FileExistsError('Use a fresh output for a full run; plots mode rebuilds without experiments')
    started=time.monotonic()
    with threadpool_limits(limits=1):
        # First fixed replay serves as a real-data validation before the full grid.
        for number,seed in enumerate(seeds,1):
            for t,inp in inputs.items():
                for representation in REPRESENTATIONS:
                    row,trace=run_replay(inp['frames'][representation],inp['y'],seed,representation,t)
                    rows.append(row);traces.append(trace)
            if number==1 or number%50==0:
                print(f'main replay {number}/{len(seeds)}; elapsed={time.monotonic()-started:.1f}s',flush=True)
        summary=pd.DataFrame(rows);trace=pd.concat(traces,ignore_index=True)
        write_table(summary,paths['tables']/'replay_summary')
        trace.to_csv(paths['tables']/'replay_trajectories.csv',index=False)
        for property_seed in SEEDS:
            seed_traces=[]
            for t,inp in inputs.items():
                card=by_seed[by_seed.temperature_c.eq(t)&by_seed.seed.eq(property_seed)].sort_values('candidate_id').reset_index(drop=True)
                np.testing.assert_array_equal(card.candidate_id,np.arange(1,45))
                frames=feature_sets(card,inp['composition'])
                for seed in seeds:
                    for representation in SENSITIVE:
                        row,tr=run_replay(frames[representation],inp['y'],seed,representation,t)
                        row['property_seed']=property_seed;tr['property_seed']=property_seed
                        sensitivity.append(row);seed_traces.append(tr)
            pd.concat(seed_traces,ignore_index=True).to_csv(sensitivity_file,index=False,
                mode='a' if sensitivity_file.exists() else 'w',header=not sensitivity_file.exists())
            pd.DataFrame(sensitivity).to_csv(paths['tables']/'property_seed_sensitivity.csv',index=False)
            print(f'property seed {property_seed} complete; elapsed={time.monotonic()-started:.1f}s',flush=True)
        sensitivity=pd.DataFrame(sensitivity)
        write_table(sensitivity,paths['tables']/'property_seed_sensitivity')
    return summary,trace,sensitivity


def reuse_local_results(paths,source):
    """Retain validated unchanged routes from local results, with clean new outputs."""
    source=Path(source).resolve()
    write_json(paths['configs']/'reuse_provenance.json',{
        'operation':'reuse existing unchanged local routes; recompute all derived tables and figures',
        'source':str(source),'raw_sha256':{name:sha256(source/'tables'/f'{name}.csv') for name in
            ['replay_summary','replay_trajectories','property_seed_sensitivity','property_seed_trajectories']}})
    frames={}
    for name in ['replay_summary','replay_trajectories','property_seed_sensitivity','property_seed_trajectories']:
        frame=pd.read_csv(source/'tables'/f'{name}.csv')
        keep=SENSITIVE if name.startswith('property_seed') else REPRESENTATIONS
        frame=frame[frame.representation.isin(keep)].copy()
        frame.to_csv(paths['tables']/f'{name}.csv',index=False)
        if name in ['replay_summary','property_seed_sensitivity']:
            write_table(frame,paths['tables']/f'{name}')
        frames[name]=frame
    return frames['replay_summary'],frames['replay_trajectories'],frames['property_seed_sensitivity']


def aggregate_results(paths,summary,trace,sensitivity):
    hits=[]
    for (t,representation),g in summary.groupby(['temperature_c','representation']):
        for budget in (6,12,18,24,30,36,42,44):
            p=float(g.batch_cost_to_optimum.le(budget).mean());n=len(g);z=1.96
            center=(p+z*z/(2*n))/(1+z*z/n);half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
            hits.append({'temperature_c':t,'representation':representation,'budget':budget,'hit_rate':p,
                         'n':n,'ci_low':center-half,'ci_high':center+half})
    hits=pd.DataFrame(hits);write_table(hits,paths['tables']/'budget_hit_rates')
    aggregate=summary.groupby(['temperature_c','representation'],as_index=False).agg(
        n=('replay_seed','size'),batch_cost_mean=('batch_cost_to_optimum','mean'),batch_cost_std=('batch_cost_to_optimum','std'),
        evaluations_mean=('evaluations_to_optimum','mean'),auc_mean=('normalized_auc','mean'),
        auc_std=('normalized_auc','std'),batch_auc_mean=('normalized_auc_batch','mean'))
    aggregate=aggregate.merge(hits[hits.budget.eq(18)][['temperature_c','representation','hit_rate']],on=['temperature_c','representation'])
    aggregate=aggregate.sort_values(['temperature_c','batch_cost_mean','hit_rate','auc_mean'],ascending=[True,True,False,False])
    aggregate['rank']=aggregate.groupby('temperature_c').cumcount()+1
    write_table(aggregate,paths['tables']/'overall_summary')
    overall=aggregate.groupby('representation',as_index=False).agg(mean_rank=('rank','mean'),
                 batch_cost_mean=('batch_cost_mean','mean'),hit18_mean=('hit_rate','mean'),auc_mean=('auc_mean','mean'))
    overall=overall.sort_values(['mean_rank','batch_cost_mean','hit18_mean','auc_mean'],ascending=[True,True,False,False])
    write_table(overall,paths['tables']/'cross_temperature_summary')
    pairs=[]
    for t in (25,45):
        for metric in METRICS:
            wide=summary[summary.temperature_c.eq(t)].pivot(index='replay_seed',columns='representation',values=metric)
            for a,b in itertools.combinations(REPRESENTATIONS,2):
                d=wide[a]-wide[b];half=1.96*d.std(ddof=1)/np.sqrt(len(d))
                pairs.append({'temperature_c':t,'metric':metric,'representation_a':a,'representation_b':b,
                              'a_minus_b':float(d.mean()),'ci_low':float(d.mean()-half),'ci_high':float(d.mean()+half),
                              'n':len(d),'a_better':int((d< -1e-12 if metric in METRICS[:2] else d>1e-12).sum()),
                              'ties':int(np.isclose(d,0,atol=1e-12,rtol=0).sum())})
    pairs=pd.DataFrame(pairs);write_table(pairs,paths['tables']/'pairwise_comparisons')
    seed_aggregate=sensitivity.groupby(['temperature_c','representation','property_seed'],as_index=False).agg(
        batch_cost_mean=('batch_cost_to_optimum','mean'),evaluations_mean=('evaluations_to_optimum','mean'),
        auc_mean=('normalized_auc','mean'),hit18=('batch_cost_to_optimum',lambda x:float(x.le(18).mean())))
    write_table(seed_aggregate,paths['tables']/'property_seed_summary')
    variance=[]
    for (t,rep),g in sensitivity.groupby(['temperature_c','representation']):
        for metric in METRICS:
            wide=g.pivot(index='replay_seed',columns='property_seed',values=metric)
            # These are descriptive crossed effects, not a formal independent variance decomposition.
            variance.append({'temperature_c':t,'representation':rep,'metric':metric,
                'property_seed_mean_variance':float(wide.mean(axis=0).var(ddof=1)),
                'replay_mean_variance':float(wide.mean(axis=1).var(ddof=1)),
                'mean_within_replay_property_variance':float(wide.var(axis=1,ddof=1).mean()),
                'mean_within_property_replay_variance':float(wide.var(axis=0,ddof=1).mean())})
    variance=pd.DataFrame(variance);write_table(variance,paths['tables']/'property_seed_vs_replay_variability')
    return aggregate,overall,hits,pairs,seed_aggregate,variance


def verify(paths,summary,trace,sensitivity,seeds):
    assert len(summary)==len(seeds)*2*len(REPRESENTATIONS) and len(trace)==len(summary)*44
    assert set(summary.representation)==set(REPRESENTATIONS)
    assert set(sensitivity.representation)==set(SENSITIVE)
    assert not summary.duplicated(['temperature_c','representation','replay_seed']).any()
    assert not trace.duplicated(['temperature_c','representation','replay_seed','candidate_id']).any()
    assert trace.groupby(['temperature_c','representation','replay_seed']).candidate_id.nunique().eq(44).all()
    assert summary.groupby(['temperature_c','replay_seed']).initial_candidates.nunique().eq(1).all()
    assert len(sensitivity)==len(seeds)*2*len(SEEDS)*len(SENSITIVE)
    assert not sensitivity.duplicated(['temperature_c','representation','replay_seed','property_seed']).any()
    assert sensitivity.groupby(['temperature_c','replay_seed']).initial_candidates.nunique().eq(1).all()
    sensitivity_trace=pd.read_csv(paths['tables']/'property_seed_trajectories.csv')
    sensitivity_keys=['temperature_c','representation','replay_seed','property_seed']
    assert len(sensitivity_trace)==len(sensitivity)*44
    assert not sensitivity_trace.duplicated(sensitivity_keys+['candidate_id']).any()
    assert sensitivity_trace.groupby(sensitivity_keys).candidate_id.nunique().eq(44).all()
    assert set(sensitivity.property_seed)==set(SEEDS)
    assert set(summary.replay_seed)==set(seeds)
    old=pd.read_csv(paths['conductivity_guided_search']/'tables/replay_summary.csv')
    old=old[old.method.eq('UCB')&old.replay_seed.isin(seeds)]
    new=summary[summary.representation.eq('Proposed 2D')]
    keys=['temperature_c','replay_seed']
    np.testing.assert_allclose(old.set_index(keys)[list(METRICS)].sort_index(),new.set_index(keys)[list(METRICS)].sort_index(),atol=1e-12,rtol=0)
    oldtrace=pd.read_csv(paths['conductivity_guided_search']/'tables/replay_trajectories.csv')
    oldtrace=oldtrace[oldtrace.method.eq('UCB')&oldtrace.replay_seed.isin(seeds)]
    newtrace=trace[trace.representation.eq('Proposed 2D')]
    k=keys+['evaluation']
    np.testing.assert_array_equal(oldtrace.set_index(k).candidate_id.sort_index(),newtrace.set_index(k).candidate_id.sort_index())
    write_json(paths['configs']/'validation.json',{'status':'passed','api_calls':0,
        'main_replays':len(summary),'main_trajectory_rows':len(trace),'property_seed_replays':len(sensitivity),
        'property_seed_trajectory_rows':len(sensitivity_trace),'matched_initializations':True,
        'unique_keys':True,'no_repeated_candidates':True,'proposed_2d_matches_step03_ucb_exactly':True})


def figures(paths,inputs,summary,trace,sensitivity,aggregate,hits,pairs,seed_aggregate,variance):
    plt.rcParams.update({'font.size':8,'axes.spines.top':True,'axes.spines.right':True,'pdf.fonttype':42})
    def save(number,name,source,draw,height=4):
        fig,axes=plt.subplots(1,2,figsize=(12,height))
        for ax,t in zip(axes,(25,45)):
            draw(ax,source[source.temperature_c.eq(t)],t);ax.set_title(f'{t}°C');ax.grid(alpha=.15)
        fig.tight_layout();save_figure(fig,paths['figures']/f'Fig{number:02d}_{name}',source)
    association=pd.read_csv(paths['tables']/'feature_rcs_associations.csv')
    # Deduplicate shared descriptors across representations for a readable heatmap.
    association=association.drop_duplicates(['temperature_c','feature'])
    def heatmap(ax,g,t):
        g=g.sort_values('feature');v=g.spearman_rcs.to_numpy(float)
        image=ax.imshow(v[:,None],cmap='RdBu_r',vmin=-1,vmax=1,aspect='auto')
        ax.set_yticks(np.arange(len(g)),g.feature,fontsize=7);ax.set_xticks([0],['RCS'])
        for i,value in enumerate(v):
            ax.text(0,i,'constant' if not np.isfinite(value) else f'{value:.2f}',ha='center',va='center',fontsize=7)
        ax.figure.colorbar(image,ax=ax,label='Spearman correlation')
    save(1,'feature_rcs_heatmap',association,heatmap,8)
    curve=trace.groupby(['temperature_c','representation','evaluation'],as_index=False).agg(
        mean=('normalized_best_so_far','mean'),std=('normalized_best_so_far','std'),n=('replay_seed','size'))
    def best(ax,g,t):
        for rep,v in g.groupby('representation'):
            ax.plot(v.evaluation,v['mean'],label=rep,color=COLORS[rep]);half=1.96*v['std']/np.sqrt(v.n)
            ax.fill_between(v.evaluation,v['mean']-half,v['mean']+half,color=COLORS[rep],alpha=.08)
        ax.set(xlabel='Evaluations (within-batch order)',ylabel='Normalized best-so-far');ax.legend(fontsize=6)
    save(2,'best_so_far_curves',curve,best)
    def hit(ax,g,t):
        for rep,v in g.groupby('representation'):
            ax.plot(v.budget,100*v.hit_rate,marker='.',label=rep,color=COLORS[rep])
        ax.set(xlabel='Evaluations at batch end',ylabel='Optimum hit rate (%)');ax.legend(fontsize=6)
    save(3,'optimum_hit_probability',hits,hit)
    def box(metric,label):
        def draw(ax,g,t):
            ax.boxplot([g[g.representation.eq(r)][metric] for r in REPRESENTATIONS],vert=False,
                       tick_labels=REPRESENTATIONS,showfliers=False)
            ax.set(xlabel=label)
        return draw
    save(4,'evaluations_to_optimum',summary,box('batch_cost_to_optimum','Actual batch-end cost to optimum'))
    save(5,'normalized_auc',summary,box('normalized_auc','Normalized best-so-far AUC'))
    contrasts=[('Random','Proposed 2D'),('Salt only','Proposed 2D'),('Conductivity only','Proposed 2D'),
               ('Proposed 2D','Composition')]
    forest=pairs[pairs.metric.eq('batch_cost_to_optimum')&pairs.apply(lambda r:(r.representation_a,r.representation_b) in contrasts,axis=1)]
    def difference(ax,g,t):
        g=g.reset_index(drop=True)
        ax.errorbar(g.a_minus_b,np.arange(len(g)),xerr=np.vstack([g.a_minus_b-g.ci_low,g.ci_high-g.a_minus_b]),fmt='o',color='#527C8E')
        ax.set_yticks(np.arange(len(g)),g.representation_a+' − '+g.representation_b,fontsize=7)
        ax.axvline(0,color='#8A8F93',ls='--');ax.set(xlabel='Paired mean batch-cost difference (95% CI)')
    save(6,'paired_difference_forest',forest,difference)
    dimension=pd.read_csv(paths['tables']/'feature_manifest.csv')[['temperature_c','representation','dimensions']]
    performance=aggregate.merge(dimension,on=['temperature_c','representation'])
    def dimensions(ax,g,t):
        for _,row in g.iterrows():
            ax.scatter(row.dimensions,row.batch_cost_mean,color=COLORS[row.representation],s=35,label=row.representation)
        ax.set(xlabel='Declared representation dimensions',ylabel='Mean batch cost to optimum');ax.margins(x=.15,y=.25)
        ax.legend(loc='upper right',fontsize=7,frameon=False)
    save(7,'performance_vs_dimension',performance,dimensions)
    spaces=[]
    for t,inp in inputs.items():
        f=inp['frames']['Proposed 2D'].copy();f['candidate_id']=np.arange(1,45);f['temperature_c']=t;f['RCS']=inp['y'];spaces.append(f)
    spaces=pd.concat(spaces,ignore_index=True)
    def proposed(ax,g,t):
        scatter=ax.scatter(g.Infor_Sigma_e,g.ce_0,c=g.RCS,cmap='viridis',s=24)
        ax.figure.colorbar(scatter,ax=ax,label='RCS');ax.set(xlabel='Our Infor. Sigma_e (S/m)',ylabel='ce_0 (mol/L)')
    save(8,'proposed_2d_space',spaces,proposed)
    marginal=spaces.groupby(['temperature_c','ce_0'],as_index=False).agg(
        mean_RCS=('RCS','mean'),min_RCS=('RCS','min'),max_RCS=('RCS','max'),count=('RCS','size'))
    def salt_effect(ax,g,t):
        ax.errorbar(g.ce_0,g.mean_RCS,yerr=np.vstack([g.mean_RCS-g.min_RCS,g.max_RCS-g.mean_RCS]),
                    marker='o',color='#C49A6C',capsize=3)
        ax.set(xlabel='ce_0 (mol/L)',ylabel='Mean RCS (range across formulations)')
    save(9,'salt_concentration_rcs',marginal,salt_effect)
    stability=[]
    order=summary.replay_seed.drop_duplicates().tolist()
    for t in (25,45):
        for n in [30,60,100,150,200,300]:
            use=summary[summary.temperature_c.eq(t)&summary.replay_seed.isin(order[:n])]
            ranks=use.groupby('representation').batch_cost_to_optimum.mean().rank(method='min')
            stability.extend({'temperature_c':t,'representation':r,'replays':len(order[:n]),'rank':v} for r,v in ranks.items())
    stability=pd.DataFrame(stability).drop_duplicates()
    def rank(ax,g,t):
        for rep,v in g.groupby('representation'):ax.plot(v.replays,v['rank'],marker='.',label=rep,color=COLORS[rep])
        ax.set(xlabel='Matched replays',ylabel='Rank by mean batch cost');ax.invert_yaxis();ax.legend(fontsize=6)
    save(10,'ranking_stability',stability,rank)
    v=variance[variance.metric.eq('batch_cost_to_optimum')]
    def variability(ax,g,t):
        g=g.set_index('representation').reindex(SENSITIVE)
        idx=np.arange(len(g));width=.25
        for j,(col,label,color) in enumerate([
            ('property_seed_mean_variance','Variance of property-seed means','#7FA6B7'),
            ('replay_mean_variance','Variance of replay means','#C49A6C'),
            ('mean_within_replay_property_variance','Mean property variance within replay','#9583A6')]):
            ax.bar(idx+(j-1)*width,g[col],width,label=label,color=color)
        ax.set_xticks(idx,SENSITIVE);ax.set(ylabel='Batch-cost variance (evaluations²)');ax.legend(fontsize=6)
    save(11,'property_seed_vs_replay_variability',v,variability)


def rebuild_figures(root, output=None):
    """Redraw Step 4 from saved results without experiments or config changes."""
    paths = paths_for(root, output)
    def read(name):
        return pd.read_csv(paths['tables'] / (name + '.csv'))
    cards = pd.read_csv(paths['conductivity_guided_search'] / 'tables/infor_sigma_summary.csv')
    inputs = {}
    for t in (25, 45):
        c = cards[cards.temperature_c.eq(t)].sort_values('candidate_id').reset_index(drop=True)
        composition = pd.read_excel(paths['design'] / f'Composition-based_Representations_{t}.xlsx')
        y = pd.read_excel(paths['design'] / f'Label_RCS_{t}.xlsx').iloc[:, 0].to_numpy(float)
        inputs[t] = {'cards': c, 'composition': composition, 'y': y,
                     'frames': feature_sets(c, composition)}
    figures(paths, inputs, read('replay_summary'), read('replay_trajectories'),
            read('property_seed_sensitivity'), read('overall_summary'),
            read('budget_hit_rates'), read('pairwise_comparisons'),
            read('property_seed_summary'), read('property_seed_vs_replay_variability'))


def run(root,output=None,mode='full',reuse_source=None):
    if mode == 'plots':
        rebuild_figures(root, output)
        return
    paths=paths_for(root,output);start=time.monotonic()
    config={'status':'running','step':4,'mode':mode,'api_calls':0,'created_utc':datetime.now(timezone.utc).isoformat(),
        'representations':list(REPRESENTATIONS),'property_seeds':list(SEEDS),'n_replays':300,
        'initial_points':6,'batch_size':6,'kappa':1.96,'alpha':.25,
        'gpr':'same conductivity_guided_search RBF(.6,bounds=(1e-3,1e3)); normalize_y=True; observed-label optimization only',
        'input_standardization':'full candidate-X only, no labels','tie_break':'ascending candidate_id',
        'random_baseline':'matched initial 6; independently seeded uniform permutation of remaining candidates',
        'primary_metric':'mean actual batch-end cost to first global optimum; hit@18 and AUC tie breaks',
        'variance_note':'crossed descriptive variances; no independent causal variance decomposition',
        'software_versions':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','matplotlib']}}
    write_json(paths['configs']/'run_config.json',config)
    try:
        inputs,by_seed,seeds=load_inputs(paths);config['replay_seeds']=seeds
        if mode=='pilot':seeds=seeds[:3];config['n_replays']=3
        if reuse_source is not None:
            summary,trace,sensitivity=reuse_local_results(paths,reuse_source)
            config['local_results_reused']=True
        else:
            summary,trace,sensitivity=experiments(paths,inputs,by_seed,seeds)
        verify(paths,summary,trace,sensitivity,seeds)
        aggregate,overall,hits,pairs,seed_aggregate,variance=aggregate_results(paths,summary,trace,sensitivity)
        figures(paths,inputs,summary,trace,sensitivity,aggregate,hits,pairs,seed_aggregate,variance)
        config['status']='completed';config['main_replays']=len(summary);config['sensitivity_replays']=len(sensitivity)
        config['winner']=str(overall.iloc[0].representation)
        print(aggregate.to_string(index=False),flush=True);print('CROSS TEMPERATURE\n'+overall.to_string(index=False),flush=True)
    except Exception as exc:
        config['status']='interrupted';config['error_type']=type(exc).__name__;raise
    finally:
        config['finished_utc']=datetime.now(timezone.utc).isoformat();config['elapsed_seconds']=time.monotonic()-start
        write_json(paths['configs']/'run_config.json',config)
        for file in [Path(__file__),project_directory()/'run_representation_ablation.py',project_directory()/'analysis/conductivity_guided_search.py',project_directory()/'analysis/ampere_conductivity.py',project_directory()/'tests/test_representation_ablation.py']:
            shutil.copy2(file,paths['snapshots']/file.name)
