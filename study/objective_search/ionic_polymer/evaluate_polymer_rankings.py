"""Offline figures and result tables. Contains no TabPFN/API calls."""
from __future__ import annotations
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

METHOD_COLORS = {'TabPFN + UCB': '#456990', 'GPR + UCB': '#EF767A', 'Random (exact)': '#48A999',
                 'Paper trajectory': '#8C78A8'}


def save_figure(fig, name, source, out):
    fig.savefig(out / 'figures' / (name + '.png'), dpi=220, bbox_inches='tight', facecolor='white')
    fig.savefig(out / 'figures' / (name + '.pdf'), bbox_inches='tight', facecolor='white')
    source.to_csv(out / 'figure_source_data' / (name + '.csv'), index=False)
    plt.close(fig)


def select_batch(prediction, eligible_ids, k=4):
    """The selector sees prediction values and candidate identities, no outcomes."""
    p = prediction[prediction.candidate_id.isin(eligible_ids)].copy()
    return p.sort_values(['ucb', 'candidate_id'], ascending=[False, True]).head(k).candidate_id.tolist()


def evaluate_first_batch(d, first_batch, pool, method, replicate):
    """Evaluator-only access to measured outcomes and the fixed reference scale."""
    eligible = d.iloc[:pool]
    initial = d.iloc[:7].candidate_id.tolist()
    winner = eligible.loc[eligible.evaluation_score.idxmax(), 'candidate_id']
    if len(first_batch) != 4 or len(set(first_batch)) != 4 or set(first_batch) & set(initial):
        raise ValueError('Invalid first-batch selection')
    if not set(first_batch) <= set(eligible.iloc[7:].candidate_id):
        raise ValueError('Selected a candidate without a label in this restricted replay')
    history = [initial, initial + first_batch, eligible.candidate_id.tolist()]
    budget = [7, 11, pool]
    cost = 7 if winner in initial else 11 if winner in first_batch else pool
    rows = []
    for selected, n in zip(history, budget):
        x = eligible[eligible.candidate_id.isin(selected)]
        best = float(x.evaluation_score.max())
        feasible = x[x.high_voltage_pass.eq(1)]
        rows.append(dict(pool=pool, method=method, replicate=replicate, budget=n,
                         best_score=best, regret=float(eligible.evaluation_score.max() - best),
                         winner_seen=int(winner in selected), discovery_cost=cost,
                         best_conductivity=float(x.conductivity_mS_cm.max()),
                         best_feasible_capacity=float(feasible.capacity_mAh_g.max()),
                         first_batch=' '.join(first_batch), stage='initial' if n == 7 else 'decision_batch' if n == 11 else 'forced_exhaustion'))
    selected = eligible.set_index('candidate_id').loc[first_batch]
    metric = dict(pool=pool, method=method, replicate=replicate, first_batch=' '.join(first_batch),
                  budget11_best_score=rows[1]['best_score'], budget11_regret=rows[1]['regret'],
                  budget11_hit=rows[1]['winner_seen'], discovery_cost=cost,
                  new_batch_mean_score=float(selected.evaluation_score.mean()),
                  new_batch_feasible_fraction=float(selected.high_voltage_pass.mean()),
                  new_batch_best_capacity=float(selected.capacity_mAh_g.max()))
    return rows, metric


def build_replay_tables(d, tab_predictions, gp):
    curves, metrics, chosen = [], [], []
    for pool in [14, 15]:
        eligible = d.iloc[7:pool].candidate_id.tolist()
        arms = [('TabPFN + UCB', str(seed), p) for seed, p in tab_predictions.items()]
        arms.append(('GPR + UCB', 'deterministic', gp))
        for method, replicate, pred in arms:
            ids = select_batch(pred, eligible)
            trace, metric = evaluate_first_batch(d, ids, pool, method, replicate)
            curves.extend(trace); metrics.append(metric)
            for rank, candidate in enumerate(ids, 1):
                chosen.append(dict(pool=pool, method=method, replicate=replicate, rank=rank, candidate_id=candidate))
        # Uniform sampling without replacement: enumerate every batch, no Monte Carlo noise.
        for i, batch in enumerate(itertools.combinations(eligible, 4)):
            trace, metric = evaluate_first_batch(d, list(batch), pool, 'Random (exact)', str(i))
            curves.extend(trace); metrics.append(metric)
        if pool == 14:
            trace, metric = evaluate_first_batch(d, d.iloc[7:11].candidate_id.tolist(), pool, 'Paper trajectory', 'historical')
            curves.extend(trace); metrics.append(metric)
    curves, metrics = pd.DataFrame(curves), pd.DataFrame(metrics)
    summaries = []
    for (pool, method), x in metrics.groupby(['pool', 'method'], sort=False):
        random_cost = metrics[(metrics.pool == pool) & metrics.method.eq('Random (exact)')].discovery_cost.mean()
        summaries.append(dict(pool=pool, method=method, n_variants=len(x),
                              variants_meaning='inference seeds, same initial 7' if method.startswith('TabPFN') else 'all equally likely batches' if method.startswith('Random') else 'one trajectory',
                              budget11_best_score_mean=x.budget11_best_score.mean(),
                              budget11_best_score_sd=x.budget11_best_score.std(ddof=1) if len(x) > 1 else 0.,
                              budget11_hit_rate=x.budget11_hit.mean(), discovery_cost_mean=x.discovery_cost.mean(),
                              discovery_cost_sd=x.discovery_cost.std(ddof=1) if len(x) > 1 else 0.,
                              random_cost_over_method_cost=random_cost/x.discovery_cost.mean(),
                              new_batch_mean_score=x.new_batch_mean_score.mean(),
                              new_batch_feasible_fraction=x.new_batch_feasible_fraction.mean()))
    return curves, metrics, pd.DataFrame(summaries), pd.DataFrame(chosen)


def generate_report(d, audit, out):
    # Import only our experiment implementation; never reads/downloads author code.
    from prepare_polymer_search import FEATURES, OUTCOMES, SEEDS, KAPPA, sha, write_json, fingerprint, initial_prediction_contract, gp_predict, full_candidate_grid, zscore_objective, validate_prediction
    from analysis.ampere_conductivity import apply_style
    apply_style()
    tabs, cache_metadata = {}, {}
    for seed in SEEDS:
        stem = out / 'cache' / f'full_grid_initial7_seed{seed}'
        if not stem.with_suffix('.json').exists():
            continue
        md = json.loads(stem.with_suffix('.json').read_text())
        contract = initial_prediction_contract(d, seed, 'full_grid')
        if md['signature'] != fingerprint(contract) or md['csv_sha256'] != sha(stem.with_suffix('.csv')) or md['raw_sha256'] != sha(stem.with_suffix('.npz')):
            raise ValueError('Refusing report from changed or corrupt prediction cache')
        p = pd.read_csv(stem.with_suffix('.csv'), dtype={'candidate_id': str})
        if p.candidate_id.tolist() != contract['query_ids']:
            raise ValueError('Prediction IDs changed')
        validate_prediction({k: p[k].to_numpy() for k in ['mean', 'std', 'ucb', 'q025', 'q05', 'q95', 'q975']}, len(p))
        tabs[seed] = p
        cache_metadata[seed] = md
    if not tabs:
        raise ValueError('No full-grid predictive-distribution caches are available')
    grid = full_candidate_grid()
    grid = grid[~grid.candidate_id.isin(d.iloc[:7].candidate_id)].copy()
    y, observed_mean, observed_std = zscore_objective(d.iloc[:7][OUTCOMES])
    gp_result = gp_predict(d.iloc[:7][FEATURES], y, grid[FEATURES])
    gp = pd.DataFrame({k: v for k, v in gp_result.items() if isinstance(v, np.ndarray)})
    gp.insert(0, 'candidate_id', grid.candidate_id.to_numpy())
    rankings = []
    for method, replicate, pred in [('TabPFN + UCB', str(seed), p) for seed, p in tabs.items()] + [('GPR + UCB', 'deterministic', gp)]:
        x = pred.merge(grid, on='candidate_id', validate='one_to_one')
        x = x.sort_values(['ucb', 'candidate_id'], ascending=[False, True]).reset_index(drop=True)
        x['rank'] = np.arange(1, len(x)+1)
        x['method'], x['replicate'] = method, replicate
        x['measured_after_initial7'] = x.candidate_id.isin(d.iloc[7:].candidate_id)
        x['paper_next_batch'] = x.candidate_id.isin(d.iloc[7:11].candidate_id)
        rankings.append(x)
    ranks = pd.concat(rankings, ignore_index=True)
    top = ranks[ranks['rank'] <= 4].copy()
    top = top.merge(d[['candidate_id'] + OUTCOMES + ['evaluation_score']], on='candidate_id', how='left')
    top['feedback_status'] = np.where(top.measured_after_initial7, 'existing measured result; retrospective only', 'UNMEASURED: cannot continue real replay')
    top['polymer_percent'] = top.P1.map({1:5,2:10,3:15})
    top['ionic_liquid'] = top.P2.map({1:'C2mimBF4',2:'C2mimTFO',3:'C2mimFSI',4:'C2mimTFSI'})
    top['salt_pair'] = top.P3.map({1:'FSI:TFSI',2:'FSI:BF4',3:'FSI:TFO',4:'TFSI:BF4',5:'TFSI:TFO',6:'BF4:TFO'})
    top['salt_ratio'] = top.P4.map(dict(enumerate([0,.2,.4,.5,.6,.8,1],1)))
    ranks.to_csv(out / 'tables/full_grid_predictions_and_ranks.csv', index=False)
    top.to_csv(out / 'tables/full_grid_top4.csv', index=False)
    curves, metrics, summary, selected = build_replay_tables(d, tabs, gp)
    for name, df in [('restricted_replay_trace', curves), ('restricted_replay_metrics', metrics), ('restricted_replay_summary', summary), ('restricted_replay_selected', selected)]:
        df.to_csv(out / 'tables' / (name+'.csv'), index=False)
    # Prediction diagnostics use the unrevealed outcomes in the evaluator only.
    diagnostic = []
    truth = d.iloc[7:].copy()
    truth['target_observed_scale'] = ((truth[OUTCOMES] - observed_mean) / observed_std).sum(axis=1)
    for method, replicate, pred in [('TabPFN + UCB', str(seed), p) for seed, p in tabs.items()] + [('GPR + UCB', 'deterministic', gp)]:
        x = truth.merge(pred, on='candidate_id', validate='one_to_one')
        x['method'], x['replicate'] = method, replicate
        diagnostic.append(x)
    diagnostic = pd.concat(diagnostic, ignore_index=True)
    diagnostic.to_csv(out / 'tables/prediction_diagnostics.csv', index=False)
    diag_rows = []
    for (method, replicate), x in diagnostic.groupby(['method', 'replicate']):
        err = x['mean'] - x.target_observed_scale
        diag_rows.append(dict(method=method, replicate=replicate, n=8,
                              mae=float(np.abs(err).mean()), rmse=float(np.sqrt((err**2).mean())),
                              interval90_coverage=float(((x.target_observed_scale >= x.q05) & (x.target_observed_scale <= x.q95)).mean()),
                              spearman=float(spearmanr(x['mean'], x.target_observed_scale).statistic)))
    pd.DataFrame(diag_rows).to_csv(out / 'tables/prediction_diagnostic_summary.csv', index=False)
    draw_figures(d, curves, metrics, summary, ranks, top, selected, diagnostic, out)
    full_summaries = []
    for (method, replicate), x in top.groupby(['method', 'replicate']):
        full_summaries.append(dict(method=method, replicate=replicate, top4=' '.join(x.candidate_id),
                                   known_feedback=int(x.measured_after_initial7.sum()), unmeasured_feedback=int((~x.measured_after_initial7).sum()),
                                   overlap_paper_first_new_batch=int(x.paper_next_batch.sum()),
                                   selected_paper_winner=int(x.candidate_id.eq('1415').any())))
    full_summary = pd.DataFrame(full_summaries)
    full_summary.to_csv(out / 'tables/full_grid_recommendation_summary.csv', index=False)
    checks = dict(passed=True, completed_inference_seeds=sorted(tabs), all_six_seeds_completed=sorted(tabs)==SEEDS,
                  full_space=504, initial_known=7, query_rows=497,
                  gpr_export_checks=audit['gpr_reference_reproduction'],
                  no_repeated_top4=bool(top.groupby(['method', 'replicate']).candidate_id.nunique().eq(4).all()),
                  no_initial_candidates_selected=not bool(top.candidate_id.isin(d.iloc[:7].candidate_id).any()),
                  full_grid_stopped_before_unknown_feedback=True,
                  random_exact_batches={str(n): int(sum((metrics.pool==n)&metrics.method.eq('Random (exact)'))) for n in [14, 15]},
                  restricted_terminal_equal=bool(curves[curves.stage.eq('forced_exhaustion')].groupby('pool').best_score.nunique().eq(1).all()),
                  evaluation_scale_not_in_predictor_contract=all('evaluation_score' not in json.dumps(md['contract']) for md in cache_metadata.values()),
                  note='Six seeds measure inference variation only. No independent BO campaigns or calibrated-uncertainty claim.')
    if not checks['no_repeated_top4'] or not checks['no_initial_candidates_selected'] or not checks['restricted_terminal_equal']:
        raise ValueError('Result invariant failed')
    checks['audit_scope'] = 'computational invariants, not evidence of calibrated uncertainty or optimization superiority'
    write_json(out / 'tables/result_audit.json', checks)
    for path in [Path(__file__), Path(__file__).with_name('prepare_polymer_search.py')]:
        snapshot = out / 'snapshots' / f'{path.stem}_{sha(path)[:12]}.py'
        if not snapshot.exists(): snapshot.write_bytes(path.read_bytes())
    write_json(out / 'tables/report_manifest.json', {'report_source_sha256': sha(__file__), 'prediction_sources': {str(k): v['source_sha256'] for k,v in cache_metadata.items()}, 'figures': {p.name: sha(p) for p in (out/'figures').glob('*')}, 'source_tables': {p.name: sha(p) for p in (out/'figure_source_data').glob('*.csv')}})
    write_readme(d, audit, summary, full_summary, cache_metadata, out)
    write_full_ranking_analysis(d, ranks, out)
    from compare_polymer_acquisitions import run as run_acquisition
    run_acquisition()


def draw_figures(d, curves, metrics, summary, ranks, top, selected, diagnostic, out):
    # 1. Actual measured outcomes and retrospective score, including the extra row.
    cols = ['conductivity_mS_cm', 'high_voltage_pass', 'capacity_mAh_g', 'evaluation_score']
    labels = ['Conductivity [mS/cm]', 'High-voltage pass [0/1]', 'Capacity [mAh/g]', 'Paper-14 reference score']
    fig, axs = plt.subplots(4, 1, figsize=(10.5, 9), sharex=True)
    colors = ['#A6BED0']*7 + ['#48A999']*4 + ['#EF767A']*3 + ['#C3C3C3']
    for ax, col, label in zip(axs, cols, labels):
        ax.bar(np.arange(15), d[col], color=colors)
        ax.set_ylabel(label)
        for pos in [6.5, 10.5, 13.5]: ax.axvline(pos, color='gray', ls='--', lw=.8)
    axs[0].set_title('ionic_polymer measured data: initial 7 | next 4 | paper final 3 | extra public row')
    axs[-1].set_xticks(range(15), d.candidate_id, rotation=45)
    axs[-1].set_xlabel('Encoded formulation (1415 = paper recommendation; 1321 = extra zero row)')
    fig.tight_layout(); save_figure(fig, 'Fig01_measured_outcomes', d, out)
    # 2. Restricted-pool learning curves; forced exhaustion explicitly marked.
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for ax, pool in zip(axs, [14, 15]):
        for method, color in METHOD_COLORS.items():
            x = curves[(curves.pool==pool)&curves.method.eq(method)]
            if x.empty: continue
            stat=x.groupby('budget').best_score.agg(['mean','min','max']).reset_index()
            ax.plot(stat.budget, stat['mean'], '-o', label=method, color=color, ms=4)
            if len(x.replicate.unique())>1:ax.fill_between(stat.budget,stat['min'],stat['max'],alpha=.12,color=color)
        ax.axvspan(11.05, pool+.05, color='#EEEEEE', alpha=.6)
        ax.text(.98,.04,'Final reveal is forced; no new model call',transform=ax.transAxes,ha='right',fontsize=8)
        ax.set(title=f'Retrospective measured pool: {pool}',xlabel='Cumulative experiments (initial 7 included)',ylabel='Best paper-reference score')
        ax.set_xticks([7,11,pool]);ax.legend(fontsize=8)
    fig.suptitle('Restricted-pool sensitivity only — not a 504-candidate closed loop',fontsize=11)
    fig.tight_layout();save_figure(fig,'Fig02_restricted_best_score',curves,out)
    # 3. Budget-11 outcomes and batch-level discovery cost.
    fig, axs = plt.subplots(1,3,figsize=(13,4.2))
    x=summary[summary.pool==15].copy();order=['TabPFN + UCB','GPR + UCB','Random (exact)'];x=x.set_index('method').loc[order].reset_index()
    for ax,col,title,ylabel in zip(axs,['budget11_best_score_mean','budget11_hit_rate','discovery_cost_mean'],
                                   ['Best result after first new batch','Paper winner found by budget 11','First-hit cost (batch resolution)'],
                                   ['Paper-reference score','Fraction','Cumulative experiments']):
        bars=ax.bar(range(len(x)),x[col],color=[METHOD_COLORS[m] for m in order])
        for bar,v in zip(bars,x[col]):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height(),f'{v:.3g}',ha='center',va='bottom',fontsize=9)
        ax.set(title=title,ylabel=ylabel);ax.set_xticks(range(len(x)),['TabPFN','GPR','Random'],rotation=20)
        ax.margins(y=.2)
        if col=='budget11_hit_rate': ax.set_ylim(0,1.12); ax.set_yticks([0,.25,.5,.75,1.])
    fig.suptitle('15 measured formulations: TabPFN inference-seed averages; exact random baseline',fontsize=11)
    fig.tight_layout();save_figure(fig,'Fig03_restricted_budget11_metrics',summary,out)
    # 4. Actual 504-space recommendation sets, no invented feedback.
    t=top.copy();t['arm']=t.method+' / '+t.replicate
    order=list(dict.fromkeys(t.arm));candidate_order=list(dict.fromkeys(t.candidate_id))
    mat=np.zeros((len(order),len(candidate_order)))
    for _,r in t.iterrows():mat[order.index(r.arm),candidate_order.index(r.candidate_id)]=5-r['rank']
    fig,ax=plt.subplots(figsize=(max(9,len(candidate_order)*.55),4.7));im=ax.imshow(mat,cmap='Blues',vmin=0,vmax=4,aspect='auto')
    ax.set_yticks(range(len(order)),order,fontsize=8)
    known=set(d.candidate_id);ax.set_xticks(range(len(candidate_order)),[c+(' *' if c in known else '') for c in candidate_order],rotation=45,ha='right')
    ax.set(title='First 4 recommendations from all 497 unobserved candidates',xlabel='* Existing measured feedback; all others require new experiments')
    fig.colorbar(im,ax=ax,label='Rank weight (4 = first recommendation, 0 = not selected)')
    fig.tight_layout();save_figure(fig,'Fig04_full_grid_recommendations',t,out)
    # 5. All-candidate UCB maps: categorical combination by polymer/ratio levels.
    avg=ranks[ranks.method.eq('TabPFN + UCB')].groupby('candidate_id',as_index=False).ucb.mean().merge(ranks[['candidate_id','P1','P2','P3','P4']].drop_duplicates(),on='candidate_id')
    gp=ranks[ranks.method.eq('GPR + UCB')].copy()
    fig,axs=plt.subplots(1,2,figsize=(12,8),sharey=True)
    for ax,data,title in zip(axs,[avg,gp],['TabPFN (UCB averaged across seeds for display)','Independent released-GPR UCB']):
        a=np.full((24,21),np.nan)
        for _,v in data.iterrows():a[(int(v.P2)-1)*6+int(v.P3)-1,(int(v.P1)-1)*7+int(v.P4)-1]=v.ucb
        im=ax.imshow(a,aspect='auto',cmap='viridis');fig.colorbar(im,ax=ax,fraction=.035,label='UCB (current observed target scale)')
        ax.set(title=title,xlabel='(P1, P4) polymer concentration / salt-ratio code')
        ax.set_xticks(range(21),[f'{p1},{p4}' for p1 in range(1,4) for p4 in range(1,8)],rotation=90,fontsize=7)
        ax.set_yticks(range(24),[f'{p2},{p3}' for p2 in range(1,5) for p3 in range(1,7)],fontsize=7)
    axs[0].set_ylabel('(P2, P3) ionic-liquid / mixed-salt category code')
    fig.suptitle('Full 504-code grid; 7 initial points blank — predictions are not measurements',fontsize=11)
    fig.tight_layout();save_figure(fig,'Fig05_full_grid_ucb_maps',pd.concat([avg.assign(method='TabPFN average'),gp.assign(method='GPR')]),out)
    # 6. Hidden-label diagnostics on only the 8 measured held-out candidates.
    fig,axs=plt.subplots(1,2,figsize=(11.5,4.5),sharey=True)
    for ax,method in zip(axs,['TabPFN + UCB','GPR + UCB']):
        x=diagnostic[diagnostic.method.eq(method)].groupby('candidate_id').agg(target=('target_observed_scale','first'),mean=('mean','mean'),lower=('q05','mean'),upper=('q95','mean')).reset_index()
        ax.errorbar(x.target,x['mean'],yerr=np.vstack([np.maximum(0,x['mean']-x.lower),np.maximum(0,x.upper-x['mean'])]),fmt='o',color=METHOD_COLORS[method],capsize=3)
        for _,row in x.iterrows():
            if row.candidate_id in {'1415','1325','1321'}:
                ax.annotate(row.candidate_id,(row.target,row['mean']),fontsize=7,xytext=(3,3),textcoords='offset points')
        lo=min(x.target.min(),x.lower.min());hi=max(x.target.max(),x.upper.max());ax.plot([lo,hi],[lo,hi],'--',color='gray',lw=1)
        ax.set_xlim(lo-.3,hi+.8)
        ax.set(title=method,xlabel='Measured outcome transformed using known-7 scale')
    axs[0].set_ylabel('Predicted score (mean and 90% marginal interval)')
    fig.suptitle('Eight-label retrospective diagnostic; intervals are not proven calibrated',fontsize=11)
    fig.tight_layout();save_figure(fig,'Fig06_prediction_intervals',diagnostic,out)
    # 7. Compare paper-figure numbers with independent reconstruction and first-batch replay.
    reference=d.iloc[:14].groupby('historical_batch').evaluation_score.mean()
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    x=np.arange(3);axs[0].bar(x-.18,[-1.14,.81,1.57],.36,label='Reported Fig.1',color='#8C78A8');axs[0].bar(x+.18,reference,.36,label='Recomputed from 14 rows',color='#48A999')
    axs[0].set(title='Paper batch mean score',ylabel='Retrospective reference score');axs[0].set_xticks(x,['Initial 7','Next 4','Final 3']);axs[0].legend(fontsize=8)
    m=metrics[metrics.pool.eq(14)].groupby('method').new_batch_mean_score.mean().reindex(['TabPFN + UCB','GPR + UCB','Random (exact)','Paper trajectory'])
    axs[1].bar(range(len(m)),m,color=[METHOD_COLORS[k] for k in m.index]);axs[1].set_xticks(range(len(m)),['TabPFN','GPR','Random','Paper'],rotation=20)
    axs[1].set(title='First new batch of 4: measured-pool comparison',ylabel='Mean score of new batch');fig.tight_layout()
    source=pd.concat([pd.DataFrame(dict(panel='paper_reconstruction',batch=[0,1,2],reported=[-1.14,.81,1.57],recomputed=reference.values)),m.rename('new_batch_mean_score').reset_index().assign(panel='first_new_batch')],ignore_index=True);save_figure(fig,'Fig07_paper_metric_comparison',source,out)
    # 8. Make normalization sensitivity visible instead of hiding an endpoint switch.
    score_parts=[]
    for n in [7,11,14,15]:
        a=d.iloc[:n][['conductivity_mS_cm','capacity_mAh_g','high_voltage_pass']];mu=a.mean();sd=a.std(ddof=0)
        tmp=d[['candidate_id']].copy();tmp['scale_rows']=n;tmp['score']=((d[a.columns]-mu)/sd).sum(axis=1);score_parts.append(tmp)
    sensitivity=pd.concat(score_parts,ignore_index=True)
    fig,ax=plt.subplots(figsize=(8,4.2))
    for c,color in [('1415','#456990'),('1325','#EF767A')]:
        v=sensitivity[sensitivity.candidate_id.eq(c)];ax.plot(v.scale_rows,v.score,'-o',label=c+(' paper recommendation' if c=='1415' else 'high conductivity; high-voltage fail'),color=color)
    ax.set(title='The scoring scale changes which formulation looks best',xlabel='Rows used to define retrospective normalization',ylabel='Sum of three standardized outcomes');ax.set_xticks([7,11,14,15]);ax.legend(fontsize=8)
    fig.tight_layout();save_figure(fig,'Fig08_normalization_sensitivity',sensitivity,out)


def write_readme(d, audit, summary, full_summary, cache_metadata, out):
    from prepare_polymer_search import SEEDS
    root=out.parent.parent
    qpath=out/'quota_history.json';q=json.loads(qpath.read_text()) if qpath.exists() else []
    credits=q[-1]['used']-q[0]['used'] if q else None
    tabfull=full_summary[full_summary.method.eq('TabPFN + UCB')]
    gpfull=full_summary[full_summary.method.eq('GPR + UCB')]
    rows=[]
    for _,r in summary.iterrows():
        rows.append(f"|{int(r.pool)}|{r.method}|{r.budget11_best_score_mean:.4f}|{r.budget11_hit_rate:.1%}|{r.discovery_cost_mean:.3f}|{r.random_cost_over_method_cost:.3f}|")
    fullrows=[]
    for _,r in full_summary.iterrows():
        fullrows.append(f"|{r.method} / {r.replicate}|{r.top4}|{int(r.known_feedback)}/4|{int(r.overlap_paper_first_new_batch)}/4|{'是' if r.selected_paper_winner else '否'}|")
    top_data = pd.read_csv(out/'tables/full_grid_top4.csv',dtype={'candidate_id':str})
    decoded=[]
    for code,x in top_data[top_data.method.eq('TabPFN + UCB')].groupby('candidate_id'):
        r=x.iloc[0]
        decoded.append(f"|{code}|{int(r.polymer_percent)}%|{r.ionic_liquid}|{r.salt_pair}|{r.salt_ratio:g}|{len(x)}/{len(cache_metadata)}|")
    decoded_text='\n'.join(decoded)
    time_sum=sum(v['wall_seconds'] for v in cache_metadata.values())
    text=f'''# objective_search：ionic_polymer / clio 配方推荐实验

本轮先运行 **ionic_polymer，Song等，2025，王颖团队**；clio目录已建立，尚未运行。clio原论文的“两工作日”指机器人配液、清洗和测量42次评价的实验时间，不是离线计算时间。

## 本轮究竟做了什么

主分析保留作者的 **504候选设计空间**：已知原始7条实测结果，TabPFN与GPR预测其余497条，并各自推荐第一批4条。**没有为未知配方伪造反馈，也没有执行504候选的模拟闭环。** 后续实验若没有实测值，就在“等待实验反馈”处停止。

另做14/15条已测池的回顾性选点比较，用来观察最早一个批次的表现。它预先限制了候选池，属于次要敏感性分析，不代表在504空间中的真实发现效率。剩余样本全部揭示是强制终点，没有再次调用模型，不作为优化优势。

**目前结果：** 在完整504空间，六种子TabPFN推荐的第一批全部缺少实测反馈，不能判断其真实优劣；GPR第一批与论文下一批4条一致。在限制后的14/15条已测池里，TabPFN六种子均在预算11时包含1415，属于单一起点下的回顾性结果。8条隐藏实测上的90%区间覆盖仅37.5%～50%，且预测均值的Spearman相关为负，不支持声称不确定性已校准或预测整体可靠。

## 数据、目标和对照

- 原论文：[Multi-Objective Optimization of Ionic Polymer Electrolytes for High-Voltage Fast-Charging and Versatile Lithium Batteries](https://doi.org/10.1002/adma.202500941)，[作者数据入口](https://github.com/WangGroupFDU/Bayes_optimization)。提交固定为`d2c5476e3a9bd54bc139ed59830aac461ee50770`。
- 只下载3份实验xlsx与2份优化参考csv，保存在[ionic_polymer数据目录](../datasets/ionic_polymer/)。没有下载作者代码。参考csv是模型输出，仅用于GPR核验，绝不作为实验标签。
- 论文Fig.1为7→11→14，三批大小7、4、3，14/504≈2.78%解释了摘要2.8%。完整公开表有15条，额外1321的三项结果均为0，其未进入图中的原因尚未确认，完整池中仍保留。
- 输入只有P1–P4；P2/P3在TabPFN中显式标为类别，GPR保留作者数字编码的RBF距离。物理配方端点的重复可能性尚未逐项化学审计，因此504首先指编码网格数。
- 模型目标：仅用当轮已知7条的电导率、容量、高电压0/1标签，各自按ddof=0标准化后相加；额外分数列和隐藏标签不进模型。
- 主选点：均值＋1.96×预测标准差（UCB），不重复选择。不是论文四种采集函数合并排名的完整复现。TabPFN方差来自返回的分布，不是六种子波动；尚未证明其校准成立。
- GPR独立复现：RBF长度尺度1、信号方差1、有效训练对角项1.01×10⁻⁸（含数值jitter）、预测噪声1×10⁻¹⁰，匹配作者发布结果；不是调参后挑出的最好GPR。
- 固定评价尺度：论文14条统一标准化的回顾性综合分数，仅评估器可见。它不进入训练合同、UCB或超参数选择。原始性能同时保留。用初始7条的尺度作为评价会把1325而非1415排第一，故不混用尺度。

## 原论文的关键指标及本轮能否比较

|项目|原论文／原数据|本轮对应内容|
|---|---|---|
|初始已知配方|7条|同样7条|
|设计空间|504编码配方|同样网格，预测剩余497条|
|真实优化批次|初始7＋新增4＋新增3|已完成第一批4条推荐；未知反馈不继续|
|三批平均综合分数|约−1.14、0.81、1.57|独立重算约{audit['paper_batch_means_recalculated'][0]:.3f}、{audit['paper_batch_means_recalculated'][1]:.3f}、{audit['paper_batch_means_recalculated'][2]:.3f}，见Fig07|
|论文推荐配方|1415；电导率3.61 mS/cm、容量63.78 mAh/g、高电压标签1|报告第一批是否包含它；完整空间未知标签不用于宣称它是全局最优|
|作者采集策略|EI、PI、UCB、Thompson sampling合并排名|当前先用共同UCB比较代理，论文真实轨迹只作参考|
|随机对照／发现成本|正文未给相同口径的随机加速数值|仅在已测池精确枚举随机批次，重新计算|
|真实日历时间|未找到优化总用时精确值|只记录模型/API耗时，不能宣称节省了多少真实实验小时|

## 完整504空间：第一批推荐

{len(cache_metadata)}个TabPFN推理种子已经完成。下面每行是独立的推理种子，不是独立初始化实验。`已知反馈`指在历史公开表中恰好有实测值的推荐，不代表本轮执行了新实验；与作者下一批的重合只反映选点相似性，不是正确率。

|方法／种子|按UCB排序的4条推荐|已知反馈|与论文下一批重合|含论文推荐1415|
|---|---|---:|---:|---|
'''+ '\n'.join(fullrows)+f'''

TabPFN推荐中的未知反馈数范围为{int(tabfull.unmeasured_feedback.min())}～{int(tabfull.unmeasured_feedback.max())}/4；GPR为{int(gpfull.unmeasured_feedback.iloc[0])}/4。因此当前不能计算完整504空间的实际发现成本、regret或“找到最优”的轮次。完整排名见[预测与排名](ionic_polymer/results/tables/full_grid_predictions_and_ranks.csv)，[待反馈推荐表](ionic_polymer/results/tables/full_grid_top4.csv)保留组成编码和反馈状态。

TabPFN的不同种子一共推荐了以下物理配方编码，全部尚无公开反馈；这里的盐比例按作者P4字典解释，不能把编码位数当作实际浓度。

|编码|PBDT浓度|离子液体|混合盐|盐比例|入选种子数|
|---|---:|---|---|---:|---:|
{decoded_text}

![完整空间第一批推荐](ionic_polymer/results/figures/Fig04_full_grid_recommendations.png)

![完整候选空间UCB分布](ionic_polymer/results/figures/Fig05_full_grid_ucb_maps.png)

## 次要结果：已测池的早期推荐

固定初始7条，在其余已测候选中选4条，预算累计11条时评价。14条视图沿用全网格预测，仅排除额外1321，不重新调用API。随机批次分别精确枚举35和70种组合，没有随机抽样误差。六推理种子的波动只描述固定数据下推理变化，不能当作统计显著性或跨起点稳定性。

|已测池|方法|预算11最好参考分数|预算11命中1415|平均首次命中成本|随机成本／方法成本|
|---:|---|---:|---:|---:|---:|
'''+ '\n'.join(rows)+f'''

成本包含初始7条。由于按批揭示，首次命中成本只能是11或最终14/15；不能给同批4条编造顺序。表中成本比仅针对这个限制后的池，不能写成504空间的加速倍数。所有候选最终揭示后结果必然相同，重点看预算11。

图中阴影分别是固定起点下六推理种子的范围和精确随机批次的范围，不是置信区间。

![已测池最好结果曲线](ionic_polymer/results/figures/Fig02_restricted_best_score.png)

![已测池预算11指标](ionic_polymer/results/figures/Fig03_restricted_budget11_metrics.png)

![原文综合指标重算与第一批对照](ionic_polymer/results/figures/Fig07_paper_metric_comparison.png)

## 数据与预测诊断

![实测性能及论文批次](ionic_polymer/results/figures/Fig01_measured_outcomes.png)

![8条已测候选的回顾性预测诊断](ionic_polymer/results/figures/Fig06_prediction_intervals.png)

上图真值采用当前已知7条定义的目标尺度，与固定评价参考分数不同。TabPFN图中对各种子预测均值和区间端点取平均，仅用于展示，不作为一个重新校准的集成分布。只有8条诊断标签，覆盖率不能证明不确定性可靠。

![评分尺度改变排名](ionic_polymer/results/figures/Fig08_normalization_sensitivity.png)

## 完整性、耗时与复现

- 数据Git blob、SHA256、累计行对应及原文均分重算均通过；独立GPR在两轮共1008条作者网格输出上的均值、方差、UCB最大绝对差均低于10⁻⁹，见[数据审计](ionic_polymer/results/data/data_audit.json)。
- TabPFN checkpoint沿用ampere_conductivity的`tabpfn-v3-regressor-v3_default.ckpt`，n_estimators=8，推理种子{sorted(cache_metadata)}。全网格预测累计耗时{time_sum:.1f}秒（不含本地绘图）；此时间不能与论文的实验时间比较。
- API累计消耗{credits} credits，包含先前一次8条接口检查；额度记录见[quota_history.json](ionic_polymer/results/quota_history.json)。实际计费预测最多7份，硬上限100000 credits，无自动重试。
- [结果核验](ionic_polymer/results/tables/result_audit.json)、[冻结协议](ionic_polymer/pilot_protocol.md)、[逐请求分布缓存](ionic_polymer/results/cache/)、源码与协议快照、逐图CSV源数据一并保留。

项目根目录运行：

```bash
# 只读数据审计，不调用API
.venv/bin/python study/objective_search/ionic_polymer/prepare_polymer_search.py audit
# 校验并复用已存预测；缓存缺失时才会发起API预测
.venv/bin/python study/objective_search/ionic_polymer/prepare_polymer_search.py run --scope full_grid --resume
# 仅重建表格/图和本README，不调用API
.venv/bin/python study/objective_search/ionic_polymer/prepare_polymer_search.py report
```

离线检查命令：`.venv/bin/python -m unittest discover -s study/objective_search/ionic_polymer -p test_polymer_search.py`，不调用API。

目录：ionic_polymer中`prepare_polymer_search.py`负责审计和预测，`evaluate_polymer_rankings.py`只读缓存生成结果，`Output/`保留输出；[clio目录](clio/)只预留说明。本轮是第一批推荐的可行性试跑，不是完成三轮的新实验，也不是TabPFN优化优势的正式结论。
'''
    (root/'README.md').write_text(text)


def write_full_ranking_analysis(d, ranks, out):
    """Evaluate frozen initial-model rankings, never impute unknown outcomes."""
    targets = d.iloc[7:14].copy()
    detail = ranks.merge(targets[['candidate_id', 'historical_batch', 'evaluation_score'] +
                                ['conductivity_mS_cm', 'capacity_mAh_g', 'high_voltage_pass']],
                         on='candidate_id', how='inner', validate='many_to_one')
    detail['static_batch_of_4'] = (detail['rank'] + 3) // 4
    detail['static_budget_at_batch_end'] = 7 + np.minimum(4 * detail.static_batch_of_4, 497)
    detail.to_csv(out/'tables/full_grid_paper_target_ranks.csv', index=False)
    all_ranks = ranks.copy()
    all_ranks['static_batch_of_4'] = (all_ranks['rank'] + 3) // 4
    all_ranks['is_paper_final_1415'] = all_ranks.candidate_id.eq('1415')
    all_ranks.to_csv(out/'tables/full_grid_static_recommendation_schedule.csv', index=False)
    summaries, curves = [], []
    initial_best = float(d.iloc[:7].evaluation_score.max())
    for (method, replicate), x in detail.groupby(['method', 'replicate']):
        winner = x[x.candidate_id.eq('1415')].iloc[0]
        summaries.append(dict(method=method, replicate=replicate,
                              winner_rank=int(winner['rank']),
                              winner_static_batch=int(winner.static_batch_of_4),
                              static_budget=int(winner.static_budget_at_batch_end),
                              recall_paper7_at4=float((x['rank'] <= 4).mean()),
                              recall_paper7_at8=float((x['rank'] <= 8).mean()),
                              recall_paper7_at12=float((x['rank'] <= 12).mean())))
        for batch in range(126):
            k = min(4 * batch, 497)
            known = x[x['rank'] <= k]
            curves.append(dict(method=method, replicate=replicate, static_batch=batch, additional_candidates=k,
                               cumulative_budget=7+k, known_paper_target_hits=len(known),
                               known_target_recall=len(known)/7,
                               winner_hit=int(winner['rank'] <= k),
                               best_public_reference_score_lower_bound=max([initial_best]+known.evaluation_score.tolist()),
                               measured_fraction=(7+len(known))/(7+k)))
    summary = pd.DataFrame(summaries)
    curve = pd.DataFrame(curves)
    summary.to_csv(out/'tables/full_grid_static_rank_summary.csv', index=False)
    curve.to_csv(out/'tables/full_grid_static_rank_curves.csv', index=False)
    # Complete permutations, explicit target availability, and exact endpoint identities.
    assert len(detail) == 7 * ranks.groupby(['method', 'replicate']).ngroups
    for _, x in ranks.groupby(['method', 'replicate']):
        assert len(x) == 497 and x.candidate_id.nunique() == 497
        assert sorted(x['rank']) == list(range(1, 498))
    assert curve.groupby(['method', 'replicate']).tail(1).winner_hit.eq(1).all()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for method, color in [('TabPFN + UCB', '#456990'), ('GPR + UCB', '#EF767A')]:
        x = summary[summary.method.eq(method)]
        axes[0].scatter(x.replicate.astype(str), x.winner_rank, label=method, color=color)
        g = curve[curve.method.eq(method)].groupby('static_batch')
        for ax, column in [(axes[1], 'known_target_recall'), (axes[2], 'best_public_reference_score_lower_bound')]:
            z = g[column].agg(['mean', 'min', 'max'])
            ax.plot(z.index, z['mean'], color=color, label=method)
            ax.fill_between(z.index, z['min'], z['max'], color=color, alpha=.15)
    axes[0].set(xlabel='Inference seed / GPR', ylabel='Rank of paper final recipe 1415', title='Full grid: lower rank is better')
    axes[0].tick_params(axis='x', rotation=45)
    axes[1].set(xlabel='Static batch (4 candidates; no refit)', ylabel='Recall of 7 paper follow-up recipes', ylim=(0, 1.05))
    axes[2].set(xlabel='Static batch (4 candidates; no refit)', ylabel='Best public reference score: lower bound')
    axes[2].axhline(2.474354498346776, ls=':', color='gray', label='Paper final recipe score')
    for ax in axes: ax.legend(fontsize=7); ax.grid(alpha=.2)
    fig.suptitle('Frozen initial-7 ranking audit: NOT an adaptive closed-loop experiment')
    fig.tight_layout()
    save_figure(fig, 'Fig09_full_grid_static_rank_audit', curve, out)
    # Preserve source data for the scatter panel as well as the curve panels.
    summary.to_csv(out/'figure_source_data/Fig09_full_grid_static_rank_audit_summary.csv', index=False)
    rows=[]
    for _, r in summary.iterrows():
        rows.append(f"|{r.method} / {r.replicate}|{int(r.winner_rank)}|{int(r.winner_static_batch)}|{int(r.static_budget)}|{r.recall_paper7_at8:.1%}|{r.recall_paper7_at12:.1%}|")
    section = """
## 完整504候选排名：找回论文配方的直接评价

**完整空间已经跑完首轮预测和全部排名；不是尚未运行。** 504条中初始7条已知，因此实际待推荐排名包含497条。六个TabPFN种子及GPR的完整推荐顺序、每4条对应的静态批次都已导出。主要问题是能否早推荐论文最终配方1415，而不是要求先测完全部候选。

下表的“静态批次”＝向上取整（排名÷4）；“累计预算”＝7＋该批次结束时累计推荐数。它回答**保持初始7条模型不变，沿排名逐批取4条，需要第几批才能覆盖1415**。它不是实验反馈后重新训练得到的真实BO轮次。原论文在新增第二批（总计第3批、累计14条）测到1415；此处静态批次从初始7条之后开始计数，不能把两种轮次直接当作同一实验条件。

|方法／种子|1415推荐排名／497|静态新增批次|静态累计预算|前8条覆盖论文新增7条|前12条覆盖论文新增7条|
|---|---:|---:|---:|---:|---:|
""" + '\n'.join(rows) + """

随机不放回排序的参考：1415的期望排名为249，按4条一批的期望静态批次为62.63，前4/8/12条命中概率分别为4/497、8/497、12/497。它是静态随机排序参照，不是原论文结果，也不是自适应随机闭环模拟。六个TabPFN种子只有推理随机性，不能视为六次独立优化实验。

![完整排名评价](ionic_polymer/results/figures/Fig09_full_grid_static_rank_audit.png)

图中综合分数曲线只使用初始7条和已覆盖的论文后续7条的真实结果。**未测配方记为缺失，不赋0分；所以曲线是已有公开证据所支持的最好分数下界，不代表每个推荐配方的实际表现。** 论文外的未知配方可能更好，也可能更差。综合分数是三项标准化结果之和，没有单位，差值不能解释为性能提升百分比。

- [完整推荐顺序与静态批次](ionic_polymer/results/tables/full_grid_static_recommendation_schedule.csv)
- [论文后续7个配方的逐项排名、原始性能及参考分数](ionic_polymer/results/tables/full_grid_paper_target_ranks.csv)
- [各预算覆盖率、命中情况与参考分数下界](ionic_polymer/results/tables/full_grid_static_rank_curves.csv)

### TabPFN与采集函数分别做什么

TabPFN是**代理模型**：学习已测配方与综合目标的关系，为每个未测候选输出预测分布，包括预测均值μ和标准差σ。UCB是**选点规则**：用μ＋1.96σ给候选排序，兼顾预测性能与探索。没有代理模型，UCB没有这些候选相关的μ、σ可以使用。

因此同一个UCB，接GPR与接TabPFN会推荐不同配方。当前比较是“GPR代理＋UCB”对“TabPFN代理＋UCB”；替换的是预测模型，采集规则固定。实际闭环应为：已测数据→代理模型预测→采集函数选4条→实验反馈→更新标准化目标与模型→重新预测。后续每一轮仍需要代理模型，不是首轮结束后只靠采集函数运行。

原论文合并EI、PI、UCB与Thompson采样；当前共同UCB用于隔离代理模型差异，不能称为原论文采集策略的完整复现。现有公开标签可支持作者真实轨迹的模型更新，但不支持TabPFN任意新选配方的真实反馈；使用作者批次统一更新两种模型可做“共享历史的逐轮推荐审计”，应另列为历史回放，而不能声称是TabPFN自主闭环的发现成本。
"""
    readme = out.parent.parent/'README.md'
    old = readme.read_text()
    marker = '\n## 完整504候选排名：找回论文配方的直接评价'
    readme.write_text(old.split(marker)[0] + section)
    manifest_path = out/'tables/report_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    from prepare_polymer_search import sha
    manifest['figures'] = {p.name:sha(p) for p in (out/'figures').glob('*')}
    manifest['source_tables'] = {p.name:sha(p) for p in (out/'figure_source_data').glob('*.csv')}
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
