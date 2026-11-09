"""Offline acquisition ablation: fixed initial-7 posterior, no API or imputed labels."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from scipy.special import softmax, ndtri
from scipy.stats import norm
from scipy.linalg import cho_factor, cho_solve
import torch
from tabpfn.architectures.shared.bar_distribution import FullSupportBarDistribution
from prepare_polymer_search import prepare_data, zscore_objective, OUTCOMES, FEATURES, SEEDS, sha
from evaluate_polymer_rankings import save_figure
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
OUT = ROOT/'results'
TS_SEEDS = list(range(410,430))


def stable_order(values, ids):
    return np.lexsort((np.asarray(ids, str), -np.asarray(values)))


def rank_values(values, ids):
    order = stable_order(values, ids)
    r = np.empty(len(order), int); r[order] = np.arange(1, len(order)+1)
    return r


def top4_vote(scores, ids):
    votes = np.zeros(len(ids))
    for s in scores:
        votes[stable_order(s, ids)[:4]] += [4,3,2,1]
    return votes


def rank_bounds(values, index):
    """Do not fabricate target rank when sparse voting leaves zero-score ties."""
    return int(1+np.sum(values > values[index])), int(np.sum(values >= values[index]))


def marginal_samples(prob, borders, rng, draws=4):
    """Independent inverse-CDF samples, half-normal tails as in cached criterion."""
    cdf = np.cumsum(prob, axis=1); cdf[:,-1] = 1.
    u = rng.random((len(prob),draws))
    result=np.empty_like(u)
    scale0=(borders[1]-borders[0])/ndtri(.75)
    scale1=(borders[-1]-borders[-2])/ndtri(.75)
    for j in range(draws):
        b=np.sum(cdf < u[:,j,None],axis=1)
        previous=np.where(b>0,cdf[np.arange(len(prob)),np.maximum(b-1,0)],0)
        f=np.clip((u[:,j]-previous)/prob[np.arange(len(prob)),b],1e-12,1-1e-12)
        result[:,j]=borders[b]+f*(borders[b+1]-borders[b])
        left=b==0; right=b==prob.shape[1]-1
        result[left,j]=borders[1]-scale0*ndtri(1-f[left]/2)
        result[right,j]=borders[-2]+scale1*ndtri((1+f[right])/2)
    return result


def exact_ei(criterion, logits, threshold):
    """Correct native EI upper-tail location offset for interior incumbents.

    Installed FullSupportBarDistribution.ei omits p_last*(b[-2]-t)
    when t is below the upper-tail origin. Keep the half-normal mean term.
    """
    assert torch.all(threshold >= criterion.borders[1])
    assert torch.all(threshold <= criterion.borders[-2])
    return criterion.ei(logits, threshold) + torch.softmax(logits,-1)[...,-1]*(criterion.borders[-2]-threshold)


def tab_scores(seed, best):
    stem=OUT/'cache'/f'full_grid_initial7_seed{seed}'
    meta=json.loads(stem.with_suffix('.json').read_text())
    assert sha(stem.with_suffix('.npz'))==meta['raw_sha256']
    assert sha(stem.with_suffix('.csv'))==meta['csv_sha256']
    pred=pd.read_csv(stem.with_suffix('.csv'),dtype={'candidate_id':str})
    raw=np.load(stem.with_suffix('.npz'))
    logits=torch.as_tensor(raw['logits'],dtype=torch.float64)
    criterion=FullSupportBarDistribution(torch.as_tensor(raw['borders'],dtype=torch.float64))
    # Validate the reconstruction against API-derived moments before using EI/PI.
    assert np.allclose(criterion.mean(logits).numpy(),pred['mean'],atol=2e-6)
    assert np.allclose(criterion.variance(logits).numpy(),pred['std']**2,atol=2e-5)
    threshold=torch.full(logits.shape[:-1],best,dtype=logits.dtype)
    scores={'EI':exact_ei(criterion,logits,threshold).numpy(), 'PI':criterion.pi(logits,threshold).numpy(),
            'UCB':pred.ucb.to_numpy()}
    prob=softmax(raw['logits'],axis=1)
    sample=lambda seed: marginal_samples(prob,raw['borders'],np.random.default_rng(seed))
    return pred.candidate_id.to_numpy(),scores,sample


def gp_scores(ranks,d,best):
    pred=ranks[ranks.method.eq('GPR + UCB')].sort_values('candidate_id')
    ids=pred.candidate_id.to_numpy(); mean=pred['mean'].to_numpy(); std=pred['std'].to_numpy()
    z=(mean-best)/std
    scores={'EI':(mean-best)*norm.cdf(z)+std*norm.pdf(z),'PI':norm.cdf(z),'UCB':mean+1.96*std}
    train=d.iloc[:7][FEATURES].to_numpy(float)
    q=pred[FEATURES].to_numpy(float)
    kernel=lambda a,b:np.exp(-.5*np.sum((a[:,None,:]-b[None,:,:])**2,axis=2))
    k=kernel(train,train)+np.eye(7)*1.01e-8
    cross=kernel(q,train)
    cov=kernel(q,q)-cross@cho_solve(cho_factor(k),cross.T)
    eig,vec=np.linalg.eigh((cov+cov.T)/2)
    factor=vec*np.sqrt(np.maximum(eig,0))
    sample=lambda seed:mean[:,None]+factor@np.random.default_rng(seed).standard_normal((len(ids),4))
    return ids,scores,sample


def evaluate_ranking(ids, values, d):
    target=int(np.flatnonzero(ids=='1415')[0])
    lo,hi=rank_bounds(values,target)
    order=stable_order(values,ids); selected=ids[order[:4]]
    known=d[d.candidate_id.isin(selected)]
    visible=pd.concat([d.iloc[:7],known]).drop_duplicates('candidate_id')
    future=set(d.iloc[7:14].candidate_id)
    return dict(winner_rank_low=lo,winner_rank_high=hi,
                static_batch_low=int(np.ceil(lo/4)),static_batch_high=int(np.ceil(hi/4)),
                top4=' '.join(selected),top4_measured=len(known),
                top4_paper_followup_hits=len(set(selected)&future),
                top12_paper_followup_hits=len(set(ids[order[:12]])&future),
                budget11_best_public_score_lower_bound=float(visible.evaluation_score.max()),
                target_has_vote=bool(values[target]>0),top4_tie_at_boundary=bool(np.sum(values==values[order[3]])>1))


def run():
    torch.set_num_threads(2)
    d,_=prepare_data(); y,_,_=zscore_objective(d.iloc[:7][OUTCOMES]);best=float(max(y))
    ranks=pd.read_csv(OUT/'tables/full_grid_predictions_and_ranks.csv',dtype={'candidate_id':str})
    arms=[('TabPFN',str(seed),tab_scores(seed,best)) for seed in SEEDS]
    arms.append(('GPR','deterministic',gp_scores(ranks,d,best)))
    full=[];summaries=[];mc=[]; nominations=[]
    for model,rep,(ids,scores,sampler) in arms:
        assert len(ids)==497 and len(set(ids))==497
        assert not set(ids)&set(d.iloc[:7].candidate_id)
        assert all(np.isfinite(v).all() for v in scores.values())
        ts=sampler(410)
        ts_votes=np.zeros(len(ids))
        for j in range(4):
            ts_votes[stable_order(ts[:,j],ids)[0]]+=4-j
        methods={**scores,'Vote3':top4_vote(list(scores.values()),ids),
                 'Vote4':top4_vote(list(scores.values()),ids)+ts_votes,
                 'FullRank3':sum((len(ids)-rank_values(s,ids))/(len(ids)-1) for s in scores.values())/3}
        for method,values in methods.items():
            metrics=evaluate_ranking(ids,values,d)
            summaries.append(dict(model=model,replicate=rep,acquisition=method,**metrics))
            order=stable_order(values,ids)
            x=pd.DataFrame({'candidate_id':ids,'score':values,'EI':scores['EI'],'PI':scores['PI'],'UCB':scores['UCB']})
            x=x.iloc[order].reset_index(drop=True);x['display_order']=np.arange(1,498)
            # score ties must not be interpreted as evidence of exact target rank.
            x['rank_low']=x.score.rank(method='min',ascending=False).astype(int)
            x['rank_high']=x.score.rank(method='max',ascending=False).astype(int)
            x['model'],x['replicate'],x['acquisition']=model,rep,method
            full.append(x)
        for ts_seed in TS_SEEDS:
            ts=sampler(ts_seed);vote=top4_vote(list(scores.values()),ids)
            for j in range(4):
                index=stable_order(ts[:,j],ids)[0];vote[index]+=4-j
                nominations.append(dict(model=model,replicate=rep,ts_seed=ts_seed,draw=j+1,candidate_id=ids[index]))
            mc.append(dict(model=model,replicate=rep,ts_seed=ts_seed,**evaluate_ranking(ids,vote,d)))
    full=pd.concat(full,ignore_index=True);summary=pd.DataFrame(summaries);mc=pd.DataFrame(mc)
    for name,x in [('acquisition_full_rankings',full),('acquisition_summary',summary),('acquisition_ts_sensitivity',mc),('acquisition_ts_nominations',pd.DataFrame(nominations))]:
        x.to_csv(OUT/'tables'/f'{name}.csv',index=False)
    draw(summary,mc)
    write_readme(summary,mc,best)
    write_ranking_profile(full,d)
    output_files=list((OUT/'tables').glob('acquisition*.csv'))+list((OUT/'tables').glob('ranking_profile*.csv'))+list((OUT/'figures').glob('Fig1[01]*'))
    audit=dict(status='completed_offline',initial_measured=7,unmeasured_candidates=497,api_calls=0,
               best_observed_model_scale=best,model_seeds=SEEDS,primary_ts_seed=410,ts_sensitivity_seeds=TS_SEEDS,
               vote_rule='Top4 each of EI/PI/UCB gets 4,3,2,1; TS four draws give winner weights 4,3,2,1; aggregate. Ties retain rank intervals.',
               tabpfn_ts='Independent marginal draws, not a GP-like joint posterior function draw.',
               gp_ts='Correlated latent GP draws; same seed label but not bitwise GPy sampling reproduction.',
               fullrank_rule='Equal-weight mean of complete-list normalized ranks of EI/PI/UCB; our extension, not original top4 voting.',
               sources={'commit':'d2c5476e3a9bd54bc139ed59830aac461ee50770','post_process_blob':'50258ae7af1f7b3e766dd9b413e56c402007ac1a','config_blob':'c7509fb36cd57f675cfab393ff7668f0fb2ea764'},
               numerical_note='Native EI corrected for omitted upper-tail location offset; tail sampler uses half-normal inverse CDF, not inherited finite-bar icdf.',
               code_sha256=sha(__file__),outputs={str(p.relative_to(OUT)):sha(p) for p in output_files})
    (OUT/'tables/acquisition_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    manifest_path=OUT/'tables/report_manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        manifest['figures']={p.name:sha(p) for p in (OUT/'figures').glob('*')}
        manifest['source_tables']={p.name:sha(p) for p in (OUT/'figure_source_data').glob('*.csv')}
        manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    print(summary[['model','replicate','acquisition','winner_rank_low','winner_rank_high','top4','budget11_best_public_score_lower_bound']].to_string(index=False))


def draw(summary,mc):
    methods=['UCB','EI','PI','Vote3','Vote4','FullRank3']
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,model in zip(axes,['TabPFN','GPR']):
        for i,method in enumerate(methods):
            x=summary[(summary.model==model)&(summary.acquisition==method)]
            jitter=np.linspace(-.15,.15,len(x)) if len(x)>1 else [0]
            for dx,(_,r) in zip(jitter,x.iterrows()):
                ax.plot([i+dx,i+dx],[r.winner_rank_low,r.winner_rank_high],color='#456990',alpha=.5)
                ax.scatter(i+dx,r.winner_rank_low,color='#456990',s=22)
                if r.winner_rank_high!=r.winner_rank_low: ax.scatter(i+dx,r.winner_rank_high,marker='_',color='#456990')
        ax.set(xticks=range(len(methods)),xticklabels=methods,yscale='log',ylabel='Rank of paper final recipe 1415 (log scale)',title=model)
        ax.grid(axis='y',alpha=.2);ax.axhline(4,color='gray',ls=':')
    fig.suptitle('Fixed initial-7 predictions: sparse-vote ties shown as rank intervals')
    fig.tight_layout();save_figure(fig,'Fig10_acquisition_comparison',summary,OUT)
    mc.to_csv(OUT/'figure_source_data/Fig10_acquisition_ts_sensitivity.csv',index=False)


def write_readme(summary,mc,best):
    methods=['UCB','EI','PI','Vote3','Vote4','FullRank3'];rows=[]
    for model in ['TabPFN','GPR']:
        for method in methods:
            x=summary[(summary.model==model)&(summary.acquisition==method)]
            lo=int(x.winner_rank_low.min());hi=int(x.winner_rank_high.max())
            rank=str(lo) if lo==hi else f'{lo}～{hi}'
            rows.append(f'|{model}|{method}|{rank}|{int(x.top4_paper_followup_hits.min())}～{int(x.top4_paper_followup_hits.max())}/7|{x.budget11_best_public_score_lower_bound.min():.3f}～{x.budget11_best_public_score_lower_bound.max():.3f}|')
    tops=[]
    for _,r in summary[summary.acquisition.isin(['EI','Vote3','Vote4','FullRank3'])].iterrows():
        tops.append(f'|{r.model}/{r.replicate}|{r.acquisition}|{r.top4}|{r.top4_measured}/4|')
    s='''
## 采集函数对照：EI与多函数投票

固定初始7条与已有TabPFN分布，离线比较，新增API调用为0。没有利用1415的位置调参，也没有选择性保留有利种子；这些方案是看过原UCB结果之后开展的探索性分析。所有方法都在同样的497个剩余候选中选择。

作者实现核实：[post_process.py](https://github.com/WangGroupFDU/Bayes_optimization/blob/d2c5476e3a9bd54bc139ed59830aac461ee50770/utils/post_process.py)与[config.json](https://github.com/WangGroupFDU/Bayes_optimization/blob/d2c5476e3a9bd54bc139ed59830aac461ee50770/config.json)。每种EI、PI、UCB各取前4名，按4、3、2、1赋票分；Thompson抽4次函数，每次最大值对应配方分别获4、3、2、1分；票分相加。不是直接相加原始采集值。

- **EI**：TabPFN使用完整bar分布计算预期改善，包含半正态尾部；修正本地库EI遗漏的右尾位置偏移项，并验证EI导数与PI一致。GPR使用高斯EI。阈值仅取初始7条中模型目标的最好值。
- **PI**：改善概率，作为诊断单列；UCB仍为均值＋1.96倍标准差。
- **Vote3**：EI＋PI＋UCB的前4名投票，无随机抽样。
- **Vote4**：加入Thompson提名，主结果固定抽样种子410，另外410～429共20个种子保留敏感性结果。TabPFN只缓存了边际分布，所以采用独立边际抽样近似，不冒充联合后验Thompson；GPR用相关联合后验抽样。不能把二者TS视为完全相同机制。
- **FullRank3**：对EI、PI、UCB的全列表名次等权平均，作为完整排名的补充方案；这是本项目扩展，不是作者原投票规则。

**前4名投票不构成完整的严格排序。** 未获票的候选全部并列0分。因此Vote3/4报告1415的名次区间；不能用候选编号打破平局后把“展示顺序”当作发现轮次。完整CSV同时保存rank_low与rank_high。

|代理模型|采集方案|1415排名范围（含种子变化及平局）|首批4条覆盖论文后续7条|预算11已知最好参考分数下界|
|---|---|---:|---:|---:|
'''+ '\n'.join(rows)+'''

**本轮结论：** 单独EI没有稳定优于原UCB；作者式前4名投票没有给1415票分。全排名融合将1415提前到21～27名，但首批4条仍没有命中1415，已知最好参考分数下界仍为1.473，不能据此声称实际性能提高。加入Thompson的20种子敏感性中，TabPFN共120个组合均未给1415票分；GPR的20个组合中有1个给票，说明抽样本身会改变提名，不能挑有利抽样报告。

分数下界只来自有真实测量值的推荐和初始7条；缺失反馈不记0分。排名不等于实际闭环轮数，所有结果均未进行反馈后模型更新。Thompson敏感性分析是对同一预测分布重复抽样，不是独立优化实验。

![采集函数比较](ionic_polymer/results/figures/Fig10_acquisition_comparison.png)

|代理／种子|采集方案|首批4个推荐（票分平局按编号展示）|有公开测量值|
|---|---|---|---:|
'''+ '\n'.join(tops)+'''

[完整采集值与排名](ionic_polymer/results/tables/acquisition_full_rankings.csv)、[汇总](ionic_polymer/results/tables/acquisition_summary.csv)、[Thompson抽样敏感性](ionic_polymer/results/tables/acquisition_ts_sensitivity.csv)、[抽样提名](ionic_polymer/results/tables/acquisition_ts_nominations.csv)、[审计与文件哈希](ionic_polymer/results/tables/acquisition_audit.json)。

重建本节及图表（离线）：`.venv/bin/python study/objective_search/ionic_polymer/compare_polymer_acquisitions.py`。
'''
    path=ROOT.parent/'README.md';old=path.read_text();marker='\n## 采集函数对照：EI与多函数投票'
    path.write_text(old.split(marker)[0]+s)


def write_ranking_profile(full, d):
    """Multi-endpoint retrospective ranking evaluation, not a fabricated BO loop."""
    targets=d.iloc[7:14].copy()
    good=set(targets.nlargest(3,'evaluation_score').candidate_id)
    x=full[full.acquisition.eq('FullRank3')].copy()
    details=x.merge(targets,on='candidate_id',validate='many_to_one')
    details['static_batch_low']=(details.rank_low+3)//4
    details['static_batch_high']=(details.rank_high+3)//4
    details['top3_public_followup_score']=details.candidate_id.isin(good)
    details.to_csv(OUT/'tables/ranking_profile_targets.csv',index=False)
    curve=[];summaries=[]
    for (model,rep),a in x.groupby(['model','replicate']):
        a=a.sort_values('display_order')
        r=details[(details.model==model)&(details.replicate==rep)]
        assert set(r.candidate_id)==set(targets.candidate_id)
        for k in range(498):
            selected=set(a.head(k).candidate_id)
            known=targets[targets.candidate_id.isin(selected)]
            visible=pd.concat([d.iloc[:7],known])
            feasible=visible[visible.high_voltage_pass.eq(1)]
            curve.append(dict(model=model,replicate=rep,additional_budget=k,total_budget=7+k,
                              public7_hits=len(known),public_top3_hits=len(selected&good),
                              winner_hit=int('1415' in selected),
                              best_public_score_lower_bound=float(visible.evaluation_score.max()),
                              best_public_conductivity_lower_bound=float(visible.conductivity_mS_cm.max()),
                              best_public_feasible_capacity_lower_bound=float(feasible.capacity_mAh_g.max()),
                              known_measured_count=len(visible)))
        summaries.append(dict(model=model,replicate=rep,
                              public7_mean_rank_low=float(r.rank_low.mean()),public7_mean_rank_high=float(r.rank_high.mean()),
                              public_top3_mean_rank_low=float(r[r.candidate_id.isin(good)].rank_low.mean()),
                              public_top3_mean_rank_high=float(r[r.candidate_id.isin(good)].rank_high.mean()),
                              all_public_top3_covered_by_rank=int(r[r.candidate_id.isin(good)].rank_high.max())))
    curve=pd.DataFrame(curve);summaries=pd.DataFrame(summaries)
    curve.to_csv(OUT/'tables/ranking_profile_budget_curves.csv',index=False)
    summaries.to_csv(OUT/'tables/ranking_profile_summary.csv',index=False)
    paper=[]
    for batch,n in [(0,7),(1,11),(2,14)]:
        visible=d.iloc[:n]
        paper.append(dict(added_batch=batch,total_measured=n,
                          public7_hits=max(n-7,0),public_top3_hits=len(set(visible.candidate_id)&good),
                          winner_hit=int('1415' in set(visible.candidate_id)),
                          best_score=float(visible.evaluation_score.max()),
                          best_conductivity=float(visible.conductivity_mS_cm.max()),
                          best_feasible_capacity=float(visible[visible.high_voltage_pass.eq(1)].capacity_mAh_g.max())))
    paper=pd.DataFrame(paper);paper.to_csv(OUT/'tables/ranking_profile_paper_trajectory.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(14,4.4))
    target_order=targets.sort_values('evaluation_score',ascending=False).candidate_id.tolist()
    for i,code in enumerate(target_order):
        for model,color,dx in [('TabPFN','#456990',-.12),('GPR','#EF767A',.12)]:
            a=details[(details.candidate_id==code)&(details.model==model)]
            lo,hi=a.rank_low.min(),a.rank_high.max()
            axes[0].plot([i+dx]*2,[lo,hi],color=color,lw=3)
            axes[0].scatter(i+dx,(lo+hi)/2,color=color,label=model if i==0 else None,s=25)
    axes[0].set(xticks=range(7),xticklabels=target_order,ylabel='Rank among 497 candidates (lower is better)',title='Paper follow-up recipes, ordered by measured score')
    for model,color in [('TabPFN','#456990'),('GPR','#EF767A')]:
        a=curve[curve.model.eq(model)]
        for ax,column in [(axes[1],'public_top3_hits'),(axes[2],'best_public_score_lower_bound')]:
            g=a.groupby('additional_budget')[column].agg(['mean','min','max'])
            ax.step(g.index,g['mean'],where='post',color=color,label=model)
            ax.fill_between(g.index,g['min'],g['max'],step='post',alpha=.15,color=color)
    axes[1].set(xlim=(0,160),xlabel='Additional recommendations, frozen model',ylabel='Hits of public follow-up top 3',yticks=[0,1,2,3])
    axes[2].set(xlim=(0,160),xlabel='Additional recommendations, frozen model',ylabel='Best public reference score: lower bound')
    for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('Complete-rank fusion: offline ranking, NOT feedback-updated BO rounds')
    fig.tight_layout();save_figure(fig,'Fig11_ranking_profile',curve,OUT)
    details.to_csv(OUT/'figure_source_data/Fig11_ranking_profile_targets.csv',index=False)
    # Exact target identities and all original published outcomes remain visible.
    rows=[]
    for _,t in targets.sort_values('evaluation_score',ascending=False).iterrows():
        a=details[(details.model=='TabPFN')&(details.candidate_id==t.candidate_id)]
        g=details[(details.model=='GPR')&(details.candidate_id==t.candidate_id)].iloc[0]
        rows.append(f"|{t.candidate_id}|{t.evaluation_score:.3f}|{t.conductivity_mS_cm:g}|{t.capacity_mAh_g:g}|{int(t.high_voltage_pass)}|新增第{int(t.historical_batch)}批|{int(a.rank_low.min())}～{int(a.rank_high.max())}|{int(a.static_batch_low.min())}～{int(a.static_batch_high.max())}|{int(g.rank_low)}|")
    budgets=[]
    for k in [4,8,12,24,28,44,80,120]:
        vals=[]
        for model in ['TabPFN','GPR']:
            a=curve[(curve.model==model)&(curve.additional_budget==k)]
            vals.extend([f"{a.public_top3_hits.min()}～{a.public_top3_hits.max()}/3",f"{a.public7_hits.min()}～{a.public7_hits.max()}/7",f"{a.best_public_score_lower_bound.min():.3f}～{a.best_public_score_lower_bound.max():.3f}"])
        budgets.append('|'+str(k)+'|'+'|'.join(vals)+'|')
    means=[]
    for model in ['TabPFN','GPR']:
        a=summaries[summaries.model==model]
        means.append(f"|{model}|{a.public_top3_mean_rank_low.min():.1f}～{a.public_top3_mean_rank_high.max():.1f}|{a.public7_mean_rank_low.min():.1f}～{a.public7_mean_rank_high.max():.1f}|{a.all_public_top3_covered_by_rank.min()}～{a.all_public_top3_covered_by_rank.max()}|")
    narrative="""# ionic_polymer：TabPFN完整排名融合与原论文对照

当前采用 **EI、PI、UCB完整排名等权融合**作为主要探索方案。输入与预测模型不变：初始7条实测，504编码候选中其余497条参与排序。六个种子仅表示推理变化，不是六次独立闭环实验。

## 结论先看

**优势集中于最终配方1415的优先排序，尚不是对所有优良配方的普遍优势。** 它从原UCB的32～37名提前到21～27名，固定GPR完整融合为44名。另两个公开后续高分配方1315、1314则分别落在107～117、76～84名，明显晚于GPR的4、3名。不能只报告1415，而忽略这些不利结果。

首批是否命中不再作为成败判据：原论文也在新增第2批才测到1415。重点报告各配方排名、预算覆盖、多个高分配方平均排名及覆盖成本。真实BO后续排名会更新，本次只有初始模型的静态顺序。

## 源代码依据与我们做的修改

已经通过GitHub连接器直接读取作者固定提交`d2c5476e3a9bd54bc139ed59830aac461ee50770`，没有仅凭论文推测规则，也没有把作者代码复制到项目：

|作者源码|核实内容|
|---|---|
|[post_process.py](https://github.com/WangGroupFDU/Bayes_optimization/blob/d2c5476e3a9bd54bc139ed59830aac461ee50770/utils/post_process.py)|EI/PI/UCB各取前4；加上4次Thompson提名；同配方的名次票分累加|
|[config.json](https://github.com/WangGroupFDU/Bayes_optimization/blob/d2c5476e3a9bd54bc139ed59830aac461ee50770/config.json)|每份名单按4、3、2、1赋票分，不是给四种采集函数各设4、3、2、1权重|
|[bayes_opt_pure_math.py](https://github.com/WangGroupFDU/Bayes_optimization/blob/d2c5476e3a9bd54bc139ed59830aac461ee50770/utils/bayes_opt_pure_math.py)|高斯EI、PI，以及均值＋1.96倍标准差的UCB公式|
|[bayes_opt_gpy.py](https://github.com/WangGroupFDU/Bayes_optimization/blob/d2c5476e3a9bd54bc139ed59830aac461ee50770/utils/bayes_opt_gpy.py)|Thompson使用相关GP后验函数抽样；每次取最大值，作者种子410|

**我们保留“综合多个采集标准”的思路，扩展为全列表名次融合，而非宣称原样复现作者投票。** 对每项采集函数按分数降序排名r，转换为(497−r)/496，再将EI、PI、UCB三项等权平均。分数越大越优先。该分数是选点优先级，不是配方实测综合分数。融合分数相同保留名次区间；CSV展示顺序仅用编号确定。

TabPFN的EI/PI使用完整预测分布；GPR使用高斯公式。主融合不含Thompson，避免把TabPFN独立边际抽样当作GP联合抽样。作者式投票、单独EI等对照仍保留于[上层完整实验报告](../README.md)。

## 哪些优良配方排得靠前

按公开后续7条中的实测综合参考分数排序。该参考分数将论文14条的三个结果各自标准化后相加，仅供事后评价，不参与训练或调权。“前三个高分配方”仅指这7条公开后续数据的前三，不代表504空间全局前三。高压通过是0/1。

|配方|实测参考分数|电导率mS/cm|容量mAh/g|高压通过|论文首次实测|TabPFN融合排名|对应静态新增批次²|GPR融合排名|
|---|---:|---:|---:|---:|---|---:|---:|---:|
"""+'\n'.join(rows)+"""

² 静态新增批次＝向上取整（排名÷4），从初始7条之后计数，不更新模型。1415对应第6～7批，1314对应第19～21批，1315对应第27～30批；原论文是真实反馈更新后的新增第1/2批，二者不属于相同闭环条件，不能直接计算优化加速比。无须测完全部504条；真实闭环只需要每批所选配方的测量反馈。

## 多目标覆盖与预算指标

|模型|公开高分前三平均排名|公开后续7条平均排名|覆盖高分前三需要取到的名次上界|
|---|---:|---:|---:|
"""+'\n'.join(means)+"""

以下预算指初始7条之外的新推荐数。同一个预算下比较两种初始模型的静态顺序；区间来自六个TabPFN种子及融合平局按编号展示的结果，GPR无种子区间。高分前三为1415、1315、1314。

|新增推荐预算|TabPFN覆盖高分前三|TabPFN覆盖公开7条|TabPFN已知最好分数下界|GPR覆盖高分前三|GPR覆盖公开7条|GPR已知最好分数下界|
|---:|---:|---:|---:|---:|---:|---:|
"""+'\n'.join(budgets)+"""

**参考分数下界不能当成全部推荐的真实最好性能。** 未公开测量的配方不填0、不替代为模型预测。另导出已有测量支持的最高电导率、最高高压通过配方容量及测量覆盖数，见预算曲线CSV。这些指标也受反馈缺失影响。

## 与原论文真实进度怎么对照

原论文使用真实实验反馈更新模型，公开轨迹为初始7→11→14条。已知最好参考分数依次为1.473→2.050→2.474；公开后续高分前三的覆盖依次为0/3→2/3→3/3。1415在新增第2批测到。

我们的完整融合在初始7条上就把1415排在21～27名，是可以报告的排序信号；但1314、1315明显更晚，且尚无TabPFN自行选点后的更新轨迹。因此不能说整体闭环效率已接近或超过原论文。GPR静态对照隔离了初始代理/融合差异；论文真实轨迹反映反馈更新后的实际进展，必须分列。

![多个配方排名与预算曲线](Output/figures/Fig11_ranking_profile.png)

## 论文草稿写法建议

### 正文推荐表述

> 在仅使用七个初始实验观测的条件下，TabPFN结合多采集函数排名融合，将原研究最终推荐配方排列于497个待评估候选的第21～27位，表现出少样本条件下优先筛选潜在优良配方的能力。相较单独UCB及固定GPR对照，该配方的推荐优先级进一步提高。上述结果来自离线候选排序，尚需实验反馈验证其闭环优化效率。

### 可以突出强调的数据

|写作重点|本次证据|建议措辞|
|---|---|---|
|少样本候选筛选|初始仅7条观测，最终推荐配方位于剩余497个候选的前约4%～5%|具备少样本候选优先筛选的潜力|
|融合策略的改善|目标配方从单独UCB的32～37名提前至21～27名|多采集函数排名融合提高了该目标配方的推荐优先级|
|相同初始数据的模型对照|固定GPR完整排名融合将该配方排在44名|在该目标配方的排序上优于固定GPR对照|
|推理结果的一致性|六个推理种子的目标排名均落在21～27名|当前固定初始数据下，目标配方排名对推理种子的变化较小|

原论文最终推荐配方是有实测支持的参考目标，不是504空间已经穷尽验证的全局最优。与作者推荐不同不直接代表错误，也不能推断作者未测配方的性能；不对作者是否存在事后选择作无证据推测。

### “接近原论文效果”如何限定

建议具体写成“能够优先筛出原研究最终推荐配方”或“与原研究最终推荐目标具有一致的筛选方向”。目前不直接写“整体效果接近原论文”，因为这容易被理解为实验次数、真实性能或闭环效率已经相当。

完整对照结果继续保留：1314、1315等其他公开高分配方的排名尚不如GPR，优势不能推广为所有优良配方的普遍改善。正文突出上述目标配方的积极排序信号，同时在补充分析中完整呈现多配方排名与预算覆盖，并在正文或图注指向该分析，避免形成整体全面优于对照的印象。

### 结果段或图注应保留的简短限定

> 结果基于固定七条初始观测的离线排名；六个种子表示推理变化，不是独立优化实验。最终配方为原研究推荐目标，未声称其为全空间全局最优。

该融合方案是在已查看UCB结果后提出的探索性扩展，不把事后方法选择当作独立验证。不将排名改善表述为实验次数减少、实际性能提高或已完成闭环验证；也不将标准化综合分数差解释为性能提升百分比。后续可用作者历史7条/11条输入分别审计更新后的排名，但那仍是共享历史的推荐回放，不是TabPFN自主闭环实验。

## 文件与复现

- [各公开配方的逐种子排名与原始结果](Output/tables/ranking_profile_targets.csv)
- [各预算的覆盖、参考分数、电导率和可行容量下界](Output/tables/ranking_profile_budget_curves.csv)
- [多配方排名汇总](Output/tables/ranking_profile_summary.csv)
- [原论文真实7→11→14轨迹](Output/tables/ranking_profile_paper_trajectory.csv)
- [全部采集函数值及候选排名](Output/tables/acquisition_full_rankings.csv)

项目根目录运行`.venv/bin/python study/objective_search/ionic_polymer/compare_polymer_acquisitions.py`可离线重建本README和相关图表，不调用API。脚本生成结果时核对缓存哈希，数值测试覆盖EI与PI关系、尾部抽样及投票平局。
"""
    (ROOT/'README.md').write_text(narrative)
    parent=ROOT.parent/'README.md';text=parent.read_text()
    text+='\n详细的多配方排名、预算覆盖与原论文真实轮次比较，见[ionic_polymer子目录报告](ionic_polymer/README.md)。\n'
    parent.write_text(text)


if __name__=='__main__':run()
