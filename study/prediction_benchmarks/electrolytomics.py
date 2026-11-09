"""electrolytomics EDB-1: author numeric descriptors, log10(mS/cm), stratified 80/10/10."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent / 'electrolytomics';sys.path.insert(0,str(HERE.parent))
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from benchmark_utils import run,sha
BASE=HERE.parents[1]/'datasets/electrolytomics/datasets'
RAW=BASE/'raw/conductivity/EDB-1_conductivity.csv';FEATURE_FILE=BASE/'featurized/conductivity/all_qsar_fgrp_cond.csv'

def prepare():
 raw=pd.read_csv(RAW);feat=pd.read_csv(FEATURE_FILE)
 features=[c for c in feat.select_dtypes('number') if c!='conductivity_log']
 if len(raw)!=len(feat) or not np.allclose(raw.conductivity_log,feat.conductivity_log):raise ValueError('Feature/label row alignment invalid')
 if not np.allclose(np.log10(raw.conductivity),raw.conductivity_log):raise ValueError('Target log units invalid')
 keys=[c for c in raw if 'conductivity' not in c]
 keyframe=raw[keys].fillna('');ids=pd.factorize(pd.MultiIndex.from_frame(keyframe),sort=True)[0]
 work=feat[features].copy();work['condition_group']=ids;work['raw_row_id']=np.arange(len(work));work['y']=raw.conductivity_log
 # An exact physical condition must have the same authored descriptors.
 if work.groupby('condition_group')[features].nunique().max().max()>1:raise ValueError('Same condition has inconsistent descriptors')
 data=work.groupby('condition_group',sort=True)[features+['y']].mean().reset_index()
 data['row_id']=np.arange(len(data));data['group']='EDB1';data['conductivity_mS_cm']=10**data.y
 bins=pd.cut(data.y,[-7,-4,-3,-2,-1,0,1,2],labels=False)
 if bins.isna().any():raise ValueError('Labels outside fixed author stratification bins')
 rest,test=train_test_split(np.arange(len(data)),test_size=.1,random_state=0,stratify=bins)
 train,val=train_test_split(rest,test_size=len(test)/len(rest),random_state=0,stratify=bins.iloc[rest])
 data['split']='train';data.loc[val,'split']='validation';data.loc[test,'split']='test'
 mapping=work[['raw_row_id','condition_group']].merge(data[['condition_group','row_id','split']],on='condition_group',validate='many_to_one')
 out=HERE/'Output/data';out.mkdir(parents=True,exist_ok=True);mapping.to_csv(out/'raw_to_condition.csv',index=False)
 audit={'raw_rows':len(raw),'conditions':len(data),'duplicate_measurements_aggregated':len(raw)-len(data),'feature_count':len(features),
  'aggregation':'mean log10 target within exact raw composition, salt, concentration and temperature; identical descriptors checked',
  'split':'fixed author bins [-7,-4,-3,-2,-1,0,1,2], random_state=0; approximately 80/10/10',
  'label_in_features':False,'log_label_verified':True,'raw_sha256':sha(RAW),'feature_sha256':sha(FEATURE_FILE),
  'comparison_limit':'Exact duplicates grouped before split, unlike direct author row split; original exact test IDs unavailable. OOD171 not used this round.'}
 return data,features,audit
SPEC={'z':'electrolytomics','title':'Electrolytomics，Kumar等，2025','doi':'https://doi.org/10.1021/acs.chemmater.4c03196','script_name':'electrolytomics.py','source_path':Path(__file__),'mode':'log10','target_label':'log10(mS/cm)',
 'notes':['- 使用作者现成数值描述符（103列），剔除目标conductivity_log和SMILES文本，不使用预测值、OOD标签或其他性质。',
 '- 原始10,196行，精确相同的组成/盐/浓度/温度先合并，取log10标签均值，避免重复条件跨集合。特征一致性已核验。',
 '- 按作者固定log标签区间[-7,-4,-3,-2,-1,0,1,2]分层，random_state=0，约8:1:1；验证只报告，不用于TabPFN调参。',
 '- 保留全部正电导率范围，不套用Z45的0.1下限。属于随机条件插值；同一配方的其他温度可跨集合。'],
 'comparison':['原文Fig.2d中最佳Chemprop（分层划分）的测试log10 MAE=0.09、RMSE=0.18、R²=0.94。本轮应比较target尺度指标，不把线性mS/cm的误差与该表混用。',
 '本轮使用作者浅层模型描述符，而非Chemprop图网络；提前聚合247条重复条件，与原文样本划分不同，因此是背景核对，不是严格同条件优劣判断。']}
if __name__=='__main__':
 d,f,a=prepare();run(HERE,d,f,a,[RAW,FEATURE_FILE],SPEC)
