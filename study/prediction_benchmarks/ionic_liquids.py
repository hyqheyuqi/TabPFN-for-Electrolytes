"""ionic_liquids: two separate IL conductivity datasets, authored descriptors plus T."""
from pathlib import Path
import sys,json
HERE=Path(__file__).resolve().parent / 'ionic_liquids';sys.path.insert(0,str(HERE.parent))
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from benchmark_utils import run,sha
BASE=HERE.parents[1]/'datasets/ionic_liquids'
FILES=[BASE/'nh_data.json',BASE/'ilthermo_data.json']

def prepare():
 parts=[];audit={'datasets':{},'unit_conversion':'JSON S/m multiplied by10 -> mS/cm','model_target':'log10(mS/cm)',
  'split':'project-defined random exact species-temperature condition 60/20/20, seed42; species can cross splits',
  'comparison_limit':'Original split IDs/precise rule unavailable; larger downloaded file3718 vs paper3933 measurements; not exact reproduction'}
 feature_names=None
 for name,file in zip(['nh','ilthermo'],FILES):
  source=json.loads(file.read_text());rows=[]
  for species,entry in source.items():
   names=entry['descriptor_names']
   if feature_names is None:feature_names=list(names)+['temperature_K']
   if list(names)+['temperature_K']!=feature_names:raise ValueError('Inconsistent descriptor order')
   if len(entry['temperature'])!=len(entry['conductivity']):raise ValueError('Temperature/label lengths differ')
   for t,k in zip(entry['temperature'],entry['conductivity']):
    rows.append({'species':species,'temperature_K':t,'conductivity_mS_cm':10*k,**dict(zip(names,entry['descriptors']))})
  raw=pd.DataFrame(rows)
  if not np.isfinite(raw[feature_names+['conductivity_mS_cm']]).all().all():raise ValueError('Nonfinite data')
  valid=raw[raw.conductivity_mS_cm.gt(0)].copy()
  d=valid.groupby(['species']+feature_names,sort=True,dropna=False).conductivity_mS_cm.mean().reset_index()
  d['group']=name;d['y']=np.log10(d.conductivity_mS_cm)
  dev,test=train_test_split(np.arange(len(d)),test_size=.2,random_state=42)
  train,val=train_test_split(dev,test_size=.25,random_state=42)
  d['split']='train';d.loc[val,'split']='validation';d.loc[test,'split']='test'
  parts.append(d)
  audit['datasets'][name]={'raw_rows':len(raw),'positive_rows':len(valid),'conditions':len(d),'ILs':len(source),'sha256':sha(file)}
 data=pd.concat(parts,ignore_index=True);data['row_id']=np.arange(len(data));return data,feature_names,audit
SPEC={'z':'ionic_liquids','title':'Ionic liquid conductivity models by symbolic regression，2026','doi':'https://doi.org/10.1039/D5CP04143K','script_name':'ionic_liquids.py','source_path':Path(__file__),'mode':'log10','target_label':'log10(mS/cm)',
 'notes':['- nh自测集（22种IL）与ILThermo文献集（338种IL）完全分开训练，输入为作者16项分子描述符和温度。JSON的S/m转为mS/cm乘10。',
 '- 重复的同一IL/温度条件在拆分前取线性电导率均值；所有正值保留，不套用Z45阈值。',
 '- 本项目采用测量条件6:2:2、种子42；同一IL的不同温度可跨集合。这评价温度插值，不是对未见离子液体的外推。',
 '- 原文采用符号回归，loss在log尺度计算；TabPFN本轮直接拟合log10电导率，以避免负的物理预测值。',
 '- 原文大集报告3933次测量，当前发布JSON实际3718次；没有强行凑齐，不能视为完全相同数据版本。'],
 'comparison':['原文小集训练/验证R²分别0.97/0.94，大集分别0.76/0.73。应对照本轮各子集验证集R²（线性电导率），而不是log target的R²。',
 '原始划分编号与精确划分规则尚未核实，本轮划分为项目自定义；数字并列仅作背景核对，不宣称严格复现或优于原文。']}
if __name__=='__main__':
 d,f,a=prepare();run(HERE,d,f,a,FILES,SPEC)
