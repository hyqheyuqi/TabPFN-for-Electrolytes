"""aqueous aqueous mixtures: reconstruct salt molality from feeder mass balance."""
from pathlib import Path
import sys,json
HERE=Path(__file__).resolve().parent / 'aqueous';sys.path.insert(0,str(HERE.parent))
import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.model_selection import GroupShuffleSplit
from benchmark_utils import run,sha
SOURCE=HERE.parents[1]/'datasets/aqueous/otto_aqueous.json'
FORMULAS={'LiNO3':{'Li':1,'N':1,'O':3},'LiClO4':{'Li':1,'Cl':1,'O':4},'Li2SO4':{'Li':2,'S':1,'O':4},
 'NaNO3':{'Na':1,'N':1,'O':3},'NaClO4':{'Na':1,'Cl':1,'O':4},'Na2SO4':{'Na':2,'S':1,'O':4},'NaBr':{'Na':1,'Br':1}}
PT=Chem.GetPeriodicTable();MW={s:sum(PT.GetAtomicWeight(e)*n for e,n in f.items()) for s,f in FORMULAS.items()}
FEATURES=[f'{s}_mol_kg_water' for s in FORMULAS]+['temperature_C']

def reconstruct(record):
 e=record['electrolyte'];moles={s:0. for s in FORMULAS};water_g=0.;volume=0.
 for stock,v in e['volumes'].items():
  density=e['source densities'][stock];m=e['source molalities'][stock]
  if v<0 or density<=0 or m<0:raise ValueError('Invalid feeder data')
  if stock!='water' and stock not in MW:raise ValueError('Unknown salt')
  grams=v*density;solvent_g=grams if stock=='water' else grams/(1+m*MW[stock]/1000)
  water_g+=solvent_g;volume+=v
  if stock!='water':moles[stock]+=m*solvent_g/1000
 if water_g<=0 or not np.isclose(volume,7.):raise ValueError('Invalid solvent mass/volume')
 features={f'{s}_mol_kg_water':n/(water_g/1000) for s,n in moles.items()}
 return features,water_g

def prepare():
 raw=json.loads(SOURCE.read_text());rows=[]
 for i,e in enumerate(raw):
  f,water=reconstruct(e)
  rows.append({'raw_row_id':i,'GUID':e['GUID'],'conductivity_mS_cm':e['conductivity'],'temperature_C':e['temperature'],
   'water_mass_g':water,**f})
 source=pd.DataFrame(rows)
 if not source.GUID.is_unique:raise ValueError('Duplicate GUIDs')
 if not np.isfinite(source[FEATURES+['conductivity_mS_cm']]).all().all() or not source.conductivity_mS_cm.gt(0).all():raise ValueError('Invalid data')
 # Same composition always on the same side, even when measured at different T.
 keys=[f'{s}_mol_kg_water' for s in FORMULAS]
 recipe=pd.MultiIndex.from_frame(source[keys].round(12));source['recipe_id']=pd.factorize(recipe,sort=True)[0]
 data=source.copy();data['row_id']=np.arange(len(data));data['group']='aqueous';data['y']=data.conductivity_mS_cm
 gss=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42);dev,test=next(gss.split(data,groups=data.recipe_id))
 train_rel,val_rel=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=42).split(data.iloc[dev],groups=data.iloc[dev].recipe_id))
 train,val=dev[train_rel],dev[val_rel];data['split']='train';data.loc[val,'split']='validation';data.loc[test,'split']='test'
 if data.groupby('recipe_id')['split'].nunique().max()!=1:raise ValueError('Repeated recipe leakage')
 audit={'rows':len(data),'unique_recipes':int(data.recipe_id.nunique()),'source_sha256':sha(SOURCE),'salt_molar_masses_g_mol':MW,
  'conversion':'solution mass=volume(mL)*density(g/mL); water mass=solution mass/(1+stock molality*salt MW/1000); moles=stock molality*water_kg; final molality=sum salt moles/total water_kg',
  'split':'project-defined grouped recipe 60/20/20, seed42; row ratios may differ; recipe keys rounded12 only for grouping',
  'features':FEATURES,'excluded_predictors':['pH','electrochemistry','timestamp','RUN_ID','GUID'],
  'conductivity_unit':'mS/cm, as robot data convention','no_label_threshold':True,'no_original_prediction_benchmark':True}
 return data,FEATURES,audit
SPEC={'z':'aqueous','title':'Autonomous Discovery of Battery Electrolytes，Dave等，2020','doi':'https://doi.org/10.1016/j.xcrp.2020.100264','script_name':'aqueous.py','source_path':Path(__file__),'mode':'linear','target_label':'线性mS/cm',
 'notes':['- 使用251条机器人实测水系电解液记录，通过原液体积、密度、molality和盐摩尔质量进行质量守恒换算。',
 '- 每份原液先求溶液质量，再由m=M_salt/mol_water关系拆分水质量和盐摩尔数；最终每种盐的molality以混合物总水质量为分母。不是简单体积加权浓度。',
 '- 输入是7种盐的最终mol/kg水和温度，共8列；pH、电化学响应、时间戳和实验编号均不作为输入。',
 '- 自定义配方分组6:2:2（种子42）；同配方的重复测量及不同温度记录归在同一侧。此比例按配方组划分，实际行数可能不同。',
 '- 全部正电导率保留，无按误差删样本，无Z45的范围下限。'],
 'comparison':['aqueous原文主要研究自主实验优化，不提供与本轮相同的静态电导率留出预测误差表；本轮仅报告项目常见预测指标，不把优化效果或仪器重复性当成预测精度基线。']}
if __name__=='__main__':
 d,f,a=prepare();run(HERE,d,f,a,[SOURCE],SPEC)
