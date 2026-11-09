"""Scientific safeguards: exact random reference, feedback and hidden-label boundary."""
import itertools
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
import replay_clio_search as bo

class ReplayChecks(unittest.TestCase):
 def test_random_against_all_permutations(self):
  for initial in [[0],[1],[3]]:
   y=np.array([1.,2.,3.,4.]);curve,costs=bo.random_reference(y,initial)
   paths=[np.r_[y[initial],y[list(p)]] for p in itertools.permutations([i for i in range(4) if i not in initial])]
   for row in curve.to_dict('records'):
    k=row['budget'];best=np.array([p[:k].max() for p in paths])
    self.assertAlmostEqual(row['best_mS_cm'],best.mean())
    for r in bo.RATIOS:self.assertAlmostEqual(row[f'hit_{r}'],np.mean(best>=r*y.max()-1e-12))
   for r in bo.RATIOS:
    empirical=np.mean([bo.hit_cost(p,r*y.max(),len(initial)) for p in paths])
    self.assertAlmostEqual(costs[r],empirical)
 def test_initial_and_new_hit_count(self):
  self.assertEqual(bo.hit_cost([9,1,2,3,4,10],8),5)
  self.assertEqual(bo.hit_cost([1,2,3,4,5,9],8),6)
  self.assertIsNone(bo.hit_cost([1,2,3],8))
 def test_contract_refuses_hidden_targets(self):
  observed=pd.DataFrame([['F0',.2,.3,0.,4.]],columns=['formulation_id']+bo.FEATURES+['conductivity'])
  query=pd.DataFrame([['F1',.3,.4,1.]],columns=['formulation_id']+bo.FEATURES)
  c=bo.prediction_contract(observed,query)
  self.assertEqual(c['y_train'],[4.]);self.assertNotIn('y_query',c)
  with self.assertRaises(ValueError):bo.prediction_contract(observed,query.assign(conductivity=900.))
  with self.assertRaises(ValueError):bo.prediction_contract(observed,query.assign(formulation_id='F0'))
 def test_feedback_is_sequential(self):
  d=pd.DataFrame(dict(formulation_id=[f'F{i}' for i in range(7)],salt_molality=np.arange(7),EC_mass_fraction=.4,
      DMC_cosolvent_ratio=1.,conductivity=[1.,2.,3.,4.,5.,6.,7.]))
  contracts=[]
  def predictor(c):
   contracts.append(c);x=np.array(c['X_query'])[:,0]
   # Choose F5 before F6 to require a real second update.
   score=-x
   return pd.DataFrame(dict(formulation_id=c['query_ids'],mean=score,std=np.zeros(len(x)),q05=score,q95=score)),{}
  with tempfile.TemporaryDirectory() as tmp:
   out=Path(tmp);(out/'tables').mkdir()
   trace,_=bo.replay(d,list(range(5)),bo.METHODS[1],42,out,'test',predictor)
   self.assertEqual([len(c['y_train']) for c in contracts],[5,6])
   self.assertEqual(contracts[1]['y_train'][-1],6.)
   self.assertNotIn('F6',contracts[1]['observed_ids'])
   self.assertEqual(trace.formulation_id.tolist(),[f'F{i}' for i in range(7)])
   bo.audit_trace(trace,d,list(range(5)),complete=True)
 def test_cache_keys_depend_on_observed_labels_and_order(self):
  c=dict(X_train=[[1],[2]],y_train=[3,4],query_ids=['F2'])
  self.assertNotEqual(bo.digest(c),bo.digest(c|{'y_train':[3,9]}))
  self.assertNotEqual(bo.digest(c),bo.digest(c|{'X_train':[[2],[1]]}))
 def test_equal_scores_tie_by_id(self):
  pred=pd.DataFrame(dict(formulation_id=['F2','F1'],mean=[1.,1.],std=[.5,.5]))
  self.assertEqual(bo.choose(pred).formulation_id,'F1')

if __name__=='__main__':unittest.main()
