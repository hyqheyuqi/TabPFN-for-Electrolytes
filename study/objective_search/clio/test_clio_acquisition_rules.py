"""Independent EI integration and strategy/feedback safeguards, without API calls."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.stats import norm
import compare_clio_acquisition_rules as opt

class AcquisitionChecks(unittest.TestCase):
 def test_ei_matches_independent_quadrature_including_tails(self):
  borders=np.array([-2.,-1.,0.,1.,2.]);p=np.array([.1,.3,.4,.2]);scale=1/norm.ppf(.75)
  def density(x):
   if x<-1:return p[0]*2*norm.pdf((-1-x)/scale)/scale
   if x>1:return p[-1]*2*norm.pdf((x-1)/scale)/scale
   if x<0:return p[1]
   return p[2]
  for threshold in [-4.,-.5,0.,1.5,4.]:
   numerical=sum(quad(lambda x:max(x-threshold,0)*density(x),a,b,epsabs=1e-10)[0]
     for a,b in [(-np.inf,-1),(-1,0),(0,1),(1,np.inf)])
   actual=opt.bar_ei(np.log(p)[None,:],borders,threshold)[0]
   self.assertAlmostEqual(actual,numerical,places=7)
 def test_uniform_bucket_ei(self):
  with np.errstate(divide='ignore'):logits=np.log([[0.,1.,0.]])
  actual=opt.bar_ei(logits,np.array([-1.,0.,1.,2.]),.25)[0]
  self.assertAlmostEqual(actual,.5*.75**2)
 def test_gp_zero_variance_ei(self):
  pred=pd.DataFrame(dict(mean=[1.,3.],std=[0.,0.]))
  self.assertTrue(np.allclose(opt.acquisition(pred,'EI',2.),[0.,1.]))
 def test_cache_key_ignores_only_legacy_acquisition_coefficient(self):
  c={'X_train':[[1]],'y_train':[2],'query_ids':['F1'],'kappa':1.96}
  self.assertEqual(opt.key(c),opt.key(c|{'kappa':.5}))
  self.assertNotEqual(opt.key(c),opt.key(c|{'y_train':[9]}))
 def test_real_feedback_changes_context(self):
  d=pd.DataFrame(dict(formulation_id=[f'F{i}' for i in range(7)],salt_molality=np.arange(7),
        EC_mass_fraction=.4,DMC_cosolvent_ratio=1.,conductivity=[1.,2.,3.,4.,5.,6.,7.]))
  class FakeCache:
   def __init__(self):self.contracts=[]
   def load(self,c):
    self.contracts.append(c);mean=-np.array(c['X_query'])[:,0].astype(float)
    pred=pd.DataFrame(dict(formulation_id=c['query_ids'],mean=mean,std=np.zeros(len(mean)),q05=mean,q95=mean))
    return pred,{},dict(key=opt.key(c),cache_hit=True,metadata_path='mock')
  cache=FakeCache()
  with tempfile.TemporaryDirectory() as tmp:
   out=Path(tmp);(out/'traces').mkdir()
   tr=opt.replay(d,list(range(5)),'development','TabPFN','Mean',42,out,cache)
   self.assertEqual([len(c['y_train']) for c in cache.contracts],[5,6])
   self.assertEqual(cache.contracts[1]['y_train'][-1],6.)
   self.assertNotIn('y_query',cache.contracts[1])
   self.assertEqual(tr.formulation_id.tolist(),[f'F{i}' for i in range(7)])

if __name__=='__main__':unittest.main()
