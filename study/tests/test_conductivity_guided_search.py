import unittest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from analysis.conductivity_guided_search import measure_replay, replay, METHODS



class ConductivityGuidedSearchTests(unittest.TestCase):

    def test_batch_hit_cost_and_auc(self):
        y=np.arange(14,dtype=float)
        order=[0,1,2,3,4,5,13,6,7,8,9,10,11,12]
        rounds=[0]*6+[1]*6+[2]*2
        s,t=measure_replay(order,rounds,y,1,'UCB',25)
        self.assertEqual(s['evaluations_to_optimum'],7)
        self.assertEqual(s['batch_cost_to_optimum'],12)
        self.assertAlmostEqual(s['normalized_auc'],np.maximum.accumulate(y[order]).mean()/13)
        self.assertTrue(t.hit.iloc[6:].all())

    def test_tied_optima_and_initial_hit(self):
        y=np.arange(10,dtype=float);y[0]=9
        s,_=measure_replay(list(range(10)),[0]*6+[1]*4,y,1,'EI',25)
        self.assertEqual(s['evaluations_to_optimum'],1)
        self.assertEqual(s['batch_cost_to_optimum'],6)

    def test_no_duplicate_and_matched_initialization(self):
        x=StandardScaler().fit_transform(np.column_stack([np.arange(14),np.sin(np.arange(14))]))
        y=np.cos(np.arange(14))
        first=[]
        for m in METHODS:
            a,t=replay(x,y,9,m,25)
            self.assertEqual(t.candidate_id.nunique(),14)
            first.append(a['initial_candidates'])
            b,u=replay(x,y,9,m,25)
            pd.testing.assert_frame_equal(t,u)
        self.assertEqual(len(set(first)),1)

    def test_batch_does_not_read_unobserved_labels(self):
        x=StandardScaler().fit_transform(np.column_stack([np.arange(14),np.sin(np.arange(14))]))
        y=np.arange(14,dtype=float)
        initial=np.random.RandomState(9).choice(14,6,replace=False)
        altered=y.copy();altered[np.setdiff1d(np.arange(14),initial)]+=1000
        for m in METHODS:
            _,a=replay(x,y,9,m,25);_,b=replay(x,altered,9,m,25)
            self.assertEqual(a.candidate_id.iloc[:12].tolist(),b.candidate_id.iloc[:12].tolist())


if __name__=='__main__':unittest.main()
