"""Offline mathematical checks of acquisition and sparse-vote semantics."""
import unittest
import numpy as np
import torch
from scipy.special import softmax
from compare_polymer_acquisitions import top4_vote, rank_bounds, marginal_samples, exact_ei
from tabpfn.architectures.shared.bar_distribution import FullSupportBarDistribution


class AcquisitionChecks(unittest.TestCase):
    def test_sparse_voting_preserves_zero_vote_rank_interval(self):
        ids=np.array(['a','b','c','d','e','f'])
        votes=top4_vote([np.arange(6.)],ids)
        np.testing.assert_equal(votes,[0,0,1,2,3,4])
        self.assertEqual(rank_bounds(votes,0),(5,6))
        self.assertEqual(rank_bounds(votes,5),(1,1))

    def test_votes_add_by_candidate_without_scale_dependence(self):
        ids=np.array(['a','b','c','d','e'])
        x=np.array([1.,4,2,5,3])
        np.testing.assert_equal(top4_vote([x,x*100],ids),2*top4_vote([x],ids))
        self.assertEqual(top4_vote([x,x*100],ids).sum(),20)

    def test_ei_derivative_matches_improvement_probability(self):
        raw = {'borders': np.array([-2., -1., 0., 1., 2.]),
               'logits': np.array([[-4., 0., 2., -4.], [-4., 2., 0., -4.], [-4., 1., 1., -4.]])}
        logits=torch.as_tensor(raw['logits'][:3],dtype=torch.float64)
        c=FullSupportBarDistribution(torch.as_tensor(raw['borders'],dtype=torch.float64))
        t=torch.full((3,),0.2,dtype=torch.float64);h=1e-4
        derivative=(exact_ei(c,logits,t+h)-exact_ei(c,logits,t-h))/(2*h)
        np.testing.assert_allclose(-derivative.numpy(),c.pi(logits,t).numpy(),atol=1e-6)
        self.assertTrue(torch.all(exact_ei(c,logits,t)>=torch.clamp(c.mean(logits)-t,min=0)))

    def test_marginal_sampler_matches_native_quantiles_including_tails(self):
        raw = {'borders': np.array([-2., -1., 0., 1., 2.]),
               'logits': np.array([[-4., 0., 2., -4.], [-4., 2., 0., -4.], [-4., 1., 1., -4.]])}
        logits=torch.as_tensor(raw['logits'][:2],dtype=torch.float64)
        c=FullSupportBarDistribution(torch.as_tensor(raw['borders'],dtype=torch.float64))
        u=np.array([1e-10,.05,.5,.95,1-1e-10])
        class FixedRNG:
            def random(self,shape):return np.broadcast_to(u,shape).copy()
        sample=marginal_samples(softmax(raw['logits'][:2],axis=1),raw['borders'],FixedRNG(),draws=len(u))
        expected=np.column_stack([c.icdf(logits,float(p)).numpy() for p in u])
        # Native icdf inherits finite bars; validate its interior only.
        np.testing.assert_allclose(sample[:,1:-1],expected[:,1:-1],rtol=2e-4,atol=2e-4)
        from scipy.special import ndtr, ndtri
        prob=softmax(raw['logits'][:2],axis=1);b=raw['borders']
        left_cdf=2*(1-ndtr((b[1]-sample[:,0])/((b[1]-b[0])/ndtri(.75))))*prob[:,0]
        right_cdf=1-prob[:,-1]+prob[:,-1]*(2*ndtr((sample[:,-1]-b[-2])/((b[-1]-b[-2])/ndtri(.75)))-1)
        np.testing.assert_allclose(left_cdf,u[0],atol=1e-15)
        np.testing.assert_allclose(right_cdf,u[-1],atol=1e-15)


if __name__=='__main__':unittest.main()
