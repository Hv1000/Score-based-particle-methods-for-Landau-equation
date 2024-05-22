import numpy as np

class Sampling():
    def __init__(self, nex, d, t, seed):
        self.nex = nex
        self.d = d
        self.t = t
        if seed==None:
            np.random.seed(seed)
        # 10000: 23
        #37: 5000

    def target_distribution(self, sample):
        vx = sample[0]
        vy = sample[1]
        K = 1 - np.exp(-self.t/8) / 2
        f = 1/(2*np.pi*K) * np.exp(-(vx**2 + vy**2)/(2*K)) * ((2*K-1)/K + (1-K)/(2*K**2)*(vx**2 + vy**2))
        return f

    def proposal_sampling(self):
        m = np.random.multivariate_normal(np.zeros(self.d), np.eye(self.d))
        return m
    
    def proposal_distribution(self, sample):
        vx = sample[0]
        vy = sample[1]
        g = 1/(np.pi) * np.exp(-(vx**2+vy**2)/2)
        return g

    def rejection_sampling(self):
        samples = []
        while len(samples) < self.nex:
            proposed_sample = self.proposal_sampling()
            ratio = self.target_distribution(proposed_sample) / (2*self.proposal_distribution(proposed_sample))
            u = np.random.rand()
            if u < ratio:
                samples.append(proposed_sample)
        return samples