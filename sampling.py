import numpy as np

class Sampling():
    def __init__(self, nex, seed):
        self.nex = nex
        
        self.mean_prop = np.sqrt(3/2)
        self.std_prop = np.sqrt(0.5)
        self.M = 0.45

        np.random.seed(seed)

    def target_dist(self, r):
        if r > 0:
            return r**3 * np.exp(-r**2)
        else:
            return 0
        
    def prop_dist(self, r, mean, std):
        return np.exp(-(r-mean)**2 / (2*std**2))
    
    def prop_sampling(self, mean, std): 
        return np.random.normal(loc=mean, scale=std)

    def rejection_sampling(self, nex, mean, std, M):
        accepted_samples = []
        while len(accepted_samples) < nex:
            prop_sample = self.prop_sampling(mean, std)
            accept_prob = self.target_dist(prop_sample) / (M * self.prop_dist(prop_sample, mean, std))
            if np.random.rand() < accept_prob:
                accepted_samples.append(prop_sample)
        return np.array(accepted_samples)

    def sampling(self):
        n = int(self.nex/4)
        r = self.rejection_sampling(n, self.mean_prop, self.std_prop, self.M)
        r = r[:,None]
        
        theta = np.pi/2 * np.random.rand(n, 1) 

        v_x = r * np.cos(theta) 
        v_y = r * np.sin(theta)
        
        samples_1 = np.hstack((v_x, v_y))
        samples_2 = np.hstack((-v_x, v_y))
        samples_3 = np.hstack((-v_x, -v_y))
        samples_4 = np.hstack((v_x, -v_y))
        
        samples = np.vstack((samples_1, samples_2, samples_3, samples_4))
        return samples