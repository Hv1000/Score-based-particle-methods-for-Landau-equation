import numpy as np
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Particle():
    def __init__(self, v, score, score_jac, dt, d, C):
        self.v = torch.tensor(v).float().to(device)
        self.dt = torch.tensor(dt).float().to(device)
        self.score = score
        self.score_jac = score_jac

        self.d = d
        self.C = C #1/16


    def vel_field(self, v, score):
        v_ij = v-self.v
        score_ij = score-self.score

        ## collision kernel
        v_ij_extend = v_ij[:,None,:]
        norm = torch.linalg.vector_norm(v_ij, ord=2, dim=1, keepdim=True)
        A = self.C * ((norm**2)[:,:,None]*torch.eye(self.d).to(device) - v_ij_extend.permute(0,2,1)*v_ij_extend)

        ## velocity field
        AS = torch.squeeze(torch.matmul(score_ij[:,None,:], A))
        vel_field = -1*torch.mean(AS, axis=0)
        return vel_field
    
    def vel_and_entropy(self):
        vmap_func = torch.vmap(self.vel_field, in_dims=0, out_dims=0)
        vel_field = vmap_func(self.v, self.score)
        v_new = self.v + self.dt*vel_field
        entropy_decay_rate = torch.mean(torch.sum(vel_field * self.score, dim=1))
        return v_new.detach().cpu().numpy(), entropy_decay_rate.detach().cpu().numpy()

    def logdet_field(self, v, score, score_jac):
        v_ij = v-self.v
        score_ij = score-self.score

        ## collision kernel
        v_ij_extend = v_ij[:,None,:]
        norm = torch.linalg.vector_norm(v_ij, ord=2, dim=1, keepdim=True)
        A = self.C * ((norm**2)[:,:,None]*torch.eye(self.d).to(device) - v_ij_extend.permute(0,2,1)*v_ij_extend)

        ## logdet field
        AS = torch.sum(torch.sum(A * score_jac, dim=1), dim=1)
        VS = torch.sum(v_ij * score_ij, dim=1)
        logdet_field = -1*torch.mean(AS - (self.d-1)*self.C*VS)
        return logdet_field

    def density(self, f):
        f = torch.tensor(f).float().to(device)
        vmap_func = torch.vmap(self.logdet_field, in_dims=0, out_dims=0)
        logdet_field = vmap_func(self.v, self.score, self.score_jac)
        f_new = f / torch.exp(self.dt*logdet_field)
        return f_new.detach().cpu().numpy()


def energy(v):
    return np.sum(v**2) / v.shape[0]

def score_ext(v, t):
    K = 1 - np.exp(-t/8) / 2
    norm = np.sum(v**2, axis=1, keepdims=True)
    return -1/K*v + (1-K)/((2*K-1)*K + (1-K)/2*norm)*v

def f_ext(v, t):
    K = 1 - np.exp(-t/8) / 2
    norm = np.sum(v**2, axis=1)
    return 1/(2*np.pi*K) * np.exp(-norm/(2*K)) * ((2*K-1)/K + (1-K)/(2*K**2)*norm)