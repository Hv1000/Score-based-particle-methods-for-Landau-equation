import numpy as np
from joblib import Parallel, delayed
from sklearn.neighbors import KernelDensity

def collision_kernel(v, i):
    v_diff = v[i,:]-v
    v_temp = v_diff[:,None,:]
    norm = np.linalg.norm(v_diff, ord=2, axis=1, keepdims=True)
    collision_kernel = 1/16*((norm**2)[:,:,None]*np.eye(v.shape[1]) - v_temp.transpose(0,2,1)*v_temp)
    return collision_kernel

def compute_velocity_field(v, score):
    def velocity_field_single(i):
        ck_i = collision_kernel(v, i)
        temp = np.squeeze(np.matmul((score[i,:]-score)[:,None,:], ck_i), axis=1)
        return -1*np.mean(temp, axis=0)
    velocity_field_list = Parallel(n_jobs=-1)(delayed(velocity_field_single)(i) for i in range(v.shape[0]))
    return np.array(velocity_field_list)

def compute_v(v, velocity_field, dt):
    return v + dt*velocity_field

def compute_logdet_field(v, score, score_jac):
    def logdet_field_single(i):
        ck_i = collision_kernel(v, i)
        term1 = np.sum(np.sum(ck_i * score_jac[i,:,:].T, axis=1), axis=1)
        term2 = 1/16*np.sum((v[i,:]-v)*(score[i,:]-score), axis=1) 
        return -1*np.mean(term1 - term2)
    logdet_field_list = Parallel(n_jobs=-1)(delayed(logdet_field_single)(i) for i in range(v.shape[0]))
    return np.array(logdet_field_list)

def compute_f(f, logdet_field, dt):
    return f / np.exp(dt*logdet_field)

def compute_f_kde(v, v_grid, eps):
    kde = KernelDensity(kernel='gaussian', bandwidth=eps).fit(v)
    return np.exp(kde.score_samples(v_grid))

def compute_energy(v):
    return np.sum(v**2) / v.shape[0]

def compute_entropy_decay_rate(velocity_field, score):
    return np.mean(np.sum(velocity_field * score, axis=1))


def score_exact(v, t):
    K = 1 - np.exp(-t/8) / 2
    norm = np.sum(v**2, axis=1, keepdims=True)
    return -1/K*v + (1-K)/((2*K-1)*K + (1-K)/2*norm)*v

def f_exact(v, t):
    K = 1 - np.exp(-t/8) / 2
    norm = np.sum(v**2, axis=1)
    return 1/(2*np.pi*K) * np.exp(-norm/(2*K)) * ((2*K-1)/K + (1-K)/(2*K**2)*norm)