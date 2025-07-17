import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import sampling as sp
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

'''fully connected neural network'''
class MLP(torch.nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.act = nn.SiLU()
        self.layers = nn.ModuleList([])
        self.layers.append(nn.Linear(dim_in, dim_hidden, bias=True))
        for _ in range(num_layers-2):
            self.layers.append(nn.Linear(dim_hidden, dim_hidden, bias=True))
        self.layers.append(nn.Linear(dim_hidden, dim_out, bias=True))

        # parameters initialization
        for layer in self.layers:
            fan_in = nn.init._calculate_correct_fan(layer.weight, mode='fan_in')
            nn.init.trunc_normal_(tensor=layer.weight, mean=0, std=np.sqrt(1/fan_in))
            nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        x = self.act(self.layers[0].forward(x))
        for i in range(1, self.num_layers-1): 
            x = self.act(self.layers[i](x))
        x = self.layers[-1](x)
        return x


'''score mathcing and particle updates'''
class ScoreMatching():
    def __init__(self, v, dt, C, u1, u2, dim_in, dim_out, dim_hidden, num_layers, use_newNN):
        self.v = torch.tensor(v).float().to(device) # particles
        self.dt = torch.tensor(dt).float().to(device) # time step 
        self.C = torch.tensor(C).float().to(device) # collision rate
        (self.nex, self.d) = self.v.shape # total number and dimension of particles

        self.u1 = torch.tensor(u1).to(device)
        self.u2 = torch.tensor(u2).to(device)

        if use_newNN:
            self.net = MLP(dim_in, dim_out, dim_hidden, num_layers).to(device)
        else:
            self.net = torch.load('model/model.pth', map_location='cuda')
    
    def s_net(self, v):
        return self.net(v)
    
    def save_model(self):
        return torch.save(self.net, 'model/model.pth')
    

    ''' initial L2 score matching'''
    def L2_loss(self, v):
        ## initial score
        t1 = v - self.u1
        t2 = v - self.u2
        g1 = torch.exp(-1/2 * torch.sum(t1**2, dim=1, keepdim=True))
        g2 = torch.exp(-1/2 * torch.sum(t2**2, dim=1, keepdim=True))
        s_init = -(g1 * t1 + g2 * t2) / (g1 + g2)
        ## compute L2 loss
        s = self.s_net(v)
        return torch.sum((s - s_init)**2) / torch.sum(s_init**2)

    def L2_train(self, lr, tol):
        optimizer = torch.optim.Adamax(self.net.parameters(), lr=lr)
        L2_loss = 1
        iter = 0
        while L2_loss > tol:
            optimizer.zero_grad()
            loss = self.L2_loss(self.v)
            loss.backward()
            optimizer.step()
            iter += 1
            L2_loss = loss.item()
            if iter % 1000 == 0:
                print('Iter %d, L2 Loss: %.5e' % (iter, L2_loss))


    ''' implicit score matching'''
    def ism_loss(self, v):
        s = self.s_net(v)
        jac_s = torch.vmap(torch.func.jacrev(self.s_net))(v)
        div_s = torch.einsum('bii->b', jac_s)
        ism_loss = (torch.sum(s**2) + 2*torch.sum(div_s)) / v.shape[0]
        return ism_loss
    
    def ism_train(self, lr, max_epoch):
        optimizer = torch.optim.Adamax(self.net.parameters(), lr=lr)
        
        for epoch in range(max_epoch):
            optimizer.zero_grad()
            loss = self.ism_loss(self.v)
            loss.backward()
            optimizer.step()            
                 
            if (epoch+1) % 5 == 0:
                print('Epoch %d, ISM Loss: %.5e' % (epoch+1, loss.item()))


    ''' update particles and densities'''
    def odefunc(self, L):
        ### neural ODE: d_t (v, l) = odefun(t, v, l) with v - particle position ; l - log determinant
       
        Id = torch.eye(self.d, device=device)
        v = L[:,:self.d]
        s = self.s_net(v).detach()
        s_jac = torch.vmap(torch.func.jacrev(self.s_net))(v).detach()

        def subodefunc(v_i, s_i, s_jac_i, v, s):
            v_diff = v_i - v
            s_diff = s_i - s
            ## Coulombian collision kernel
            v_diff_new = v_diff[:,None,:]
            norm_diff = torch.linalg.vector_norm(v_diff, ord=2, dim=1)
            norm_diff_new = norm_diff + 1e-8 # avoid division by zero
            A = ((norm_diff**2)[:,None,None] * Id - v_diff_new.permute(0,2,1) * v_diff_new) / (norm_diff_new**3)[:,None,None]
            ## dv
            As = torch.squeeze(torch.matmul(A, s_diff[:,:,None]))
            dv = -torch.mean(As, axis=0)
            ## dl
            dl1 = torch.sum(torch.mean(A, dim=0) * s_jac_i)
            dl2 = (self.d-1) * torch.mean(torch.sum(v_diff * s_diff, dim=1) / norm_diff_new**3)
            dl = -(dl1 - dl2)
            return (dv, dl)
        
        vmap_func = torch.vmap(subodefunc, in_dims=(0, 0, 0, None, None), out_dims=0)
        dv, dl = vmap_func(v, s, s_jac, v, s)
        return torch.cat((dv, dl[:,None]), dim=1)
    
    def stepRK1(self, L):
        ## forward Euler
        L += self.dt*self.C*self.odefunc(L)
        return L

    def updateRBM(self, v, f, B):
        ## random batch method
        v = torch.tensor(v).float().to(device) # particles
        f = torch.tensor(f).float().to(device) # densities
        
        v_new = torch.empty(0).to(device)
        f_new = torch.empty(0).to(device)
        
        dataset = torch.utils.data.TensorDataset(v, f)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=B, shuffle=True)

        for (vb, fb) in data_loader:
            ## initial condition
            n = vb.shape[0]
            L = torch.zeros((n, self.d+1)).to(device)
            L[:,:self.d] = vb

            ## update particles and densities
            L = self.stepRK1(L)
            vb_new = L[:,:self.d]
            logdet = L[:,-1]
            fb_new = fb / torch.exp(logdet)
            
            v_new = torch.cat((v_new, vb_new), dim=0)
            f_new = torch.cat((f_new, fb_new), dim=0)
        return v_new.detach().cpu().numpy(), f_new.detach().cpu().numpy()
    


''' initial sampling from a mixture of Gaussian distributions'''
def sampling_mix_gaussian(nex, d, u1, u2):
    samples = np.zeros((nex, d))       
    prob = np.random.rand(nex)

    for i in range(nex):
        m = np.random.multivariate_normal(np.zeros(d), np.eye(d))
        if prob[i] < 0.5:
            samples[i,:] = m + u1
        else:
            samples[i,:] = m + u2
    return samples

''' reconstruction by kernel density estimation'''
def kde(v, v_grid, eps):
    kde = KernelDensity(kernel='gaussian', bandwidth=eps).fit(v)
    f_kde = np.exp(kde.score_samples(v_grid))
    return f_kde

def f_init(v, u1, u2):
    f1 = np.exp(-1/2 * np.sum((v - u1)**2, axis=1))
    f2 = np.exp(-1/2 * np.sum((v - u2)**2, axis=1))
    return (f1 + f2) / (4 * np.pi)

## main function
def main():
    ## time
    T = 40
    t = 0
    dt = 0.1
    Nt = int((T-t)/dt)

    ## collision strength
    C = 1/16
    d = 2

    ## network params
    dim_in = 2
    dim_out = 2
    dim_hidden = 32
    num_layers = 4

    ## initial data
    u1 = np.array([[-2., 1.]])
    u2 = np.array([[0., -1.]])

    nex = 120**2
    v = sampling_mix_gaussian(nex, d, u1, u2)
    f = f_init(v, u1, u2)

    V = np.empty((Nt+1, nex, d)) # particles
    F = np.empty((Nt+1, nex)) # density
    E = np.empty(Nt+1) # energy
    H = np.empty(Nt+1) # entropy
    V[0] = v
    F[0] = f
    E[0] = np.mean(np.sum(v**2, axis=1), axis=0)
    H[0] = np.mean(np.log(f))


    for nt in range(Nt):
        t += dt
        print('current time t=%f '% t)
        
        ## training
        if nt==0:
            model = ScoreMatching(v, dt, C, u1, u2, dim_in, dim_out, dim_hidden, num_layers, True)
            model.L2_train(lr=1e-4, tol=1e-5)
        else:
            model = ScoreMatching(v, dt, C, u1, u2, dim_in, dim_out, dim_hidden, num_layers, False)
            model.ism_train(lr=1e-4, max_epoch=25)
        model.save_model()
        
        ## update particles and densities by random batch method
        v, f = model.updateRBM(v=v, f=f, B=1280)
        V[nt+1] = v
        E[nt+1] = np.mean(np.sum(v**2, axis=1), axis=0)
        F[nt+1] = f
        H[nt+1] = np.mean(np.log(f))
    

    ## kernel density estimation
    L = 10
    Nr = 120
    h = 2*L/Nr
    V0 = np.linspace(-L+h/2,L-h/2,Nr)
    VX, VY = np.meshgrid(V0, V0)
    V_mesh = np.hstack((VX.flatten()[:,None], VY.flatten()[:,None]))

    f_kde = kde(v, V_mesh, eps=0.3)
    f_kde = f_kde.reshape(Nr, Nr)
    
    
    plt.plot(V0, f_kde[59,:])
    plt.xlabel('v_x')
    plt.legend(['score-based', 'exact'])
    plt.title('f(v_x, 0) at t=40')
    plt.show()


if __name__ == '__main__':
    main()
