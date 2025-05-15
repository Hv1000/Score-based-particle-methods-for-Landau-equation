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
    def __init__(self, v, t, dt, C, dim_in, dim_out, dim_hidden, num_layers, use_newNN):
        self.v = torch.tensor(v).float().to(device) # particles
        self.t = torch.tensor(t).float().to(device) # time
        self.dt = torch.tensor(dt).float().to(device) # time step 
        self.C = torch.tensor(C).float().to(device) # collision rate
        (self.nex, self.d) = self.v.shape # total number and dimension of particles

        if use_newNN:
            self.net = MLP(dim_in, dim_out, dim_hidden, num_layers).to(device)
        else:
            self.net = torch.load('model.pth', map_location='cuda')
    
    def s_net(self, v):
        return self.net(v)
    
    def save_model(self):
        return torch.save(self.net, 'model.pth')
    

    '''L2 score matching'''
    def s_ext(self, v, t):
        ## analytical score for 2D BKW
        K = 1 - torch.exp(-2*self.C*t) / 2
        norm = torch.sum(v**2, dim=1, keepdim=True) 
        return (-1/K + (1-K) / ((2*K-1)*K + (1-K)/2 * norm)) * v

    def L2_loss(self, v, t):
        s = self.s_net(v)
        s_ext = self.s_ext(v, t)
        return torch.sum((s - s_ext)**2) / torch.sum(s_ext**2)

    def L2_train(self, lr, tol):
        optimizer = torch.optim.Adamax(self.net.parameters(), lr=lr)
        L2_loss = 1
        iter = 0
        while L2_loss > tol:
            optimizer.zero_grad()
            loss = self.L2_loss(self.v, self.t)
            loss.backward()
            optimizer.step()
            iter += 1
            L2_loss = loss.item()
            if iter % 1000 == 0:
                print('Iter %d, L2 Loss: %.5e' % (iter, L2_loss))


    '''implicit score matching'''
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
    
            ISM_loss = loss.item()            
            if (epoch+1) % 10 == 0:
                L2_loss = self.L2_loss(self.v, self.t).item()
                print('Epoch %d, ISM Loss: %.5e, L2 Loss: %.5e' % (epoch+1, ISM_loss, L2_loss))
        

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
            ## Maxwellian collision kernel
            v_diff_new = v_diff[:,None,:]
            norm_diff = torch.linalg.vector_norm(v_diff, ord=2, dim=1)
            A = (norm_diff**2)[:,None,None] * Id - v_diff_new.permute(0,2,1) * v_diff_new
            ## dv
            As = torch.squeeze(torch.matmul(A, s_diff[:,:,None]))
            dv = -torch.mean(As, axis=0)
            ## dl
            dl1 = torch.sum(torch.mean(A, dim=0) * s_jac_i)
            dl2 = (self.d-1) * torch.mean(torch.sum(v_diff * s_diff, dim=1))
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
    
    
    def ent_rate(self, v, B):
        ## entropy production rate
        Id = torch.eye(self.d, device=device)
        v = torch.tensor(v).float().to(device)
        s = self.s_net(v).detach()

        def sub_ent_rate(v_i, s_i, v, s):
            v_diff = v_i - v
            s_diff = s_i - s
            ## Maxwellian collision kernel
            v_diff_new = v_diff[:,None,:]
            norm_diff = torch.linalg.vector_norm(v_diff, ord=2, dim=1)
            A = (norm_diff**2)[:,None,None] * Id - v_diff_new.permute(0,2,1) * v_diff_new
            ## dH
            As_diff = torch.squeeze(torch.matmul(A, s_diff[:,:,None]))
            dH = -torch.mean(torch.sum(s_diff * As_diff, dim=1)) / 2
            return dH
        
        dH = torch.empty(0).to(device)

        dataset = torch.utils.data.TensorDataset(v, s)
        loader = torch.utils.data.DataLoader(dataset, batch_size=B, shuffle=False)
        vmap_func = torch.vmap(sub_ent_rate, in_dims=(0, 0, None, None), out_dims=0)
        
        for (vb, sb) in loader:
            dHb = vmap_func(vb, sb, v, s)
            dH = torch.cat((dH, dHb), dim=0)

        ent_rate = self.C*torch.mean(dH)
        return ent_rate.detach().cpu().numpy()


## 2D BKW analytic solution
def f_bkw2d(v, t, C):
    norm = np.sum(v**2, axis=1)
    d = 2
    K = 1 - 1/2*np.exp(-2*C*(d-1)*t)
    P = ((d+2)*K - d) / (2*K)
    Q = (1-K) / (2*K**2)
    f_bkw = 1/((2*np.pi*K)**(d/2)) * np.exp(-norm/(2*K)) * (P + Q*norm)
    return f_bkw


def kde(v, v_grid, eps):
    kde = KernelDensity(kernel='gaussian', bandwidth=eps).fit(v)
    f_kde = np.exp(kde.score_samples(v_grid))
    return f_kde


def main():
    ## time
    T = 1
    t = 0
    dt = 0.01
    Nt = int((T-t)/dt)

    ## collision strength
    C = 1/16
    d = 2

    ## network params
    dim_in = 2
    dim_out = 2
    dim_hidden = 32
    num_layers = 5

    ## initial data
    nex = 160**2
    model_sp = sp.Sampling(nex=nex, seed=24)
    v = np.array(model_sp.sampling())
    f = f_bkw2d(v, t, C)

    V = np.empty((Nt+1, nex, d))
    F = np.empty((Nt+1, nex))
    E = np.empty(Nt+1)
    H = np.empty(Nt+1)
    V[0] = v
    F[0] = f
    E[0] = np.mean(np.sum(v**2, axis=1), axis=0)
    H[0] = np.mean(np.log(f))


    for nt in range(Nt):
        t += dt
        print('current time t=%f '% t)
        
        ## training
        if nt==0:
            model = ScoreMatching(v, t, dt, C, dim_in, dim_out, dim_hidden, num_layers, True)
            model.L2_train(lr=1e-3, tol=1e-4)
        else:
            model = ScoreMatching(v, t, dt, C, dim_in, dim_out, dim_hidden, num_layers, False)
            model.ism_train(lr=5e-4, max_epoch=50)
        model.save_model()
        
        ## update particles and densities
        v, f = model.updateRBM(v=v, f=f, B=1280)
        V[nt+1] = v
        E[nt+1] = np.mean(np.sum(v**2, axis=1), axis=0)
        F[nt+1] = f
        H[nt+1] = np.mean(np.log(f))
    

    ## kernel density estimation
    L = 4
    Nr = 100
    h = 2*L/Nr
    V0 = np.linspace(-L+h/2,L-h/2,Nr)
    VX, VY = np.meshgrid(V0, V0)
    V_mesh = np.hstack((VX.flatten()[:,None], VY.flatten()[:,None]))

    f_kde = kde(v, V_mesh, eps=0.12)
    f_kde = f_kde.reshape(Nr, Nr)
    f_ext = f_bkw2d(V_mesh, T, C)
    f_ext = f_ext.reshape(Nr, Nr)
    
    plt.plot(V0, f_kde[49,:])
    plt.plot(V0, f_ext[49,:])
    plt.xlabel('v_x')
    plt.legend(['score-based', 'exact'])
    plt.title('f(v_x, 0) at t=1')
    plt.show()


if __name__ == '__main__':
    main()
