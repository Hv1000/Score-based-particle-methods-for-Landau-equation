import torch
import math
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class DNN(torch.nn.Module):
    def __init__(self, d, m, nTh):
        super().__init__()
        self.d = d
        self.m = m
        self.nTh = nTh
        # self.act = torch.relu
        self.act = torch.nn.SiLU()

        self.layers = torch.nn.ModuleList([])
        self.layers.append(torch.nn.Linear(d, m, bias=True))
        for i in range(nTh-2):
            self.layers.append(torch.nn.Linear(m, m, bias=True))
        self.layers.append(torch.nn.Linear(m, d, bias=True))

        # parameters initialization
        for layer in self.layers:
            fan_in = torch.nn.init._calculate_correct_fan(layer.weight, mode='fan_in')
            torch.nn.init.trunc_normal_(tensor=layer.weight, mean=0, std=math.sqrt(1/fan_in))
            torch.nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        x = self.act(self.layers[0].forward(x))
        for i in range(1, self.nTh-1):
            x = self.act(self.layers[i](x))
        x = self.layers[-1](x)
        return x
    
    
class ScoreMatching():
    def __init__(self, samples, time):
        self.samples = torch.tensor(samples, requires_grad=True).float().to(device)
        self.nex = samples.shape[0]
        self.d = samples.shape[1]
        self.t = torch.tensor(time).float()

        if time==0:
            self.score_net = DNN(d=self.d, m=32, nTh=4).to(device)
        else:
            self.score_net = torch.load('model.pth', map_location='cuda')

        self.optimizer_L2 = torch.optim.Adamax(self.score_net.parameters(), lr=1e-3)
        self.optimizer_ism = torch.optim.Adamax(self.score_net.parameters(), lr=1e-4)
    
    def L2_loss(self, samples, score_exact):
        score_net = self.score_net(samples)
        L2_loss = torch.sum((score_net - score_exact)**2) / torch.sum(score_exact**2)
        return L2_loss

    def train_L2(self, tol):
        L2_loss = 1
        iter = 0
        while L2_loss > tol:
            self.optimizer_L2.zero_grad()
            score_exact = self.score_exact(self.samples, self.t)
            loss = self.L2_loss(self.samples, score_exact)
            loss.backward()
            self.optimizer_L2.step()
            L2_loss = loss.item()
            iter += 1
            if iter % 1000 == 0:
                ism_loss = self.ism_loss(self.samples).item()
                print('Iter %d, L2 Loss: %.5e, ISM Loss: %.5e' % (iter, L2_loss, ism_loss))
        print('Iter %d, Loss: %.5e, ISM Loss: %.5e' % (iter, L2_loss, ism_loss))
        return L2_loss
    
    def ism_loss(self, samples):
        score_net = self.score_net(samples)
        grad_1 = torch.autograd.grad(outputs=score_net[:,0], grad_outputs = torch.ones_like(score_net[:,0]), inputs=samples, retain_graph=True, create_graph=True)[0]
        grad_2 = torch.autograd.grad(outputs=score_net[:,1], grad_outputs = torch.ones_like(score_net[:,1]), inputs=samples, retain_graph=True, create_graph=True)[0]
        div = torch.sum(grad_1[:,0] + grad_2[:,1])
        ism_loss = (torch.sum(score_net**2) + 2*div) / self.nex
        return ism_loss
    
    def train_ism(self, max_iter):
        iter = 0
        score_exact = self.score_exact(self.samples, self.t)

        while iter <= max_iter:
            self.optimizer_ism.zero_grad()
            loss = self.ism_loss(self.samples)
            loss.backward()
            self.optimizer_ism.step()
            ism_loss = loss.item()
            iter += 1

            # for p in self.score_net.parameters():
            #     param_norm = p.grad.data.norm(2)
            #     total_norm += param_norm.item() ** 2
            # total_norm = total_norm**(1./2)

            if iter % 5 == 0:
                L2_loss = self.L2_loss(self.samples, score_exact).item()
                print('Iter %d, ISM Loss: %.5e, L2 Loss: %.5e' % (iter, ism_loss, L2_loss))
        return L2_loss

################################################################################################################
    def compute_score(self, samples):
        samples = torch.tensor(samples).float().to(device)
        score = self.score_net(samples)
        return score.detach().cpu().numpy()

    def compute_score_jac(self, samples):
        samples = torch.tensor(samples, requires_grad=True).float().to(device)
        score_net = self.score_net(samples)
        grad_1 = torch.autograd.grad(outputs=score_net[:,0], grad_outputs = torch.ones_like(score_net[:,0]), inputs=samples, retain_graph=True, create_graph=True)[0]
        grad_2 = torch.autograd.grad(outputs=score_net[:,1], grad_outputs = torch.ones_like(score_net[:,1]), inputs=samples, retain_graph=True, create_graph=True)[0]
        
        jac_pre_1 = torch.cat((grad_1, grad_2), dim=1)
        jac_pre_2 = torch.stack(torch.split(jac_pre_1, 1, dim=0))
        jac = jac_pre_2.reshape(self.nex, self.d, self.d)
        return jac.detach().cpu().numpy()
    
    def score_exact(self, samples, t):
        K = 1 - torch.exp(-t/8) / 2
        norm = torch.sum(samples**2, dim=1, keepdim=True)
        score = (-1/K + (1-K) / ((2*K-1)*K + (1-K)/2*norm))*samples
        return score

    def save_model(self):
        return torch.save(self.score_net,'model.pth')