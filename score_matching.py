import torch
from torch.func import jacrev, vmap
import networks
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class ScoreMatching():
    def __init__(self, v, t, nex, d, m, nTh):
        self.v = torch.tensor(v, requires_grad=True).float().to(device)
        self.t = torch.tensor(t).float().to(device)
        self.nex = nex

        ## network for score
        if t==0:
            self.score_net = networks.DNN(d=d, m=m, nTh=nTh).to(device)
        else:
            self.score_net = torch.load('model.pth', map_location='cuda')

        ## optimizer
        self.optimizer_L2 = torch.optim.Adamax(self.score_net.parameters(), lr=1e-3)
        self.optimizer_ism = torch.optim.Adamax(self.score_net.parameters(), lr=1e-4)
    

    def score_ext(self, v, t):
        K = 1 - torch.exp(-t/8) / 2
        norm = torch.sum(v**2, dim=1, keepdim=True)
        score_ext = (-1/K + (1-K) / ((2*K-1)*K + (1-K)/2*norm))*v
        return score_ext

    def L2_loss(self, v, t):
        score_ext = self.score_ext(v, t)
        score_net = self.score_net(v)
        L2_loss = torch.sum((score_net - score_ext)**2) / torch.sum(score_ext**2)
        return L2_loss

    def L2_train(self, tol):
        L2_loss = 1
        iter = 0
        while L2_loss > tol:
            self.optimizer_L2.zero_grad()
            loss = self.L2_loss(self.v, self.t)
            loss.backward()
            self.optimizer_L2.step()
            iter += 1

            L2_loss = loss.item()
            if iter % 1000 == 0:
                ism_loss = self.ism_loss(self.v).item()
                print('Iter %d, L2 Loss: %.5e, ISM Loss: %.5e' % (iter, L2_loss, ism_loss))
        print('Iter %d, Loss: %.5e, ISM Loss: %.5e' % (iter, L2_loss, ism_loss))
        return L2_loss
    
    def ism_loss(self, v):
        score_net = self.score_net(v)
        grad_1 = torch.autograd.grad(outputs=score_net[:,0], grad_outputs = torch.ones_like(score_net[:,0]), inputs=v, retain_graph=True, create_graph=True)[0]
        grad_2 = torch.autograd.grad(outputs=score_net[:,1], grad_outputs = torch.ones_like(score_net[:,1]), inputs=v, retain_graph=True, create_graph=True)[0]
        div = torch.sum(grad_1[:,0] + grad_2[:,1])
        ism_loss = (torch.sum(score_net**2) + 2*div) / self.nex
        return ism_loss
    
    def ism_train(self, max_iter):
        iter = 0
        while iter <= max_iter:
            self.optimizer_ism.zero_grad()
            loss = self.ism_loss(self.v)
            loss.backward()
            self.optimizer_ism.step()
            iter += 1

            # for p in self.score_net.parameters():
            #     param_norm = p.grad.data.norm(2)
            #     total_norm += param_norm.item() ** 2
            # total_norm = total_norm**(1./2)

            ism_loss = loss.item()
            if iter % 5 == 0:
                L2_loss = self.L2_loss(self.v, self.t).item()
                print('Iter %d, ISM Loss: %.5e, L2 Loss: %.5e' % (iter, ism_loss, L2_loss))
        return L2_loss


    def score(self, v):
        v = torch.tensor(v).float().to(device)
        score = self.score_net(v)
        return score.detach()

    def score_jac(self, v):
        v = torch.tensor(v, requires_grad=True).float().to(device)
        score_jac = vmap(jacrev(self.score_net))(v)
        return score_jac.detach()

    def save_model(self):
        return torch.save(self.score_net,'model.pth')