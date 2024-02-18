# Phi.py
# neural network to model the potential function
import torch

def antiderivTanh(x): # activation function aka the antiderivative of tanh
    return torch.abs(x) + torch.log(1+torch.exp(-2.0*torch.abs(x)))

def derivTanh(x): # act'' aka the second derivative of the activation function antiderivTanh
    return 1 - torch.pow( torch.tanh(x) , 2 )

class ResNN(torch.nn.Module):
    def __init__(self, nTh, m, d):
        """
        :param d:   int, dimension of space input (expect inputs to be d+1 for space-time)
        :param m:   int, hidden dimension
        :param nTh: int, number of resNet layers , (number of theta layers)
        """
        super(ResNN, self).__init__()

        if nTh < 2:
            print("nTh must be an integer >= 2")
            exit(1)

        self.nTh  = nTh
        self.m    = m
        self.d    = d
        self.act = antiderivTanh

        self.layers = torch.nn.ModuleList([])
        self.layers.append(torch.nn.Linear(self.d + 1, self.m, bias=True)) # opening layer
        for i in range(nTh-1): 
            self.layers.append(torch.nn.Linear(self.m, self.m, bias=True)) # resnet layers
        
    def forward(self, x):
        """
        :param x: tensor nex-by-d+1, inputs
        :return:  tensor nex-by-1,   outputs
        """
        x = self.act(self.layers[0].forward(x))

        for i in range(1,self.nTh):
            x = x + self.act(self.layers[i](x))
        return x

    

class Phi(torch.nn.Module):
    def __init__(self, nTh, m, d):
        """
            neural network approximating Phi

            Phi( x,t ) = w'*ResNet( [x;t])

        :param nTh:  int, number of resNet layers , (number of theta layers)
        :param m:    int, hidden dimension
        :param d:    int, dimension of space input (expect inputs to be d+1 for space-time)
        """
        super(Phi,self).__init__()

        self.nTh  = nTh
        self.m    = m
        self.d    = d

        self.ResNN = ResNN(self.nTh, self.m, self.d)

        self.w = torch.nn.Linear(self.m, 1, bias=False)
        # set initial values
        self.w.weight.data = torch.randn(self.w.weight.data.shape)
    

    def forward(self, x):
        return self.w(self.ResNN(x))


    def Grad_Hess(self, x, justGrad=False):
        """
        :param x: input data, torch Tensor nex-by-d
        :param justGrad: boolean, if True only return gradient, if False return (grad, Hess)
        :return: gradient, Hessian    OR    just gradient
        """
    
        u = [] # hold the u_0,u_1,...,u_M for the forward pass
        z = self.nTh*[None] # hold the z_0,z_1,...,z_M for the backward pass
        affine = [] # hold the K_i^T diag(act'(K_i u_{i-1} + b_i))

        # Forward of ResNet N and fill u
        affine.append(self.ResNN.layers[0].forward(x)) # K_0 * S + b_0
        u.append(self.ResNN.act(affine[0])) # u0
        u_mid = u[0]

        for i in range(1, self.nTh):
            affine.append(self.ResNN.layers[i](u_mid))
            u_mid = u_mid + self.ResNN.act(affine[i])
            u.append(u_mid)

        # compute gradient and fill z
        for i in range(self.nTh-1,0,-1): # work backwards, placing z_i in appropriate spot
            if i == self.nTh-1:
                z_mid = self.w.weight.t()  # m by 1
            else:
                z_mid = z[i+1]

            # z_i = z_{i+1} + h K_i' diag(...) z_{i+1}
            z[i] = z_mid + torch.mm( self.ResNN.layers[i].weight.t(), torch.tanh(affine[i]).t() * z_mid )  

        # z_0 = K_0' diag(...) z_1
        z[0] = torch.mm( self.ResNN.layers[0].weight.t(), torch.tanh(affine[0]).t() * z[1] )
        grad = z[0]

        if justGrad:
            return grad.t()[:,:self.d] # nex by d+1


        # -----------------
        # Hessian
        #------------------


        # t_0
        W = self.ResNN.layers[0].weight
        diag = derivTanh(affine[0]) * z[1].t()
        Hess = torch.matmul(W.t(), diag[:,:,None] * W)
        J = torch.tanh(affine[0])[:,:,None] * W

        # t_i
        for i in range(1,self.nTh-1):
            W = self.ResNN.layers[i].weight
            diag = derivTanh(affine[i]) * z[i+1].t()
            temp = torch.matmul(W.t(), diag[:,:,None] * W)
            Hess = Hess + torch.matmul(J.permute(0,2,1), torch.matmul(temp, J))
            J = J + torch.matmul( torch.tanh(affine[i])[:,:,None] * W, J)

        # t_M
        W = self.ResNN.layers[self.nTh-1].weight
        diag = derivTanh(affine[self.nTh-1]) * self.w.weight
        temp = torch.matmul(W.t(), diag[:,:,None] * W)
        Hess = Hess + torch.matmul(J.permute(0,2,1), torch.matmul(temp, J))

        return grad.t()[:,:self.d], Hess[:,:self.d,:self.d]


    


# if __name__ == "__main__":
#     import time

#     # test 
#     d = 2
#     m = 64
#     nex = 3

#     net = Phi(nTh=5, m=m, d=d)
#     net.eval()
#     x = torch.randn(nex,d+1)
#     y = net(x)

#     # end = time.time()
#     # g,h = net.Grad_Hess(x)
#     # print('Hessian takes ', time.time()-end)
#     # print(g)
#     # print(h)

#     end = time.time()
#     g = net.Grad_Hess(x, justGrad=True)
#     print('JustGrad takes  ', time.time()-end)


#     end2 = time.time()
#     v_new = torch.zeros((nex, d))
#     for i in range(nex):
#         v_diff = x[i,:d]-x[:,:d] # matrix
#         norm = torch.norm(v_diff, dim=1, keepdim=True)

#         v_temp = v_diff[:,None,:]
#         proj_i = (norm**2)[:,:,None] * torch.eye(d) - v_temp.permute(0,2,1) * v_temp
       
#         temp = (norm**(0)) * torch.squeeze(torch.matmul((g[i,:]-g)[:,None,:], proj_i), dim=1)
#         v_new[i,:] = x[i,:d] - torch.sum(temp, dim=0) / nex
    
#     print('new_v takes ', time.time()-end2)

#     # check conservation of momentum
#     mo_old = torch.sum(x[:,:d], dim=0)
#     mo_new = torch.sum(v_new, dim=0)
#     print(mo_new-mo_old)
