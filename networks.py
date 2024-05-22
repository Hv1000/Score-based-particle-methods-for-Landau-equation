import torch
import math
#torch.set_default_tensor_type(torch.DoubleTensor)

class DNN(torch.nn.Module):
    def __init__(self, d, m, nTh):
        super().__init__()
        self.d = d
        self.m = m
        self.nTh = nTh
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
    

class ResNet(torch.nn.Module):
    def __init__(self, d, m, nTh):
        super().__init__()
        self.d = d
        self.m = m
        self.nTh = nTh
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
            x = x + self.act(self.layers[i](x))
        x = self.layers[-1](x)
        return x