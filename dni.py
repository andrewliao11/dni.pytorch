import torch
import torch.nn as nn
import numpy as np

class dni_linear(nn.Module):
    def __init__(self, input_dims, num_classes, dni_hidden_size=1024, conditioned=False):
        super(dni_linear, self).__init__()
        self.conditioned = conditioned
        if self.conditioned:
            dni_input_dims = input_dims+num_classes
        else:
            dni_input_dims = input_dims
        self.layer1 = nn.Sequential(
                      nn.Linear(dni_input_dims, dni_hidden_size),
                      nn.BatchNorm1d(dni_hidden_size),
                      nn.ReLU()
                      )
        self.layer2 = nn.Sequential(
                      nn.Linear(dni_hidden_size, dni_hidden_size),
                      nn.BatchNorm1d(dni_hidden_size),
                      nn.ReLU()
                      )
        self.layer3 = nn.Linear(dni_hidden_size, input_dims)

    def forward(self, x, y):
        if self.conditioned:
            assert y is not None
            x = torch.cat((x, y), 1)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return out

class dni_Conv2d(nn.Module):
    def __init__(self, input_dims, input_size, num_classes, dni_hidden_size=64, conditioned=False):
        super(dni_Conv2d, self).__init__()
        self.conditioned = conditioned
        if self.conditioned:
            dni_input_dims = input_dims+1
        else:
            dni_input_dims = input_dims

        self.input_size = list(input_size)
        self.label_emb = nn.Linear(num_classes, np.prod(np.array(input_size)))

        self.layer1 = nn.Sequential(
                      nn.Conv2d(dni_input_dims, dni_hidden_size, kernel_size=5, padding=2),
                      nn.BatchNorm2d(dni_hidden_size),
                      nn.ReLU())
        self.layer2 = nn.Sequential( 
                      nn.Conv2d(dni_hidden_size, dni_hidden_size, kernel_size=5, padding=2),
                      nn.BatchNorm2d(dni_hidden_size),
                      nn.ReLU())
        self.layer3 = nn.Sequential(
                      nn.Conv2d(dni_hidden_size, input_dims, kernel_size=5, padding=2))

    def forward(self, x, y):
        if self.conditioned:
            assert y is not None
            y = self.label_emb(y)
            y = y.view([-1, 1]+self.input_size)
            x = torch.cat((x, y), 1)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return out

