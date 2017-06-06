import torch.nn as nn
from dni import *

# CNN Model (2 conv layer)
class cnn(nn.Module):
    def __init__(self, conditioned_DNI, num_classes):
        super(cnn, self).__init__()
       
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, num_classes)

        # DNI module
        self._layer1 = dni_Conv2d(16, (14, 14), num_classes, conditioned=conditioned_DNI)
        self._layer2 = dni_Conv2d(32, (7, 7), num_classes, conditioned=conditioned_DNI)
        self._fc = dni_linear(num_classes, num_classes, conditioned=conditioned_DNI)

        self.cnn = nn.Sequential(
                   self.layer1, 
                   self.layer2, 
                   self.fc)
        self.dni = nn.Sequential(
                   self._layer1, 
                   self._layer2, 
                   self._fc)
        self.optimizers = []
        self.forwards = []
        self.init_optimzers()
        self.init_forwards()

    def init_optimzers(self, learning_rate=0.001):
        self.optimizers.append(torch.optim.Adam(self.layer1.parameters(), lr=learning_rate))
        self.optimizers.append(torch.optim.Adam(self.layer2.parameters(), lr=learning_rate))
        self.optimizers.append(torch.optim.Adam(self.fc.parameters(), lr=learning_rate))
        self.optimizer = torch.optim.Adam(self.cnn.parameters(), lr=learning_rate)
        self.grad_optimizer = torch.optim.Adam(self.dni.parameters(), lr=learning_rate)

    def init_forwards(self):
        self.forwards.append(self.forward_layer1)
        self.forwards.append(self.forward_layer2)
        self.forwards.append(self.forward_fc)

    def forward_layer1(self, x, y=None):
        out = self.layer1(x)
        grad = self._layer1(out, y)
        return out, grad
 
    def forward_layer2(self, x, y=None):
        out = self.layer2(x)
        grad = self._layer2(out, y)
        return out, grad
    
    def forward_fc(self, x, y=None):
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        grad = self._fc(out, y)
        return out, grad
    
    def forward(self, x, y=None):
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer2_flat = layer2.view(layer2.size(0), -1)
        fc = self.fc(layer2_flat)
        if y is not None:
            grad_layer1 = self._layer1(layer1, y)
            grad_layer2 = self._layer2(layer2, y)
            grad_fc = self._fc(fc, y)
            return (layer1, layer2, fc), (grad_layer1, grad_layer2, grad_fc)
        else:
            return layer1, layer2, fc
        
# Neural Network Model (1 hidden layer)
class mlp(nn.Module):
    def __init__(self, conditioned_DNI, input_size, num_classes, hidden_size=256):
        super(mlp, self).__init__()
        # classify network
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
        # dni network
        self._fc1 = dni_linear(hidden_size, num_classes, conditioned=conditioned_DNI)
        self._fc2 = dni_linear(num_classes, num_classes, conditioned=conditioned_DNI)

        self.mlp = nn.Sequential(self.fc1, self.relu, self.fc2)
        self.dni = nn.Sequential(self._fc1, self._fc2)
        self.optimizers = []
        self.forwards = []
        self.init_optimzers()
        self.init_forwards()
        
    def init_optimzers(self, learning_rate=3e-5):
        self.optimizers.append(torch.optim.Adam(self.fc1.parameters(), lr=learning_rate))
        self.optimizers.append(torch.optim.Adam(self.fc2.parameters(), lr=learning_rate))
        self.optimizer = torch.optim.Adam(self.mlp.parameters(), lr=learning_rate)
        self.grad_optimizer = torch.optim.Adam(self.dni.parameters(), lr=learning_rate)

    def init_forwards(self):
        self.forwards.append(self.forward_fc1)
        self.forwards.append(self.forward_fc2)

    def forward_fc1(self, x, y=None):
        x = x.view(-1, 28*28)
        out = self.fc1(x)
        grad = self._fc1(out, y)
        return out, grad
       
    def forward_fc2(self, x, y=None):
        x = self.relu(x)
        out = self.fc2(x)
        grad = self._fc2(out, y)
        return out, grad
 
    def forward(self, x, y=None):
        x = x.view(-1, 28*28)
        fc1 = self.fc1(x)
        relu1 = self.relu(fc1)
        fc2 = self.fc2(relu1)
        
        if y is not None:
            grad_fc1 = self._fc1(fc1, y)
            grad_fc2 = self._fc2(fc2, y)
            return (fc1, fc2), (grad_fc1, grad_fc2)
        else:
            return fc1, fc2
