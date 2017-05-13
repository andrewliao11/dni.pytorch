import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import pickle as pkl
from dni import *
from plot import *

# Hyper Parameters
num_epochs = 300
batch_size = 100
learning_rate = 0.001
model_name = 'CNN'
conditioned_DNI = False

# MNIST Dataset
train_dataset = dsets.MNIST(root='../data/',
                            train=True, 
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='../data/',
                           train=False, 
                           transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)
def save_grad(name):
    def hook(grad):
        backprop_grads[name] = grad
        backprop_grads[name].volatile = False
    return hook

# CNN Model (2 conv layer)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
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
        self.fc = nn.Linear(7*7*32, 10)

        # DNI module
        self._layer1 = dni_Conv2d(16, conditioned=conditioned_DNI)
        self._layer2 = dni_Conv2d(32, conditioned=conditioned_DNI)
        self._fc = dni_linear(10, conditioned=conditioned_DNI)

        self.cnn = nn.Sequential(
                   self.layer1, 
                   self.layer2, 
                   self.fc)
        self.dni = nn.Sequential(
                   self._layer1, 
                   self._layer2, 
                   self._fc)
       
    def forward_layer1(self, x, y=None):
        out = self.layer1(x)
        grad = self._layer1(out, y)
        return out, grad
 
    def forward_layer2(self, x, y=None):
        out = self.layer2(x)
        grad = self._layer2(out, y)
        return out, grad
    
    def forward_fc(self, x, y=None):
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
            return layer1, layer2, fc, grad_layer1, grad_layer2, grad_fc
        else:
            return layer1, layer2, fc
        
net = CNN()
net.cuda()

# Param, Optimizer and Criterion
optimizer = torch.optim.Adam(net.cnn.parameters(), lr=learning_rate)
optimizer_layer1 = torch.optim.Adam(net.layer1.parameters(), lr=learning_rate)
optimizer_layer2 = torch.optim.Adam(net.layer2.parameters(), lr=learning_rate)
optimizer_fc = torch.optim.Adam(net.fc.parameters(), lr=learning_rate)
grad_optimizer = torch.optim.Adam(net.dni.parameters(), lr=learning_rate)
classificationCriterion = nn.CrossEntropyLoss()
syntheticCriterion = nn.MSELoss()

def test_model(epoch):
    # Test the Model
    net.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = Variable(images).cuda()
        layer1, layer2, fc = net(images)
        outputs = fc
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels).sum()

    perf = 100 * correct / total
    print('Epoch %d: Accuracy of the network on the 10000 test images: %d %%' % (epoch, perf))
    return perf

stats = dict(grad_loss=[], classify_loss=[])
best_perf = 0.
# Train the Model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()

        # Layer1
        # Forward + Backward + Optimize
        optimizer_layer1.zero_grad()
        out, grad1 = net.forward_layer1(images)
        out.backward(grad1.detach().data)
        optimizer_layer1.step()
        
        # Layer2
        # Forward + Backward + Optimize
        optimizer_layer2.zero_grad()
        out, grad2 = net.forward_layer2(out.detach())
        out.backward(grad2.detach().data)
        optimizer_layer2.step()

        # flatten+Fc
        # Forward + Backward + Optimize
        out = out.detach()
        out = out.view(out.size(0), -1)
        optimizer_fc.zero_grad()
        out, grad3 = net.forward_fc(out)
        out.backward(grad3.detach().data)
        optimizer_fc.step()

        # synthetic model
        # Forward + Backward + Optimize
        grad_optimizer.zero_grad()
        optimizer.zero_grad()
        layer1, layer2, fc, grad_layer1, grad_layer2, grad_fc = net(images, labels)
        backprop_grads = {}
        handle_layer1 = layer1.register_hook(save_grad('layer1'))
        handle_layer2 = layer2.register_hook(save_grad('layer2'))
        handle_fc = fc.register_hook(save_grad('fc'))
        outputs = fc
        loss = classificationCriterion(outputs, labels)
        loss.backward(retain_variables=True)
        handle_layer1.remove()
        handle_layer2.remove()
        handle_fc.remove()
        grad_loss = syntheticCriterion(grad_layer1, backprop_grads['layer1'].detach())
        grad_loss += syntheticCriterion(grad_layer2, backprop_grads['layer2'].detach())
        grad_loss += syntheticCriterion(grad_fc, backprop_grads['fc'].detach())

        grad_loss.backward()
        grad_optimizer.step()

        stats['grad_loss'].append(grad_loss.data[0])
        stats['classify_loss'].append(loss.data[0])
        if (i+1) % 100 == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Grad Loss: %.4f'
                   %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0], grad_loss.data[0]))


    if (epoch+1) % 10 == 0:
        perf = test_model(epoch+1)
        if perf > best_perf:
            torch.save(net.state_dict(), model_name+'_cnn_best.pkl')
        net.train()

# Save the Model ans Stats
pkl.dump(stats, open(model_name+'_stats.pkl', 'w'))
torch.save(net.state_dict(), model_name+'_cnn.pkl')
plot(stats, name=model_name)
