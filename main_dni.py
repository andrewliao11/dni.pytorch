import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import pickle as pkl
from torch.autograd import Variable
from plot import *
from dni import *
import ipdb

# Hyper Parameters 
input_size = 784
hidden_size = 256
num_classes = 10
num_epochs = 300
batch_size = 100
learning_rate = 3e-5
model_name = 'DNI'
conditioned_DNI = False

# MNIST Dataset 
train_dataset = dsets.MNIST(root='../data', 
                            train=True, 
                            transform=transforms.ToTensor(),  
                            download=True)

test_dataset = dsets.MNIST(root='../data', 
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

# Neural Network Model (1 hidden layer)
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        # classify network
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
        # dni network
        self._fc1 = dni_linear(hidden_size, conditioned=conditioned_DNI)
        self._fc2 = dni_linear(num_classes, conditioned=conditioned_DNI)

        self.mlp = nn.Sequential(self.fc1, self.relu, self.fc2)
        self.dni = nn.Sequential(self._fc1, self._fc2)

    def forward_fc1(self, x, y=None):
        out = self.fc1(x)
        grad = self._fc1(out, y)
        return out, grad
       
    def forward_fc2(self, x, y=None):
        out = self.fc2(x)
        grad = self._fc2(out, y)
        return out, grad
 
    def forward(self, x, y=None):
        fc1 = self.fc1(x)
        relu1 = self.relu(fc1)
        fc2 = self.fc2(relu1)
        
        if y is not None:
            grad_fc1 = self._fc1(fc1, y)
            grad_fc2 = self._fc2(fc2, y)
            return fc1, fc2, grad_fc1, grad_fc2
        else:
            return fc1, fc2
    
net = Net(input_size, hidden_size, num_classes)
net.cuda()   

# Param, Optimizer and Criterion
optimizer_fc1 = torch.optim.Adam(net.fc1.parameters(), lr=learning_rate)
optimizer_fc2 = torch.optim.Adam(net.fc2.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(net.mlp.parameters(), lr=learning_rate)
grad_optimizer = torch.optim.Adam(net.dni.parameters(), lr=learning_rate)
classificationCriterion = nn.CrossEntropyLoss()
syntheticCriterion = nn.MSELoss()

def test_model(epoch):
    # Test the Model
    net.eval()
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = Variable(images.view(-1, 28*28)).cuda()
        _, outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels).sum()
    perf = 100 * correct / total
    print('Epoch %d: Accuracy of the network on the 10000 test images: %d %%' % (epoch, perf))
    return perf

best_perf = 0.
stats = dict(grad_loss=[], classify_loss=[])
# Train the Model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  
        # Convert torch tensor to Variable
        images = Variable(images.view(-1, 28*28)).cuda()
        labels = Variable(labels).cuda()
        # Fc1        
        # Forward + Backward + Optimize
        optimizer_fc1.zero_grad()
        out, grad1 = net.forward_fc1(images)
        out.backward(grad1.detach().data)
        optimizer_fc1.step()

        # relu+Fc2
        # Forward + Backward + Optimize
        out = net.relu(out.detach())
        optimizer_fc2.zero_grad()
        out, grad2 = net.forward_fc2(out)
        out.backward(grad2.detach().data)
        optimizer_fc2.step()

        # synthetic model
        # Forward + Backward + Optimize
        grad_optimizer.zero_grad()
        optimizer.zero_grad()
        fc1, fc2, grad_fc1, grad_fc2 = net(images, labels)
        backprop_grads = {}
        handle_fc1 = fc1.register_hook(save_grad('fc1'))
        handle_fc2 = fc2.register_hook(save_grad('fc2'))
        outputs = fc2
        loss = classificationCriterion(outputs, labels)
        loss.backward(retain_variables=True)
        handle_fc1.remove()
        handle_fc2.remove()
        grad_loss = syntheticCriterion(grad_fc1, backprop_grads['fc1'].detach())
        grad_loss += syntheticCriterion(grad_fc2, backprop_grads['fc2'].detach())

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
            torch.save(net.state_dict(), model_name+'_model_best.pkl')
        net.train()

# Save the Model ans Stats
pkl.dump(stats, open(model_name+'_stats.pkl', 'w'))
torch.save(net.state_dict(), model_name+'_model.pkl')
plot(stats, name=model_name)
