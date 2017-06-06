from model import *
from plot import *
import torch
from torch.autograd import Variable
import ipdb

class classifier():

    def __init__(self, args, data):
        self.train_loader = data.train_loader
        self.test_loader = data.test_loader
        self.batch_size = args.batch_size
        self.num_train = data.num_train
        self.num_classes = data.num_classes
        assert args.model_type == 'mlp' or args.model_type == 'cnn'
        if args.model_type == 'mlp':
            self.net = mlp(args.conditioned, data.input_dims, data.num_classes, hidden_size=256)
        elif args.model_type == 'cnn':
            self.net = cnn(args.conditioned, data.num_classes)

        if args.use_gpu:
            self.net.cuda()

        self.classificationCriterion = nn.CrossEntropyLoss()
        self.syntheticCriterion = nn.MSELoss()
        self.plot = args.plot
        self.num_epochs = args.num_epochs
        self.model_name = args.model_name
        if args.conditioned:
            self.model_name += '.conditioned'
        self.conditioned = args.conditioned
        self.best_perf = 0.
        self.stats = dict(grad_loss=[], classify_loss=[])

    def optimizer_module(self, optimizer, forward, out, label_onehot=None):
        optimizer.zero_grad()
        out, grad = forward(out, label_onehot)
        out.backward(grad.detach().data)
        optimizer.step()
        out = out.detach()
        return out

    def save_grad(self, name):
        def hook(grad):
            self.backprop_grads[name] = grad
            self.backprop_grads[name].volatile = False
        return hook

    def optimizer_dni_module(self, images, labels, label_onehot, grad_optimizer, optimizer, forward):
        # synthetic model
        # Forward + Backward + Optimize
        grad_optimizer.zero_grad()
        optimizer.zero_grad()
        outs, grads = forward(images, label_onehot)
        self.backprop_grads = {}
        handles = {}
        keys = []
        for i, (out, grad) in enumerate(zip(outs, grads)):
            handles[str(i)] = out.register_hook(self.save_grad(str(i)))
            keys.append(str(i))
        outputs = outs[-1]
        loss = self.classificationCriterion(outputs, labels)
        loss.backward(retain_variables=True)
        for (k, v) in handles.items():
            v.remove()
        grad_loss = 0.
        for k in keys:
            grad_loss += self.syntheticCriterion(grads[int(k)], self.backprop_grads[k].detach())

        grad_loss.backward()
        grad_optimizer.step()
        self.stats['grad_loss'].append(grad_loss.data[0])
        self.stats['classify_loss'].append(loss.data[0])
        return loss, grad_loss

    def train_model(self):
        for epoch in range(self.num_epochs):
            for i, (images, labels) in enumerate(self.train_loader):  
                # Convert torch tensor to Variable
                labels_onehot = torch.zeros([labels.size(0), self.num_classes])
                labels_onehot.scatter_(1, labels.unsqueeze(1), 1)  
                images = Variable(images).cuda()
                labels = Variable(labels).cuda()
                labels_onehot = Variable(labels_onehot).cuda()
                out = images
                # Forward + Backward + Optimize
                for (optimizer, forward) in zip(self.net.optimizers, self.net.forwards):
                    if self.conditioned:
                        out = self.optimizer_module(optimizer, forward, out, labels_onehot)
                    else:
                        out = self.optimizer_module(optimizer, forward, out)
                # synthetic model
                # Forward + Backward + Optimize
                loss, grad_loss = self.optimizer_dni_module(images, labels, labels_onehot, 
                                          self.net.grad_optimizer, self.net.optimizer, self.net)
                
                if (i+1) % 100 == 0:
                    print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Grad Loss: %.4f' 
                         %(epoch+1, self.num_epochs, i+1, self.num_train//self.batch_size, loss.data[0], grad_loss.data[0]))

            if (epoch+1) % 10 == 0:
                perf = self.test_model(epoch+1)    
                if perf > self.best_perf:
                    torch.save(self.net.state_dict(), self.model_name+'_model_best.pkl')
                    self.net.train()

        # Save the Model ans Stats
        pkl.dump(self.stats, open(self.model_name+'_stats.pkl', 'wb'))
        torch.save(self.net.state_dict(), self.model_name+'_model.pkl')
        if self.plot:
            plot(self.stats, name=self.model_name)

    def test_model(self, epoch):
        # Test the Model
        self.net.eval()
        correct = 0
        total = 0
        for images, labels in self.test_loader:
            images = Variable(images).cuda()
            outputs = self.net(images)
            outputs = outputs[-1]
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels).sum()
        perf = 100 * correct / total
        print('Epoch %d: Accuracy of the network on the 10000 test images: %d %%' % (epoch, perf))
        return perf

        

