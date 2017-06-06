import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms

class mnist():
    def __init__(self, args):
        # MNIST Dataset 
        train_dataset = dsets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

        test_dataset = dsets.MNIST(root='./data',
                           train=False,
                           transform=transforms.ToTensor())

        # Data Loader (Input Pipeline)
        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=False)
        self.input_dims = 784
        self.num_classes = 10
        self.in_channel = 1
        self.num_train = len(train_dataset)

class cifar10():
    def __init__(self, args):
        # CIFAR 10 Dataset
        transform = self.image_transform()
        train_dataset = dsets.CIFAR10(root='./data/',
                               train=True, 
                               transform=transform,
                               download=True)

        test_dataset = dsets.CIFAR10(root='./data/',
                              train=False, 
                              transform=transforms.ToTensor())

        # Data Loader (Input Pipeline)
        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=100, 
                                           shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=100, 
                                          shuffle=False)
        self.num_classes = 10
        self.in_channel = 3
        self.num_train = len(train_dataset)

    def image_transform(self):
        # Image Preprocessing 
        # WARNING: difference from other settings: crop to 28*28
        transform = transforms.Compose([
            transforms.Scale(40),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(28),
            transforms.ToTensor()])
        return transform
