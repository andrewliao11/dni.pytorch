import argparse
from train import *
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms

class mnist():
    def __init__(self, args):
        # MNIST Dataset 
        train_dataset = dsets.MNIST(root='../data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

        test_dataset = dsets.MNIST(root='../data',
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
        self.num_train = len(train_dataset)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='DNI')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--model_type', choices=['mlp', 'cnn'], default='mlp',
                    help='currently support mlp and cnn')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--model_name', type=str, default='mlp_dni',
                    help='used to save stats and model')
    parser.add_argument('--conditioned', type=bool, default=False)
    parser.add_argument('--plot', type=bool, default=False)
    parser.add_argument('--use_gpu', type=bool, default=True)

    args = parser.parse_args()
    data = mnist(args)
    m = classifier(args, data)
    m.train_model()
