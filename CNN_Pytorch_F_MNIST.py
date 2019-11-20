import torch
from itertools import product

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from NetworkF_MNIST import Network
from NetworkF_MNIST import Train
from NetworkF_MNIST import RunBuilder

from collections import OrderedDict

from torch.utils.tensorboard import SummaryWriter

torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)


# hyper parameters
# batch_size = 100
# lr = 0.01

def get_all_preds(model, data_loader):
    all_preds = torch.tensor([])
    for batch in data_loader:
        images, labels = batch

        preds = model(images)
        all_preds = torch.cat((all_preds, preds), dim=0)

    return all_preds


def train_with_multiple_params():
    epochs = 10
    network = Network()
    train_set = torchvision.datasets.FashionMNIST(
        root='./data/FashionMNIST',
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()])
    )

    parameters = OrderedDict(
        lr=[0.01, 0.001],
        batch_size=[10, 100, 1000]
    )

    for run in RunBuilder.get_runs(parameters):
        print(f'##### {run} ######')
        train = Train()
        ret = train.train_data_set(train_set, network, run, epochs)

        pred_loader = torch.utils.data.DataLoader(train_set, batch_size=10000)
        train_preds = get_all_preds(network, pred_loader).argmax(dim=1)

        cm = confusion_matrix(train_set.targets, train_preds)
        print(cm)


def train_and_test():
    epochs = 10
    parameters = OrderedDict(
        lr=[0.01],
        batch_size=[1000]
    )

    network = Network()
    train_set = torchvision.datasets.FashionMNIST(
        root='./data/FashionMNIST',
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()])
    )

    train = Train()
    ret = train.train_data_set(train_set, network, RunBuilder.get_runs(parameters)[0], epochs)
    network = ret['network']

    test_set = torchvision.datasets.FashionMNIST(
        root='F_MNIST_data_test/',
        download=True,
        train=False,
        transform=transforms.Compose([transforms.ToTensor()])
    )

    ret = train.test_data_set(test_set, network, RunBuilder.get_runs(parameters)[0])
    print(f"total loss test: {ret['total_loss']}")
    print(f"correctly predicted: {ret['total_correct']}")
    print(f"actual correct: {test_set.targets.numpy().shape[0]}")
    print(f"% correct: {ret['total_correct']}/{test_set.targets.numpy().shape[0]}")


def exec_main():
    # train_with_multiple_params
    train_and_test()


if __name__ == '__main__':
    exec_main()
