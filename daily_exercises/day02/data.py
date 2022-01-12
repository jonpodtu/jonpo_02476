import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])

def combine_data():
    data_path = '../data/corruptmnist/'
    data_all = [np.load('%strain_%s.npz' % (data_path, i)) for i in range(5)]

    merged_data = {}
    for data in data_all:
        [merged_data.update({k: v}) for k, v in data.items()]
    np.savez('%strain_data.npz' % data_path, **merged_data)    


def mnist():
    # Check if traindata is combined
    mnist_path = '../data/corruptmnist/'
    if not os.path.isfile('%strain_data.npz' % mnist_path):
        combine_data()
    else:
        try:
            test = np.load('%stest.npz' % mnist_path)
            train = np.load('%strain_data.npz' % mnist_path)  
            print('Successfully imported existing files')
        except OSError:
            print('Could not load the files')

    # Let's reorganize the NPZ files
    images_train = torch.Tensor(train.f.images)
    labels_train = torch.Tensor(train.f.labels).type(torch.LongTensor)
    trainset = TensorDataset(images_train, labels_train)

    images_test = torch.Tensor(test.f.images)
    labels_test = torch.Tensor(test.f.labels).type(torch.LongTensor)
    testset = TensorDataset(images_test, labels_test)

    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = DataLoader(testset, batch_size=64, shuffle=True)

    return trainloader, testloader