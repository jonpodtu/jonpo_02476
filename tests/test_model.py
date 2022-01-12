from src.models.model import MyAwesomeModel
from tests import _PATH_DATA
import torch
from torch.utils.data import DataLoader, TensorDataset
import os
import unittest

class TestClass(unittest.TestCase):
    # Load data
    tr_images = torch.load(os.path.join(_PATH_DATA, "images_train.pt"))
    tr_labels = torch.load(os.path.join(_PATH_DATA, "labels_train.pt"))
    train_set = DataLoader(TensorDataset(tr_images, tr_labels),
                           batch_size=64,
                           shuffle=True)
    
    model = MyAwesomeModel()
    orig_parameters = list(model.parameters())
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.NLLLoss()

    def test_optimization(self):
        # We simply take one training step
        for images, labels in self.train_set:
            images = images.unsqueeze(1)
            self.optimizer.zero_grad()

            output = self.model(images)
            loss = self.criterion(output, labels)
            loss.backward()
            self.optimizer.step()
            self.assertListEqual(
                self.orig_parameters, list(self.model.parameters())
            ), "The model parameters are not being optimized"
            break
