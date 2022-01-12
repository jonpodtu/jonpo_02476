from src.models.model import MyAwesomeModel
from tests import _PATH_DATA
import torch
from torch.utils.data import DataLoader, TensorDataset
import os

class TestClass():
    # Load data
    tr_images = torch.load(os.path.join(_PATH_DATA, "images_train.pt"))
    tr_labels = torch.load(os.path.join(_PATH_DATA, "labels_train.pt"))
    train_set = DataLoader(TensorDataset(tr_images, tr_labels),
                           batch_size=64,
                           shuffle=True)
    
    model = MyAwesomeModel()
    model.train()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.NLLLoss()

    def test_loss_not_zero(self):
        # We simply take one training step
        for images, labels in self.train_set:
            images = images.unsqueeze(1)
            self.optimizer.zero_grad()

            output = self.model(images)
            loss = self.criterion(output, labels)
            assert loss != 0, "The loss was zero. This causes problems learning."
            break

    def test_change_of_weights(self):
        for images, labels in self.train_set:
            images = images.unsqueeze(1)

            # Get model parameters
            before = {}
            for key in self.model.state_dict():
                before[key] = self.model.state_dict()[key].clone()
            
            self.optimizer.zero_grad()
            output = self.model(images)
            loss = self.criterion(output, labels)
            loss.backward()
            self.optimizer.step()

            # Get new model parameters
            after = {}
            for key in self.model.state_dict():
                after[key] = self.model.state_dict()[key].clone()

            # Compare bias and weights
            for key in before:
                assert (before[key] != after[key]).any()
            break