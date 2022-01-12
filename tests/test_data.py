from tests import _PATH_DATA
import torch
import os
import numpy as np
import pytest

def dataload():
    # Fetching train
    tr_images = torch.load(os.path.join(_PATH_DATA, "images_train.pt"))
    tr_labels = torch.load(os.path.join(_PATH_DATA, "labels_train.pt"))

    # Fetching test
    test = np.load(os.path.join(_PATH_DATA, "test.npz"))
    test_images = torch.Tensor(test.f.images)
    test_labels = torch.Tensor(test.f.labels)

    return tr_images, tr_labels, test_images, test_labels


@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
class TestClass:
    tr_images, tr_labels, test_images, test_labels = dataload()
    N_train = 40000
    N_test = 5000

    # Testing trainingdata
    def test_traindata(self):
        # Image structure
        assert len(self.tr_images) == self.N_train, "Train data did not have the correct number of images"
        assert all([img.size() == torch.Size([28, 28]) for img in self.tr_images]), "At least one train image didn't have the right dimensions."

        # Labels
        assert len(self.tr_labels) == self.N_train, "Train data did not have the correct number of labels"
        assert all(i in torch.unique(self.tr_labels) for i in range(10)), "At least one train data label wasn't correct."

    def test_testdata(self):
        # Image structure
        assert len(self.test_images) == self.N_test, "Test data did not have the correct number of images"
        assert all([img.size() == torch.Size([28, 28]) for img in self.test_images]), "At least one test image didn't have the right dimensions."

        # Labels
        assert len(self.test_labels) == self.N_test, "Test data did not have the correct number of labels"
        assert all(i in torch.unique(self.test_labels) for i in range(10)), "At least one test data label wasn't correct."