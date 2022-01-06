from logging import raiseExceptions
import click
import torch
from torch.jit import Error
from model import MyAwesomeModel
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# We use click for to pass which model we want to load in
@click.command()
@click.argument("load_model_from", type=click.Path(exists=True))
@click.argument("data_images", type=click.Path(exists=True))
def main(load_model_from, data_images):
    """
    Tests the given model using the standard testset
    """
    print("Evaluating until hitting the ceiling")

    def normalize(tensor):
        mu = torch.mean(tensor)
        std = torch.std(tensor)
        output = (tensor - mu) / std
        return output

    # First we load in the already trained model
    model = MyAwesomeModel()
    state_dict = torch.load(load_model_from)
    model.load_state_dict(state_dict)

    # Load in the test set given arguments
    if not data_images.lower().endswith('.npz'):
        raise Exception("Only accepts .npz files")
    else:
        test = np.load(data_images)
        print(test)
        testset = TensorDataset(
            normalize(torch.Tensor(test.f.images)),
            torch.Tensor(test.f.labels).type(torch.LongTensor),
        )

    test_set = DataLoader(testset, batch_size=64, shuffle=True)

    model.eval()
    criterion = torch.nn.NLLLoss()

    # Turning off gradient will improve speed
    with torch.no_grad():
        running_loss = 0
        accuracies = []
        for images, labels in test_set:
            images = images.unsqueeze(1)

            # Test loss
            output = model.forward(images)
            running_loss += criterion(output, labels).item()

            # Test Accuracy
            ps = torch.exp(output)
            _, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy = torch.mean(equals.type(torch.FloatTensor))
            accuracies.append(accuracy.item())
        mean_acc = sum(accuracies) / len(accuracies)
        print(f"Accuracy: {mean_acc*100}%")


if __name__ == "__main__":
    main()
