import click

import torch
from torch.utils.data import TensorDataset, DataLoader

from model import MyAwesomeModel

@click.command()
@click.argument('load_model_from', type=click.Path(exists=True))

def main(load_model_from):
    print("Evaluating until hitting the ceiling")

    # TODO: Implement evaluation logic here
    # First we load in the already trained model
    model = MyAwesomeModel()
    state_dict = torch.load(load_model_from)
    model.load_state_dict(state_dict)

    # Load in the test set given arguments (TODO: Make more versatile - extend to different formats)
    testset = TensorDataset(torch.load("data/processed/images_test.pt"), torch.load("data/processed/labels_test.pt"))
    test_set = DataLoader(testset, batch_size=64, shuffle=True)

    # We ensure our model is set to eval mode and turn of gradient
    # Turning off gradient will improve speed
    model.eval()

    # Let's save the test loss as well
    test_losses = []
    criterion = torch.nn.NLLLoss()

    with torch.no_grad():
        running_loss = 0
        accuracies = []
        for images, labels in test_set:
            images = images.unsqueeze(1)

            # Test loss
            output = model.forward(images)
            running_loss += criterion(output, labels).item()
            
            # Test Accuracy
            ps = (torch.exp(output))
            _, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy = torch.mean(equals.type(torch.FloatTensor))
            accuracies.append(accuracy.item())
        mean_acc = sum(accuracies) / len(accuracies)
        print(f'Accuracy: {mean_acc*100}%')

if __name__ == '__main__':
    main()