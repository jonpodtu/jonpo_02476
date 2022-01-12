import argparse
import sys

import matplotlib.pyplot as plt
import torch
from model import MyAwesomeModel
from torch import optim

from data import mnist


class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
    
    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.1)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement training loop here
        model = MyAwesomeModel()
        train_set, _ = mnist()

        # Criterion: We use the negative log likelihood as our output is logSoftMax
        criterion = torch.nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        # Epochs and train_loss
        epochs = 30
        train_loss = []

        for e in range(epochs):
            # Dropout should be one ie. we set model to training mode
            model.train()
            running_loss = 0

            '''
            The for-loop does the following:
                We use convolutional network, so first we unsqueeze
                Resets the gradients
                
                1. Makes a forward pass through the network
                2. Use the logits to calculate the loss. We use the computed logits from our output.
                3. Perform a backward pass through the network with loss.backward() to calculate the gradients
                4. Take a step with the optimizer to update the weights
            '''

            for images, labels in train_set:
                images = images.unsqueeze(1)
                optimizer.zero_grad()
                
                
                output = model(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            train_loss.append(loss.item())
            print('[%d] loss: %.3f' % (e + 1, running_loss / len(train_set)))
            
        torch.save(model.state_dict(), 'trained_model.pt')

        plt.plot(train_loss, label = "Training loss")
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig("loss.png")

    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('load_model_from', default="")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement evaluation logic here
        # First we load in the already trained model
        model = MyAwesomeModel()
        state_dict = torch.load(args.load_model_from)
        model.load_state_dict(state_dict)

        # Load in the test set 
        _, test_set = mnist()
        
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
    TrainOREvaluate()
    