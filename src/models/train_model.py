import torch
from torch import optim
from torch.utils.data import TensorDataset, DataLoader


import matplotlib.pyplot as plt

from model import MyAwesomeModel

def main():
    print("Training day and night")

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    trainset = TensorDataset(torch.load('data/processed/images_train.pt'), torch.load('data/processed/labels_train.pt'))
    train_set = DataLoader(trainset, batch_size=64, shuffle=True)

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
        
    torch.save(model.state_dict(), 'models/trained_model.pt')

    plt.plot(train_loss, label = "Training loss")
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig("reports/figures/loss.png")

if __name__ == '__main__':
    main()