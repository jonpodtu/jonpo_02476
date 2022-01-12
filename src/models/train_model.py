import os

import hydra
import matplotlib.pyplot as plt
import torch
from hydra.utils import to_absolute_path
from model import MyAwesomeModel
from omegaconf import DictConfig
from torch import optim
from torch.utils.data import DataLoader, TensorDataset


@hydra.main(config_path="config", config_name="training_conf.yaml")
def main(cfg: DictConfig):
    print("Training day and night...")

    trainset = TensorDataset(
        torch.load(to_absolute_path(cfg.paths["images"])),
        torch.load(to_absolute_path(cfg.paths["labels"])),
    )
    train_set = DataLoader(
        trainset, batch_size=cfg.hyperparameters["batch_size"], shuffle=True
    )
    print("The trainingset is {} long!".format(len(trainset)))
    # Criterion: We use the negative log likelihood as our output is logSoftMax
    criterion = torch.nn.NLLLoss()
    if cfg.hyperparameters["optimizer"].lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=cfg.hyperparameters["lr"])
    elif cfg.hyperparameters["optimizer"].lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=cfg.hyperparameters["lr"])
    else:
        print('Not a valid optimizer! Please choose "adam" or "sgd".')

    # Epochs and train_loss
    epochs = cfg.hyperparameters["epochs"]
    train_loss = []

    for e in range(epochs):
        # Dropout should be one ie. we set model to training mode
        model.train()
        running_loss = 0

        """
        The for-loop does the following:
            We use convolutional network, so first we unsqueeze
            Resets the gradients
            1.  Makes a forward pass through the network
            2.  Use the logits to calculate the loss. We use the computed
                logits from our output.
            3.  Perform a backward pass through the network with
                loss.backward() to calculate the gradients
            4.  Take a step with the optimizer to update the weights
        """

        for images, labels in train_set:
            images = images.unsqueeze(1)
            optimizer.zero_grad()

            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        train_loss.append(loss.item())
        print("[%d] loss: %.3f" % (e + 1, running_loss / len(train_set)))

    models_dir = to_absolute_path(cfg.paths["model_save"])
    os.makedirs(models_dir, exist_ok=True)
    torch.save(model, to_absolute_path(os.path.join(models_dir, "trained_model.pt")))

    fig_dir = to_absolute_path(cfg.paths["figures"])
    os.makedirs(fig_dir, exist_ok=True)
    plt.plot(train_loss, label="Training loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(fig_dir, "loss.png"))


if __name__ == "__main__":
    model = MyAwesomeModel()
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    main()
