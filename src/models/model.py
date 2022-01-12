import torch.nn.functional as F
from torch import nn

import logging
log = logging.getLogger(__name__)

class MyAwesomeModel(nn.Module):
    """
    A 2D convolutional network.

    We use convolutional layers because they are known to be
    translational invariant, which is relevant as we work with
    rotated corrupted data.
    """
    def __init__(self, cfg):
        super().__init__()

        # Input layer.
        self.conv1 = nn.Conv2d(cfg.conv1['in'], 
                               cfg.conv1['out'], 
                               kernel_size=cfg.conv1['kernel_size'])

        # Hidden layer(s)
        self.conv2 = nn.Conv2d(cfg.conv2['in'], 
                               cfg.conv2['out'], 
                               kernel_size=cfg.conv2['kernel_size'])
        self.fc1_in = cfg.fc1['in']
        self.fc1 = nn.Linear(self.fc1_in, cfg.fc1['out'])

        # Output layer
        self.output = nn.Linear(cfg.output['in'], cfg.output['out'])

        # Dropout module
        self.dropout = nn.Dropout(p=cfg.fc1['dropout'])

    def forward(self, x):
        """
        Forward passes the input through multiple convolutional
        and fully connected layers.

            Parameters:
                x: Input values from a 28x28 picture given in
                grayscale (1 channel)

            Returns:
                x: Output of the neural network given the input
                and the current parameters
        """
        # Convolutional and maxpoolinf
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))

        # Flatten for fully connected (fc) layers
        x = x.view(-1, self.fc1_in)

        #  Using dropout on fc layers
        x = self.dropout(F.relu(self.fc1(x)))

        # Forward the output layer.
        x = F.log_softmax(self.output(x), dim=1)

        return x
