import torch.nn.functional as F
from torch import nn


class MyAwesomeModel(nn.Module):
    '''
    A 2D convolutional network. 
    
    We use convolutional layers because they are known to be 
    translational invariant, which is relevant as we work with
    rotated corrupted data. 
    '''
    def __init__(self):
        super().__init__()

        # Input layer.
        self.conv1 = nn.Conv2d(1, 10, kernel_size = 5)
        
        # Hidden layer(s)
        self.conv2 = nn.Conv2d(10, 20, kernel_size = 5)
        self.fc1 = nn.Linear(320, 50)
        
        # Output layer
        self.output = nn.Linear(50, 10)

        # Dropout module
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # Convolutional and maxpoolinf
        x = F.relu(F.max_pool2d(self.conv1(x),2))
        x = F.relu(F.max_pool2d(self.conv2(x),2))
        
        # Flatten for fully connected (fc) layers
        x = x.view(-1, 320)

        #  Using dropout on fc layers
        x = self.dropout(F.relu(self.fc1(x)))
        
        # Forward the output layer. 
        x = F.log_softmax(self.output(x), dim = 1)

        return x