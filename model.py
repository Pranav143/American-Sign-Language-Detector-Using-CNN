import torch.nn as nn
import torch
import torch.nn.functional as F

class BasicCNN(nn.Module): # The CNN used in the first part to overfit data
    def __init__(self):
        super(BasicCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, stride=1) # This reduces 56x56 to 54x54
        self.pool1 = nn.MaxPool2d(2, 2) # Reduces 54x54 to 27x27
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=5, stride=1) #Reduces 27x27 to 23x23
        self.pool2 = nn.MaxPool2d(2,2) # Reduces 22x22 to 11x11
        self.fc1 = nn.Linear(8 * 11 * 11, 100) # There are 11*11 'pixels' from 8 feature maps, and this layer has 100 neurons
        self.fc2 = nn.Linear(100, 10) # 10 outputs since we have 10 letter classes from A-K

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1) # Restructure the 2D grid into a 1 dimensional list to feed to the fully connected layer
        x = F.relu(self.fc1(x))
        x = (self.fc2(x)) # Lastly, simply run through another linear layer.
        return x
    
class CNN(nn.Module): # The CNN used when testing different combinations of hyperparameters. Below model is the best found
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=15, kernel_size=3, stride=1) # This reduces 56x56 to 54x54
        self.pool1 = nn.MaxPool2d(2, 2) # Reduces 54x54 to 27x27
        self.conv2 = nn.Conv2d(in_channels=15, out_channels=15, kernel_size=3, stride=1) #Reduces 27x27 to 25x25
        self.pool2 = nn.MaxPool2d(2,2) # Reduces 25x25 to 12x12
        self.fc1 = nn.Linear(15 * 12 * 12, 30) # There are 12*12 'pixels' from 15 feature maps
        self.fc2 = nn.Linear(30, 10) # 10 outputs since we have 10 letter classes from A-K
            
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1) # Restructure the 2D grid into a 1 dimensional list to feed to the fully connected layer
        x = F.relu(self.fc1(x))
        x = (self.fc2(x)) # Lastly, simply run through another linear layer.
        return x  

class BnCNN(nn.Module): # Batch norm using CNN
    def __init__(self):
        super(BnCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=15, kernel_size=3, stride=1) # This reduces 56x56 to 54x54
        self.batch1 = nn.BatchNorm2d(15)
        self.pool1 = nn.MaxPool2d(2, 2) # Reduces 54x54 to 27x27
        self.conv2 = nn.Conv2d(in_channels=15, out_channels=15, kernel_size=3, stride=1) #Reduces 27x27 to 25x25
        self.batch2 = nn.BatchNorm2d(15) # 15 input channels into the batchnorm
        self.pool2 = nn.MaxPool2d(2,2) # Reduces 25x25 to 12x12
        self.fc1 = nn.Linear(15 * 12 * 12, 30) # There are 12*12 'pixels' from 15 feature maps
        self.batch3 = nn.BatchNorm1d(30) # 1 dimension batch norm since we flattened things out
        self.fc2 = nn.Linear(30, 10) # 10 outputs since we have 10 letter classes from A-K
            
    def forward(self, x):
        x = self.pool1(F.relu(self.batch1((self.conv1(x)))))
        x = self.pool2(F.relu(self.batch2((self.conv2(x)))))
        x = x.view(x.size(0), -1) # Restructure the 2D grid into a 1 dimensional list to feed to the fully connected layer
        x = F.relu(self.batch3((self.fc1(x))))
        x = (self.fc2(x)) # Lastly, simply run through another linear layer.
        return x 

class CeCNN(nn.Module): # Cross entropy using CNN
    def __init__(self):
        super(CeCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=15, kernel_size=3, stride=1) # This reduces 56x56 to 54x54
        self.pool1 = nn.MaxPool2d(2, 2) # Reduces 54x54 to 27x27
        self.conv2 = nn.Conv2d(in_channels=15, out_channels=15, kernel_size=3, stride=1) #Reduces 27x27 to 25x25
        self.pool2 = nn.MaxPool2d(2,2) # Reduces 25x25 to 12x12
        self.fc1 = nn.Linear(15 * 12 * 12, 30) # There are 12*12 'pixels' from 15 feature maps
        self.fc2 = nn.Linear(30, 10) # 10 outputs since we have 10 letter classes from A-K
            
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1) # Restructure the 2D grid into a 1 dimensional list to feed to the fully connected layer
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=10) # Lastly, simply run through a softmax function to prep for cross entropy loss
        return x      
    
class BothCNN(nn.Module): # Both Batch norm and cross entropy using CNN
    def __init__(self):
        super(BothCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=15, kernel_size=3, stride=1) # This reduces 56x56 to 54x54
        self.batch1 = nn.BatchNorm2d(15)
        self.pool1 = nn.MaxPool2d(2, 2) # Reduces 54x54 to 27x27
        self.conv2 = nn.Conv2d(in_channels=15, out_channels=15, kernel_size=3, stride=1) #Reduces 27x27 to 25x25
        self.batch2 = nn.BatchNorm2d(15)
        self.pool2 = nn.MaxPool2d(2,2) # Reduces 25x25 to 12x12
        self.fc1 = nn.Linear(15 * 12 * 12, 30) # There are 12*12 'pixels' from 15 feature maps
        self.batch3 = nn.BatchNorm1d(30)
        self.fc2 = nn.Linear(30, 10) # 10 outputs since we have 10 letter classes from A-K
            
    def forward(self, x):
        x = self.pool1(F.relu(self.batch1((self.conv1(x)))))
        x = self.pool2(F.relu(self.batch2((self.conv2(x)))))
        x = x.view(x.size(0), -1) # Restructure the 2D grid into a 1 dimensional list to feed to the fully connected layer
        x = F.relu(self.batch3((self.fc1(x))))
        x = F.softmax(self.fc2(x), dim=10) # Lastly, run through softmax for cross entropy
        return x 
