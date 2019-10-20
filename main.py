import torch
import torch.nn as nn
import torch.utils.data 
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
import time
import matplotlib.pyplot as plt

import numpy as np
from torchsummary import summary 

from try_model import CNN, BnCNN, CeCNN, BothCNN

start_time = time.time() #Note down start of code run 

epochs = 40
batch_size = 25
learning_rate = 0.01
seed = 1
torch.manual_seed(seed)

"""DATA LOADING"""
"""MAKE SURE YOU CHANGE THE ROOT HERE TO YOUR FOLDER, IF YOU END UP RUNNING THIS CODE"""
personal_data = torchvision.datasets.ImageFolder(root=r'\Users\Pranav\PycharmProjects\assignment 4\Agnihotri_1004148914', 
                                        transform=torchvision.transforms.ToTensor()) # Only my personal pictures

data = torchvision.datasets.ImageFolder(root=r'\Users\Pranav\PycharmProjects\assignment 4\asl_images', 
                                        transform=torchvision.transforms.ToTensor()) # Full dataset

#_____________________________________________________________________________________________________________
"""PART 1 CODE: VISUALIZING A BATCH"""

letters = ('A', 'B', 'C', 'D',
           'E', 'F', 'G', 'H', 'I', 'K')

data_loader_for_visualization = DataLoader(personal_data, batch_size=4, shuffle=True)

# Function below taken from PyTorch guide, some modifications made, but credit where credit is due 
def imshow(img):
    
    npimg = img.numpy() #Convert from a tensor to numpy 
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

data_iterater = iter(data_loader_for_visualization) #allows us to iterate through batches using the next function, which is what enumerate does automatically
images, labels = data_iterater.next() # Gives us the first batch 


imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%7s' % letters[labels[j]] for j in range(4))) #Print out the labels spaced out 7 spaces

#__________________________________________________________________________________________________________________

"""Data Transforming with mean and standard deviation calculations"""
# These are here just to not have to recompute this everytime. Uncomment these out and comment out the below code for finding mean and std if you don't want to waste computational power
#channel_mean = [0.7754194140434265, 0.6673466563224792, 0.5820994973182678]
#channel_std = [0.09336044, 0.1683979, 0.19541572]

# Now we need to calculate the mean for each channel
channel_mean = [0,0,0]
channel_sum = [0,0,0]
channel_std = [0,0,0]

for i in range(0,len(data),1): # We sum each channel
    channel_sum[0] += data[i][0][0].sum() 
    channel_sum[1] += data[i][0][1].sum()
    channel_sum[2] += data[i][0][2].sum()
    
# To get the mean, we divide by the number of "pixels" times number of images. The code below translates to 30 * 56 * 56
channel_mean[0] += channel_sum[0]/ (len(data) * len(data[0][0][0]) * len(data[0][0][0][0])) 
channel_mean[1] += channel_sum[1]/ (len(data) * len(data[0][0][0]) * len(data[0][0][0][0]))
channel_mean[2] += channel_sum[2]/ (len(data) * len(data[0][0][0]) * len(data[0][0][0][0]))

# Our output are tensors, so we just need to grab the number from there 
for i in range(len(channel_mean)):
    channel_mean[i] = channel_mean[i].item() 
    
# Now we calculated the std for each channel 
r = [0]*len(data)
g = [0]*len(data)
b = [0]*len(data)

# Now we separate the dataset into individual channels
for i in range(len(data)):
    r[i] = data[i][0][0].numpy()
    g[i] = data[i][0][1].numpy()
    b[i] = data[i][0][2].numpy()  

# Amalgamate all the data for each channel into their own numpy arrays
r = np.asarray(r)
g = np.asarray(g)
b = np.asarray(b)

# Finally, calculate the std and store them in the std channel array
channel_std[0] = np.std(r)
channel_std[1] = np.std(g)
channel_std[2] = np.std(b)


# Now we can create a full transformation pipeline including the normalizaton
transforms= torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((channel_mean[0], channel_mean[1], channel_mean[2]), (channel_std[0], channel_std[1], channel_std[2]))])

# Get the same dataset again but this time, we have the normalization component 
data = torchvision.datasets.ImageFolder(root=r'\Users\Pranav\PycharmProjects\assignment 4\asl_images', 
                                        transform=transforms) 

#_________________________________________________________________________________________________
"""TRAIN VALIDATION SPLITTING OF DATA"""
# So allow me to just explain my process for splitting the dataset. When I import the dataset,
# I get all 1370 images in order of class, just the way it is in the folders. This means that
# if I take the first 80% of images in each class for training and the latter 20% of each class
# for validation, then I effectively create those sets with DIFFERENT people's photos, which 
# is the method that I think is best. So the code below does precisely that.
# It may seem a little janky but I didn't really find any in built function to do this kind of splitting
# At the very end I use the DataLoader class to create my train_data and valid_data that are shuffled to 
# take away their inherent order. 

class_index_end = [137,272,409,547,684,821,957,1095,1231,1369] # This is the index of the last element of each class in the dataset. Example, last element of class 0 is at element 137, and so on
class_index_start = [0, 138, 273, 410, 548, 685, 822, 958, 1096, 1232] # This is the index that each class starts at 

class_split = [0]*len(class_index_end) # This notes down the length of each class in terms of how many elements they have, because for some reason this dataset isn't fully even. There aren't the same number of As and Bs and Cs etc
class_split_train = [0]*len(class_index_end)
class_split_valid = [0]*len(class_split_train)

for i in range(len(class_index_end)):
    if i == 0 :
        class_split[i] += class_index_end[i]
    else:
        class_split[i] += class_index_end[i] - class_index_end[i-1]

for i in range(len(class_split)):
    class_split_train[i] = int(class_split[i] * 0.80) # We note down how many elements 80% of a class is, rounded to an int
    class_split_valid[i] += class_split[i] - class_split_train[i] # The rest are for validation set 

# Just turning these into np array for easy manipulations
class_split = np.asarray(class_split)
class_split_train = np.asarray(class_split_train)
class_split_valid = np.asarray(class_split_valid)

class_index_end = np.asarray(class_index_end)
class_index_start = np.asarray(class_index_start)

class_index_train_end = class_index_start + class_split_train # This is where the train set ends for each class 

# Now, to split into test and train, I want to take the first 80% of each class and put it into test, and take the latter 20% and put it into train
data_for_train = []
label_for_train = []
data_for_valid = []
label_for_valid = []

for i in range(len(class_index_end)):
    start = class_index_start[i]
    train_end = class_index_train_end[i]
    end = class_index_end[i]
    for j in range(start, train_end + 1, 1):
        data_for_train.append([data[j][0]]) # 0th element is the actual image data
        label_for_train.append(data[j][1])  # 1st element of the tuple is the label
    for m in range(train_end + 1, end + 1, 1):
        data_for_valid.append([data[j][0]])
        label_for_valid.append(data[j][1])

# So we have these arrays aren't of type dataset, and in order to load them into the data loader, 
# I wrote a class that will turn them into map-style datasets 
        
class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y
        

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        sample_train = self.X[index][0]
        sample_label = self.y[index]

        return sample_train, sample_label

training_set = ImageDataset(data_for_train, label_for_train)
validation_set = ImageDataset(data_for_valid, label_for_valid)

# And finally, after all that, we have our train_data and valid_data dataloaders 
train_data = DataLoader(training_set, batch_size=batch_size, shuffle=True) 
valid_data = DataLoader(validation_set, batch_size=batch_size, shuffle=True) # We take the entire validation set as one batch 

#________________________________________________________________________________________________________________________

"""TRAINING LOOP"""

#Please note that in this training loop, read the comments very carefully. Depending
#on if you're usng MSE loss or Cross Entropy etc there are different things you have to
#comment out/comment in. Instructions are provided, just read thoroughly

def accuracy(prediction, label): # Need a way to measure accuracy. We do this by taking the max index of the output tensor
    num_correct = 0
    
    pred = prediction.clone().detach().numpy() # We need to copy the values of these tensors since they have autograd and we dont wanna mess that up by detaching that or returning a processed version
    lab = label.clone().detach().numpy() # So, we use .clone() to copy, .detach() to remove from the autograd computation graph, and .numpy() to make manipulations easier

    for i in range(len(pred)):
        if np.argmax(pred[i]) == np.argmax(lab[i]): # Each element of prediction and label is a length 10 vector. In each vector, the index of the max element refers to the class that's most strongly predicted.
            num_correct += 1                                # If the indices where they have their max elements line up, it means the predicted strongest class is the same as the actual class. Hence, it's a correct prediction.
    #acc = float(num_correct/len(pred))
    return num_correct

def turn1hot(tensor_array): # A function to one-hot encode the labels of each batch 
    nparr = tensor_array.clone().detach().numpy() # turn the tensor into a numpy array after detaching it from any computational graphs
    arr = nparr.tolist() # Turn the array into a normal list for easy manipulation
    for i in range(len(arr)): # For loop does the process for the whole batch
        val_at_index = nparr[i]
        zero_vec = [0]*10 # Create a list of 0s of length number of classes 
        zero_vec[int(val_at_index)] += 1 # Make the appropriate class's index a 1 
        arr[i] = zero_vec
    return np.asarray(arr) # Return the one hot encoded list as an array

# All the kinds of CNN we have. Uncomment whichever you want to run. Note, with CeCNN and BothCNN you need to choose loss_fnc = crossentropy

#cnn = BasicCNN() # Over fitting model
cnn = CNN() # best combination tested model 
#cnn = BnCNN() # batch norm model
#cnn = CeCNN() # cross entropy model
#cnn = BothCNN() # both cross entropy and batch norm model

# Pick whichever loss you want based on network chosen above
loss_fnc = nn.MSELoss(reduction='sum') # Initialize a MSE loss object, and specify it is to sum the loss of the entire batch of elements, since default is taking the mean
#loss_fnc = nn.CrossEntropyLoss(reduce='sum') #Initialize cross entropy loss object

optimizer = optim.SGD(cnn.parameters(), lr=learning_rate) #Set up the SGD optimizer

valid_acc_array = [0]*epochs
train_acc_array = [0]*epochs
num_pred_correct_valid = 0
num_pred_correct_train = 0

loss_valid = 0
loss_train= 0

epoch_array = [0]*epochs
loss_array_train = [0]*epochs
loss_array_valid = [0]*epochs
for i in range(len(epoch_array)):
    epoch_array[i] += i


for i in range(0,epochs,1):
    for batch_index, data_list in enumerate(train_data):

        optimizer.zero_grad() # Zero gradients at the very start of each batch
        
        input_data = data_list[0].float()  # First element is the actual data tensors
        input_label_raw = data_list[1]  # Don't make this a float yet, since we need to onehot encode labels first
        input_label = turn1hot(input_label_raw) # Now we have an array of one-hot labels
        
        #Depending on what loss function you use, comment either one of the below labels out. First one is for MSE, second for Cross entropy. Cross entropy just needs class indices
        labels = torch.from_numpy(input_label).float() # Turns out the loss function works best with tensor labels, so converting the numpy array of labels into a tensor.float()
        #labels = input_label_raw.long() # If using cross entropy, comment the above line out and use this line
        
        predictions = cnn(input_data) # Make a forward pass through the CNN
        loss = loss_fnc(predictions, labels) # Compute the loss real quick ya feel 
        
        loss_train += loss
        num_pred_correct_train += accuracy(predictions, labels) # Get the accuracy for this epoch by running this through the accuracy function
    
        loss.backward()
        optimizer.step()
    
    # Every epoch I want to calculate the accuracy of the model 
    for batch_index, data_list in enumerate(valid_data):
        
        input_data = data_list[0].float()  # First element is the actual data tensors
        input_label_raw = data_list[1]  # Don't make this a float yet, since we need to onehot encode labels first
        input_label = turn1hot(input_label_raw) # Now we have an array of one-hot labels
 
        #Depending on what loss function you use, comment either one of the below labels out. First one is for MSE, second for Cross entropy. Cross entropy just needs class indices
        labels = torch.from_numpy(input_label).float() # Turns out the loss function works best with tensor labels, so converting the numpy array of labels into a tensor.float()
        #labels = input_label_raw.long()
        
        predictions = cnn(input_data)
        loss = loss_fnc(predictions, labels)
        
        loss_valid += loss
        
        num_pred_correct_valid += accuracy(predictions, labels) # Get the accuracy for this epoch by running this through the accuracy function
    
    # Now amalgamate the validation accuracy for the epoch, and reset the number of correct prediction counter
    valid_acc_array[i] += float(num_pred_correct_valid/len(valid_data.dataset))
    num_pred_correct_valid = 0
    loss_array_valid[i] += float(loss_valid/batch_size)
    loss_valid = 0
    
    train_acc_array[i] += float(num_pred_correct_train/len(train_data.dataset))
    num_pred_correct_train = 0
    loss_array_train[i] += float(loss_train/batch_size)
    loss_train = 0

#_________________________________________________________________________________________________________
"""PLOTTING DATA"""

font = {'size': 6}
plt.figure(1)  # Figure 1 is for training and validation loss + accuracy
plt.subplot(211)
plt.plot(epoch_array, loss_array_train, 'r', epoch_array, loss_array_valid, 'g')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Training and Validation Loss by Epoch', fontdict= font)
plt.legend(['training loss', 'validation loss'])
plt.subplot(212)
plt.plot(epoch_array, train_acc_array, 'r', epoch_array, valid_acc_array, 'g')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Training and Validation Accuracy by Epoch', fontdict= font)
plt.legend(['training accuracy', 'validation accuracy'])
plt.tight_layout()
plt.show()

summary(cnn, input_size=(3,56,56))

print("---%s seconds ---" % (time.time() - start_time)) # Print out how long it took

