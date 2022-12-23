import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import os
import torch.optim as optim

save_flag = 0
N_epoch = 200
batch_size = 512

# Construct the network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(3, 32, 3, padding='same') # 16*32*32
        self.batch0 = nn.BatchNorm2d(32)

        self.conv1 = nn.Conv2d(32, 64, 3, padding='same') # 32*32*32
        self.batch1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2) # 32*16*16
        
        self.conv2 = nn.Conv2d(64, 64, 3, padding='same') # 64*26*26
        self.batch2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, padding='same') # 64*22*22
        self.batch3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, 3, padding='same') # 128*16*16
        self.batch4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(2) # 128*4*4

        self.conv5 = nn.Conv2d(128, 256, 3, padding='same') # 128*12*12
        self.batch5 = nn.BatchNorm2d(256)
        self.pool5 = nn.MaxPool2d(2) # 128*4*4

        self.conv6 = nn.Conv2d(256, 256, 3, padding='same') # 128*16*16
        self.batch6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 256, 3, padding='same') # 128*12*12
        self.batch7 = nn.BatchNorm2d(256)

        self.pool6 = nn.MaxPool2d(2) # 128*4*4

        self.dropout1 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(256*2*2, 64*2*2)
        self.dropout2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64*2*2, 32*2*2)
        self.dropout3 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(32*2*2, 16*2*2)
        self.dropout4 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(16*2*2, 10)

    def forward(self, x):
        x = self.conv0(x)
        x = self.batch0(x)
        x = F.relu(x)

        x = self.conv1(x)
        x = self.batch1(x)
        x = F.relu(x)
        x1 = self.pool1(x)

        x = self.conv2(x1)
        x = self.batch2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.batch3(x)
        x = F.relu(x) + x1

        x = self.conv4(x)
        x = self.batch4(x)
        x = F.relu(x)
        x = self.pool4(x)

        x = self.conv5(x)
        x = self.batch5(x)
        x = F.relu(x)
        x1 = self.pool5(x)

        x = self.conv6(x1)
        x = self.batch6(x)
        x = F.relu(x)
        x = self.conv7(x)
        x = self.batch7(x)
        x = F.relu(x) + x1
        x = self.pool6(x)

        x = torch.flatten(x, 1)

        x = self.dropout1(x)
        x = self.fc1(x)
        x = F.relu(x)

        x = self.dropout2(x)
        x = self.fc2(x)
        x = F.relu(x)

        x = self.dropout3(x)
        x = self.fc3(x)
        x = F.relu(x)

        x = self.dropout4(x)
        x = self.fc4(x)
        
        return x       




# Making the code device-agnostic
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device being used is "{device}"')
torch.device(device)

# Load the data
train_transformer = transforms.Compose( [transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)), transforms.RandomHorizontalFlip(0.5), transforms.RandomErasing(ratio=(0.25, 4), scale = (0.3, 0.5))] )
test_transformer = transforms.Compose( [transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))] )
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transformer)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transformer)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

# Report number of labels, training images, image size etc.
num_labels = torch.unique( torch.tensor( [sample[1] for sample in trainset] ) )
num_trainset = len(trainset)
num_testset = len(testset)
image_size = trainset[0][0].shape
print( f'Class labels consist of {num_labels} with a total of { len(num_labels) } classes.'  )
print( 'Number of training images is equal to: ' + str( num_trainset ) )
print( 'Number of test images is equal to: ' + str( num_testset ) )
print( 'Size of each image is equal to: ' + str( image_size ) )



# Define a loss function
net = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.05, patience=2)

# Train the network
epoch_num = N_epoch
loss_batch = []
accuracy_batch = []
for epoch in range(epoch_num):    
    running_loss = 0
    accuracy_epoch = 0
    for batch_ind, batch in enumerate(trainloader):
        if device == 'cpu':
            images, labels = batch[0], batch[1]
        else:
            images, labels = batch[0].to(device), batch[1].to(device)
        # refresh optimizer for the new batch
        optimizer.zero_grad()
        # feedforward the batch
        outputs = net(images)
        # backward the error
        loss = criterion(outputs, labels)
        loss.backward()
        # optimize
        optimizer.step()
        # report error/accuracy of the batch
        running_loss += loss.item()
        loss_batch.append( loss.item() )
        accuracy_this_batch = ( torch.eq(labels, outputs.argmax(axis=1)).sum().detach().cpu().numpy() ) 
        accuracy_batch.append( accuracy_this_batch / len(labels) * 100 )
        accuracy_epoch += accuracy_this_batch
    
    # print loss
    running_loss /= len(trainloader)
    accuracy_epoch = accuracy_epoch / (len(trainloader) * batch_size) * 100
    text = f'Epoch: {epoch+1} -----> [Error: {running_loss}] and [Accuracy: {accuracy_epoch}]'
    print(text)

    scheduler.step(running_loss)

# Plot the error over time
fig, ax = plt.subplots(2,1, figsize =(20, 10))
ax = ax.ravel()
# plot error over time
_ = ax[0].plot(loss_batch, linewidth=0.5)
ax[0].set_xlabel('Batch number', fontsize=15)
ax[0].set_ylabel('Running error', fontsize=15)
# plot accuracy over time
_ = ax[1].plot(accuracy_batch, linewidth=0.5)
ax[1].set_xlabel('Batch number', fontsize=15)
ax[1].set_ylabel('Training accuracy', fontsize=15)

# test the model
correct = 0
total = 0
with torch.no_grad():
    for batch in testloader:
        if device == 'cpu':
            images, labels = batch[0], batch[1]
        else:
            images, labels = batch[0].to(device), batch[1].to(device)
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('\n ---------------------------------------------------------------------------------------- \n')
print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')

# Save the model
if save_flag:
    save_path = os.path.join('Model.pt')
    torch.save(net.state_dict(), save_path)


