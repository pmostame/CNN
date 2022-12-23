import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

# Define architecture
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

        self.dropout1 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(256*2*2, 64*2*2)
        self.dropout2 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64*2*2, 32*2*2)
        self.dropout3 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(32*2*2, 16*2*2)
        self.dropout4 = nn.Dropout(0.2)
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

# Load the model
net = Net().to(device)
load_path = 'Model.pt'
net.load_state_dict( torch.load(load_path, map_location=device) )

# load test dataset
batch_size = 512
test_transformer = transforms.Compose( [transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))] )
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transformer)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

# Test the network
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
print('\n------------------------------------------------------------------------\n')        
print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')
