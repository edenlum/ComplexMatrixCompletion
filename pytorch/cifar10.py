import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=False)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MLP(torch.nn.Module):  # 继承 torch 的 Module
    def __init__(self):
        super(MLP,self).__init__()  
        self.fc1 = torch.nn.Linear(32 * 32 * 3,64) 
        self.fc2 = torch.nn.Linear(64,64) 
        self.fc3 = torch.nn.Linear(64,64)
        self.fc4 = torch.nn.Linear(64,10) 
     
    def forward(self,x):
        x = x.view(-1,32 * 32 * 3)    
        x = F.relu(self.fc1(x))  
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.softmax(self.fc4(x), dim=1) 
        return x
    

def train_cuda(running_loss, net, optimizer, criterion, epoch):
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        
        if i % 20 == 19: 
            print("Epoch:{}, Iteration:{}, Loss:{:.3f}".format(epoch + 1, i + 1, running_loss / (i+1)))

def test_cuda(net):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

def main(expand):
    n_epochs = 20
    mlp = MLP().to(device)
    if expand:
        print('#'*30)
        print('Expanding Net!!!')
        print('#'*30)
        from expand_net import replace_linear_layers
        replace_linear_layers(mlp, 2, mode=expand, init_scale=0.5)

    criterion = nn.CrossEntropyLoss()
    optimizer_mlp = optim.Adam(mlp.parameters(), lr=1e-4)
    # optimizer_mlp = optim.SGD(mlp.parameters(), lr=0.01, momentum=0.9)
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        train_cuda(running_loss, mlp, optimizer_mlp, criterion, epoch)
        test_cuda(mlp)

    print('Finished Training')

if __name__=='__main__':
    # EXPAND = False
    EXPAND = 'complex'
    main(EXPAND)