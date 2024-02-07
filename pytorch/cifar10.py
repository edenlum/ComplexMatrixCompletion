import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb

from utils import experiments
from expand_net import replace_linear_layers

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

class MLP(torch.nn.Module): 
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
    

def train(running_loss, net, optimizer, criterion, epoch):
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
            wandb.log({"loss": running_loss / (i+1)})
            print("Epoch:{}, Iteration:{}, Loss:{:.3f}".format(epoch + 1, i + 1, running_loss / (i+1)))

def test(net):
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
        wandb.log({"accuracy": 100 * correct / total})
        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

def run_experiment(expand, d, lr, init_scale, diag_init_scale):
    n_epochs=10
    model = MLP().to(device)
    if expand:
        print('#'*30)
        print('Expanding Net!!!')
        print('#'*30)
        replace_linear_layers(model, d, mode=expand, init_scale=init_scale, diag_init_scale=diag_init_scale)

    criterion = nn.CrossEntropyLoss()
    optimizer_mlp = optim.Adam(model.parameters(), lr=lr)
    # optimizer_mlp = optim.SGD(mlp.parameters(), lr=0.01, momentum=0.9)
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        train(running_loss, model, optimizer_mlp, criterion, epoch)
        test(model)

def name(kwargs):
    # short name for display on wandb
    return f"expand_{kwargs['expand']}_d_{kwargs['d']}_init_{kwargs['init']}_diag_{kwargs['diag_init_scale']}_lr_{kwargs['lr']}"

def main():
    for i, kwargs in enumerate(experiments({
            "expand":               ["complex", "real", False],
            "d":                    [3],
            "init_scale":           [0],
            "lr":                   [1e-4],
            "diag_init_scale":      [0.1],
    })):
        exp_name = name(kwargs)
        config = {"comment": ""}
        config.update(kwargs)
        wandb.init(
            project="expand-nets",
            entity="complex-team",
            name=exp_name,
            config=config
        )
        run_experiment(**kwargs)
        wandb.finish()
    print('Finished Training')

if __name__=='__main__':
    main()