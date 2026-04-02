import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

train_batch_size = 1
test_batch_size = 256
epoch_size = 5
seed = 245

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
)

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
g = torch.Generator()
g.manual_seed(seed)

# num_workers > 0일경우, windows 상에서 sub프로세스 하나 새로 띄움. 멀티프로세싱 꼬여서 RuntimeERR 발생가능
trainset = torchvision.datasets.CIFAR10(root='./dataPT',train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
                                          shuffle=True, num_workers=0,
                                          generator=g)

testset = torchvision.datasets.CIFAR10(root='./dataPT',train=False,
                                        download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                          shuffle=False, num_workers=0)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
net = Net()
        
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
net.train()

for epoch in range(epoch_size):
    delta_time = time.perf_counter()
    total_loss = 0.0
    train_cases = 0

    for inputs, labels in trainloader:
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        train_cases += labels.size(0)

    delta_time = time.perf_counter() - delta_time
    print(f'FINISHED EPOCH[{epoch+1}] TRAIN, ELAPSED TIME : [{delta_time*1000:.3f}] ms')
    print(f"TRAIN EPOCH [{epoch+1}] LOSS : {total_loss/train_cases:.3f}")


PATH = './dataPT/cifar_net_batchsize1.pth'
torch.save(net.state_dict(), PATH)

# EVALUATE
running_loss = 0.0
running_cases = 0
total_loss = 0.0
total_test_cases = 0
net.eval()

with torch.no_grad():
    for i, data in enumerate(testloader, 0):
        inputs, labels = data

        outputs = net(inputs)
        loss = criterion(outputs, labels)

        # batch size 별 가중치 부여, batch size보다 작은 loss data는 total loss 평균 계산 보정함
        total_loss += loss.item() * labels.size(0)
        total_test_cases += labels.size(0)
        running_loss += loss.item() * labels.size(0)
        running_cases += labels.size(0)
        
        if(i % 500 == 499):
            print(f"TEST LOOP [{i/500}] LOSS : {running_loss/running_cases:.3f}")
            running_loss = 0.0
            running_cases = 0

if(running_cases > 0) :
    print(f"TEST LOOP [LAST] LOSS : {running_loss/running_cases:.3f}")


print(f"TOTAL VERIFY LOSS : [{total_loss/total_test_cases:.3f}]")
print("EVALUATE DONE")
