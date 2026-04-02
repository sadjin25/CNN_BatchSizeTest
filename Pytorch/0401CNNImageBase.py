#TODO : 저장된 grad (pth파일) 불러오고, 이거 학습후 기존 데이터 업데이트(zero_grad 할지?)
#TODO : train / test 데이터 구분하기(지금은 둘다 겹침)
#TODO : 현재 4mini-batch gd방식. 한번 학습하는 batch size 조절하면서 step loss/epoch avg loss 실험하기.
# => 통제변수(모델, 데이터셋, optimizer,lr,epoch 일치), 비교(trainloss, val accuracy, train time, memory)
# =>... 결과예상은 메모리/타임/loss 관점에서 비교할 것. 어느 환경에서 쓸만할지도 고민해볼까.
# 졸업논문 뚝딱이네여  

import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
)

batch_size = 4

# num_workers > 0일경우, windows 상에서 sub프로세스 하나 새로 띄움. 멀티프로세싱 꼬여서 RuntimeERR 발생가능
trainset = torchvision.datasets.CIFAR10(root='./dataPT',train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./dataPT',train=False,
                                        download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

dataiter = iter(trainloader)
images, labels = next(dataiter)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        #6으로 된 채널 차원수는 바꾸지 않음, 데이터 사이즈만 줄일뿐..
        self.pool = nn.MaxPool2d(2, 2)
        #따라서 pool 거쳐도 차원수 그대로 input으로넣음
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

#전체 data를 2회, 배치사이즈4로 묶어서 루프당 학습한다.
for epoch in range(2):
    running_loss = 0.0
    #batch_size 묶음 하나를 루프당 loader에서 내보냄. dataloader 전체 data를 순회하는것.
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        #net.zero_grad() 가능. model 하나고, 모든 grad 초기화니까.
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        #왜 loss에서 backward 하고 자빠짐 : 끝단에서 거슬러 올라간다는 의미 + 여러 모델 접근가능성.
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if(i % 2000 == 1999):
            print(f"[{epoch+1}, {i+1:5d}] loss : {running_loss/2000:.3f}")
            running_loss = 0.0

print('FINISHED TRAIN')

PATH = './dataPT/cifar_net.pth'
torch.save(net.state_dict(), PATH)

dataiter = iter(testloader)
images, labels = next(dataiter)

imshow(torchvision.utils.make_grid(images))
print('GroundTruth : ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))