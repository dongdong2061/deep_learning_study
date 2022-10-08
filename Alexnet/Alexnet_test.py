import torch
import torchvision
import torch.nn as nn 
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as transforms
import torch.optim as optim

"""
cifar10有50000个训练图像和10000个测试图像
"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
model部分
"""
#建立AlexNet模型，5个卷积层，3个池化层，3个全连接层
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),

            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32,64,3,padding=1),

            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.ReLU()
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64,128,3,padding=1),
        )
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(128,256,3,padding=1),
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(256,256,3,padding=1),

            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.ReLU()
        )

        #定义全连接层
        self.dense = nn.Sequential(
            nn.Linear(256*4*4,1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,10)
        )

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        # print(x.shape)
        x = x.view(-1,256*4*4)
        self.dense(x)
        return x

"""
train部分
"""
def train():
    transform = transforms.Compose(
        [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    #数据集加载,这里是在kaggle中应用
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data',train=True,
        download=False,transform=transform
    )
    train_loader = DataLoader(train_dataset,batch_size=200,shuffle=True,num_workers=0)

    test_dataset = torchvision.datasets.CIFAR10(
        root='./data',train=False,
        download=False,transform=transform
    )
    test_loader = DataLoader(test_dataset,batch_size=50,shuffle=False)

    test_data_iter = iter(test_loader)  #迭代数据

    test_image, test_label = test_data_iter.next()

    #设定基本参数
    epochs = 20
    lr = 0.01

    net = AlexNet().cuda(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(),lr=lr,momentum=0.9)

    train_loss = []
    for epoch in range(epochs):
        sum_loss = 0
        total = 0
        correct = 0
        for batch_idx,(x,y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            pred = net(x)

            optimizer.zero_grad()
            # print(pred.shape)
            # print(y.shape)
            loss = loss_func(pred,y)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()

            if batch_idx == 249:
                with torch.no_grad():
                        outputs = net(test_image)  # [batch, 10]
                        predict_y = torch.max(outputs, dim=1)[1]  
                        accuracy = torch.eq(predict_y, test_label).sum().item() / test_label.size(0)
                        print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                            (epoch + 1, batch_idx + 1, sum_loss / 250, accuracy))
                        sum_loss = 0.0
            train_loss.append(loss.item())
            
        print(["epoch:%d , batch:%d , loss:%.3f" %(epoch,batch_idx,loss.item())])
        torch.save(net.state_dict(), './AlexNet.pth')
