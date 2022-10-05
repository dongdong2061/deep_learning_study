import torchvision
import torch
import matplotlib.pyplot as plt 
import numpy as np
import torch.nn as nn
from model import LeNet
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as transforms
import glob
from model import MyDataset

def main():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # 50000张训练图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    # train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
    #                                          download=False, transform=transform)
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=36,
    #                                            shuffle=True, num_workers=0)

    # # 10000张验证图片
    # # 第一次使用时要将download设置为True才会自动去下载数据集
    # val_set = torchvision.datasets.CIFAR10(root='./data', train=False,
    #                                        download=False, transform=transform)
    # val_loader = torch.utils.data.DataLoader(val_set, batch_size=5000,
    #                                          shuffle=False, num_workers=0)

    im_train_list = glob.glob("cifar10/train/*/*.png")
    im_test_list = glob.glob("cifar10/test/*/*.png")

    train_dataset = MyDataset(im_train_list,transform=transform)
    test_dataset = MyDataset(im_test_list,transform=transforms.ToTensor())

    train_loader = DataLoader(dataset=train_dataset,batch_size=36,shuffle=True,num_workers=0)
    test_loader = DataLoader(dataset=test_dataset,batch_size=10000,shuffle=False,num_workers=0)



    test_data_iter = iter(test_loader)
    test_image, test_label = test_data_iter.next()
    
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # print(' '.join('%5s' % classes[test_label[j]] for j in range(4)))

    # imshow(torchvision.utils.make_grid(test_image))

    net = LeNet()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(5):  # loop over the dataset multiple times

        running_loss = 0.0  #累加损失
        for step, data in enumerate(train_loader, start=0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()  #将历史损失梯度为零
            # forward + backward + optimize
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics，验证测试阶段
            running_loss += loss.item()
            if step % 500 == 499:    # print every 500 mini-batches
                with torch.no_grad():
                    outputs = net(test_image)  # [batch, 10]
                    predict_y = torch.max(outputs, dim=1)[1]  
                    accuracy = torch.eq(predict_y, test_label).sum().item() / test_label.size(0)

                    print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                          (epoch + 1, step + 1, running_loss / 500, accuracy))
                    running_loss = 0.0

    print('Finished Training')

    save_path = './Lenet.pth'
    torch.save(net.state_dict(), save_path)

def imshow(img):
    img = img / 2 + 0.5  #反标准差
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))  #改变维度，转化为numpy的shape格式
    plt.show()


if __name__ == '__main__':
    main()
