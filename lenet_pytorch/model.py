import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
from PIL import Image
"""
LeNet(1998)网络结构
Pytorch Tensor的通道排序:[batch,channel,height,width]
"""

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)  #定义卷积层 3代表输入图像的维度，16代表卷积核数量 5代表卷积核15*15
        self.pool1 = nn.MaxPool2d(2, 2) #池化核为2 stride为2，池化层只改变高度和宽度
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32*5*5, 120)  #全连接层
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))    # input(3, 32, 32) output(16, 28, 28)
        x = self.pool1(x)            # output(16, 14, 14)
        x = F.relu(self.conv2(x))    # output(32, 10, 10)
        x = self.pool2(x)            # output(32, 5, 5)
        x = x.view(-1, 32*5*5)       # output(32*5*5),特征矩阵展平
        x = F.relu(self.fc1(x))      # output(120)
        x = F.relu(self.fc2(x))      # output(84)
        x = self.fc3(x)              # output(10)
        return x


label_name = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"] 

label_dict = {}

for idx,name in enumerate(label_name):
    label_dict[name] = idx

def default_loader(path):
    return Image.open(path).convert("RGB")  
    
#根据cifar10创建自己的数据集
class MyDataset(Dataset):
    def __init__(self,im_list,transform = None,loader = default_loader):  #初始化,im_list为train或test文件夹下文件名列表
        super(MyDataset,self).__init__()
        imgs = []
        for im_item in im_list:
            """
            cifar10\train\airplane\aeroplane_s_000004.png
            """
            im_label_name = im_item.split("\\")[-2]
            imgs.append([im_item,label_dict[im_label_name]])
        
        self.imgs = imgs  #数据
        self.transform = transform  #数据增强
        self.loader = loader #数据加载
    

    def __getitem__(self,index):  
        im_path,im_label = self.imgs[index]
        im_data = self.loader(im_path)

        if self.transform is not None:
            im_data = self.transform(im_data)

        return im_data , im_label
    
    def __len__(self):
        return len(self.imgs)
