import torch
import torch.nn as nn
import torch.utils 
from torchvision import transforms,datasets,utils
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
import json 
import time
from GoogLeNet.model import GoogLeNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]),
    "val":transforms.Compose([transforms.Resize((224,224)),  #cannot 224 ,must (224,224)
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
}

data_root = os.path.abspath(os.path.join(os.getcwd(),"../"))  #获得数据源路径
image_path = data_root+"/data_set/flower_data" #flower data set path
train_dataset = datasets.ImageFolder(root=image_path+"/train",transform=data_transform["train"])

train_num = len(train_dataset)

# {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3,'tulips':4}
flower_list = train_dataset.class_to_idx  #获得每个类别对应的索引值
cla_dict = dict((val,key) for key,val in flower_list.items())
#write dict into json file
json_str = json.dumps(cla_dict, indent=4)  #编码为json格式
with open("./GoogLeNet/class_indices.json","w") as json_file:
    json_file.write(json_str)

batch_size = 32
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers = 0)

validate_dataset = datasets.ImageFolder(root=image_path + '/val',transform=data_transform['val'])

val_num = len(validate_dataset)
validate_loader = torch.utils.data.DataLoader(validate_dataset,batch_size=4,shuffle=True,num_workers=0)

test_data_iter = iter(validate_loader)
test_image,test_label = test_data_iter.next()


# def imshow(img):
#     img = img / 2 + 0.5 #unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg,(1,2,0)))
#     plt.show()

# print(' '.join('%5s' % cla_dict[test_label[j].item()] for j in range(4)))
# imshow(utils.make_grid(test_image))

net = GoogLeNet(num_classes=5,aux_logits=True,init_weights=True)

net.to(device)
loss_function = nn.CrossEntropyLoss()
# pata = list(net.parameters())  #查看模型参数
optimizer = optim.Adam(net.parameters(),lr=0.0002)  #优化对象为模型的所有参数

save_path = "GoogleNet/GoogleNet.pth"
best_acc = 0.0
for epoch in range(10):
    #train
    net.train()  #用于管理dropout方法，启动
    running_loss = 0.0
    t1 = time.perf_counter()
    for step,data in enumerate(train_loader,start=0):
        images,labels = data
        optimizer.zero_grad()
        output ,outputs_2,outputs_1= net(images.to(device))
        loss0 = loss_function(output,labels.to(device))
        loss1 = loss_function(outputs_1,labels.to(device))
        loss2 = loss_function(outputs_2,labels.to(device))
        loss = loss0 + loss1*0.3 + loss2*0.3  #这里0.3是论文中指出的参数值
        loss.backward() #损失反向传播
        optimizer.step()

        running_loss += loss.item()
        #输出训练进度
        rate = (step + 1) / len(train_loader)
        a = "*" * int(rate*50)
        b = "." * int((1-rate)*50)
        print("\rtrain loss:{:^3.0f}%[{}->{}]{:.3f}".format(int(rate*100),a,b,loss,end=""))
    
    print(time.perf_counter()-t1)

    #validate
    net.eval() #用于管理dropout方法，禁用
    acc =0.0
    with torch.no_grad(): 
        for date_test in validate_loader:
            test_images,test_labels = date_test
            outputs = net(test_images.to(device))
            predict_y = torch.max(outputs,dim=1)[1]
            acc += (predict_y == test_labels.to(device)).sum().item()
        accurate_test = acc / val_num
        if accurate_test > best_acc:
            torch.save(net.state_dict(),save_path)
        print("[epoch %d] train_loss: %.3f  test_accuracy: %.3f" % (epoch+1, running_loss/step, acc/val_num))
 
print("Finished Training!")
