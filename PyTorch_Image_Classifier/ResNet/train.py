import json
import sys
from tqdm import tqdm
import torch.nn as nn
import torch
from torchvision import transforms,datasets
import os
import torch.optim as optim
from model import resnet34

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

data_transform = {
    'train': transforms.Compose(
        [transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]  #这些数据来源于transform learning官网中resnet的参数
    ),
    'test': transforms.Compose(
        [transforms.Resize(256), 
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]
    )
}

root_path = os.path.abspath(os.path.join(os.getcwd(),"../"))
data_path = root_path + "/data_set/flower_data"
train_dataset = datasets.ImageFolder(root=data_path+'/train',transform=data_transform['train'])
train_num = len(train_dataset)
print(train_num)
#根目录 root 下存储的是类别文件夹（如cat，dog），每个类别文件夹下存储相应类别的图像（如xxx.png）
test_dataset = datasets.ImageFolder(root=data_path+'/val',transform=data_transform['test'])
test_num = len(test_dataset)
print(test_num)

# {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3,'tulips':4}
flower_list = train_dataset.class_to_idx
cla_list = dict((cla,key) for key,cla in flower_list.items())
print(cla_list)

json_str = json.dumps(cla_list,indent=4)
with open('ResNet/class_indices.json','w') as json_file:
    json_file.write(json_str)

batch_size = 32
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=0)

test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=True,num_workers=0)

#利用迁移学习减少计算量
net = resnet34()
model_weight_file = 'ResNet/resnet34-pre.pth'
assert os.path.exists(model_weight_file), "file {} is not exist".format(model_weight_file)
net.load_state_dict(torch.load(model_weight_file,map_location=device),strict=False)

#改变加载模型的fc层的结构
inchannel = net.fc.in_features #in_features表示原有模型中fc层的输入深度
net.fc = nn.Linear(inchannel,5) #将输入深度改为5
net.to(device)

#定义损失函数、优化器
loss_function = nn.CrossEntropyLoss()
params = [p for p in net.parameters() if p.requires_grad]
optimizer = optim.Adam(params, lr=0.0001)

best_acc = 0.0
save_path = 'ResNet/resnet34.pth'
for epoch in range(3):
    #train model
    net.train()
    running_loss = 0.0
    train_bar = tqdm(train_loader,file = sys.stdout)
    for step,data in enumerate(train_loader,start=0):
        images,labels = data
        optimizer.zero_grad()
        outputs = net(images.to(device))
        loss = loss_function(outputs,labels.to(device))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        #进度条描述
        train_bar.desc = "train epoch [{0}/{1}] loss :{2} ".format(3,epoch+1,loss)

    #validate
    net.eval()
    acc = 0.0 
    with torch.no_grad():
        test_bar = tqdm(test_loader,file=sys.stdout)
        for data_test in test_loader:
            test_images,test_labels = data_test
            outputs = net(test_images.to(device))
            predict_y = torch.max(outputs,dim=1)[1]
            acc += torch.eq(predict_y,test_labels.to(device)).sum().item() #torch.eq().sum()就是将所有值相加，但是得到的仍然是一个tensor

            test_bar.desc = "validate epoch [{}/{}] ".format(3,epoch+1)
        
    val_accurate = acc/test_num
    print("[epoch {}] train_loss :{:.3f} accurate:{:.3f}".format(epoch+1,running_loss/test_num,val_accurate))

    if val_accurate > best_acc:
        best_acc = val_accurate
        torch.save(net.state_dict(),save_path)

print("train finished!")
