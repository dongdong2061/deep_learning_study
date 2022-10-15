import torch

import torch.nn as nn
from torchvision import transforms,datasets,utils
import os,json,sys 
import torch.optim as optim
from model import vgg
from tqdm import *


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

def main():
    data_transform = {
        'train':transforms.Compose(
            #将给定图像随机裁剪为不同的大小和宽高比，然后缩放所裁剪得到的图像为制定的大小；
            # (即先随机采集，然后对裁剪得到的图像缩放为同一大小)
            [transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
        ),
        'val':transforms.Compose(
            [transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
        )
     }
    #返回绝对路径
    data_root = os.path.abspath(os.path.join(os.getcwd(),"../")) #定义数据集的路径
    image_path = data_root+"/data_set/flower_data" #flower data set path
    train_dataset = datasets.ImageFolder(root=image_path+"/train",transform=data_transform["train"])
    train_num = len(train_dataset)
    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3,'tulips':4}
    flower_list = train_dataset.class_to_idx #返回class_name,class_id
    cla_dict = dict((class_id,class_name) for class_name,class_id in flower_list.items()) #生成 序号：类名 的字典
    #建立json文件,
    #将python对象编码成Json字符串,indent意思为4个空格
    json_str = json.dumps(cla_dict,indent=4)
    with open('VGGnet/class_indices.json', 'w') as json_file:
        json_file.write(json_str)
    
    batch_size = 32
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers = 0)
   

    validata_dataset = datasets.ImageFolder(root=image_path+'/val',transform=data_transform['val'])
    validata_loader = torch.utils.data.DataLoader(validata_dataset,batch_size = batch_size,shuffle=True,num_workers=0)
    val_num = len(validata_dataset)

    #测试集迭代
    test_data_iter = iter(validata_loader)
    test_images,test_labels = test_data_iter.next()

    model_name = 'vgg16'
    VGGnet = vgg(model_name=model_name, class_num = 5,init_weights=True)
    VGGnet.to(device)
    loss_function = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(VGGnet.parameters(),lr=0.0001) #定义优化器

    epochs = 30 #定义循环次数
    bset_acc = 0 #记录最优精确度
    save_path = "./{}net.pth".format(model_name)
    for epoch in range(epochs):
        #train,进入训练模式
        VGGnet.train() 
        running_loss = 0.0
        #设置输出进度条，file为设置输出方向
        train_bar = tqdm(train_loader,file=sys.stdout)
        for step,data in enumerate(train_bar):
            optimizer.zero_grad() 
            images,labels = data
            output = VGGnet(images.to(device))
            loss = loss_function(output,labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            #增加进度条描述
            train_bar.desc = "train epoch[{}/{}] loss {:.3f}".format(epoch+1,epochs,loss)

        #validation
        VGGnet.eval() #进入验证模式
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validata_loader,file=sys.stdout)
            for val_data in val_bar:
                val_images,val_labels = val_data
                outputs = VGGnet(val_images.to(device))
                pred = torch.max(outputs,dim=1)[1]
                acc += torch.eq(pred,outputs).sum().item()

        val_accurate = acc / val_num
        print("[epoch {}] train_loss :{:.3f} accurate:{:.3f}".format(epoch+1,running_loss/val_num,val_accurate))

        if val_accurate > bset_acc:
            torch.save(VGGnet.state_dict(),save_path)

    print("train finished!")

if __name__ == "__main__":
    main()


