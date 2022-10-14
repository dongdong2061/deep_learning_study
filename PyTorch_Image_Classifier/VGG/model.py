from mimetypes import init
from turtle import forward
import torch
import torch.nn as nn


class VGG(nn.module):
    def __init__(self,features,class_num=1000,init_weights=False):
        super(VGG,self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512*7*7),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(2048,2048),
            nn.ReLU(True),
            nn.Linear(2048,class_num)
        )

        if init_weights:
            self._initialize_weights()

    def forward(self,x):
        #N * 3 *224 *224
        x = self.features(x)
        #N * 512 * 7 * 7
        x = torch.flatten(x,start_dim=1) #转换为一维
        # N * (512*7*7)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):  #初始化网络结构参数
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias,0)

cfgs = {
    'vgg11': [64,'M',128,'M',256,256,'M',512,512,'M',512,512,'M'],
    'vgg13': [64,64,'M',128,128,'M',256,256,'M',512,512,'M',512,512,'M'],
    'vgg16': [64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M',512,512,512,'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

#建立特征提取网络
def make_features(cfg:list):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2,stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels,v,kernel_size=3,padding=1)
            layers += [conv2d,nn.ReLU(True)]
            in_channels = v
    return nn.Sequential(*layers) #*layers代表传入一个元组

# **kwargs 传入字典 即 features,class_num=1000,init_weights=False
def vgg(model_name='vgg16',**kwargs):
    try:
        cfg = cfgs[model_name]
    except:
        print("Warning: model number {} not in cfgs dict!".format(model_name))
        exit(-1)
    
    model = VGG(make_features(cfg),**kwargs)
    return model
