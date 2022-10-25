from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json,torch
from model import GoogLeNet

data_transform = transforms.Compose(
    [transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#load Image
img = Image.open("AlexNet/tulip.jpg")
plt.imshow(img)
#[N,C,H,W]
img = data_transform(img)
#expand batch dimension
img = torch.unsqueeze(img,dim=0) #主要是对数据维度进行扩充。第二个参数为0数据为行方向扩，为1列方向扩

#read class_dict
try:
    json_file = open('GoogLeNet\class_indices.json','r')
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

#初始化模型
model = GoogLeNet(num_classes=5,aux_logits=False)
model.to(device)
model_weight_path = "GoogLeNet/GoogleNet.pth"
missing_keys ,unexpected_keys =  model.load_state_dict(torch.load(model_weight_path,map_location=device),strict=False) #由于GoogLeNet的网络结构严格来说和设定的结构并不完全一样
model.eval()  #网络含有dropout操作，此处抑制
with torch.no_grad():
    #predict class
    output = torch.squeeze(model(img))
    predict = torch.softmax(output,dim=0) #使用每一列值和为1
    predict_cla = torch.argmax(predict).numpy() #返回指定维度最大值的序号
print(class_indict[str(predict_cla)],predict[predict_cla].item())  #item:get a Python number from a tensor containing a single value
plt.show()
