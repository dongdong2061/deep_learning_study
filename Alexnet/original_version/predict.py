from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json,torch
from model import AlexNet

data_transform = transforms.Compose(
    [transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
)

#load Image
img = Image.open("AlexNet/tulip.jpg")
plt.imshow(img)
#[N,C,H,W]
img = data_transform(img)
#expand batch dimension
img = torch.unsqueeze(img,dim=0)

#read class_dict
try:
    json_file = open('AlexNet\class_indices.json','r')
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

#初始化模型
model = AlexNet(num_classes=5)

model_weight_path = "AlexNet\AlexNet.pth"
model.load_state_dict(torch.load(model_weight_path))
model.eval()  #网络含有dropout操作，此处抑制
with torch.no_grad():
    #predict class
    output = torch.squeeze(model(img))
    predict = torch.softmax(output,dim=0) #使用每一列值和为1
  
    predict_cla = torch.argmax(predict).numpy() #返回指定维度最大值的序号
print(class_indict[str(predict_cla)],predict[predict_cla].item())  #item:get a Python number from a tensor containing a single value
plt.show()
