from symbol import except_clause
import torch
import torch.nn as nn
import os,json
from torchvision import transforms
from model import resnet34
from PIL import Image
import matplotlib.pyplot as plt

data_transform = transforms.Compose(
    [transforms.Resize(256), 
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]
)

#load Image
img = Image.open("AlexNet/tulip.jpg")
plt.imshow(img)
img = data_transform(img)
img = torch.unsqueeze(img,dim=0)

device = torch.device("gpu" if torch.cuda.is_available() else "cpu")
print("using {} device".format(device))

resnet = resnet34(num_classes=5)
model_weights_path = "ResNet/resnet34.pth"

resnet.load_state_dict(torch.load(model_weights_path,map_location=device),strict=False)

try:
    json_file = open("ResNet\class_indices.json","r")
    class_indict = json.load(json_file)
except:
    print("file {} does not exist!".format("json"))
    exit(-1)

resnet.eval()
with torch.no_grad():
    output = torch.squeeze(resnet(img))
    predict = torch.softmax(output,dim=0)

    predict_cla = torch.argmax(predict).numpy()

print(class_indict[str(predict_cla)],predict[predict_cla].item())

plt.show()
