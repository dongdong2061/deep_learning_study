import torch,os,json
import torch.nn as nn 
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from model import vgg


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

data_transform = transforms.Compose(
    [transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
)

image_path = "AlexNet/tulip.jpg"
assert os.path.exists(image_path), "file {} don't exit!".format(image_path)
image = Image.open(image_path)
plt.imshow(image)

# [N, C, H, W]
img = data_transform(image)
# expand batch dimension
img = torch.unsqueeze(img, dim=0)

json_file = "VGGnet\class_indices.json"
assert os.path.exists(json_file), "file: '{}' dose not exist.".format(json_file)
with open(json_file,'r') as js:
    class_dict = json.load(js)

model = vgg(model_name='vgg16',class_num=5,init_weight=True)
weights = ''  #权重文件路径
model.load_state_dict(torch.load(weights,map_location=device))

model.eval()
with torch.no_grad():
    output = torch.squeeze(model(img.to(device))).cpu()
    predict = torch.softmax(output, dim=0)
    predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_dict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_dict[str(i)],
                                                  predict[i].numpy()))
    plt.show()
