# LeNet网络
Lenet是一个 7 层的神经网络，包含2 个卷积层，2 个池化层，3 个全连接层。其中所有卷积层的所有卷积核都为 5x5，步长 strid=1，池化方法都为全局 pooling，
激活函数为 Sigmoid，不过我们实现的时候利用的是ReLu函数，网络结构如下：

![image](https://user-images.githubusercontent.com/86656412/193977837-d2469411-fc9c-46f8-b62b-ffdfaeae314e.png)

