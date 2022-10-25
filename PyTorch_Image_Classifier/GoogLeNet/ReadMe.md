# GoogLeNet模型学习





### 在Google colaboratory 中上传数据集并训练自己的模型

一、数据集解压

```
!tar zxvf /content/drive/MyDrive/deep_learning/flower_photos.tgz -C ./
```

![image](https://user-images.githubusercontent.com/86656412/197728283-367058e2-a731-4f76-bad1-840498fe29a4.png)


二、建立数据集

![image](https://user-images.githubusercontent.com/86656412/197728337-d5ef1e51-a871-43df-a177-01d534886ca0.png)




![image](https://user-images.githubusercontent.com/86656412/197728405-2ea59943-bf14-4059-b5b9-f90a02708293.png)




三、调试，训练模型





### 一、解决的问题/特点/贡献

1. **保证算力情况下增大宽度和深度**
2. **宽度：利用Inception结构同时执行多个网络结构**
3. **深度：利用辅助分类器防止梯度消失**
4. **使用1*1的卷积核进行降维以及映射处理**
5. **丢弃了全连接层，使用平均池化层（大大减少模型参数）**

为什么要提出Inception：

一般来说，提升网络性能最直接的办法就是增加网络深度和宽度，但一味地增加，会带来诸多问题：
1）参数太多，如果训练数据集有限，很容易产生过拟合；
2）网络越大、参数越多，计算复杂度越大，难以应用；
3）网络越深，容易出现梯度弥散问题（梯度越往后穿越容易消失），难以优化模型。
我们希望在增加网络深度和宽度的同时减少参数，为了减少参数，自然就想到将全连接变成稀疏连接。但是在实现上，全连接变成稀疏连接后实际计算量并不会有质的提升，因为大部分硬件是针对密集矩阵计算优化的，稀疏矩阵虽然数据量少，但是计算所消耗的时间却很难减少。在这种需求和形势下，Google研究人员提出了Inception的方法。

##### Inception结构

![image](https://user-images.githubusercontent.com/86656412/197728834-6ed0cbe7-0856-483a-80a7-347956b70e66.png)


Inception就是把多个卷积或池化操作，放在一起组装成一个网络模块，设计神经网络时以模块为单位去组装整个网络结构。模块如上图所示，**注意：每个分支所得的特征矩阵高和宽必须相同**。

图（a）这个结构存在很多问题，是不能够直接使用的。首要问题就是参数太多，导致特征图厚度太大。为了解决这个问题，作者在其中加入了1X1的卷积核，改进后的Inception结构如（b）,减少参数原理如下：
![image](https://user-images.githubusercontent.com/86656412/197728905-0ef0ecd8-595c-4b8b-90fc-c7d1e1bf19f6.png)


### 二、模型结构

![image](https://user-images.githubusercontent.com/86656412/197728957-aaac52f6-2c6f-46a8-a0d8-f1ea5442dcbf.png)

![image](https://user-images.githubusercontent.com/86656412/197729008-6b078917-e901-43b5-a6ef-658015a5dba9.png)



### 三、代码

[deep_learning_study/PyTorch_Image_Classifier/GoogLeNet at main · dongdong2061/deep_learning_study (github.com)](https://github.com/dongdong2061/deep_learning_study/tree/main/PyTorch_Image_Classifier/GoogLeNet)

模型训练

![image](https://user-images.githubusercontent.com/86656412/197729085-8570fb88-1748-47d8-8f11-c44aeadc7472.png)


结果展示

![image](https://user-images.githubusercontent.com/86656412/197729147-c6a9eefd-0fc6-4ff4-9dc9-d97da6089c05.png)

