## VGG(visual Geometry Group)

VGG模型是2014年ILSVRC竞赛的第二名，第一名是[GoogLeNet](https://baike.baidu.com/item/GoogLeNet/22689587?fromModule=lemma_inlink)。但是VGG模型在多个[迁移学习](https://baike.baidu.com/item/迁移学习/22768151?fromModule=lemma_inlink)任务中的表现要优于googLeNet。而且，从图像中提取CNN特征，VGG模型是首选算法。它的缺点在于，参数量有140M之多，需要更大的存储空间。

VGG的特点：

- 小卷积核。作者将卷积核全部替换为3x3（极少用了1x1）；
- 小池化核。相比AlexNet的3x3的池化核，VGG全部为2x2的池化核；
- 层数更深特征图更宽。基于前两点外，由于卷积核专注于扩大通道数、池化专注于缩小宽和高，使得模型架构上更深更宽的同时，计算量的增加放缓；
- 全连接转卷积。网络测试阶段将训练阶段的三个全连接替换为三个卷积，测试重用训练时的参数，使得测试得到的全卷积网络因为没有全连接的限制，因而可以接收任意宽或高为的输入。

![image](https://user-images.githubusercontent.com/86656412/195987304-b77aec57-2909-4fb1-8bb0-261a65bf8d37.png)
![image](https://user-images.githubusercontent.com/86656412/195987307-ee67d322-c3e9-4905-b002-74fdfaa72d29.png)


