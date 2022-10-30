### ResNet网络模型学习

残差神经网络(ResNet)是由微软研究院的何恺明、张祥雨、任少卿、孙剑等人提出的。ResNet 在2015 年的ILSVRC（ImageNet Large Scale Visual Recognition Challenge）中取得了冠军。

残差神经网络的主要贡献是发现了“退化现象（Degradation）”，并针对退化现象发明了 “快捷连接（Shortcut connection）”，极大的消除了深度过大的神经网络训练困难问题。神经网络的“深度”首次突破了100层、最大的神经网络甚至超过了1000层。

#### 一、解决的问题/特点/贡献

1、超深的网络结构（超过1000层）

2、提出residual模块，解决了退化问题

3、使用Batch Normalization加速训练丢弃(dropout)

#### 二、深度的退化问题

网络达到一定深度后，模型性能会暂时陷入一个瓶颈很难增加，当网络继续加深后，模型在测试集上的性能反而会下降！这其实就是深度学习退化（degradation）！在**MobileNet V2**的论文中提到，由于非线性激活函数Relu的存在，每次输入到输出的过程都几乎是不可逆的，这也造成了许多**不可逆的信息损失**。

#### 三、模型结构
![image](https://user-images.githubusercontent.com/86656412/198867927-628e21c1-e29d-40b0-aefd-20330d7346af.png)
![image](https://user-images.githubusercontent.com/86656412/198867929-90d55f14-b435-4ad7-b09b-e7681b6a65f0.png)


**错误记录：**在编写代码运行的过程中，出现了运行几次的准确率为0.25上下，经过仔细排查后发现是在一个forward函数里的bn2层写为了bn1层，并未造成直接错误，但是导致模型训练错误严重。

模型训练结果一

这里通过迁移学习实现一次训练结果较好
![image](https://user-images.githubusercontent.com/86656412/198867764-db6d32c3-17f6-43ac-8dae-6c0a8a80ef30.png)
模型训练结果二

这里同样利用迁移学习，epoch为3
![image](https://user-images.githubusercontent.com/86656412/198867777-f91bb34b-c084-494c-a6f3-ab1546d7edc6.png)
