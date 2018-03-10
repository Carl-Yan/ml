# Paper reading

[TOC]

## CNN 分类

### LeNet

得分解释：得分高的神经元组合成的路径为一条可能的解释路径

bias/norm能够保证平移不变性

subsampling：减轻小变化所带来的影响，减小敏感性

### AlexNet

在cpu上做出图像变换，喂给gpu；测试的时候将图片经过不同变换之后取预测概率平均值

dropout：每个神经元dropout概率为0.5，测试的时候每个的输出×2

LRN不work

### ZFNet

反卷积：“Adaptive deconvolutional networks for mid and high level feature learning”，根据可视化的特征进行网络结构的修改

前面的层收敛，然后后面的层再收敛：越深收敛越慢，不是同时以相同速度收敛的（梯度弥散 => sudden jumps）

卷积核太大：会把noise（低频与高频）吸收进来，冲淡原始signal

？网络对图片操作具有鲁棒性（较高层次提取的feature对于平移和尺度变化具有不变性，对非中心对称的旋转不具有不变性）：是原始图像变换之后的数据喂的好？（AlexNet里面说没做图像变换效果很差） => 给出差异，神经网络能够从中学出共性

### VGG

Train：pre-init，先把小的训好，然后不断加入新的层/改变图像的缩放比例再训。但是提到了Xavier初始化(2010)发现他们可以不用这么干2333

卷积核：多用小卷积核能够【减少计算成本，增加非线性性】

？Table6：两个model的ensemble比七个还要好？

### OverFeat

**将FC看成1\*1 conv**，能够处理任意图像大小输入，输出不止一张feature map；然后综合输出的feature map得到得分 => 避免了在原图上的裁剪，实际上是增加采样数，但不增加计算量

### NIN

对feature map进行1*1 conv，增加非线性性

### GoogLeNet v1

1*1 conv：降维，增加表达能力（本质是对feature map进行线性组合），称之为bottleneck

Inception modules 用在 higher layers，低层还是conv，说是technical reasons

提了一句在数据中随机增加光照，用来增加光照和颜色的不变性。[链接](http://blog.csdn.net/sheng_ai/article/details/40652193)

训练的时候在中间层注入梯度，乘上一个weight decay因子