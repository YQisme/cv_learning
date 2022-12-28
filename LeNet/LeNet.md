## 安装pytorch

pytorch可以直接进[仓库](https://download.pytorch.org/whl/torch/)选择下载

https://download.pytorch.org/whl/torch/

选择其中一个链接，比如`pip install https://download.pytorch.org/whl/cu102/torch-1.10.2%2Bcu102-cp38-cp38-win_amd64.whl`

## 安装torchxision

<font color='red'>不能直接安装</font>使用 `pip install torchvision`，因为会安装最新的torchvison和pytorch，这样会替换掉原来安装的旧的pytorch

torchvision版本和torch版本有匹配关系，安装前先[查表](https://github.com/pytorch/vision#installation)

在进入仓库https://download.pytorch.org/whl/torchvision/下载配对的版本。

比如`pip install https://download.pytorch.org/whl/cu102/torchvision-0.11.3%2Bcu102-cp38-cp38-linux_x86_64.whl`



---

使用`torch.nn`包构建神经网络

`nn`依赖于`autograd`来定义模型并区分它们。``nn.Module``包含层和返回输出的方法 `forward(input)` 。

![convnet](E:\Typora_Picture\LeNet.assets\mnist.png)