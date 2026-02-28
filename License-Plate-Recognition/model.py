import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(6) #BN 可以让网络在训练时更加稳定，从而加快收敛速度。  #在深层神经网络中，随着网络的加深，输入数据的分布会不断发生变化（例如，经过激活函数后的数据分布）。#BN 使得网络对于权重初始化的选择不那么敏感，因此可以使用更大的学习率进行训练。#减少过拟合
        self.pool1 = nn.MaxPool2d(2, 2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(400, 84)
        self.fc2 = nn.Linear(84, 65)
        self.relu3 = nn.ReLU()


    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool1(x)#最大池化能保留特征图中的显著特征，这些通常是最重要的视觉特征，最大池化有助于增加对局部位移和噪声的鲁棒性，尤其是当图像中有背景噪声时
        x = self.relu1(x)#relu表达式简单，避免梯度爆炸或消失，加速训练
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool2(x)
        x = self.relu2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        #softmax  使用了 CrossEntropyLoss，就不需要手动添加 Softmax，因为它已经内建在损失函数中了。
        #Softmax 的引入可能会使得模型的梯度变得非常小，导致更新缓慢,所以我只使用了学习率衰减StepLR来在训练过程中调整学习率，避免重复




        return x
