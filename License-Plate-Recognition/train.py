import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from model import Model
import numpy as np
import random
import  os
#添加随机种  权重 偏置
# def setup_seed(seed: int):  #包含了大部分的随机函数
#     np.random.seed(seed)  # 固定np随机种子
#     random.seed(seed)
#     os.environ["PYTHONHASHSEED"] = str(seed)  # 固定python的哈希种子
#     torch.manual_seed(seed)  # 固定torch
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
#         torch.backends.cudnn.benchmark = False  # 关闭cudnn加速
#         torch.backends.cudnn.deterministic = True  # 设置cudnn为确定性算法
#
#
# setup_seed(0)

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((32, 32)),  #重新裁剪
    transforms.Grayscale(),#灰度化
    transforms.ToTensor(),#张量
    transforms.Normalize((0.5,), (0.5,))#标准化
])

# 加载数据集
full_dataset = datasets.ImageFolder(root='./datasets', transform=transform)#imagefolder加载图像的数据集进行预处理，数据加载

#数据集划分
dataset_size = len(full_dataset)#接收整个数据集，大小，长度
#训练集测试集分配空间
train_size = int(0.8 * dataset_size)
test_size = dataset_size - train_size
#分配数据
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])#随机切割分配
#数据迭代器  把数据从数据集拿出来
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)#128批次
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")#cuda计算快
# 初始化模型
model = Model().to(device)#安排在gpu计算
criterion = nn.CrossEntropyLoss()#实例化模型
optimizer = Adam(model.parameters(), lr=0.001)
from torch.optim.lr_scheduler import StepLR
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  # 每10个epoch，学习率衰减为原来的0.1，在训练过程中有时使用学习率衰减会帮助模型更好地收敛

# 训练过程
most_acc = 0  # 初始化最高准确率
patience = 5  # 允许连续多少轮不提升
counter = 0  # 计数器

for epoch in range(30):  # 训练30轮
    model.train()
    running_loss = 0.0#累加本轮的总损失，方便后续计算平均损失
    for images, labels in train_loader:#遍历训练集 批量返回 images ；labels

        images = images.to(device)  # 将输入数据移到设备上
        labels = labels.to(device)  # 将标签移到设备上

        optimizer.zero_grad()#清空上一次的梯度，避免梯度累积导致计算错误。

        outputs = model(images)#前向传播：输入 images，模型输出 outputs（预测结果）。
        loss = criterion(outputs, labels)#计算损失（CrossEntropyLoss），衡量预测结果 outputs 和真实 labels 的差距。
        loss.backward()#反向传播，计算梯度
        optimizer.step()#更新模型参数，让损失更小（梯度下降）。

        running_loss += loss.item()#记录本批次的损失，loss.item() 将张量转换为普通数值，累加到 running_loss。

    print(f"Epoch [{epoch + 1}/30], Loss: {running_loss / len(train_loader):.4f}")#计算 平均损失（总损失 ÷ 批次数）。

    # 评估
    model.eval()#切换评估模式
    correct = 0
    total = 0
    with torch.no_grad():#with语句固定梯度，不计算参数
        for images, labels in test_loader:#从测试集里加载数据
            images = images.to(device)#gpu
            labels = labels.to(device)
            outputs = model(images)#输出outputs
            _, predicted = torch.max(outputs, 1)#输出最大值作为预测值，1代表列
            total += labels.size(0)#正确率汇总
            correct += (predicted == labels).sum().item()
#早停
    acc = 100 * correct / total  # 计算准确率
    print(f"Accuracy on test set: {acc:.2f}%")

    if acc > most_acc:
        torch.save(model.state_dict(), 'model.pth') # 保存最优模型
        most_acc = acc  # 更新最高准确率
        counter = 0  # 重新计数
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break  # 训练提前终止
    # if epoch % 10 == 0:												  #每10轮存储一次可以防止在模型计算的时候出现突发情况
        # torch.save(model.state_dict(), f"save_model/model_{epoch}.pth")

# 保存模型
#保存参数

