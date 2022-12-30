# import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("log/mnist")

# 定义网络模型
class Net(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # 除了batch展平所有维度
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 设置参数
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
in_channels = 1
num_classes = 10
batch_size = 32
num_worker = 0
num_epochs = 5
learning_rate = 0.1


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            (0.1307,), (0.3081,)
        ),  # 注意Mnist是灰度图，彩色图像应该是  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)
# 加载数据
trainset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
trainloader = DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=num_worker
)
# 初始化网络
net = Net(in_channels, num_classes).to(device)

# 损失和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate)

# 训练网络并记录log
step = 0
net.train()
for epoch in range(num_epochs):
    losses = []
    accuraies = []
    # for i, (images, labels) in enumerate(trainloader, 0):
    # trainloader自己就可以迭代，如果需要批次的索引i来定时输出log，使用上面的enumerate
    for (images, labels) in trainloader:
        # 得到训练的图片和标签并传入cuda
        inputs, labels = images.to(device), labels.to(device)
        # forward 前向传播
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        # 参数梯度归零并反向传播backward
        optimizer.zero_grad()
        loss.backward()
        # 梯度下降，更新参数
        optimizer.step()
        # 计算mini-batch(一个iteration后)的loss和accuracy，并加入tensorboard
        losses.append(loss.item())
        # shape是[batch_size,num_classes]，得到每一行中最大值的下标，就是第几类
        _, predicted = outputs.max(1)
        num_correcct = (predicted == labels).sum()
        accuracy = num_correcct / outputs.shape[0]
        accuraies.append(accuracy)
        writer.add_scalar("Training Loss", loss, step)
        writer.add_scalar("Training Accuracy", accuracy, step)
        step += 1

    print("-" * 20)
    print(
        f"{epoch + 1} epoch\nloss：{sum(losses) / len(losses):.3f}\naccuracy: {100*sum(accuraies) / len(accuraies):.3f}%"
    )

print("训练完成！！！")
PATH = "./mnist_lenet5.pth"
torch.save(net.state_dict(), PATH)
print("模型已保存！！！")
