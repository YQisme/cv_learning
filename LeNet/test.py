import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim

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
        x = torch.flatten(x, 1)
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


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            (0.1307,), (0.3081,)
        ),  # 注意Mnist是灰度图，彩色图像应该是  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)
# 加载数据

testset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
testloader = DataLoader(
    testset, batch_size=batch_size, shuffle=True, num_workers=num_worker
)

# 初始化网络
net = Net(in_channels, num_classes).to(device)
net.load_state_dict(torch.load("./mnist_lenet5.pth"))

# 计算总精度
correct = 0
total = 0
net.eval()
# 没有训练就不需要计算输出的梯度
with torch.no_grad():
    for (images, labels) in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"在10000张测试集图片上的精度为: {100 * correct // total} %")

# 计算各种类的精度
classes = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")

# 每个类的预测概率
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}
# 输出  {'0': 0,'1': 0,'2': 0,'3': 0,'4': 0,'5': 0,'6': 0,'7': 0,'8': 0,'9': 0}
net.eval()
# 同样不需要计算梯度
with torch.no_grad():
    for (images, labels) in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# 打印每个类的精度
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f"数字{classname:3s} 的精度是 {accuracy:.1f} %")
