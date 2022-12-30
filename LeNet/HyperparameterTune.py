import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            (0.1307,), (0.3081,)
        ),  # 注意Mnist是灰度图，彩色图像应该是  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
trainset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)


classes = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # 除了batch展平所有维度
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


batch_sizes = [4, 8, 16, 32]
learning_rates = [0.1, 0.01, 0.0001, 0.0001]
for batch_size in batch_sizes:
    for learning_rate in learning_rates:
        step = 0
        # 每次更换超参数都要重新生成新的模型
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        net = Net().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=learning_rate)
        writer = SummaryWriter(f"runs/mnist1/BS{batch_size} LR{learning_rate}")

        net.train()
        for epoch in range(1):
            losses = []
            accuraies = []
            for i, (images, labels) in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = images.to(device), labels.to(device)

                # forward + backward + optimize
                outputs = net(inputs)

                loss = criterion(outputs, labels)
                # zero the parameter gradients
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 计算mini-batch(一个iteration后)的loss和accuracy，并加入tensorboard
                losses.append(loss.item())
                # the class with the highest energy is what we choose as prediction
                _, predicted = outputs.max(1)
                num_correcct = (predicted == labels).sum()
                accuracy = num_correcct / outputs.shape[0]
                accuraies.append(accuracy)
                writer.add_scalar("Training Loss", loss, step)
                writer.add_scalar("Training Accuracy", accuracy, step)
                step += 1
            writer.add_hparams(
                {"bsize": batch_size, "lr": learning_rate},
                {
                    "loss": sum(losses) / len(losses),
                    "accuracy": sum(accuraies) / len(accuraies),
                },
            )
print("完成！！！")
