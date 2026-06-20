import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.lenet = nn.Sequential(
            # Input [B, 1, 28, 28]
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            # [B, 6, 24, 24]
            nn.MaxPool2d(kernel_size=2),
            # [B, 6, 12, 12]
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            # [B, 16, 8, 8]
            nn.MaxPool2d(kernel_size=2),
            # [B, 16, 4, 4]
            nn.Flatten(),   # 16 * 4 * 4 = 256
            nn.Linear(256, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )   

    def forward(self, image):
        return self.lenet(image)


if __name__ == "__main__":
    # 数据加载
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = torchvision.datasets.FashionMNIST(root='C:/Users/Rize/Desktop/Lee/playground/learning/rl/data', 
                                                train = True,
                                                download=True,
                                                transform=transform
                                                )
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=64, shuffle=True)

    # 训练循环
    LeN = LeNet()
    optimizer = optim.Adam(LeN.parameters())
    loss_fn = nn.CrossEntropyLoss()

    num_epochs = 10
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in trainloader:
            outputs = LeN(images)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(trainloader)
        print(f"Epoch {epoch+1}: avg loss = {avg_loss:.4f}")

    # 测试集上算准确率
    testset = torchvision.datasets.FashionMNIST(root='C:/Users/Rize/Desktop/Lee/playground/learning/rl/data', 
                                                train = False,
                                                download=True,
                                                transform=transform
                                                )
    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=64, shuffle=False)

    correct = 0
    total = 0
    for images, labels in testloader:
        outputs = LeN(images)
        predict = outputs.argmax(dim=1)
        correct += (predict == labels).sum().item()
        total = total + len(labels)

    accuracy = 100 * correct / total
    print(accuracy)
    torch.save(LeN.state_dict(), 'lenet_fashionmnist.pth')
