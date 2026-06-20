import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_channels,  out_channels, stride=1):
        super().__init__()
        if (in_channels != out_channels) or (stride == 2):
            self.shortcut = nn.Conv2d(in_channels,out_channels,kernel_size=1, stride=stride)
        else:
            self.shortcut = nn.Identity()
        self.net = nn.Sequential(
            nn.Conv2d(kernel_size=3, in_channels=in_channels, out_channels=out_channels, padding=1,stride=stride),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(kernel_size=3, in_channels=out_channels, out_channels=out_channels, padding=1),
            nn.BatchNorm2d(num_features=out_channels),            
        )

    
    def forward(self, x):
        outputs = self.net(x) + self.shortcut(x)
        
        outputs = F.relu(outputs)
        return outputs

class ResNet(nn.Module):
    """   Input [B,1,28,28]
    → conv1 (in=1, out=32, kernel=3, padding=1, no pool)   [B,32,28,28]
    → BasicBlock(32, 32) x2                                 [B,32,28,28]
    → BasicBlock(32, 64, stride=2) + BasicBlock(64, 64)     [B,64,14,14]
    → BasicBlock(64, 128, stride=2) + BasicBlock(128, 128)  [B,128,7,7]
    → AdaptiveAvgPool2d(1)                                   [B,128,1,1]
    → Flatten → Linear(128, 10) """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32,kernel_size=3,padding=1),
            BasicBlock(32,32),
            BasicBlock(32,32),
            BasicBlock(32, 64, stride=2),
            BasicBlock(64, 64),
            BasicBlock(64, 128, stride=2),
            BasicBlock(128, 128),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128,10)
        )

    def forward(self, x):
        return self.net(x)







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
ReN = ResNet()
optimizer = optim.Adam(ReN.parameters())
loss_fn = nn.CrossEntropyLoss()

num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in trainloader:
        outputs = ReN(images)
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
    outputs = ReN(images)
    predict = outputs.argmax(dim=1)
    correct += (predict == labels).sum().item()
    total = total + len(labels)

accuracy = 100 * correct / total
print(accuracy)
