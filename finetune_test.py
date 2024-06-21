import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_cifar100_data_loaders(download, shuffle=False, batch_size=256):
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR100('.', train=True, download=download, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=shuffle)

    test_dataset = datasets.CIFAR100('.', train=False, download=download, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=2 * batch_size, num_workers=0, shuffle=shuffle)

    return train_loader, test_loader


class ModifiedResNet18(nn.Module):
    def __init__(self):
        super(ModifiedResNet18, self).__init__()
        self.model = torchvision.models.resnet18(pretrained=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 100)
        )

    def forward(self, x):
        return self.model(x)


model = ModifiedResNet18().to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

cifar100_train_loader, cifar100_test_loader = get_cifar100_data_loaders(download=True, shuffle=True)

epochs = 100
train_losses = []
test_losses = []
accuracies = []

early_stopping_patience = 10
best_accuracy = 0
patience_counter = 0

# 初始化Tensorboard的SummaryWriter
writer = SummaryWriter()

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for images, labels in cifar100_train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
    train_loss /= len(cifar100_train_loader.dataset)
    train_losses.append(train_loss)

    scheduler.step()  # Adjust learning rate

    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in cifar100_test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_loss /= len(cifar100_test_loader.dataset)
    test_losses.append(test_loss)

    accuracy = 100 * correct / total
    accuracies.append(accuracy)

    print(
        f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')

    # 在Tensorboard中记录loss和accuracy
    writer.add_scalar('Loss/Train', train_loss, epoch)
    writer.add_scalar('Loss/Test', test_loss, epoch)
    writer.add_scalar('Accuracy/Test', accuracy, epoch)

    # Check for early stopping
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= early_stopping_patience:
        print("Early stopping triggered")
        break

writer.close()

# 使用Matplotlib绘制loss和accuracy曲线
plt.figure(figsize=(16, 9))
plt.rcParams.update({'font.size': 14})
plt.title('CIFAR100 Loss Plot')
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
plt.plot(range(1, len(test_losses) + 1), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(16, 9))
plt.rcParams.update({'font.size': 14})
plt.title('CIFAR100 Accuracy Plot')
plt.plot(range(1, len(accuracies) + 1), accuracies, label='Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()