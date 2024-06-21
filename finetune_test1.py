import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_cifar100_data_loaders(download, shuffle=False, batch_size=256):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR100('.', train=True, download=download, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=shuffle)

    test_dataset = datasets.CIFAR100('.', train=False, download=download, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=2 * batch_size, num_workers=0, shuffle=shuffle)

    return train_loader, test_loader

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

model = torchvision.models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 100)

if device == 'cuda':
    model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

cifar100_train_loader, cifar100_test_loader = get_cifar100_data_loaders(download=True)

epochs = 100
train_losses = []
test_losses = []
accuracies = []

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

    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')

plt.figure(figsize=(16, 9))
plt.rcParams.update({'font.size': 14})
plt.title('CIFAR100 Accuracy Plot')
plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, epochs + 1), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(16, 9))
plt.rcParams.update({'font.size': 14})
plt.title('CIFAR100 Accuracy Plot')
plt.plot(range(1, epochs + 1), accuracies, label='Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
