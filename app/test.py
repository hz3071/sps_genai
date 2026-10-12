import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from cnn_model import CNN

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

test_data = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(test_data, batch_size=64)

model = CNN()
model.load_state_dict(torch.load("cnn_model.pth", map_location="cpu"))
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)

        total += labels.size(0)
        correct += (predictions == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")