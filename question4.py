import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

# Step 1: Load and preprocess the SVHN dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data = datasets.SVHN(root='./data', split='train', download=True, transform=transform)
test_data = datasets.SVHN(root='./data', split='test', download=True, transform=transform)

# Split the dataset into train and validation sets
train_size = int(0.75 * len(train_data))
val_size = len(train_data) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_data, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Step 2: Choose pretrained models
# You can choose any of the pretrained models available in torchvision

# Step 3: Fine-tune the pretrained models
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=5):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

        # Validate the model
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = correct / total
        print(f"Validation Accuracy: {val_accuracy:.4f}")

# Step 4: Compare the performance of different models
# Let's try out LeNet-5, AlexNet, VGG, ResNet-18, ResNet-50, and ResNet-101

# AlexNet
alexnet = torchvision.models.alexnet(pretrained=True)
for param in alexnet.parameters():
    param.requires_grad = False
alexnet.classifier[6] = nn.Linear(4096, 10)  # Modify the last fully connected layer for 10 classes
alexnet.classifier[6].requires_grad = True  # Set requires_grad to True for the fully connected layer
alexnet = alexnet.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(alexnet.classifier[6].parameters(), lr=0.001)  # Only optimize parameters of the fully connected layer
train_model(alexnet, criterion, optimizer, train_loader, val_loader)

# VGG
vgg = torchvision.models.vgg16(pretrained=True)
for param in vgg.parameters():
    param.requires_grad = False
vgg.classifier[6] = nn.Linear(4096, 10)  # Modify the last fully connected layer for 10 classes
vgg.classifier[6].requires_grad = True  # Set requires_grad to True for the fully connected layer
vgg = vgg.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vgg.classifier[6].parameters(), lr=0.001)  # Only optimize parameters of the fully connected layer
train_model(vgg, criterion, optimizer, train_loader, val_loader)

# ResNet-18
resnet18 = torchvision.models.resnet18(pretrained=True)
for param in resnet18.parameters():
    param.requires_grad = False
resnet18.fc = nn.Linear(resnet18.fc.in_features, 10)  # Modify the last fully connected layer for 10 classes
resnet18.fc.requires_grad = True  # Set requires_grad to True for the fully connected layer
resnet18 = resnet18.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet18.fc.parameters(), lr=0.001)  # Only optimize parameters of the fully connected layer
train_model(resnet18, criterion, optimizer, train_loader, val_loader)

# ResNet-50
resnet50 = torchvision.models.resnet50(pretrained=True)
for param in resnet50.parameters():
    param.requires_grad = False
resnet50.fc = nn.Linear(resnet50.fc.in_features, 10)  # Modify the last fully connected layer for 10 classes
resnet50.fc.requires_grad = True  # Set requires_grad to True for the fully connected layer
resnet50 = resnet50.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet50.fc.parameters(), lr=0.001)  # Only optimize parameters of the fully connected layer
train_model(resnet50, criterion, optimizer, train_loader, val_loader)

# ResNet-101
resnet101 = torchvision.models.resnet101(pretrained=True)
for param in resnet101.parameters():
    param.requires_grad = False
resnet101.fc = nn.Linear(resnet101.fc.in_features, 10)  # Modify the last fully connected layer for 10 classes
resnet101.fc.requires_grad = True  # Set requires_grad to True for the fully connected layer
resnet101 = resnet101.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet101.fc.parameters(), lr=0.001)  # Only optimize parameters of the fully connected layer
train_model(resnet101, criterion, optimizer, train_loader, val_loader)

