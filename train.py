import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights

# ---------------------------------------------------------
# 1. Device Configuration & Data Pipelines
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_data_loaders(batch_size=32):
    # Data Augmentation & Normalization
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # CIFAR-10 Dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    return trainloader, testloader

# ---------------------------------------------------------
# 2. Transfer Learning Model (ResNet50)
# ---------------------------------------------------------
def build_resnet50_model(num_classes=10):
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)

    # Freeze feature backbone
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier head
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes)
    )
    return model

# ---------------------------------------------------------
# 3. Training Function
# ---------------------------------------------------------
def train():
    os.makedirs("models", exist_ok=True)
    print(f"⚡ Using device: {device}")
    
    trainloader, testloader = get_data_loaders()
    model = build_resnet50_model().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    epochs = 5
    print("🔥 Starting Model Training...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Step [{i+1}/{len(trainloader)}] | Loss: {running_loss/100:.4f} | Acc: {100.*correct/total:.2f}%")
                running_loss = 0.0

    # Save model weights
    torch.save(model.state_dict(), "models/transfer_resnet50.pth")
    print("🎉 Model saved successfully to models/transfer_resnet50.pth!")

if __name__ == "__main__":
    train()