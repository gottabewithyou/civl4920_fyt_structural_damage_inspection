import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from dataset import CrackDataset  # Your dataset class
from unet_model import UNet  # Your model class

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define dataset transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Load datasets
train_dataset = CrackDataset("./datasets/CRACK500/traindata/traindata", "./datasets/CRACK500/traindata/traindata_mask", transform=transform)
val_dataset = CrackDataset("./datasets/CRACK500/valdata/valdata", "./datasets/CRACK500/valdata/valdata_mask", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Instantiate model, loss, optimizer
model = UNet().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
num_epochs = 25
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        loss = criterion(outputs, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {total_loss/len(train_loader):.4f}")

    # Validation step
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
    print(f"Validation Loss: {val_loss/len(val_loader):.4f}")

# Save trained model
torch.save(model.state_dict(), "unet_crack.pth")