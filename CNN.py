import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import cv2
import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 0.0001
DATA_PATH = "./Dataset_BUSI_with_GT/" # Point this to your unzipped folder

print(f"Using device: {DEVICE}")

# --- 1. DATA PREPARATION ---
class BUSIDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # Read image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Load file paths
# Assuming folder structure: .../benign/img.png, .../malignant/img.png, .../normal/img.png
def load_data(root_dir):
    image_paths = []
    labels = []
    classes = ['normal', 'benign', 'malignant']
    
    for label_idx, class_name in enumerate(classes):
        # BUSI images often have masks named similar to images. We need to filter only the original images.
        # Usually original images don't have "_mask" in the name.
        path_pattern = os.path.join(root_dir, class_name, "*).png") 
        # Note: BUSI naming can be tricky (e.g., "benign (1).png"). Adjust glob pattern as needed.
        # A safer generic way:
        full_class_path = os.path.join(root_dir, class_name)
        for filename in os.listdir(full_class_path):
            if 'mask' not in filename and filename.endswith('.png'):
                image_paths.append(os.path.join(full_class_path, filename))
                labels.append(label_idx)
                
    return image_paths, labels, classes

# Get data
print("Loading data paths...")
all_images, all_labels, class_names = load_data(DATA_PATH)

# Split Data (80% Train, 20% Val)
X_train, X_val, y_train, y_val = train_test_split(all_images, all_labels, test_size=0.2, stratify=all_labels, random_state=42)

# Transforms (Crucial for Medical Imaging)
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNet stats
])

val_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Create Loaders
train_dataset = BUSIDataset(X_train, y_train, transform=train_transforms)
val_dataset = BUSIDataset(X_val, y_val, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- 2. MODEL SETUP (BASELINE CNN) ---
print("Setting up ResNet50...")
model = models.resnet50(pretrained=True)

# Freeze early layers (optional, but good for small datasets)
for param in model.parameters():
    param.requires_grad = False

# Replace the last layer for our 3 classes
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))

model = model.to(DEVICE)

# --- 3. TRAINING LOOP ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

def train_epoch(model, loader):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    return running_loss / len(loader), 100 * correct / total

def validate_epoch(model, loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return 100 * correct / total, all_preds, all_labels

# Run Training
print("Starting Training...")
for epoch in range(EPOCHS):
    train_loss, train_acc = train_epoch(model, train_loader)
    val_acc, _, _ = validate_epoch(model, val_loader)
    
    print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

print("Training Complete. Saving Baseline Model...")
torch.save(model.state_dict(), 'baseline_resnet50.pth')