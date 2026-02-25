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
import json, random, time #imports for run folder
import numpy as np
from sklearn.metrics import confusion_matrix

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 0.0001
DATA_PATH = "./Dataset_BUSI_with_GT/" # Point this to your unzipped folder

print(f"Using device: {DEVICE}")

#Adding a run folder + seed

RUN_NAME = time.strftime("cnn_busi_baseline_%Y%m%d_%H%M%S")
RUN_DIR = os.path.join("runs", RUN_NAME)
os.makedirs(os.path.join(RUN_DIR, "checkpoints"), exist_ok = True)
os.makedirs(os.path.join(RUN_DIR, "preds"), exist_ok = True)
os.makedirs(os.path.join(RUN_DIR, "plots"), exist_ok = True)

SEED = 42
def seed_everything(seed = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(SEED)

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

        return image, label, img_path


#Helper Evaluate
def evaluate_and_save(model, loader, split_name = "val"):
    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []
    all_paths = []

    with torch.no_grad():
        pass

# Load file paths
# Assuming folder structure: .../benign/img.png, .../malignant/img.png, .../normal/img.png
def load_data(root_dir):
    image_paths = []
    labels = []
    classes = ['normal', 'benign', 'malignant']
    
    for label_idx, class_name in enumerate(classes):
        # BUSI images often have masks named similar to images. We need to filter only the original images.
        # Usually original images don't have "_mask" in the name.
        # Note: BUSI naming can be tricky (e.g., "benign (1).png"). Adjust glob pattern as needed.
        # A safer generic way:]
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
#We are replacing this with a 70/10/20 split -> X_train, X_val, y_train, y_val = train_test_split(all_images, all_labels, test_size=0.2, stratify=all_labels, random_state=42)

#First: Split out test = 20%
X_trainval, X_test, y_trainval, y_test = train_test_split(
    all_images, all_labels,
    test_size=0.20,
    stratify=all_labels,
    random_state=SEED
)

#Second : we split Train/Val from the rest of 80%
#val = 10% of total so 10% out of 80  = 0.125

X_train,X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval,
    test_size=0.125,
    stratify=y_trainval,
    random_state=SEED
)
#Save Split
split = {
    "train": X_train,
    "val":X_val,
    "test":X_test,
    "y_train":y_train,
    "y_val":y_val,
    "y_test":y_test,
    "classes":class_names

}
with open(os.path.join(RUN_DIR,"split.json"),"w") as f:
    json.dump(split, f, indent = 2)
# Save Config
config = { 
    "device": str(DEVICE),
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "learning_rate": LEARNING_RATE,
    "data_path": DATA_PATH,
    "seed": SEED,
    "img_size":[224, 224],
    "model":"resnet50_imagenet",
    "frozen":"all except layer 3, layer 4, fc",
    "classes": class_names,
    "normalize_mean":[0.485, 0.456, 0.406],
    "normalize_std":[0.229, 0.224, 0.225],
}
with open(os.path.join(RUN_DIR, "config.json"),"w") as f:
    json.dump(config, f, indent = 2)

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
test_dataset = BUSIDataset(X_test, y_test, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- 2. MODEL SETUP (BASELINE CNN) ---
print("Setting up ResNet50...")
model = models.resnet50(pretrained=True)

# Freeze the early layers (Layer 1 and 2), but UNFREEZE Layer 3 and 4 
# so the model can learn ultrasound-specific textures.
for name, param in model.named_parameters():
    if "layer3" in name or "layer4" in name or "fc" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# Replace the last layer for our 3 classes
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))

model = model.to(DEVICE)


# --- 3. TRAINING LOOP ---
criterion = nn.CrossEntropyLoss()
# Tell the optimizer to only update the unfrozen parameters
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)

#Save the best checkpoint
best_val_acc = -1.0
best_path = os.path.join(RUN_DIR, "checkpoints", "best.pt")
last_path = os.path.join(RUN_DIR, "checkpoints", "last.pt")

def train_epoch(model, loader):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels, _ in loader:
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
        for images, labels, _ in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return 100 * correct / total, all_preds, all_labels

def evaluate_and_save(model, loader, split_name = "val"):
    model.eval()
    all_preds, all_labels, all_paths = [], [], []
    all_probs = []

    with torch.no_grad():
        for images, labels, paths in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim = 1).cpu().numpy()
            preds = probs.argmax(axis=1)

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_paths.extend(paths)

    # report

    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    with open(os.path.join(RUN_DIR, f"{split_name}_classification_report.txt"),"w") as f:
        f.write(f"Accuracy:{acc:.4f}\n\n")
        f.write(report)
        f.write("\n\nConfusion Matrix: \n")
        f.write(np.array2string(cm))
    # save preds csv

    probs_df = pd.DataFrame(all_probs, columns=[f"prob_{c}" for c in class_names])
    df = pd.DataFrame({
        "path": all_paths,
        "y_true": all_labels,
        "y_pred": all_preds,
    })
    df = pd.concat([df, probs_df], axis=1)
    df.to_csv(os.path.join(RUN_DIR, "preds", f"{split_name}_predictions.csv"), index=False)

    return acc, cm, report


# Run Training

print("Starting Training")
for epoch in range(EPOCHS):
    train_loss, train_acc = train_epoch(model, train_loader)
    val_acc, _, _ = validate_epoch(model, val_loader)

    #Save LAST checkpoint every epoch
    torch.save({
        "model_state": model.state_dict(),
        "epoch": epoch + 1,
        "val_acc": val_acc
    }, last_path)

    #Save BEST checkpoint
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            "model_state":model.state_dict(),
            "epoch": epoch+ 1,
            "val_acc": val_acc
        }, best_path)
    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Train Loss: {train_loss:.4f} | "
          f"Train Acc: {train_acc:.2f}% | "
          f"Val Acc:{val_acc:.2f}%")
print(f"Training Complete. Best Val Acc: {best_val_acc:.2f}%")

ckpt = torch.load(best_path, map_location=DEVICE)
model.load_state_dict(ckpt["model_state"])

#Evaluate on val
evaluate_and_save(model, val_loader,"val")

#Evaluate on test
evaluate_and_save(model, test_loader, "test")

print("Saved test report to: ", os.path.join(RUN_DIR, "test_classification_report.txt"))
print("Saved test preds to :", os.pat.join(RUN_DIR, "preds", "test_predictions.csv"))

val_acc,cm,rep = evaluate_and_save(model,val_loader, "val")
print("Saved:", os.path.join(RUN_DIR, "val_classification_report.txt"))
print("Saved:", os.path.join(RUN_DIR, "preds", "val_predictions.csv"))
torch.save(model.state_dict(), 'baseline_resnet50.pth')