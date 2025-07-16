
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
import numpy as np
import os 

# Import from your custom modules
from src.utils.data_loader import set_seed, load_and_split_data, get_dataloaders, device, global_label_map, global_inv_label_map, num_output_classes
from src.utils.transforms import train_transform_b3, val_transform_b3
from src.utils.metrics import FocalLoss
from src.models.custom_models import get_efficientnet_b3_model # Assuming EfficientNet-B3 is the chosen one

# --- CONFIGURATION FOR TRAINING ---
set_seed(42) # Ensure reproducibility
BATCH_SIZE = 32
NUM_EPOCHS = 20
PATIENCE = 5
MODEL_SAVE_PATH = 'models/efficientnet_b3_focal_sampler.pth'
NUM_WORKERS = 4 

print(f"Training on device: {device}")

# --- LOAD DATA AND CREATE DATALOADERS ---
train_df, val_df, test_df = load_and_split_data(test_size=0.2, val_size_ratio=0.5, random_state=42)

# Get DataLoaders with WeightedRandomSampler for training
train_loader, val_loader, test_loader = get_dataloaders(
    train_df, val_df, test_df,
    train_transform=train_transform_b3,
    val_transform=val_transform_b3,
    test_transform=val_transform_b3, # Use val_transform for test set
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    use_weighted_sampler=True
)

# --- MODEL, CRITERION, OPTIMIZER ---
model = get_efficientnet_b3_model(num_output_classes)
model.to(device)

# Focal Loss (your provided code uses alpha=None initially)
# class_weights = 1.0 / np.bincount(train_df['label'].map(global_label_map).values)
# class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
criterion = FocalLoss(alpha=None, gamma=2.0) # Or alpha=class_weights if you want

optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.1, patience=5, verbose=True, min_lr=1e-7
)

# --- TRAINING FUNCTION ---
def train_model(model, criterion, optimizer, train_loader, val_loader, scheduler,
                num_epochs=NUM_EPOCHS, patience=PATIENCE):
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    epochs_no_improve = 0

    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss, correct = 0.0, 0
        total_samples = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Train"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total_samples += inputs.size(0)

        train_loss = running_loss / total_samples
        train_acc = correct / total_samples
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # --- Validation Phase ---
        model.eval()
        val_loss, val_correct = 0.0, 0
        total_val_samples = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} Val"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                total_val_samples += inputs.size(0)

        val_loss_avg = val_loss / total_val_samples
        val_acc_avg = val_correct / total_val_samples
        history['val_loss'].append(val_loss_avg)
        history['val_acc'].append(val_acc_avg)

        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss_avg:.4f}, Acc: {val_acc_avg:.4f}")

        scheduler.step(val_loss_avg) # Step scheduler on validation loss

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
    
    model.load_state_dict(best_model_wts)
    return model, history

# --- START TRAINING ---
trained_model, training_history = train_model(model, criterion, optimizer, train_loader, val_loader, scheduler, NUM_EPOCHS, PATIENCE)

# --- SAVE CHECKPOINT ---
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True) # Ensure models/ directory exists
checkpoint = {
    'model_state_dict': trained_model.state_dict(),
    'label_map': global_label_map, # Save the label map for consistent inference
    'model_name': 'efficientnet_b3',
    'history': training_history
}
torch.save(checkpoint, MODEL_SAVE_PATH)
print(f"Checkpoint saved to {MODEL_SAVE_PATH}")