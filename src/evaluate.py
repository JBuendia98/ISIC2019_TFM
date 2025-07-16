
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import os

# Import from your custom modules
from src.utils.data_loader import set_seed, load_and_split_data, get_dataloaders, device, global_label_map, global_inv_label_map, num_output_classes
from src.utils.transforms import val_transform_b3 # Use validation transform for evaluation
from src.models.custom_models import get_efficientnet_b3_model # Assuming EfficientNet-B3 is the chosen one

# --- CONFIGURATION FOR EVALUATION ---
set_seed(42)
MODEL_LOAD_PATH = 'models/efficientnet_b3_focal_sampler.pth' # Path to your trained model
BATCH_SIZE = 32
NUM_WORKERS = 4

print(f"Evaluating on device: {device}")

# --- LOAD DATA (just for test set) ---
# We load the full data to ensure correct test set split and label mapping
train_df, val_df, test_df = load_and_split_data(test_size=0.2, val_size_ratio=0.5, random_state=42)

# Get DataLoaders (only need test_loader for evaluation)
_, _, test_loader = get_dataloaders(
    train_df, val_df, test_df,
    train_transform=None, # Not needed for evaluation
    val_transform=val_transform_b3,
    test_transform=val_transform_b3,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    use_weighted_sampler=False # Not for test set
)

# --- LOAD MODEL ---
model = None
try:
    checkpoint = torch.load(MODEL_LOAD_PATH, map_location=device)
    # Ensure label map is loaded correctly from checkpoint for consistency
    loaded_label_map = checkpoint.get('label_map', global_label_map)
    loaded_num_classes = len(loaded_label_map) if loaded_label_map else num_output_classes

    model = get_efficientnet_b3_model(loaded_num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval() # Set model to evaluation mode
    print(f" Model loaded successfully from {MODEL_LOAD_PATH}")

    # Update global_inv_label_map based on loaded_label_map
    global_inv_label_map = {v: k for k, v in loaded_label_map.items()}

except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_LOAD_PATH}. Please train the model first.")
    exit()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- EVALUATION ON TEST SET ---
all_preds = []
all_labels = []
all_probs = [] # To store probabilities for ROC AUC

print("Starting evaluation on test set...")
with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Evaluating on test set"):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1) # Get probabilities
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

# Get class names for reporting and plotting
class_names = [global_inv_label_map[i] for i in range(num_output_classes)]

# --- CLASSIFICATION REPORT ---
print("\n📊 Classification Report on Test Set:")
print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

# --- CONFUSION MATRIX ---
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix on Test Set')
    plt.tight_layout()
    plt.show()

print("\n--- Confusion Matrix on Test Set ---")
plot_confusion_matrix(all_labels, all_preds, class_names)

# --- ROC CURVES AND AUC ---
def plot_roc_curves(y_true, y_probs, class_names):
    n_classes = len(class_names)
    # Binarize labels for ROC curve calculation (one-hot encoding)
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

    plt.figure(figsize=(12, 8))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], np.array(y_probs)[:, i])
        auc_score = roc_auc_score(y_true_bin[:, i], np.array(y_probs)[:, i])
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {auc_score:.2f})")

    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.50)') # Diagonal line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves by Class on Test Set")
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

print("\n--- ROC Curves on Test Set ---")
plot_roc_curves(all_labels, all_probs, class_names)

# Calculate Macro-average AUC
all_labels_onehot = label_binarize(all_labels, classes=list(range(num_output_classes)))
macro_avg_auc = roc_auc_score(all_labels_onehot, np.array(all_probs), average='macro')
print(f"\n📈 Macro-average AUC on Test Set: {macro_avg_auc:.4f}")

# --- PLOT TRAINING HISTORY (if available in checkpoint) ---
def plot_training_history(history):
    plt.figure(figsize=(14, 5))
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()

if 'history' in checkpoint:
    print("\n--- Training History Plots ---")
    plot_training_history(checkpoint['history'])
else:
    print("\nNo training history found in the checkpoint to plot.")

print("\n--- Evaluation on Test Set Completed ---")