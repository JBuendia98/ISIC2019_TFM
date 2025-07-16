import os
import copy
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import models, transforms
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, accuracy_score

# Import functions and classes from our utility and model modules
from src.utils.data_loader import set_seed, get_train_val_test_dfs, get_dataloaders, global_label_map, num_output_classes, df, input_dir, batch_size, num_workers, device
from src.utils.transforms import train_transform_224, val_transform_224, test_transform_224
from src.utils.transforms import train_transform_b3, val_transform_b3, test_transform_b3
from src.utils.transforms import train_transform_inception, val_transform_inception, test_transform_inception
from src.utils.metrics import FocalLoss # Optional: if you want to test FocalLoss in the comparison, use it here.

from src.models.custom_models import get_resnet50_model, get_densenet121_model, get_efficientnet_b3_model, get_vit_b16_model, get_inception_v3_model

# --- SET SEED AT THE START OF THE SCRIPT ---
set_seed(42)

# --- GLOBAL TRAINING PARAMETERS FOR COMPARISON ---
num_epochs_compare = 5      # Maximum number of epochs per model in the comparison
patience_compare = 3        # Patience for Early Stopping

# --- PREPARE SUB-DATASET AND DATALOADERS FOR COMPARISON ---
# A smaller subset is used for a faster comparison.
train_df_small, val_df_small, test_df_small = get_train_val_test_dfs(df, n_per_class=1000)

print(f"Using {num_workers} workers for the DataLoaders.")

# DataLoaders for 224x224 models (ResNet, DenseNet, ViT)
train_loader_224, val_loader_224, test_loader_224 = get_dataloaders(
    train_df_small, val_df_small, test_df_small,
    train_transform_224, val_transform_224, test_transform_224,
    batch_size, num_workers
)

# DataLoaders for EfficientNet-B3 (300x300)
train_loader_b3, val_loader_b3, test_loader_b3 = get_dataloaders(
    train_df_small, val_df_small, test_df_small,
    train_transform_b3, val_transform_b3, test_transform_b3,
    batch_size, num_workers
)

# DataLoaders for Inception V3 (299x299)
train_loader_inception, val_loader_inception, test_loader_inception = get_dataloaders(
    train_df_small, val_df_small, test_df_small,
    train_transform_inception, val_transform_inception, test_transform_inception,
    batch_size, num_workers
)

# --- TRAINING AND EVALUATION FUNCTION ---
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=num_epochs_compare, patience=patience_compare, model_name="Model", class_names=None):
    print(f"Training {model_name} on device: {device}")
    model = model.to(device)

    # Inception identification (to handle its auxiliary logits in training mode)
    is_inception = isinstance(model, nn.Sequential) and any(isinstance(m, models.Inception3) for m in model if isinstance(m, nn.Module)) or \
                   isinstance(model, models.Inception3)

    best_val_f1_weighted = -1.0 # We use f1-weighted as the saving metric
    best_epoch_metrics = {}
    epochs_no_improve = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=patience_compare, verbose=True, min_lr=1e-7
    )

    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Train ({model_name})"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            if is_inception:
                outputs = model(inputs)
                if isinstance(outputs, models.inception.InceptionOutputs):
                    # Sum the loss from the main and auxiliary classifiers
                    loss = criterion(outputs.logits, labels) + 0.4 * criterion(outputs.aux_logits, labels)
                else:
                    loss = criterion(outputs, labels) # In case Inception returns a tensor directly
            else:
                logits = model(inputs)
                loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}")

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} Val ({model_name})"):
                inputs, labels = inputs.to(device), labels.to(device)

                # In eval mode, Inception always returns the logits tensor directly.
                logits = model(inputs)
                loss = criterion(logits, labels)
                val_loss += loss.item()

                _, preds_val = torch.max(logits, 1)
                all_preds.extend(preds_val.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss_avg = val_loss / len(val_loader)
        val_accuracy = accuracy_score(all_labels, all_preds)
        val_f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        val_f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        val_precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        val_recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)

        print(f"Epoch {epoch+1}/{num_epochs} - Val Loss: {val_loss_avg:.4f}, Val Accuracy: {val_accuracy:.4f}")
        print(f" Val F1-Macro: {val_f1_macro:.4f}, Val F1-Weighted: {val_f1_weighted:.4f}")
        print(f" Val Precision-Macro: {val_precision_macro:.4f}, Val Recall-Macro: {val_recall_macro:.4f}")

        scheduler.step(val_f1_weighted) # Pass the f1_weighted score to the scheduler

        if val_f1_weighted > best_val_f1_weighted:
            best_val_f1_weighted = val_f1_weighted
            epochs_no_improve = 0
            best_epoch_metrics = {
                "accuracy": val_accuracy,
                "f1_macro": val_f1_macro,
                "f1_weighted": val_f1_weighted,
                "precision_macro": val_precision_macro,
                "recall_macro": val_recall_macro,
                "loss": val_loss_avg,
                "epoch": epoch + 1
            }
            best_model_wts = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
            print(f"Early Stopping counter: {epochs_no_improve} of {patience}")
            if epochs_no_improve == patience:
                print(f"Early stopping triggered for {model_name} at epoch {epoch+1}. Restoring best weights.")
                break

    model.load_state_dict(best_model_wts) # Load the best weights at the end

    print(f"\n--- Confusion Matrix for {model_name} (Best Epoch: {best_epoch_metrics.get('epoch', 'N/A')}) ---")
    model.eval()
    all_preds_best_epoch = []
    all_labels_best_epoch = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs) # In eval mode, Inception returns the logits tensor directly.
            _, preds_val = torch.max(logits, 1)
            all_preds_best_epoch.extend(preds_val.cpu().numpy())
            all_labels_best_epoch.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels_best_epoch, all_preds_best_epoch)
    if class_names:
        plt.figure(figsize=(num_output_classes+1, num_output_classes+1))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix for {model_name} (Best Validation Epoch)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
    else:
        print(cm)
    print("--------------------------------------------------\n")

    print(f"\n--- Best Validation Metrics for {model_name} ---")
    for metric, value in best_epoch_metrics.items():
        print(f" {metric.replace('_', ' ').title()}: {value:.4f}")
    print("--------------------------------------------------\n")

    return model, best_epoch_metrics

# --- INSTANTIATION AND TRAINING OF MODELS ---
results = {}
trained_models = {}
criterion = nn.CrossEntropyLoss() # Or you can use FocalLoss() if you defined it in metrics.py

# ==============================================================================
# Models to compare
# ==============================================================================

# ResNet50
print("\n" + "="*50)
print(" STARTING TRAINING: ResNet50")
print("="*50)
model_resnet = get_resnet50_model(num_output_classes)
optimizer_resnet = torch.optim.Adam(model_resnet.parameters(), lr=1e-4, weight_decay=1e-5)
trained_resnet, metrics_resnet = train_model(
    model_resnet, criterion, optimizer_resnet, train_loader_224, val_loader_224,
    model_name="ResNet50", class_names=list(global_label_map.keys())
)
results["ResNet50"] = metrics_resnet
trained_models["ResNet50"] = trained_resnet
del model_resnet, optimizer_resnet
if torch.cuda.is_available(): torch.cuda.empty_cache()

# DenseNet121
print("\n" + "="*50)
print(" STARTING TRAINING: DenseNet121")
print("="*50)
model_densenet = get_densenet121_model(num_output_classes)
optimizer_densenet = torch.optim.Adam(model_densenet.parameters(), lr=1e-4, weight_decay=1e-5)
trained_densenet, metrics_densenet = train_model(
    model_densenet, criterion, optimizer_densenet, train_loader_224, val_loader_224,
    model_name="DenseNet121", class_names=list(global_label_map.keys())
)
results["DenseNet121"] = metrics_densenet
trained_models["DenseNet121"] = trained_densenet
del model_densenet, optimizer_densenet
if torch.cuda.is_available(): torch.cuda.empty_cache()

# EfficientNet-B3
print("\n" + "="*50)
print(" STARTING TRAINING: EfficientNet-B3")
print("="*50)
model_eff_b3 = get_efficientnet_b3_model(num_output_classes)
optimizer_eff_b3 = torch.optim.Adam(model_eff_b3.parameters(), lr=1e-4, weight_decay=1e-5)
trained_eff_b3, metrics_eff_b3 = train_model(
    model_eff_b3, criterion, optimizer_eff_b3, train_loader_b3, val_loader_b3,
    model_name="EfficientNet-B3", class_names=list(global_label_map.keys())
)
results["EfficientNet-B3"] = metrics_eff_b3
trained_models["EfficientNet-B3"] = trained_eff_b3
del model_eff_b3, optimizer_eff_b3
if torch.cuda.is_available(): torch.cuda.empty_cache()

# Vision Transformer (ViT-B16)
print("\n" + "="*50)
print(" STARTING TRAINING: Vision Transformer (ViT-B16)")
print("="*50)
model_vit = get_vit_b16_model(num_output_classes)
optimizer_vit = torch.optim.Adam(model_vit.parameters(), lr=1e-4, weight_decay=1e-5)
trained_vit, metrics_vit = train_model(
    model_vit, criterion, optimizer_vit, train_loader_224, val_loader_224,
    model_name="ViT-B16", class_names=list(global_label_map.keys())
)
results["ViT-B16"] = metrics_vit
trained_models["ViT-B16"] = trained_vit
del model_vit, optimizer_vit
if torch.cuda.is_available(): torch.cuda.empty_cache()

# Inception V3
print("\n" + "="*50)
print(" STARTING TRAINING: InceptionV3")
print("="*50)
model_incep = get_inception_v3_model(num_output_classes)
optimizer_incep = torch.optim.Adam(model_incep.parameters(), lr=1e-4, weight_decay=1e-5)
trained_incep, metrics_incep = train_model(
    model_incep, criterion, optimizer_incep, train_loader_inception, val_loader_inception,
    model_name="InceptionV3", class_names=list(global_label_map.keys())
)
results["InceptionV3"] = metrics_incep
trained_models["InceptionV3"] = trained_incep
del model_incep, optimizer_incep
if torch.cuda.is_available(): torch.cuda.empty_cache()


# --- FINAL SUMMARY OF MODEL PRE-FILTERING ---
print("\n" + "="*60)
print(" FINAL SUMMARY OF MODEL PRE-FILTERING")
print("="*60)
sorted_results = sorted(results.items(), key=lambda item: item[1].get('f1_weighted', 0), reverse=True) # Sort by F1-Weighted

print("\nValidation Metrics Sorted by F1-Weighted:")
print("-" * 50)
for model_name, metrics in sorted_results:
    print(f"\n--- {model_name} --- (Best Epoch: {metrics.get('epoch', 'N/A')})")
    print(f" Validation F1-Weighted: {metrics.get('f1_weighted', 0):.4f} (KEY METRIC)")
    print(f" Validation F1-Macro: {metrics.get('f1_macro', 0):.4f}")
    print(f" Validation Accuracy: {metrics.get('accuracy', 0):.4f}")
    print(f" Validation Loss: {metrics.get('loss', 0):.4f}")
    print(f" Validation Precision (Macro): {metrics.get('precision_macro', 0):.4f}")
    print(f" Validation Recall (Macro): {metrics.get('recall_macro', 0):.4f}")
print("\n" + "="*60)

# --- EVALUATION OF THE BEST MODEL ON THE TEST SET ---
if sorted_results:
    best_model_name = sorted_results[0][0]
    print(f"\n--- Evaluating the BEST MODEL ({best_model_name}) on the Test Set ---")
    best_model = trained_models[best_model_name]
    best_model.eval()

    test_loader_final = None
    final_class_names = list(global_label_map.keys())

    # Determine which test_loader to use based on the name of the best model
    if "ResNet" in best_model_name or "DenseNet" in best_model_name or "ViT" in best_model_name:
        test_loader_final = test_loader_224
    elif "EfficientNet" in best_model_name:
        test_loader_final = test_loader_b3
    elif "Inception" in best_model_name:
        test_loader_final = test_loader_inception
    else:
        print("Error: Could not determine the test DataLoader for the best model.")

    if test_loader_final:
        all_preds_test = []
        all_labels_test = []
        test_loss_total = 0.0
        best_model = best_model.to(device)

        with torch.no_grad():
            for inputs, labels in tqdm(test_loader_final, desc=f"Evaluating {best_model_name} on Test Set"):
                inputs, labels = inputs.to(device), labels.to(device)
                logits = best_model(inputs) # In eval mode, Inception returns the logits tensor directly.
                loss = criterion(logits, labels)
                test_loss_total += loss.item()
                _, preds = torch.max(logits, 1)
                all_preds_test.extend(preds.cpu().numpy())
                all_labels_test.extend(labels.cpu().numpy())

        test_accuracy = accuracy_score(all_labels_test, all_preds_test)
        test_f1_macro = f1_score(all_labels_test, all_preds_test, average='macro', zero_division=0)
        test_f1_weighted = f1_score(all_labels_test, all_preds_test, average='weighted', zero_division=0)
        test_precision_macro = precision_score(all_labels_test, all_preds_test, average='macro', zero_division=0)
        test_recall_macro = recall_score(all_labels_test, all_preds_test, average='macro', zero_division=0)
        test_loss_avg = test_loss_total / len(test_loader_final)

        print(f"\n--- Test Set Metrics for {best_model_name} ---")
        print(f" Test Loss: {test_loss_avg:.4f}")
        print(f" Test Accuracy: {test_accuracy:.4f}")
        print(f" Test F1-Macro: {test_f1_macro:.4f}")
        print(f" Test F1-Weighted: {test_f1_weighted:.4f}")
        print(f" Test Precision (Macro): {test_precision_macro:.4f}")
        print(f" Test Recall (Macro): {test_recall_macro:.4f}")
        print("--------------------------------------------------\n")

        cm_test = confusion_matrix(all_labels_test, all_preds_test)
        print(f"\n--- Confusion Matrix for {best_model_name} on Test Set ---")
        plt.figure(figsize=(num_output_classes+1, num_output_classes+1))
        sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', xticklabels=final_class_names, yticklabels=final_class_names)
        plt.title(f'Confusion Matrix for {best_model_name} on Test Set')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        print("--------------------------------------------------\n")
    else:
        print("Test set evaluation was not performed due to a data loader error.")

# --- SAVE THE BEST MODEL ---
if sorted_results and best_model_name in trained_models:
    final_best_model = trained_models[best_model_name]
    # Make sure the 'models' folder exists
    os.makedirs('models', exist_ok=True)
    checkpoint_name = os.path.join("models", f"best_model_{best_model_name.lower().replace('-', '_')}.pth")
    torch.save({
        'model_state_dict': final_best_model.state_dict(),
        'label_map': global_label_map, # Save the label mapping
        'model_name': best_model_name,
        'metrics': results[best_model_name]
    }, checkpoint_name)
    print(f" The best model ({best_model_name}) has been saved as '{checkpoint_name}'")
print("\n--- Model comparison completed! ---")