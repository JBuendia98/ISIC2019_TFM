# mi_proyecto_isic/src/utils/data_loader.py

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import random


# --- GLOBAL CONFIGURATIONS ---
# IMPORTANT: Adjust this path to where you place the 'ISIC_2019_Training_Input' folder
# and the CSV files.
# For a GitHub clone, users will create a 'data' folder and place the downloaded ISIC files inside.
_BASE_DATA_DIR = 'data' # This assumes 'data' folder is at the project root
_INPUT_DIR = os.path.join(_BASE_DATA_DIR, 'ISIC_2019_Training_Input/ISIC_2019_Training_Input')
_LABELS_CSV = os.path.join(_BASE_DATA_DIR, 'ISIC_2019_Training_GroundTruth.csv')
_METADATA_CSV = os.path.join(_BASE_DATA_DIR, 'ISIC_2019_Training_Metadata.csv')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- REPRODUCIBILITY ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# --- GLOBAL LABEL MAPPING (loaded once) ---
# This will be populated when `load_and_split_data` is first called.
global_label_map = None
global_inv_label_map = None
num_output_classes = None

def load_and_split_data(test_size=0.2, val_size_ratio=0.5, random_state=42):
    """
    Loads the ISIC dataset, merges labels and metadata, and splits it into
    training, validation, and test sets.
    
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    global global_label_map, global_inv_label_map, num_output_classes

    print("Loading full dataset...")
    df_labels = pd.read_csv(_LABELS_CSV)
    df_meta = pd.read_csv(_METADATA_CSV)
    df = pd.merge(df_labels, df_meta, on='image')

    label_cols = [col for col in df_labels.columns if col != 'image']
    df[label_cols] = df[label_cols].apply(pd.to_numeric, errors='coerce')
    df['label'] = df[label_cols].idxmax(axis=1)

    unique_labels = sorted(df['label'].unique())
    global_label_map = {label: idx for idx, label in enumerate(unique_labels)}
    global_inv_label_map = {v: k for k, v in global_label_map.items()}
    num_output_classes = len(global_label_map)

    print(f"Total images: {len(df)}")
    print(f"Number of classes: {num_output_classes}")

    # Split into train and temporary (val+test)
    train_df, temp_df = train_test_split(df, test_size=test_size, stratify=df['label'], random_state=random_state)
    # Split temporary into validation and test
    val_df, test_df = train_test_split(temp_df, test_size=val_size_ratio, stratify=temp_df['label'], random_state=random_state)
    
    print(f"Train set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    print(f"Test set size: {len(test_df)}")

    return train_df, val_df, test_df

# --- CUSTOM DATASET CLASS ---
class ISICDataset(Dataset):
    def __init__(self, dataframe, img_dir=_INPUT_DIR, transform=None, label_map=None):
        self.data = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        # Use the global_label_map if not provided, for consistency
        self.label_map = label_map if label_map is not None else global_label_map
        if self.label_map is None:
            raise ValueError("label_map must be provided or global_label_map must be set by calling load_and_split_data first.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_id = self.data.iloc[idx]['image']
        label_name = self.data.iloc[idx]['label']
        label = self.label_map[label_name]

        img_path = os.path.join(self.img_dir, img_id + '.jpg')

        # Handle missing images gracefully (e.g., return a black image)
        if not os.path.exists(img_path):
            print(f"Warning: Image not found at {img_path}. Returning a black placeholder.")
            image = Image.new('RGB', (224, 224), color='black') # Default size, will be transformed
            if self.transform:
                image = self.transform(image)
            return image, list(self.label_map.values())[0] # Return the first valid label

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

def get_dataloaders(train_df, val_df, test_df, train_transform, val_transform, test_transform,
                    batch_size, num_workers=4, use_weighted_sampler=False):
    """
    Creates and returns PyTorch DataLoaders for the given DataFrames and transforms.
    """
    print(f"Using {num_workers} workers for DataLoaders.")

    train_dataset = ISICDataset(train_df, transform=train_transform)
    val_dataset = ISICDataset(val_df, transform=val_transform)
    test_dataset = ISICDataset(test_df, transform=test_transform)

    if use_weighted_sampler:
        # Calculate weights for WeightedRandomSampler
        labels = train_df['label'].map(global_label_map).values
        class_counts = np.bincount(labels)
        # Handle classes with zero count to avoid division by zero
        class_counts[class_counts == 0] = 1 
        weights = 1.0 / class_counts
        sample_weights = weights[labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader