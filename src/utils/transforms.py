
from torchvision import transforms

IMAGE_SIZE_224 = 224
IMAGE_SIZE_B3 = 300
IMAGE_SIZE_INCEPTION = 299 

NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

train_transform_b3 = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE_B3, scale=(0.8, 1.0)), 
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
])

# Validation/Test transformations are typically the same (resizing + normalization)
val_transform_b3 = transforms.Compose([
    transforms.Resize((IMAGE_SIZE_B3, IMAGE_SIZE_B3)),
    transforms.ToTensor(),
    transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
])

# If you were to use other model input sizes, you'd define them here:
# train_transform_224 = transforms.Compose([
#     transforms.RandomResizedCrop(IMAGE_SIZE_224, scale=(0.8, 1.0)),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(20),
#     transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
#     transforms.ToTensor(),
#     transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
# ])
# val_transform_224 = transforms.Compose([
#     transforms.Resize((IMAGE_SIZE_224, IMAGE_SIZE_224)),
#     transforms.ToTensor(),
#     transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
# ])