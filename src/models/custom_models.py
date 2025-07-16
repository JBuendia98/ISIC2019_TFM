
import torch.nn as nn
from torchvision import models
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
# You would also add imports for ResNet, DenseNet, ViT, Inception if you used them in train_compare.py
# from torchvision.models import resnet50, ResNet50_Weights
# from torchvision.models import densenet121, DenseNet121_Weights
# from torchvision.models import vit_b_16, ViT_B_16_Weights
# from torchvision.models import inception_v3, Inception_V3_Weights

def get_efficientnet_b3_model(num_classes):
    """
    Loads a pre-trained EfficientNet-B3 model and adapts its classifier for num_classes.
    """
    model = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
    # The classifier of EfficientNet is Sequential(Linear, Dropout)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

# Add functions for other models if you need them outside of train_compare.py
# def get_resnet50_model(num_classes):
#     model = resnet50(weights=ResNet50_Weights.DEFAULT)
#     model.fc = nn.Linear(model.fc.in_features, num_classes)
#     return model

# def get_densenet121_model(num_classes):
#     model = densenet121(weights=DenseNet121_Weights.DEFAULT)
#     model.classifier = nn.Linear(model.classifier.in_features, num_classes)
#     return model

# def get_vit_b16_model(num_classes):
#     model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
#     model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
#     return model

# def get_inception_v3_model(num_classes):
#     model = inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=True)
#     model.fc = nn.Linear(model.fc.in_features, num_classes)
#     if model.AuxLogits is not None:
#         model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)
#     return model