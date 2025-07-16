import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM # Make sure you have installed: pip install pytorch_grad_cam
from pytorch_grad_cam.utils.image import show_cam_on_image as show_cam_on_image_lib # Rename to avoid conflict with PIL Image

# This transform matches your val_transform_b3 in src/utils/transforms.py
# It's crucial for CAM to be applied on normalized images
_CAM_TRANSFORM = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_grad_cam_image(model, image_pil: Image.Image, target_layer, target_category=None, device='cpu'):
    """
    Generates a Grad-CAM heatmap overlayed on the original image.

    Args:
        model (torch.nn.Module): The trained model.
        image_pil (PIL.Image.Image): The input image.
        target_layer (torch.nn.Module): The target convolutional layer for Grad-CAM.
        target_category (int, optional): The target class index for explanation.
                                         If None, the predicted class will be used.
        device (str): The device to run the model on ('cpu' or 'cuda').

    Returns:
        PIL.Image.Image: The image with Grad-CAM heatmap overlayed.
    """
    # Resize the PIL image to 300x300 before converting to numpy for CAM library
    # The CAM library expects the RGB image to be in [0, 1]
    resized_original_image = image_pil.convert("RGB").resize((300, 300))
    rgb_image_np = np.array(resized_original_image).astype(np.float32) / 255.0

    input_tensor = _CAM_TRANSFORM(image_pil).unsqueeze(0).to(device)

    # If target_category is not provided, predict it
    if target_category is None:
        model.eval()
        with torch.no_grad():
            outputs = model(input_tensor)
            target_category = torch.argmax(outputs, dim=1).item()

    with GradCAM(model=model, target_layers=[target_layer], use_cuda=(device=='cuda')) as cam:
        # Pass the input_tensor, and optionally target_category if you want explanation for a specific class
        grayscale_cam = cam(input_tensor=input_tensor, targets=None if target_category is None else [torch.nn.functional.one_hot(torch.tensor(target_category), num_classes=model.classifier[1].out_features)])[0, :]
        # `show_cam_on_image_lib` expects RGB image in [0, 1]
        cam_image_np = show_cam_on_image_lib(rgb_image_np, grayscale_cam, use_rgb=True)
        
    return Image.fromarray(cam_image_np)

# Example of how to get the target_layer for EfficientNet-B3:
# This needs to be done where the model is loaded (e.g., in app.py)
# from src.models.custom_models import get_efficientnet_b3_model
# model = get_efficientnet_b3_model(num_classes)
# target_layer = model.features[-1] # This is typically the last convolutional block