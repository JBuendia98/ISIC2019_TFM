import gradio as gr
import torch
import torch.nn as nn # For nn.Linear when re-initializing model classifier
from PIL import Image
import numpy as np # For numpy arrays required by show_cam_on_image
from torchvision import models, transforms


from src.models.custom_models import get_efficientnet_b3_model
from src.utils.data_loader import device # Import the device setting
from src.explain.grad_cam import get_grad_cam_image # Import the Grad-CAM function

# === CONFIGURATION ===
MODEL_PATH = "models/efficientnet_b3_focal_sampler.pth" # Path to your saved model checkpoint
MODEL_NAME = "EfficientNet-B3" # Name of the model used

# === LOAD MODEL AND LABEL MAP ===
# This part runs once when the Gradio app starts
print(f"Loading the model '{MODEL_NAME}' from '{MODEL_PATH}'...")
model = None
label_map = None # This will store the label_map loaded from the checkpoint
inv_label_map = None # Inverse mapping for display
target_layer = None # Specific layer for Grad-CAM

try:
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    label_map = checkpoint['label_map']
    inv_label_map = {v: k for k, v in label_map.items()}
    num_classes = len(label_map)

    if MODEL_NAME == "EfficientNet-B3":
        model = get_efficientnet_b3_model(num_classes)
        # Define target_layer for Grad-CAM for EfficientNet-B3
        target_layer = model.features[-1] # Common choice for EfficientNet
    else:
        raise ValueError(f"Model {MODEL_NAME} is not explicitly configured for loading in this app.py.")

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval() # Set model to evaluation mode
    print("Model loaded successfully.")

except FileNotFoundError:
    print(f"ERROR: Model checkpoint not found at {MODEL_PATH}. Please ensure the model is trained and saved.")
    print("The application will start, but predictions will not be possible.")
    model = None
except Exception as e:
    print(f"ERROR during model loading: {e}")
    print("The application will start, but predictions will not be possible.")
    model = None

# === TRANSFORMATIONS (matching val_transform_b3 from src/utils/transforms.py) ===
# This transform is used for model inference
inference_transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# === PREDICTION FUNCTION FOR GRADIO ===
def predict_image(image_pil: Image.Image, generate_cam: bool):
    if model is None or label_map is None or inv_label_map is None:
        return "Model not loaded. Check server logs for errors.", {}, None

    if image_pil is None:
        return "Please upload an image.", {}, None

    # Preprocess image for model
    input_tensor = inference_transform(image_pil).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0] # Get probabilities for the single image

    predicted_idx = torch.argmax(probabilities).item()
    predicted_label = inv_label_map[predicted_idx]

    # Format probabilities for Gradio Label output
    confidences = {inv_label_map[i]: float(probabilities[i]) for i in range(len(label_map))}

    cam_image = None
    if generate_cam and model is not None and target_layer is not None:
        try:
            # Pass the original PIL image and the model's target layer
            cam_image = get_grad_cam_image(model, image_pil, target_layer, device=device)
        except Exception as e:
            print(f"Error generating Grad-CAM: {e}")
            cam_image = None

    return f"Predicted Class: {predicted_label}", confidences, cam_image

# === GRADIO INTERFACE ===
iface = gr.Interface(
    fn=predict_image,
    inputs=[
        gr.Image(type="pil", label="Upload an Image of a Skin Lesion"),
        gr.Checkbox(label="Generate Grad-CAM Explanation")
    ],
    outputs=[
        gr.Textbox(label="Prediction Result"),
        gr.Label(num_top_classes=len(label_map) if label_map else 8, label="Class Probabilities"), # Use actual num_classes if loaded
        gr.Image(type="pil", label="Grad-CAM Heatmap (if generated)"),
    ],
    title="ISIC 2019 Skin Lesion Classification - EfficientNet-B3 with Grad-CAM",
    description="Upload a dermatoscopic image to classify the type of skin lesion. You can also generate a Grad-CAM heatmap to visualize which parts of the image contributed most to the prediction.",
    allow_flagging="auto", # Allows users to flag interesting examples
    examples=[
        # Add paths to example images here for easy testing in Gradio
        # E.g., ["data/ISIC_2019_Training_Input/ISIC_0000000.jpg", False],
        # E.g., ["data/ISIC_2019_Training_Input/ISIC_0000010.jpg", True],
    ]
)

# === LAUNCH GRADIO APP ===
if __name__ == "__main__":
    iface.launch(share=True) # `share=True` generates a public link for demos