# 🔬 ISIC 2019 Skin Lesion Classification with EfficientNet-B3

This project implements a deep learning solution for classifying skin lesions using the ISIC 2019 Challenge dataset. It leverages a pre-trained EfficientNet-B3 model, employs techniques like Focal Loss and Weighted Random Sampling to handle class imbalance, and provides explainability through Grad-CAM via a Gradio web interface.

## 🌟 Features

* **EfficientNet-B3:** Utilizes a powerful convolutional neural network pre-trained on ImageNet.
* **Class Imbalance Handling:** Implements Focal Loss for robust training on imbalanced medical datasets and Weighted Random Sampler during training.
* **Data Augmentation:** Comprehensive image transformations to improve model generalization.
* **Modular Codebase:** Organized into `src` for better maintainability and readability.
* **Performance Metrics:** Detailed evaluation including Classification Report, Confusion Matrix, and ROC Curves.
* **Explainable AI (XAI):** Integrated Grad-CAM visualizations to understand model predictions.
* **Interactive Web App:** A user-friendly Gradio interface for real-time inference and Grad-CAM explanations.

## 📂 Project Structure

mi_proyecto_isic/
├── data/                       # Directory for raw and processed data (excluded from Git)
│   └── README.md               # Instructions on how to download and set up the ISIC 2019 dataset
│   └── .gitkeep                # Placeholder to keep the 'data/' folder in Git
├── models/                     # Directory for trained model checkpoints (excluded from Git)
│   └── README.md               # Instructions on how to get the pre-trained model
│   └── .gitkeep                # Placeholder to keep the 'models/' folder in Git
├── notebooks/                  # Jupyter notebooks for experimentation and EDA
│   └── EDA.ipynb               # Exploratory Data Analysis
├── src/                        # Main source code
│   ├── models/
│   │   └── custom_models.py    # Definitions of model architectures (e.g., EfficientNet)
│   ├── utils/
│   │   ├── data_loader.py      # ISICDataset class, data loading/splitting, DataLoaders, global maps
│   │   ├── transforms.py       # Image transformation definitions
│   │   └── metrics.py          # Loss functions (e.g., FocalLoss)
│   ├── train.py                # Script for training the EfficientNet-B3 model
│   ├── evaluate.py             # Script for comprehensive evaluation on the test set
│   └── explain/
│       └── grad_cam.py         # Grad-CAM implementation
├── app.py                      # Gradio web application for inference and XAI
├── .gitignore                  # Specifies files/folders to exclude from Git
├── README.md                   # This file
└── requirements.txt            # Python dependencies


## 🚀 Getting Started

Follow these steps to set up the project locally and run the code.

### Prerequisites

* Python 3.8+
* `pip` (Python package installer)
* A GPU (CUDA enabled) is highly recommended for faster training and inference. The code will fall back to CPU if no GPU is available.

### 1. Clone the Repository

```bash
git clone https://github.com/JBuendia98/ISIC2019_TFM.git
cd ISIC2019_TFM

### 2. Set up Python Environment

It's recommended to use a virtual environment.

Bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

### 3. Install Dependencies

Install all necessary Python packages:

Bash
pip install -r requirements.txt

### 4. Download and Prepare the Dataset

The ISIC 2019 Challenge Dataset is not included in this repository due to its substantial size and licensing. You need to download it separately from the official ISIC archive.

Source: Access the official ISIC 2019 download page:
https://challenge.isic-archive.com/data/#2019

Required files: For this project, you will need to download the following files from the ISIC 2019 Challenge:

ISIC_2019_Training_Input.zip (Contains all training images)

ISIC_2019_Training_GroundTruth.csv (Ground truth labels for training)

ISIC_2019_Training_Metadata.csv (Additional metadata for training)
(Note: Validation and test data are also available, but this project focuses on training data for the local division.)


### 5. Run Exploratory Data Analysis (EDA)

Before training, you can explore the dataset characteristics using the provided Jupyter Notebook:

Bash
jupyter notebook notebooks/EDA.ipynb
Follow the cells in the notebook to understand class distribution, age/sex demographics, and image properties. Make sure to update the paths in EDA.ipynb from /kaggle/input/... to ../data/... for local execution.

### 6. Compare and Select Models

Execute the train_compare.py script to train and evaluate various pre-trained backbone models (e.g., ResNet, DenseNet, EfficientNet, ViT, Inception) on a subset of your data. This script helps in pre-selecting the best performing architecture for further fine-tuning. The best model checkpoint from this comparison will be saved in the models/ directory.

Bash
python src/train_compare.py

### 7. Train the Model

Once the dataset is prepared and you've identified a suitable backbone (e.g., EfficientNet-B3, as suggested by the project title), you can train the chosen model. The train.py script will handle data loading, augmentation, and model training with class imbalance techniques. The best model checkpoint will be saved in the models/ directory.

Bash
python src/train_final_model.py
(Note: Training can take a significant amount of time depending on your hardware.)

### 8. Evaluate the Model

After training, evaluate the model's performance on the test set using the evaluate.py script. This will print detailed metrics (Classification Report, Confusion Matrix, etc.) to the console.

Bash
python src/evaluate.py

### 9. Run the Gradio Web Application

Launch the interactive web application to perform real-time inference on new images and visualize Grad-CAM explanations.

Bash
python app.py
Open your web browser and navigate to the local address provided by Gradio (e.g., http://127.0.0.1:7860). You can upload an image and see its predicted class and the Grad-CAM heatmap highlighting the regions most influential for the prediction.
