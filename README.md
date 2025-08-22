# Celebrity-Identifier

A machine learning project to identify 17 celebrities from uploaded images using a fine-tuned ResNet18 CNN. The application features a Streamlit-based web interface for user-friendly interaction.

---

## Overview

The Celebrity Recognition model leverages a fine-tuned ResNet18 convolutional neural network (CNN) using PyTorch for classifying 17 celebrities from uploaded images. It employs torchvision for image preprocessing with data augmentation and Streamlit for an interactive web interface. The project includes complete scripts for model training, evaluation, and a user-friendly UI, with all necessary files for setup and deployment.

---

## Features

- **Model**: Pre-trained ResNet18 fine-tuned on a custom dataset of 17 celebrities.
- **Training**: Includes data augmentation techniques such as random crops and horizontal flips to improve generalization. A validation set is used to ensure robust performance.
- **UI**: Streamlit app for uploading images and viewing predictions.
- **Tech Stack**: 
  - [PyTorch](https://pytorch.org/)
  - [torchvision](https://pytorch.org/vision/stable/index.html)
  - [Streamlit](https://streamlit.io/)
  - [Pillow](https://python-pillow.org/)
  
 ---

## 📚 Dataset

### 📂 Structure
The dataset follows a folder-based structure, organized by celebrity name:


- `train/` → used for model training  
- `val/` → used for validation/testing  

---

### 🖼️ Requirements
- Image formats: **JPEG** / **PNG**  
- Recommended: **50+ images per celebrity** for better training performance  
- Images should be clear and contain the celebrity’s face prominently  

---

⚠️ **Note**: The dataset is **not included** in this repository due to privacy and licensing concerns.

---

## 🔮 Future Improvements

- **Face Detection**: Integrate preprocessing with a face detection model (e.g., **MTCNN**) to crop and align faces before classification.  
- **Dataset Expansion**: Add more celebrities and increase image diversity to improve generalization.  
- **Deployment**: Host the app on **Streamlit Community Cloud** for public access and easy sharing.  

---

## Watch a demo

[🎥 Watch Demo](demo.mp4)


