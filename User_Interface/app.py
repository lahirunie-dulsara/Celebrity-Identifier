import torch
from torchvision import transforms, models
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import streamlit as st
import os

# Streamlit page configuration
st.set_page_config(page_title="Celebrity Recognition", page_icon="⭐", layout="centered")

# Load the model
num_classes = 17
model = resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, num_classes)
model.load_state_dict(torch.load('celebrity_model.pth'))
model.eval()

# Celebrity names
celebrities = sorted(os.listdir('Celebrity Faces Dataset/train'))

# Preprocess
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Streamlit UI
st.title("Who's the Celebrity ?")
st.write("Upload a photo to identify .")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Predict
    try:
        img = Image.open(uploaded_file).convert('RGB')
        img = transform(img).unsqueeze(0)
        with torch.no_grad():
            outputs = model(img)
            _, predicted = torch.max(outputs, 1)
        st.success(f"Predicted: {celebrities[predicted.item()]}")
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")

# Footer
st.markdown("---")
st.write("Built with Streamlit and PyTorch. Trained on a dataset of 17 celebrities.")