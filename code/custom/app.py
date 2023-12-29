import streamlit as st
from PIL import Image
import torchvision.transforms as transforms
import torch

# Load the ResNet50 model
from model import ResNet50
num_classes = 18
model = ResNet50(num_classes)
model.eval()  # Set the model to evaluation mode

def predict(image):
    # Transform the image to the format expected by the model
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return predicted.item()

# Streamlit code for UI
st.title("Image Classification App")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Classifying...")
    label = predict(image)
    st.write(f'Predicted Class: {label}')