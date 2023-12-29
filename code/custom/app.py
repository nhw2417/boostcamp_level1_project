import os
import streamlit as st
from PIL import Image
import torchvision.transforms as transforms
import torch

# Load the ResNet50 model
from model import ResNet50
num_classes = 18
model = ResNet50(num_classes)

# Load weights from the specified path
weights_path = "C:/Users/82104/Desktop/boostcamp/boostcamp_level1_project/code/custom/model/resnet50/best.pth"
model.load_state_dict(torch.load(weights_path))

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

class_descriptions = {
    0: {"Mask": "Wear", "Gender": "Male", "Age": "< 30"},
    1: {"Mask": "Wear", "Gender": "Male", "Age": ">= 30 and < 60"},
    2: {"Mask": "Wear", "Gender": "Male", "Age": ">= 60"},
    3: {"Mask": "Wear", "Gender": "Female", "Age": "< 30"},
    4: {"Mask": "Wear", "Gender": "Female", "Age": ">= 30 and < 60"},
    5: {"Mask": "Wear", "Gender": "Female", "Age": ">= 60"},
    6: {"Mask": "Incorrect", "Gender": "Male", "Age": "< 30"},
    7: {"Mask": "Incorrect", "Gender": "Male", "Age": ">= 30 and < 60"},
    8: {"Mask": "Incorrect", "Gender": "Male", "Age": ">= 60"},
    9: {"Mask": "Incorrect", "Gender": "Female", "Age": "< 30"},
    10: {"Mask": "Incorrect", "Gender": "Female", "Age": ">= 30 and < 60"},
    11: {"Mask": "Incorrect", "Gender": "Female", "Age": ">= 60"},
    12: {"Mask": "Not Wear", "Gender": "Male", "Age": "< 30"},
    13: {"Mask": "Not Wear", "Gender": "Male", "Age": ">= 30 and < 60"},
    14: {"Mask": "Not Wear", "Gender": "Male", "Age": ">= 60"},
    15: {"Mask": "Not Wear", "Gender": "Female", "Age": "< 30"},
    16: {"Mask": "Not Wear", "Gender": "Female", "Age": ">= 30 and < 60"},
    17: {"Mask": "Not Wear", "Gender": "Female", "Age": ">= 60"}
}

# Streamlit code for UI
st.title("Mask Classification App")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Classifying...")
    label = predict(image)
    st.write(f'Predicted Class: {class_descriptions[label]}')