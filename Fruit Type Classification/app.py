import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Load the saved model
@st.cache_resource
def load_model():
    checkpoint = torch.load('fruit_classification_regnety.pth', map_location=torch.device('cpu'))
    model = checkpoint['model_architecture']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

# Define transformation for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
     transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
])

# Load class names (ensure they match your training dataset's classes)
class_names = ['Pointed gourd', 'Red Jamaican Cherry', 'anjir', 'apple', 'apricot', 'avocado', 'banana', 'blue berrie', 'chiku', 'cluster_fig', 'custed apple', 'dragonfruit', 'grape', 'guava', 'indian_strawberry', 'jackfruit', 'kiwi', 'lemon', 'mango', 'mangosteen', 'mulberry', 'orange', 'papaya', 'pear', 'peas', 'pineapple', 'pomegranate', 'raspberry', 'waterapple', 'watermelon']
# Prediction function
def predict(image, model):
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return class_names[predicted.item()]

# Streamlit app
st.title("Fruit Classification Application")
st.sidebar.title("Options")

# Load the model
model = load_model()
st.sidebar.success("Model loaded successfully!")

# Option 1: Upload an image
st.header("Upload an Image")
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.write("Classifying...")
    label = predict(image, model)
    st.write(f"Prediction: **{label}**")

# Option 2: Capture image using webcam
st.header("Capture Image Using Webcam")
st.write("Use your webcam to capture an image of the fruit.")
img_data = st.camera_input("Take a picture")  # Streamlit's camera input widget

if img_data is not None:
    # Load the image taken from the webcam
    image = Image.open(img_data).convert('RGB')
    st.image(image, caption="Captured Image", use_container_width=True)
    st.write("Classifying...")
    label = predict(image, model)
    st.write(f"Prediction: **{label}**")
