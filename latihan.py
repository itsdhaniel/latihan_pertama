import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import timm
import torch
from torchvision import transforms
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz

# Load the categories for ImageNet
categories = np.array([line.strip() for line in open("imagenet_classes.txt")])

# Preprocessing function for ResNet50 v1.5
preprocess_func = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@st.cache_resource
def load_model():
    model = timm.create_model('resnet50', pretrained=True)
    model.eval()
    return model

def make_prediction(model, processed_img):
    with torch.no_grad():
        probs = model(processed_img.unsqueeze(0))
        probs = probs.softmax(1)
        probs = probs[0].detach().numpy()

    prob, idxs = probs[probs.argsort()[-5:][::-1]], probs.argsort()[-5:][::-1]
    return prob, idxs

def interpret_prediction(model, processed_img, target):
    interpretation_algo = IntegratedGradients(model)
    feature_imp = interpretation_algo.attribute(processed_img.unsqueeze(0), target=int(target))
    feature_imp = feature_imp[0].numpy()
    feature_imp = feature_imp.transpose(1, 2, 0)

    return feature_imp 

# Dashboard GUI 
st.title("ResNet 50 Image :tea: :coffee:")
upload = st.file_uploader(label="Upload Image:", type=["png", "jpg", "jpeg"])

if upload: 
    img = Image.open(upload)

    model = load_model()
    preprocessed_img = preprocess_func(img)
    probs, idxs = make_prediction(model, preprocessed_img)
    feature_imp = interpret_prediction(model, preprocessed_img, idxs[0])
  
    main_fig = plt.figure(figsize=(12, 3))

    ax = main_fig.add_subplot(111)
    plt.barh(y=categories[idxs][::-1], width=probs[::-1], color=["dodgerblue"] * 4 + ["tomato"])
    plt.title("Top 5 Probability", loc="center", fontsize=15)
    st.pyplot(main_fig, use_container_width=True)

    interp_fig, ax = viz.visualize_image_attr(feature_imp, show_colorbar=True, fig_size=(6, 6))

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        main_fig = plt.figure(figsize=(6, 6))
        ax = main_fig.add_subplot(111)
        plt.imshow(img)
        plt.xticks([], [])
        plt.yticks([], [])
        st.pyplot(main_fig, use_container_width=True)

    with col2:
        st.pyplot(interp_fig, use_container_width=True)
