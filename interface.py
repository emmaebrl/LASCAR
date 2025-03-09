import streamlit as st
import matplotlib.pyplot as plt
from tifffile import TiffFile
import os
import numpy as np

st.title("LASCAR")
st.write("Welcome to LASCAR (Land Analysis & Segmentation for Cover And Recognition)")

idx = st.number_input("Enter the index of the image you want to analyse:", min_value=0, value=0, step=1)

image_path = os.path.join('../dataset/train/images', f'{idx + 1}.tif') 

try:
    with TiffFile(image_path) as tif:
        image_arr = tif.asarray()
    
    image_display = np.clip(image_arr, 0, 2200)
    image_display = (image_display - 0) / (2200 - 0)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image_display)
    ax.axis('off') 
    
    st.pyplot(fig)

except Exception as e:
    st.error(f"Impossible de charger l'image {idx}: {e}")