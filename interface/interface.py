import streamlit as st
import matplotlib.pyplot as plt
from tifffile import TiffFile
import os
import numpy as np
import pandas as pd
import torchvision.transforms as T
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from models import SimpleCNN
import plotly.express as px






st.title("LASCAR")
st.write("Welcome to LASCAR (Land Analysis & Segmentation for Cover And Recognition)")

st.markdown("""
    <style>
    /* Fond global + texte blanc */
    .stApp {
        background-color: #0E1117;
        color: white;
    }

    h1, h2, h3, h4, h5, h6, p, label {
        color: white !important;
    }

    /* Forcer couleur du texte dans les radios */
    .stRadio > div {
        color: white !important;
    }

    /* Options radio */
    div[role="radiogroup"] > label > div[data-testid="stMarkdownContainer"] > p {
        color: white !important;
    }

    /* Widgets : input / select */
    .stNumberInput input,
    .stSelectbox div,
    .stTextInput input {
        background-color: #262730;
        color: white;
    }

    /* Titres de sections */
    .css-1y4p8pa, .css-qrbaxs {
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Forcer le style du bouton
st.markdown("""
    <style>
    div.stButton > button {
        background-color: #333333;
        color: white;
        border: 1px solid white;
        padding: 0.5em 1em;
        border-radius: 8px;
    }

    div.stButton > button:hover {
        background-color: #555555;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)


class_names = ['cultivated', 'herbaceous', 'broadleaf', 'coniferous', 'artificial', 'water', 'natural', 'snow', 'no_data', 'clouds']

test_img_dir = "../../folder_drive_emma/data/test/images"
test_ids = [f.replace(".tif", "") for f in os.listdir(test_img_dir) if f.endswith(".tif")]


st.subheader("Choisissez un modèle à utiliser :")

model_choice = st.radio(
    "Quel type de modèle souhaitez-vous utiliser ?",
    [
        " Modèle à proportion directe",
        " Modèle simple prédiction pixel",
        " Modèle existant )"
    ],
    index=1
)

st.write(f" Modèle sélectionné : {model_choice}")





class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return torch.log_softmax(self.fc(x), dim=1)
    

@st.cache_resource
def load_proportion_model(path="../model_proportion.pth"):
    model = SimpleCNN(num_classes=10)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


selected_id = st.selectbox("Sélectionnez une image de test à prédire :", test_ids)
image_path = os.path.join(test_img_dir, f"{selected_id}.tif")



try:
    with TiffFile(image_path) as tif:
        image_arr = tif.asarray()
    
    image_display = np.clip(image_arr, 0, 2200)
    image_display = (image_display - 0) / (2200 - 0)
    
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image_display)
    ax.axis('off') 
    
    st.pyplot(fig)

    if st.button("Prédire sur cette image"):
        if model_choice.strip() == "Modèle à proportion directe":
            model = load_proportion_model()

            transform = A.Compose([
                A.Resize(256, 256),
                A.Normalize(),
                ToTensorV2()
            ])

            if image_arr.shape[2] < 4:
                st.warning("L’image ne contient pas 4 canaux.")
            else:
                image_input = image_arr[:, :, :4].astype(np.float32)
                transformed = transform(image=image_input)
                input_tensor = transformed["image"].unsqueeze(0)

                with torch.no_grad():
                    output = model(input_tensor)
                    proportions = torch.exp(output.squeeze()).numpy()

                    # Affichage Plotly amélioré
                    st.subheader("Proportions prédites (graphiques)")

                    # Format proportions à 3 décimales max
                    proportions_rounded = [round(p, 3) for p in proportions]

                    df_plot = pd.DataFrame({
                        "Classe": [name.capitalize() for name in class_names],
                        "Proportion": proportions_rounded
                    })

                    fig = px.bar(
                        df_plot,
                        x="Classe",
                        y="Proportion",
                        text="Proportion",
                        range_y=[0, 1],
                        color="Classe",
                        color_discrete_sequence=px.colors.qualitative.Set3,
                        title="Distribution des classes dans l'image prédite"
                    )

                    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")

                    fig.update_layout(
                        plot_bgcolor="#0E1117",
                        paper_bgcolor="#0E1117",
                        font=dict(color="white", size=16),
                        title={
                            "text": "Distribution des classes dans l'image prédite",
                            "x": 0.5,
                            "xanchor": "center",
                            "font": dict(color="white", size=24)
                        },
                        xaxis_tickangle=-45,
                        margin=dict(l=20, r=20, t=80, b=60),
                        height=400,
                        width = 200, # ⬅️ Plus grand en hauteur
                        showlegend=False,
                        uniformtext_minsize=8,
                        uniformtext_mode="hide"
                    )


                    st.plotly_chart(fig, use_container_width=True)



except Exception as e:
    st.error(f"Impossible de charger l'image {selected_id}: {e}")




