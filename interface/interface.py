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
from models import SimpleCNN, SimpleSegNet
import plotly.express as px
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

if "started" not in st.session_state:
    st.session_state.started = False

st.title("LASCAR")
st.write("Welcome to LASCAR (Land Analysis & Segmentation for Cover And Recognition)")

st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)

# Forcer le style du bouton
st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)

CLASSES_COLORPALETTE = {
    "cultivated": "#e41a1c",
    "herbaceous": "#377eb8",
    "broadleaf": "#4daf4a",
    "coniferous": "#984ea3",
    "artificial": "#ff7f00",
    "water": "#a65628",
    "natural": "#f781bf",
    "snow": "#999999",
    "no_data": "#ffffff",
    "clouds": "#cccccc",  # ← ajoute cette ligne !
}



class_names = [
    "cultivated",
    "herbaceous",
    "broadleaf",
    "coniferous",
    "artificial",
    "water",
    "natural",
    "snow",
    "no_data",
    "clouds",
]


tab1, tab2, tab3 = st.tabs([
    "🔢 Modèle à proportion directe",
    "🧠 Modèle simple prédiction pixel",
    "📦 Modèle existant",
])

def load_proportion_model(path="model_proportion.pth"):
    model = SimpleCNN(num_classes=10)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

@st.cache_resource
def load_segmentation_model(path="model2_segmentationcomplexified.pth"):
    model = SimpleSegNet(in_channels=4, num_classes=10)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def get_image_ids(split: str, base_dir: str = "../dataset") -> tuple[list, str]:
    img_dir = os.path.join(base_dir, split, "images")
    image_ids = [f.replace(".tif", "") for f in os.listdir(img_dir) if f.endswith(".tif")]
    return image_ids, img_dir

def load_tiff_image(image_path: str) -> np.ndarray:
    with TiffFile(image_path) as tif:
        return tif.asarray()

def convert_to_rgb(img: np.ndarray) -> np.ndarray:
    return np.stack([
        normalize_band(img[:, :, 2]),
        normalize_band(img[:, :, 1]),
        normalize_band(img[:, :, 0]),
    ], axis=-1)

def display_rgb_image(rgb: np.ndarray, title: str = "Image RGB"):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(rgb)
    ax.axis("off")
    st.markdown(f"**{title}**")
    st.pyplot(fig)

def plot_proportions(proportions: list[float], title: str) -> None:
    df_plot = pd.DataFrame({
        "Classe": [name.capitalize() for name in class_names],
        "Proportion": [round(p, 3) for p in proportions],
    })
    fig = px.bar(
        df_plot,
        x="Classe",
        y="Proportion",
        text="Proportion",
        range_y=[0, 1],
        color="Classe",
        color_discrete_sequence=px.colors.qualitative.Set3,
        title=title,
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
        font=dict(color="white", size=16),
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
            font=dict(color="white", size=24),
        ),
        xaxis_tickangle=-45,
        margin=dict(l=20, r=20, t=80, b=60),
        height=400,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)



def normalize_band(band, low=2, high=98):
    p_low, p_high = np.percentile(band, (low, high))
    band = np.clip(band, p_low, p_high)
    return (band - p_low) / (p_high - p_low + 1e-8)
import json

@st.cache_data
def load_split_ids(path="splits/train_val_ids.json"):
    with open(path, "r") as f:
        return json.load(f)

split_ids_dict = load_split_ids()

def get_image_ids_custom(split: str, base_dir: str = "../dataset") -> tuple[list, str]:
    if split == "validation":
        img_dir = os.path.join(base_dir, "train", "images")  # <- ici on pointe vers train/images
        val_ids = split_ids_dict.get("val", [])
        image_ids = [img_id for img_id in val_ids if os.path.isfile(os.path.join(img_dir, f"{img_id}.tif"))]
    else:
        img_dir = os.path.join(base_dir, split, "images")
        image_ids = [f.replace(".tif", "") for f in os.listdir(img_dir) if f.endswith(".tif")]
    return image_ids, img_dir

@st.cache_data
def load_validation_labels(path="../data/train_labels_GY1QjFw.csv"):
    df = pd.read_csv(path)
    df["sample_id"] = df["sample_id"].astype(str)  # Convertir en str pour correspondre aux IDs
    df.set_index("sample_id", inplace=True)

    # 🔧 Réordonner les colonnes selon l’ordre de class_names (sécurisé)
    df = df[[col for col in class_names if col in df.columns]]

    return df


labels_df = load_validation_labels()

def plot_proportions_vs_truth(predicted: list[float], true: list[float], title: str):
    st.text(f"🔍 Taille prédite : {len(predicted)}")
    st.text(f"🔍 Taille réelle : {len(true)}")
    st.text(f"Prédite : {predicted}")
    st.text(f"Réelle : {true}")

    # Pour être sûr d’avoir 10 valeurs
    predicted = (predicted + [0] * 10)[:10]
    true = (true + [0] * 10)[:10]

    df_plot = pd.DataFrame({
        "Classe": class_names * 2,
        "Proportion": predicted + true,
        "Type": ["Prédite"] * 10 + ["Réelle"] * 10,
    })

    fig = px.bar(
        df_plot,
        x="Classe",
        y="Proportion",
        color="Type",
        barmode="group",
        text="Proportion",
        color_discrete_map={"Prédite": "#1f77b4", "Réelle": "#ff7f0e"},
        title=title
    )

    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
        font=dict(color="white", size=16),
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
            font=dict(color="white", size=24),
        ),
        xaxis_tickangle=-45,
        margin=dict(l=20, r=20, t=80, b=60),
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)






with tab1:
    st.header("Modèle à proportion directe")

    split = st.radio("Choisissez le jeu de données :", ["test", "validation"], key="split1")
    image_ids, img_dir = get_image_ids_custom(split)
    selected_id = st.selectbox("Sélectionnez une image à afficher :", image_ids, key="select_proportion")

    image_path = os.path.join(img_dir, f"{selected_id}.tif")

    try:
        img = load_tiff_image(image_path)
        rgb = convert_to_rgb(img)
        display_rgb_image(rgb, f"Image RGB : `{selected_id}`")

        if st.button("Prédire", key="predict_proportion"):
            st.subheader("Prédiction des proportions...")

            model = load_proportion_model()
            transform = A.Compose([A.Resize(256, 256), A.Normalize(), ToTensorV2()])

            if img.shape[2] < 4:
                st.error("L’image ne contient pas les 4 canaux requis.")
            else:
                image_input = img[:, :, :4].astype(np.float32)
                transformed = transform(image=image_input)
                input_tensor = transformed["image"].unsqueeze(0)

                with torch.no_grad():
                    output = model(input_tensor)
                    proportions = torch.exp(output.squeeze()).numpy()

                if split == "validation" and selected_id in labels_df.index:
                    true_props = np.array(labels_df.loc[selected_id, class_names].tolist(), dtype=np.float32).tolist()

                    if any([v is None or np.isnan(v) for v in proportions]):
                        st.error("❌ Valeurs `NaN` détectées dans les prédictions.")
                    elif any([v is None or np.isnan(v) for v in true_props]):
                        st.error("❌ Valeurs `NaN` détectées dans les valeurs réelles.")
                    else:
                        plot_proportions_vs_truth(proportions, true_props, "Comparaison Prédite vs Réelle")

                else:
                    plot_proportions(proportions, "Distribution des classes dans l'image prédite")


    except Exception as e:
        st.warning(f"Erreur lors du chargement de l'image {selected_id}: {e}")


with tab2:
    st.header("Modèle simple prédiction pixel")

    split = st.radio("Choisissez le jeu de données :", ["test", "validation"], key="split2")
    image_ids, img_dir = get_image_ids_custom(split)
    selected_id = st.selectbox("Sélectionnez une image à afficher :", image_ids, key="select_segnet")

    image_path = os.path.join(img_dir, f"{selected_id}.tif")

    try:
        img = load_tiff_image(image_path)
        rgb = convert_to_rgb(img)
        display_rgb_image(rgb, f"Image RGB : `{selected_id}`")

        if st.button("Prédire (SegNet)", key="predict_segnet"):
            st.subheader("Prédiction du masque...")

            model = load_segmentation_model()
            transform = A.Compose([A.Resize(256, 256), A.Normalize(), ToTensorV2()])

            if img.shape[2] < 4:
                st.error("L’image ne contient pas les 4 canaux requis.")
            else:
                image_input = img[:, :, :4].astype(np.float32)
                transformed = transform(image=image_input)
                input_tensor = transformed["image"].unsqueeze(0)

                with torch.no_grad():
                    logits = model(input_tensor)
                    pred_mask = torch.argmax(logits.squeeze(0), dim=0).cpu().numpy()

                # Proportions
                unique, counts = np.unique(pred_mask, return_counts=True)
                total = pred_mask.size
                proportions = np.zeros(len(class_names))
                for u, c in zip(unique, counts):
                    if u < len(class_names):
                        proportions[u] = c / total

                if split == "validation" and selected_id in labels_df.index:
                    true_props = np.array(labels_df.loc[selected_id, class_names].tolist(), dtype=np.float32).tolist()
                    plot_proportions_vs_truth(proportions, true_props, "Comparaison Prédite vs Réelle")
                else:
                    plot_proportions(proportions, "Distribution des classes dans l'image prédite")

                # Affichage du masque
                cmap = ListedColormap([CLASSES_COLORPALETTE[cls] for cls in class_names])
                fig_mask, axs = plt.subplots(1, 2, figsize=(12, 6))
                axs[0].imshow(rgb)
                axs[0].set_title("Image RGB")
                axs[0].axis("off")
                axs[1].imshow(pred_mask, cmap=cmap, vmin=0, vmax=len(class_names) - 1)
                axs[1].set_title("Masque prédit")
                axs[1].axis("off")

                handles = [
                    mpatches.Patch(color=CLASSES_COLORPALETTE[cls], label=cls)
                    for cls in class_names
                ]
                axs[1].legend(
                    handles=handles,
                    bbox_to_anchor=(1.05, 1),
                    loc="upper left",
                    fontsize=9,
                    title="Classes"
                )

                st.pyplot(fig_mask)

    except Exception as e:
        st.error(f"Erreur lors du traitement de l'image {selected_id}: {e}")


with tab3:
    st.header("Modèle existant (?)")
    st.info("Fonctionnalité à définir selon le modèle que vous voulez intégrer.")