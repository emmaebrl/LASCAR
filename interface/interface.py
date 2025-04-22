import streamlit as st
import matplotlib.pyplot as plt
from tifffile import TiffFile
import os
import numpy as np
import pandas as pd
import torch
import json

import albumentations as A
from albumentations.pytorch import ToTensorV2
from models import SimpleCNN, SimpleSegNet
import plotly.express as px
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import segmentation_models_pytorch as smp
import streamlit.components.v1 as components

if "started" not in st.session_state:
    st.session_state.started = False
st.markdown("<h1 style='text-align: center;'>LASCAR</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; font-style: italic;'>(Land Analysis & Segmentation for Cover And Recognition)</p>",
    unsafe_allow_html=True,
)
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
    "no_data": "#ffffff",
    "clouds": "#cccccc",
    "cultivated": "#e41a1c",
    "herbaceous": "#377eb8",
    "broadleaf": "#4daf4a",
    "coniferous": "#984ea3",
    "artificial": "#ff7f00",
    "water": "#a65628",
    "natural": "#f781bf",
    "snow": "#999999",
}

CLASS_NAMES = CLASSES_COLORPALETTE.keys()

tab1, tab2, tab3 = st.tabs(
    [
        "Régression des proportions",
        "Segmentation pixel-wise",
        "Finetuning U-Net",
    ]
)


########### FONCTIONS DE CHARGEMENT DES MODÈLES ###########
@st.cache_resource
def load_proportion_model(
    path=r"models\model_proportion_final.pth",
):
    model = SimpleCNN(num_classes=10)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


@st.cache_resource
def load_segmentation_model(
    path=r"models\model2_segmentation_final.pth",
):
    model = SimpleSegNet(in_channels=4, num_classes=10)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


@st.cache_resource
def load_unet_model(path=r"models\unet_resnet34_finetuned.pth"):
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=4,
        classes=10,
    )
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


############## FONCTIONS DE CHARGEMENT DES DONNEES ###########
def load_tiff_image(image_path: str) -> np.ndarray:
    with TiffFile(image_path) as tif:
        return tif.asarray()


def get_image_ids(split: str, base_dir: str = "dataset") -> tuple[list, str]:
    img_dir = os.path.join(base_dir, split, "images")
    image_ids = [
        f.replace(".tif", "") for f in os.listdir(img_dir) if f.endswith(".tif")
    ]
    return image_ids, img_dir


def get_data(
    split: str,
    base_dir: str = "dataset",
    split_path: str = "splits/train_val_ids.json",
    proportion_path: str = "dataset/train_labels_GY1QjFw.csv",
) -> tuple[list, str]:
    if split == "validation":
        with open(split_path, "r") as f:
            split_ids_dict = json.load(f)
        split_ids_dict_val = split_ids_dict["val"]
        img_dir = os.path.join(base_dir, "train", "images")
        image_ids = [
            img_id
            for img_id in split_ids_dict_val
            if os.path.isfile(os.path.join(img_dir, f"{img_id}.tif"))
        ]
        validation_masks = [
            os.path.join(base_dir, "train", "masks", f"{img_id}.tif")
            for img_id in split_ids_dict_val
        ]
        labels_df = pd.read_csv(proportion_path, index_col=0)  # id as index
        image_ids = [int(i) for i in image_ids]
        labels_df = labels_df[labels_df.index.isin(image_ids)]
        return (
            image_ids,
            img_dir,
            validation_masks,
            labels_df,
        )
    else:
        img_dir = os.path.join(base_dir, "test", "images")
        image_ids = [
            f.replace(".tif", "") for f in os.listdir(img_dir) if f.endswith(".tif")
        ]
        return image_ids, img_dir, None, None


############# FONCTIONS DE TRAITEMENT ET D'AFFICHAGE ###########
def convert_to_rgb(img: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            normalize_band(img[:, :, 2]),
            normalize_band(img[:, :, 1]),
            normalize_band(img[:, :, 0]),
        ],
        axis=-1,
    )


def display_rgb_image(rgb: np.ndarray, title: str = "Image RGB"):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(rgb)
    ax.axis("off")
    st.markdown(f"**{title}**")
    st.pyplot(fig)


def plot_proportions(proportions: list[float], title: str) -> None:
    df_plot = pd.DataFrame(
        {
            "Classe": [name.capitalize() for name in CLASS_NAMES],
            "Proportion": [round(p, 3) for p in proportions],
        }
    )
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
        xaxis=dict(
            tickangle=-45,
            tickfont=dict(color="white"),
            title_font=dict(color="white")
        ),
        yaxis=dict(
            tickfont=dict(color="white"),
            title_font=dict(color="white")
        ),
        margin=dict(l=20, r=20, t=80, b=60),
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)


def normalize_band(band, low=2, high=98):
    p_low, p_high = np.percentile(band, (low, high))
    band = np.clip(band, p_low, p_high)
    return (band - p_low) / (p_high - p_low + 1e-8)


def plot_proportions_vs_truth(pred: list[float], true: list[float], title: str):
    class_names_list = list(CLASS_NAMES)
    df_plot = pd.DataFrame({
        "Classe": class_names_list * 2,
        "Proportion": [round(p, 3) for p in pred] + [round(t, 3) for t in true],
        "Type": ["Prédite"] * len(pred) + ["Réelle"] * len(true),
    })

    fig = px.bar(
        df_plot,
        x="Classe",
        y="Proportion",
        color="Type",
        barmode="group",
        text="Proportion",
        color_discrete_map={"Prédite": "#1f77b4", "Réelle": "#ff7f0e"},
        title=title,
    )

    fig.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside",
        marker_line_width=0
    )
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
        xaxis=dict(
            tickangle=-45,
            tickfont=dict(color="white"),
            title_font=dict(color="white")
        ),
        yaxis=dict(
            tickfont=dict(color="white"),
            title_font=dict(color="white")
        ),
        margin=dict(l=20, r=20, t=80, b=60),
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)


# ========== PAGE PRINCIPALE ==========
with tab1:
    st.header("Modèle 1: Prédiction des proportions")

    # 💡 Intégration du schéma Draw.io (SVG)
    with open("graphs/simplecnn.svg", "r", encoding="utf-8") as f:
        svg_code = f.read()
    components.html(
        f"""
    <div style="zoom: 0.8; transform: scale(0.8); transform-origin: top left;">
        {svg_code}
    </div>
    """,
        height=350,
        scrolling=False,
    )

    split = st.radio(
        "Choisissez le jeu de données :", ["test", "validation"], key="split1"
    )
    image_ids, img_dir, validation_masks, labels_df = get_data(split)
    selected_id = st.selectbox(
        "Sélectionnez une image à afficher :", image_ids, key="select_proportion"
    )

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

                if split == "validation":
                    true_props = np.array(
                        labels_df.loc[selected_id, CLASS_NAMES].tolist(),
                        dtype=np.float32,
                    )

                    if any([v is None or np.isnan(v) for v in proportions]):
                        st.error("❌ Valeurs `NaN` détectées dans les prédictions.")
                    elif any([v is None or np.isnan(v) for v in true_props]):
                        st.error("❌ Valeurs `NaN` détectées dans les valeurs réelles.")
                    else:
                        plot_proportions_vs_truth(
                            proportions, true_props, "Comparaison Prédite vs Réelle"
                        )

                else:
                    plot_proportions(
                        proportions, "Distribution des classes dans l'image prédite"
                    )

    except Exception as e:
        st.warning(f"Erreur lors du chargement de l'image {selected_id}: {e}")


with tab2:
    st.header("Modèle 2: Segmentation pixel-wise")

    split = st.radio(
        "Choisissez le jeu de données :", ["test", "validation"], key="split2"
    )
    image_ids, img_dir, validation_masks, labels_df = get_data(split)
    selected_id = st.selectbox(
        "Sélectionnez une image à afficher :", image_ids, key="select_segnet"
    )

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
                proportions = np.zeros(len(CLASS_NAMES))
                for u, c in zip(unique, counts):
                    if u < len(CLASS_NAMES):
                        proportions[u] = c / total

                if split == "validation" and selected_id in labels_df.index:
                    true_props = np.array(
                        labels_df.loc[selected_id, CLASS_NAMES].tolist(),
                        dtype=np.float32,
                    ).tolist()
                    plot_proportions_vs_truth(
                        proportions, true_props, "Comparaison Prédite vs Réelle"
                    )
                else:
                    plot_proportions(
                        proportions, "Distribution des classes dans l'image prédite"
                    )

                # Affichage du masque
                # Affichage du masque
                cmap = ListedColormap(
                    [CLASSES_COLORPALETTE[cls] for cls in CLASS_NAMES]
                )

                # Chargement du masque réel si en validation
                true_mask = None
                if split == "validation":
                    gt_mask_path = os.path.join(
                        "dataset", "train", "masks", f"{selected_id}.tif"
                    )
                    if os.path.exists(gt_mask_path):
                        with TiffFile(gt_mask_path) as tif:
                            true_mask = tif.asarray()

                # Image de comparaison verte/rouge
                comparison_mask = None
                if true_mask is not None and pred_mask.shape == true_mask.shape:
                    comparison_mask = np.zeros((*true_mask.shape, 3), dtype=np.uint8)
                    comparison_mask[(pred_mask == true_mask)] = [0, 255, 0]  # vert
                    comparison_mask[(pred_mask != true_mask)] = [255, 0, 0]  # rouge

                # Affichage dynamique : RGB / prédiction / vrai masque / comparaison
                n_cols = (
                    4
                    if comparison_mask is not None
                    else 3 if true_mask is not None else 2
                )
                fig_mask, axs = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6), dpi=120)

                axs[0].imshow(rgb)
                axs[0].set_title("Image RGB")
                axs[0].axis("off")

                axs[1].imshow(pred_mask, cmap=cmap, vmin=0, vmax=len(CLASS_NAMES) - 1)
                axs[1].set_title("Masque prédit")
                axs[1].axis("off")

                if true_mask is not None:
                    axs[2].imshow(
                        true_mask, cmap=cmap, vmin=0, vmax=len(CLASS_NAMES) - 1
                    )
                    axs[2].set_title("Masque réel")
                    axs[2].axis("off")

                if comparison_mask is not None:
                    axs[3].imshow(comparison_mask)
                    axs[3].set_title("Comparaison (vert = OK)")
                    axs[3].axis("off")

                # Légende des classes sur le masque prédit
                handles = [
                    mpatches.Patch(color=CLASSES_COLORPALETTE[cls], label=cls)
                    for cls in CLASS_NAMES
                ]

                st.pyplot(fig_mask)
                legend_fig, legend_ax = plt.subplots(figsize=(12, 1))
                legend_ax.axis("off")

                legend_handles = [
                    mpatches.Patch(color=CLASSES_COLORPALETTE[cls], label=cls)
                    for cls in CLASS_NAMES
                ]
                legend_ax.legend(
                    handles=legend_handles,
                    loc="center",
                    ncol=5,
                    fontsize=10,
                    frameon=False,
                )

                st.pyplot(legend_fig)

    except Exception as e:
        st.error(f"Erreur lors du traitement de l'image {selected_id}: {e}")


with tab3:
    st.header("Modèle 3: Segmentation avec U-Net (fine-tuning)")

    split = st.radio(
        "Choisissez le jeu de données :", ["test", "validation"], key="split3"
    )
    image_ids, img_dir, validation_masks, labels_df = get_data(split)
    selected_id = st.selectbox(
        "Sélectionnez une image à afficher :", image_ids, key="select_unet"
    )

    image_path = os.path.join(img_dir, f"{selected_id}.tif")

    try:
        img = load_tiff_image(image_path)
        rgb = convert_to_rgb(img)
        display_rgb_image(rgb, f"Image RGB : `{selected_id}`")

        if st.button("Prédire (U-Net)", key="predict_unet"):
            st.subheader("Prédiction du masque (U-Net)...")

            model = load_unet_model()
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
                proportions = np.zeros(len(CLASS_NAMES))
                for u, c in zip(unique, counts):
                    if u < len(CLASS_NAMES):
                        proportions[u] = c / total

                if split == "validation" and selected_id in labels_df.index:
                    true_props = np.array(
                        labels_df.loc[selected_id, CLASS_NAMES].tolist(),
                        dtype=np.float32,
                    ).tolist()
                    plot_proportions_vs_truth(
                        proportions, true_props, "Comparaison Prédite vs Réelle (U-Net)"
                    )
                else:
                    plot_proportions(
                        proportions, "Distribution des classes dans l'image prédite"
                    )

                # Affichage du masque
                cmap = ListedColormap(
                    [CLASSES_COLORPALETTE[cls] for cls in CLASS_NAMES]
                )

                # Chargement du masque réel si en validation
                true_mask = None
                if split == "validation":
                    gt_mask_path = os.path.join(
                        "dataset", "train", "masks", f"{selected_id}.tif"
                    )
                    if os.path.exists(gt_mask_path):
                        with TiffFile(gt_mask_path) as tif:
                            true_mask = tif.asarray()

                # Image de comparaison verte/rouge
                comparison_mask = None
                if true_mask is not None and pred_mask.shape == true_mask.shape:
                    comparison_mask = np.zeros((*true_mask.shape, 3), dtype=np.uint8)
                    comparison_mask[(pred_mask == true_mask)] = [0, 255, 0]  # vert
                    comparison_mask[(pred_mask != true_mask)] = [255, 0, 0]  # rouge

                # Affichage dynamique : RGB / prédiction / vrai masque / comparaison
                n_cols = (
                    4
                    if comparison_mask is not None
                    else 3 if true_mask is not None else 2
                )
                fig_mask, axs = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6), dpi=120)

                axs[0].imshow(rgb)
                axs[0].set_title("Image RGB")
                axs[0].axis("off")

                axs[1].imshow(pred_mask, cmap=cmap, vmin=0, vmax=len(CLASS_NAMES) - 1)
                axs[1].set_title("Masque prédit (U-Net)")
                axs[1].axis("off")

                if true_mask is not None:
                    axs[2].imshow(true_mask, cmap=cmap, vmin=0, vmax=len(CLASS_NAMES) - 1)
                    axs[2].set_title("Masque réel")
                    axs[2].axis("off")

                if comparison_mask is not None:
                    axs[3].imshow(comparison_mask)
                    axs[3].set_title("Comparaison (vert = OK)")
                    axs[3].axis("off")

                # Légende des classes
                handles = [
                    mpatches.Patch(color=CLASSES_COLORPALETTE[cls], label=cls)
                    for cls in CLASS_NAMES
                ]


                st.pyplot(fig_mask)
                legend_fig, legend_ax = plt.subplots(figsize=(12, 1))
                legend_ax.axis("off")

                legend_handles = [
                    mpatches.Patch(color=CLASSES_COLORPALETTE[cls], label=cls)
                    for cls in CLASS_NAMES
                ]
                legend_ax.legend(
                    handles=legend_handles,
                    loc="center",
                    ncol=5,
                    fontsize=10,
                    frameon=False,
                )

                st.pyplot(legend_fig)


    except Exception as e:
        st.error(f"Erreur lors du traitement de l'image {selected_id}: {e}")
