# 🛰️ LASCAR – Land cover Analysis with Satellite Classification and Automated Recognition

**LASCAR** is a land cover classification project using Sentinel-2 satellite images, developed as part of the Preligens challenge. Its goal is to predict the distribution of land cover classes for a given image using deep learning models.

## 🌍 Project Objective
To develop a robust prediction system for estimating the distribution of land cover types in satellite imagery, either through semantic segmentation or directly via regression.

---

## 🧠 Developed Models

### 1. **SimpleCNN (Class Proportion Regression)**
- 📌 Goal: Directly predict the land cover class distribution vector for a given image.
- 🛠️ Method:
  - Compact CNN architecture with 3 convolutional blocks followed by **Global Average Pooling**.
  - Final fully-connected layer outputs a log-softmax vector.
  - Loss function: **KL-Divergence (batchmean)**.

<p align="center">
  <img src="graphs/simplecnn.svg" width="500"/>
</p>

### 2. **SimpleSegNet (Class Segmentation)**
- 📌 Goal: Predict a **multi-class segmentation mask** for each image and derive class proportions from it.
- 🛠️ Method:
  - **Encoder-decoder** style architecture:
    - Encoder: stacked convolutions with **BatchNorm**, ReLU, and **MaxPool** for downsampling.
    - Decoder: stacked **ConvTranspose2D** (upsampling) and standard convolutions.
  - Loss function: **CrossEntropyLoss**.

<p align="center">
  <img src="graphs/segnet.svg" width="500"/>
</p>

### 3. **U-Net ResNet34 (Pretrained Segmentation)**
- 📌 Goal: Predict **multi-class segmentation masks**, then extract **class proportions** for each image.
- 🛠️ Method:
  - Uses the `segmentation_models_pytorch` (SMP) package.
  - **U-Net** architecture with a **ResNet34 encoder pretrained on ImageNet**.
  - Inputs: 4-channel satellite images (R, G, B, NIR).
  - Loss function: `CrossEntropyLoss`.

## 🚀 Running the Project

### 1. Installation

```bash
git clone https://github.com/emmaebrl/LASCAR.git
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
pip install -r requirements.txt
```
### 2. Training the Models
For models 1 and 2, use the `Models.ipynb` notebook.
To fine-tune the U-Net model, use the ``Finetuning_Unet.py`` script.

### 3. Test the Models in our Streamlit App
We provide an interactive interface to test the models. To launch it from the project root:
```bash
streamlit run interface/interface.py 
```

### Features :
- Trois modèles disponibles :
    - SimpleCNN (proportion regression)

    - SimpleSegNet (segmentation)

    - U-Net ResNet34 (fine segmentation)

-Example predictions on the test set

-Compare predictions vs. ground truth on the validation set

### ✍️ Contributors
- **[Emma Eberle](https://github.com/emmaebrl)**
- **[Alexis Christien](https://github.com/AlexChrst)**
