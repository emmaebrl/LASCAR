# 🛰️ LASCAR – Land cover Analysis with Satellite Classification and Automated Recognition

**LASCAR** est un projet de classification de couverture terrestre à partir d'images satellites Sentinel-2, dans le cadre du challenge Preligens. Il vise à prédire la distribution des classes de terrain pour une image donnée, à l’aide de modèles d’apprentissage profond.

## 🌍 Objectif du projet
Développer un système de prédiction robuste de la distribution des types de couverture terrestre dans des images satellitaires, à partir d’une segmentation sémantique ou directement par régression.

---

## 🧠 Modèles développés

### 1. **SimpleCNN (Régression des proportions)**
- 📌 Objectif : Prédire directement le vecteur de distribution des classes de couverture terrestre pour une image donnée.
- 🛠️ Méthode :
  - Architecture CNN compacte avec 3 blocs convolutionnels suivis d'un **Average Pooling global**.
  - Couche fully-connected finale produisant un vecteur log-softmax.
  - Fonction de perte : **KL-Divergence (batchmean)**.


<p align="center">
  <img src="graphs/simplecnn.svg" width="400"/>
</p>

### 2. **SimpleSegNet (Segmentation des classes)**
- 📌 Objectif : Prédire un **masque de segmentation multi-classes** pour chaque image, et en dériver la proportion de chaque classe.
- 🛠️ Méthode :
  - Architecture de type **encoder-decoder** :
    - Encoder : empilement de convolutions avec **BatchNorm**, ReLU, et **MaxPool** (réduction de résolution).
    - Decoder : empilement de **ConvTranspose2D** (upsampling) et convolutions classiques.
  - Fonction de perte : **CrossEntropyLoss**.

<p align="center">
  <img src="graphs/segnet.svg" width="400"/>
</p>

### 3. **U-Net ResNet34 (Segmentation avec pré-entraînement)**
- 📌 Objectif : Prédire les **masques de segmentation multi-classes**, puis en extraire les **proportions de classes** pour chaque image.
- 🛠️ Méthode :
  - Utilisation du package `segmentation_models_pytorch` (SMP).
  - Architecture **U-Net** avec encoder **ResNet34 pré-entraîné sur ImageNet**.
  - Entrées : images satellites en 4 canaux (R, G, B, NIR).
  - Fonction de perte : `CrossEntropyLoss`.

## 🚀 Lancer le projet

### 1. Installation

```bash
git clone https://github.com/emmaebrl/LASCAR.git
python -m venv venv
source venv/bin/activate  # ou .\venv\Scripts\activate sous Windows
pip install -r requirements.txt
```
### 2. Entrainer les modèles
Pour les modèles 1 et 2, utiliser le notebook ``Models.ipynb``.
Pour fine-tuner un modèle U-net, utiliser le script pytjon ``Finetuning_Unet.py``.

### 3. Tester les modèles dans notre application Streamlit
Nous proposons une interface interactive pour tester les modèles. Pour la découvrir, lancer depuis la source du projet:
```bash
streamlit run interface/interface.py 
```

### Fonctionnalités :
- Trois modèles disponibles :
  - SimpleCNN (proportions)
  - SimpleSegNet (segmentation)
  - U-Net ResNet34 (segmentation fine)
- Exemple de prédiction pour le jeu de test
- Comparaison prédictions / vraie valeurs sur le jeu de validation

### ✍️ Contributeurs
- **[Emma Eberle](https://github.com/emmaebrl)**
- **[Alexis Christien](https://github.com/AlexChrst)**