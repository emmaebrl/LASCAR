# LASCAR
Land Analysis &amp; Segmentation for Cover And Recognition

EDA : 
- Vérifier la corrélation entre les couleurs des pixels et les classes qui leurs sont associées

Modélisation :
- Approche 'simple' : Donner les images en entrée et chercher à prédire les proportions directement
- Approche conventionnelle : Donner les images en entrées, classifier chaque pixel puis calculer les proportions

Les modèles
- Faire un modèle très simple, "from scratch", en codant nous mêmes les couches pour nous servir de benchmark
- Fine-Tuner un modèle pré-existant
- Prendre une architecture existante (U-Net ou autre) et travailler sur les hyper-paramètres
