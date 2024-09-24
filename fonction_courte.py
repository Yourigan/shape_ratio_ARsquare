# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 13:52:27 2024

@author: abelr
"""

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import math
# from PIL import Image
from pathlib import Path
from SR_ARsquare_V2 import CalculateShapeFactorFromImage
import matplotlib.pyplot as plt

# Chemin du dossier
dossier = Path(r"F:\Post-doc CERVELLON Alice - RAPETTI Abel (21-22) (J. CORMIER)\13- papiers\03-misfit\python\images\test2")
dossier

# Récupérer les chemins d'accès des fichiers dans une liste
fichiers = [fichier for fichier in dossier.iterdir() if fichier.is_file()]

# Afficher la liste des chemins de fichiers
print(fichiers)

S = []
for fichier in fichiers:
    percent = 0.0001
    S = S + [CalculateShapeFactorFromImage(fichier, percent)]
    
print(S)

image = S[0][3]
# Tracer l'image en niveaux de gris
plt.imshow(image, cmap='gray')
plt.show()


# Extraire le premier élément de chaque tuple
premiers_elements = [t[0] for t in S]

# Générer un plot
plt.plot(premiers_elements, marker='o')

# Ajouter un titre et des labels
plt.title('Facteur de forme pour différentes orientations')
plt.xlabel('Index')
plt.ylabel('Valeur')
plt.ylim(0,1)

# Afficher le plot
plt.show()


image_path = r"F:\Post-doc CERVELLON Alice - RAPETTI Abel (21-22) (J. CORMIER)\13- papiers\03-misfit\python\images\scale\imageRef_downscale.png"
percent = 0.0003
CalculateShapeFactorFromImage(image_path, percent)


# Boucle sur la liste de tuples pour afficher chaque figure dans un sous-graphe
fig, axes = plt.subplots(1, len(S),figsize=(200, 200))
for i in range(len(S)):
    V = S[i][3]
    axes[i].imshow(V)
# Optionnel : Désactiver les axes pour un affichage plus propre
for ax in axes.flat:
    ax.axis('off')

# Ajuster l'espacement entre les subplots
plt.tight_layout()

# Afficher la figure
plt.show()

# Ajuster l'espacement entre les sous-graphiques
plt.tight_layout()

# Afficher la figure
plt.show()
 