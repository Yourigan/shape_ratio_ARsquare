# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 16:09:33 2024

Pour calculer le ShapeFactor d'une image qui comporte plusieurs cellules'

@author: abelr
"""

from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from SR_ARsquare_V2 import find_enclosing_quadrilateral_from_contour
from SR_ARsquare_V2 import particle_average_size
from SR_ARsquare_V2 import find_close_points
from SR_ARsquare_V2 import divide_points_by_quadrilateral_edges
from SR_ARsquare_V2 import get_first_and_last_points
from SR_ARsquare_V2 import visualize_full_info_red_points
from SR_ARsquare_V2 import CalculateA
from SR_ARsquare_V2 import CalculateB
from SR_ARsquare_V2 import CalculateShapeFactor
from infos_fichiers import demander_chemin_dossier
from infos_fichiers import demander_informations
import csv



def CalculateShapeFactorFromBinaryAndContour(largest_contour, binary_img_otsu, percent):
    box_points, largest_contour, binary_img_otsu = find_enclosing_quadrilateral_from_contour(largest_contour, binary_img_otsu)
    #Calculate the min_len
    min_len = particle_average_size(largest_contour, percent)
    #get the close_points
    close_points = find_close_points(largest_contour, box_points, min_len)
    #get the subsets
    subsets = divide_points_by_quadrilateral_edges(close_points, box_points)
    # Get the first and last points for each subset
    first_and_last_points = get_first_and_last_points(subsets)
    fig = visualize_full_info_red_points(binary_img_otsu.shape, first_and_last_points, largest_contour, box_points)
    # get a
    a, average_1_3, average_2_4, A = CalculateA(first_and_last_points)
    #get b
    b, Baverage_1_3, Baverage_2_4, B = CalculateB(box_points)
    #get shape factor
    ShapeFactor, a, b = CalculateShapeFactor(a,b)
    
    return ShapeFactor, a, b, fig


def CreateContourMultiple(image_path):
    image = Image.open(image_path)
    # Convert the image to grayscale
    gray_image = np.array(image.convert("L"))

    # Apply Otsu's thresholding method
    _, binary_image_otsu = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours with Otsu's thresholded image
    contours_otsu, _ = cv2.findContours(binary_image_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    particles_data_otsu = {}
    
    # Iterate over the contours and store contour and binary mask for each particle
    for i, contour in enumerate(contours_otsu):
        # Create a blank image to store binary mask for this particle
        mask = np.zeros_like(binary_image_otsu)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        
        # Store contour and binary mask
        particles_data_otsu[f"Particle_{i+1}"] = {
            "contour": contour,
            "binary_mask": mask
        }
    
    # Show the binary masks for each particle using Otsu's thresholding
    fig, axs = plt.subplots(1, len(particles_data_otsu), figsize=(120, 60))
    for i, (particle_name, data) in enumerate(particles_data_otsu.items()):
        axs[i].imshow(data["binary_mask"], cmap='gray')
        axs[i].set_title(particle_name)
        axs[i].axis('off')
    
    plt.show()
    
    return particles_data_otsu


def GetShapeFactorAllImage(particles_data_otsu, percent, alliage, temperature, duree_tth, FileIdInDir):
    ShapeFactorAllImage = np.array([])
    # Créer le nom du fichier CSV en fonction des informations
    nom_fichier = f"{alliage}_{temperature}_{duree_tth}.csv"
    # Remplacer les espaces ou caractères spéciaux dans le nom du fichier (facultatif, pour éviter des problèmes)
    nom_fichier = nom_fichier.replace(" ", "_").replace("/", "-")
    # Écrire les informations dans le fichier CSV
    with open(nom_fichier, mode='a', newline='', encoding='utf-8') as fichier_csv:
        writer = csv.writer(fichier_csv)

        # Écrire l'en-tête
        writer.writerow(["Alliage", "Temperature", "Duree du TTH", "ParticleShapeFactor", "A particle", "B particle", "FileIdInDir"])

        for particle_name, data in particles_data_otsu.items():
            # particle_name contiendra les clés, par exemple "Particle_1", "Particle_2", etc.
            # data contiendra le contour et le masque binaire pour chaque particule
        
            contour = data["contour"]  # Accéder au contour
            binary_mask = data["binary_mask"]  # Accéder au masque binaire
        
            box_points, largest_contour, binary_img_otsu = find_enclosing_quadrilateral_from_contour(contour, binary_mask)
            ShapeFactor, a, b, fig = CalculateShapeFactorFromBinaryAndContour(largest_contour, binary_img_otsu, percent)
            print(ShapeFactor)
        
            # Ajouter les données
            writer.writerow([alliage, temperature, duree_tth, ShapeFactor, a, b, FileIdInDir])
            
            ShapeFactorAllImage = np.append(ShapeFactorAllImage,ShapeFactor)
            NbParticles = len(ShapeFactorAllImage)

    ShapeFactorAllImage = np.append(ShapeFactorAllImage,np.mean(ShapeFactorAllImage))
    print(ShapeFactorAllImage)
    return ShapeFactorAllImage, NbParticles


def GetShapeAndStoreInCSV(percent):
    
    dossier = demander_chemin_dossier()
    alliage, temperature, duree_tth = demander_informations()
    fichiers = [fichier for fichier in dossier.iterdir() if fichier.is_file()]
    S = []
    i = 0
    # Créer le nom du fichier CSV en fonction des informations
    nom_fichier = f"{alliage}_{temperature}_{duree_tth}_Mean.csv"
    # Remplacer les espaces ou caractères spéciaux dans le nom du fichier (facultatif, pour éviter des problèmes)
    nom_fichier = nom_fichier.replace(" ", "_").replace("/", "-")
    # Écrire les informations dans le fichier CSV
    with open(nom_fichier, mode='a', newline='', encoding='utf-8') as fichier_csv:
        writer = csv.writer(fichier_csv)
        # Écrire l'en-tête
        writer.writerow(["Alliage", "Temperature", "Duree du TTH", "MeanShapeFactor", "NbParticles"])
    
        for fichier in fichiers:
            percent = percent
        
            # S = S + [GetShapeFactorAllImage(CreateContourMultiple(fichier), percent, alliage, temperature, duree_tth, i)]
            ShapeFactorAllImage, NbParticles = GetShapeFactorAllImage(CreateContourMultiple(fichier), percent, alliage, temperature, duree_tth, i)

            # Ajouter les données au fichier csv
            writer.writerow([alliage, temperature, duree_tth, ShapeFactorAllImage[-1] ,NbParticles])
        
            i = i+1
            
    return S, alliage, temperature, duree_tth


GetShapeAndStoreInCSV(0.001)



# dossier = demander_chemin_dossier()
# dossier
# # # Chemin du dossier
# # dossier = Path(r"F:\Post-doc CERVELLON Alice - RAPETTI Abel (21-22) (J. CORMIER)\13- papiers\03-misfit\python\images\test2")
# # dossier

# # Récupérer les chemins d'accès des fichiers dans une liste
# fichiers = [fichier for fichier in dossier.iterdir() if fichier.is_file()]





# fichiers[1]


# # Load the image
# image_path = r"F:\Post-doc CERVELLON Alice - RAPETTI Abel (21-22) (J. CORMIER)\13- papiers\03-misfit\python\images\multiples\test_grandeur_nature2.png"
# image = Image.open(image_path)

# particles_data_otsu = CreateContourMultiple(fichiers[1])

# i = 0


# alliage = "alliage"
# temperature = "temperature"
# duree_tth = "duree tth"

# ShapeFactorAllImage = GetShapeFactorAllImage(CreateContourMultiple(image_path), 0.0008, alliage, temperature, duree_tth, i)[0][-1]
# ShapeFactorAllImage
# i = 0
# # Afficher la liste des chemins de fichiers
# print(fichiers)
# S = []
# for fichier in fichiers:
#     fichier
#     S = S + [GetShapeFactorAllImage(CreateContourMultiple(fichier), 0.0008, alliage, temperature, duree_tth, i)]
#     i = i+1
# i    
# print(S)


# GetShapeFactorAllImage(particles_data_otsu, percent=0.0008, alliage=alliage, temperature=temperature, duree_tth=duree_tth)


