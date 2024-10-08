# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 16:09:33 2024

Pour calculer le ShapeFactor d'une image qui comporte plusieurs cellules'

@author: abelr
"""

import sys
import os
# from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from SR_ARsquare_V2 import find_enclosing_quadrilateral_from_contour
from SR_ARsquare_V2 import define_min_length
from SR_ARsquare_V2 import find_close_points
from SR_ARsquare_V2 import divide_points_by_quadrilateral_edges
from SR_ARsquare_V2 import get_first_and_last_points
from SR_ARsquare_V2 import visualize_full_info_red_points
from SR_ARsquare_V2 import CalculateA
from SR_ARsquare_V2 import CalculateB
from SR_ARsquare_V2 import remove_border_objects
# from SR_ARsquare_V2 import get_contour
from infos_fichiers import demander_chemin_dossier
from infos_fichiers import demander_informations

import csv

# Ajouter le chemin du fichier à sys.path
sys.path.append(r"F:\Post-doc CERVELLON Alice - RAPETTI Abel (21-22) (J. CORMIER)\13- papiers\03-misfit\python\shape_ratio_ARsquare")

# from pathlib import Path


def CalculateShapeFactorFromBinaryAndContour(largest_contour, binary_img_otsu, percent):
    """
    Calculates the shape factor based on a given contour and binary image, 
    following the Van Sluytman Shape Factor (2012). This function processes 
    the provided contour and binary image to calculate parameters `a` and 
    `b`, and determine the shape factor.

    Args:
        largest_contour (numpy.ndarray): The contour of the particle or object 
            to be analyzed, typically obtained from contour detection methods.
        binary_img_otsu (numpy.ndarray): The binary image of the object used 
            for visualizing the contour and quadrilateral.
        percent (float): The percentage used to define the minimum length for 
            finding close points on the contour.

    Returns:
        tuple: A tuple containing:
            - ShapeFactor (float): The calculated shape factor, which is the 
              ratio `a/b` based on the bounding quadrilateral of the particle.
            - a (float): The value of the A parameter, calculated from the 
              distances between the first and last points in each subset.
            - b (float): The value of the B parameter, calculated from the 
              distances between consecutive points on the quadrilateral.
            - fig (numpy.ndarray): The visual representation of the contour, 
              quadrilateral, and first/last points.

    Description:
        This function processes the provided largest contour and binary image 
        to calculate the shape factor. It first finds the enclosing 
        quadrilateral around the contour, defines the minimum length to find 
        close points on the contour, and divides those points into subsets 
        based on proximity to the edges of the quadrilateral. It then 
        calculates the A and B parameters from the distances between points 
        and computes the shape factor as `a / b`. A visual representation of 
        the contour, quadrilateral, and first/last points is also generated.

    Steps:
        - The function uses the largest contour and binary image to find 
          the enclosing quadrilateral.
        - Close points on the contour are identified based on a distance 
          threshold.
        - These points are divided into subsets corresponding to the edges 
          of the quadrilateral.
        - The first and last points from each subset are used to calculate 
          the A parameter.
        - The B parameter is calculated from the distances between 
          consecutive points on the quadrilateral.
        - The shape factor is computed as the ratio `a / b`.
        - A visual output is created, displaying the relevant features.

    Notes:
        - The function assumes that the contour and binary image have been 
          properly extracted from the input image or object.
        - The visualization is returned as a NumPy array representing the 
          generated figure.
    """
    box_points, largest_contour, binary_img_otsu = find_enclosing_quadrilateral_from_contour(largest_contour, binary_img_otsu)
    #Calculate the min_len
    min_len = define_min_length(largest_contour, percent)
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
    ShapeFactor = round(a/b,2)
    
    return ShapeFactor, a, b, fig


def CreateContourMultiple(image_path):
    """
    Creates and displays contours for multiple particles in an image after 
    applying Otsu's thresholding. The function extracts contours of all 
    detected particles and generates binary masks for each one.

    Args:
        image_path (str): The file path to the input image containing 
            multiple particles or objects.

    Returns:
        dict: A dictionary containing:
            - "contour" (numpy.ndarray): The contour of each particle 
              detected in the image.
            - "binary_mask" (numpy.ndarray): The binary mask of each 
              particle, where the particle is filled and the background 
              is black.

    Description:
        This function processes an image to remove border objects and 
        then applies Otsu's thresholding to convert the image to binary. 
        It then finds contours in the thresholded image, detects multiple 
        particles, and creates binary masks for each particle. The binary 
        mask fills the detected particle with white (255) and the rest of 
        the image remains black (0). Each particle's contour and binary 
        mask are stored in a dictionary, and the binary masks are displayed 
        using Matplotlib.

    Steps:
        - The image is processed using the `remove_border_objects` function 
          to remove unwanted objects at the border.
        - Otsu's thresholding is applied to binarize the image.
        - Contours are found in the thresholded image.
        - A binary mask is created for each particle, where the particle is 
          filled in and the background is black.
        - The function stores each particle's contour and binary mask in a 
          dictionary.
        - The binary masks for each particle are displayed using Matplotlib.

    Notes:
        - The function is intended for images containing multiple particles 
          or objects, each of which will be separated into its own binary mask.
        - The `particles_data_otsu` dictionary stores the contour and mask 
          for each particle, labeled as "Particle_1", "Particle_2", etc.
        - The visualization shows the binary masks for each detected particle 
          in a subplot.
    """
    gray_image = remove_border_objects(image_path)
    
    # # Convert the image to grayscale
    # gray_image = np.array(image.convert("L"))

    # Apply Otsu's thresholding method
    _, binary_image_otsu = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours with Otsu's thresholded image
    contours_otsu, _ = cv2.findContours(binary_image_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # # image = Image.open(image_path)
    # contours_otsu, binary_image_otsu = get_contour(image_path)
    
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
    fig, axs = plt.subplots(1, len(particles_data_otsu), figsize=(15, 15))
    for i, (particle_name, data) in enumerate(particles_data_otsu.items()):
        axs[i].imshow(data["binary_mask"], cmap='gray')
        axs[i].set_title(particle_name)
        axs[i].axis('off')
    
    plt.show()
    
    return particles_data_otsu


def GetShapeFactorAllImage(particles_data_otsu, percent, alliage, temperature, duree_tth, FileIdInDir, PathToFileDirectory):
    """
    Calculates the shape factor for each particle in an image and saves the 
    results to a CSV file. This function iterates over all detected particles, 
    computes their shape factors, and logs the results along with experimental 
    metadata such as alloy type, temperature, and heat treatment duration.

    Args:
        particles_data_otsu (dict): A dictionary containing the contour and 
            binary mask of each particle, where each key represents a particle 
            (e.g., "Particle_1") and each value contains:
            - "contour" (numpy.ndarray): The contour of the particle.
            - "binary_mask" (numpy.ndarray): The binary mask of the particle.
        percent (float): The percentage used to define the minimum length for 
            finding close points on the contour.
        alliage (str): The type of alloy being analyzed.
        temperature (float or int): The temperature of the heat treatment 
            applied to the alloy (in degrees Celsius).
        duree_tth (float or int): The duration of the heat treatment (in hours).
        FileIdInDir (str): An identifier for the file within the directory, 
            used to differentiate between multiple files.
        PathToFileDirectory (str): The directory path where the CSV file will 
            be saved.

    Returns:
        tuple: A tuple containing:
            - ShapeFactorAllImage (numpy.ndarray): An array of shape factors 
              for all particles in the image, with an appended mean value 
              at the end.
            - NbParticles (int): The number of particles analyzed.

    Description:
        This function processes multiple particles detected in an image and 
        calculates the shape factor for each one using the Van Sluytman Shape 
        Factor method. For each particle:
        - The function finds the enclosing quadrilateral and calculates the 
          shape factor by dividing parameter `a` by parameter `b`.
        - The results are logged to a CSV file that includes metadata such as 
          the alloy type, temperature, and heat treatment duration.
        - A summary array `ShapeFactorAllImage` is created, containing all the 
          calculated shape factors, with the mean value appended at the end.
        - The total number of particles analyzed is also returned.

    Steps:
        - A CSV file is created (or appended) with the name based on the alloy, 
          temperature, and heat treatment duration.
        - For each particle, the shape factor is calculated and logged to the 
          CSV file along with metadata.
        - The function prints each calculated shape factor and the final array 
          of shape factors including the mean.

    Notes:
        - The CSV file is named based on the alloy, temperature, and heat 
          treatment duration, with spaces and special characters replaced.
        - The function assumes that the `particles_data_otsu` dictionary 
          contains valid contours and binary masks for each particle.
        - The final array `ShapeFactorAllImage` includes all the individual 
          shape factors followed by their mean.
    """

    ShapeFactorAllImage = np.array([])
    # Créer le nom du fichier CSV en fonction des informations
    nom_fichier = f"{alliage}_{temperature}_{duree_tth}.csv"
    # Remplacer les espaces ou caractères spéciaux dans le nom du fichier (facultatif, pour éviter des problèmes)
    nom_fichier = nom_fichier.replace(" ", "_").replace("/", "-")
    nom_fichier = os.path.join(PathToFileDirectory, nom_fichier)
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
    """
   Calculates the shape factors for multiple images, stores the results in a 
   CSV file, and logs the mean shape factor for each image along with the 
   number of particles analyzed.

   Args:
       percent (float): The percentage used to define the minimum length for 
           finding close points on the contour of each particle.

   Returns:
       tuple: A tuple containing:
           - S (list): A list to store any additional results (currently unused).
           - alliage (str): The alloy type entered by the user.
           - temperature (float or int): The temperature of the heat treatment 
             entered by the user.
           - duree_tth (float or int): The duration of the heat treatment 
             entered by the user.

   Description:
       This function processes multiple images from a user-specified folder, 
       calculates the shape factors for all detected particles in each image, 
       and stores the results in a CSV file. The CSV file logs the alloy type, 
       temperature, heat treatment duration, mean shape factor for each image, 
       and the number of particles analyzed.

   Steps:
       - The user is prompted to select input and output directories.
       - Information regarding the alloy type, temperature, and heat treatment 
         duration is collected from the user.
       - The function iterates over all image files in the selected folder, 
         computes the shape factors for each particle in each image, and 
         appends the results to a CSV file.
       - The CSV file is named based on the alloy type, temperature, and heat 
         treatment duration, with any spaces or special characters replaced 
         for compatibility.
       - For each image, the mean shape factor and the number of particles 
         are written to the CSV file.

   Notes:
       - The function assumes that the images contain multiple particles and 
         uses the `CreateContourMultiple` and `GetShapeFactorAllImage` 
         functions to process them.
       - The CSV file is created (or appended) in the user-specified output 
         directory with a name based on the alloy, temperature, and duration 
         information provided.
       - The `percent` parameter influences the minimum length for finding 
         close points on the particle contours, impacting the shape factor 
         calculation.
    """   
    #Rentrer le dossier dans lequel il y a les fichiers imae d'entrée
    dossier = demander_chemin_dossier()
    dossier_sortie = demander_chemin_dossier()
    alliage, temperature, duree_tth = demander_informations()
    fichiers = [fichier for fichier in dossier.iterdir() if fichier.is_file()]
    S = []
    i = 0
    # Créer le nom du fichier CSV en fonction des informations
    nom_fichier = f"{alliage}_{temperature}_{duree_tth}_Mean.csv"
    # Remplacer les espaces ou caractères spéciaux dans le nom du fichier (facultatif, pour éviter des problèmes)
    nom_fichier = nom_fichier.replace(" ", "_").replace("/", "-")
    nom_fichier = os.path.join(dossier_sortie, nom_fichier)
    # Écrire les informations dans le fichier CSV
    with open(nom_fichier, mode='a', newline='', encoding='utf-8') as fichier_csv:
        writer = csv.writer(fichier_csv)
        # Écrire l'en-tête
        writer.writerow(["Alliage", "Temperature", "Duree du TTH", "MeanShapeFactor", "NbParticles"])
    
        for fichier in fichiers:
            percent = percent
        
            # GetShapeFactorAllImage(particles_data_otsu, percent, alliage, temperature, duree_tth, FileIdInDir, PathToFileDirectory)
            ShapeFactorAllImage, NbParticles = GetShapeFactorAllImage(CreateContourMultiple(fichier), percent, alliage, temperature, duree_tth, i, dossier_sortie)

            # Ajouter les données au fichier csv
            writer.writerow([alliage, temperature, duree_tth, ShapeFactorAllImage[-1] ,NbParticles])
        
            i = i+1
            
    return S, alliage, temperature, duree_tth


GetShapeAndStoreInCSV(0.05)
