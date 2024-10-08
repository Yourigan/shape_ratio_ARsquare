# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 14:15:39 2024

@author: abelr
"""

import tkinter as tk
from tkinter import filedialog
from tkinter import simpledialog
import csv
# import os
from pathlib import Path

# def demander_chemin_dossier():
#     # Initialiser la fenêtre Tkinter
#     root = tk.Tk()
#     root.withdraw()  # Masquer la fenêtre principale

    # # Forcer la fenêtre de dialogue à être toujours au premier plan
    # root.attributes('-topmost', True)

#     # Ouvrir la boîte de dialogue pour sélectionner un dossier
#     chemin_dossier = filedialog.askdirectory(title="Sélectionnez un dossier", parent=root)

#     # Détruire la fenêtre Tkinter après la sélection
#     root.destroy()
    
#     # chemin_dossier = os.path.normpath(chemin_dossier)
    
#     # Ajouter le préfixe 'r' pour simuler une chaîne brute
#     chemin_dossier_brut = f"r'{chemin_dossier}'"
    
#     # Retourner le chemin sélectionné
#     return chemin_dossier_brut


def demander_chemin_dossier():
    # Initialiser la fenêtre Tkinter
    root = tk.Tk()
    root.withdraw()  # Masquer la fenêtre principale

    # Forcer la fenêtre de dialogue à être toujours au premier plan
    root.attributes('-topmost', True)

    # Ouvrir la boîte de dialogue pour sélectionner un dossier
    chemin_dossier = filedialog.askdirectory(title="Sélectionnez un dossier")

    # Vérifier si un dossier a été sélectionné
    if chemin_dossier:
        # Convertir le chemin en WindowsPath
        chemin_dossier_path = Path(chemin_dossier)
        return chemin_dossier_path
    else:
        return None  # Si aucun dossier n'a été sélectionné


# chemin_dossier = demander_chemin_dossier()
# print(chemin_dossier)

def demander_informations():
    # Initialiser la fenêtre Tkinter
    root = tk.Tk()
    root.withdraw()  # Masquer la fenêtre principale

    # Demander les informations
    alliage = simpledialog.askstring("Alliage", "Quel est l'alliage ?")
    temperature = simpledialog.askstring("Température", "Quelle est la température ?")
    duree_tth = simpledialog.askstring("Durée du TTH", "Quelle est la durée du TTH ?")

    # Retourner les informations
    return alliage, temperature, duree_tth

# def ecrire_dans_csv(alliage, temperature, duree_tth):
#     # Créer le nom du fichier CSV en fonction des informations
#     nom_fichier = f"{alliage}_{temperature}_{duree_tth}.csv"

#     # Remplacer les espaces ou caractères spéciaux dans le nom du fichier (facultatif, pour éviter des problèmes)
#     nom_fichier = nom_fichier.replace(" ", "_").replace("/", "-")

#     # Écrire les informations dans le fichier CSV
#     with open(nom_fichier, mode='w', newline='', encoding='utf-8') as fichier_csv:
#         writer = csv.writer(fichier_csv)

#         # Écrire l'en-tête
#         writer.writerow(["Alliage", "Température", "Durée du TTH"])

#         # Ajouter les données
#         writer.writerow([alliage, temperature, duree_tth])

#     print(f"Les informations ont été enregistrées dans {nom_fichier}.")

# # Fonction principale
# def main_infos():
#     # Demander les informations via la fenêtre de dialogue
#     alliage, temperature, duree_tth = demander_informations()
    
#     if alliage and temperature and duree_tth:  # Vérifier si toutes les informations sont fournies
#         # Écrire les informations dans le fichier CSV
#         ecrire_dans_csv(alliage, temperature, duree_tth)
#     else:
#         print("Toutes les informations n'ont pas été fournies.")

# # Appeler la fonction principale
# if __name__ == "__main_infos__":
#     main_infos()

