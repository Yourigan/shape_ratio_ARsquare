# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 09:13:24 2024

@author: abelr
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything_hq import sam_model_registry, SamPredictor
import os
import sys
sys.path.append("..")
from segment_anything_hq import SamAutomaticMaskGenerator


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    


def show_res(masks, scores, input_point, input_label, input_box, filename, image):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        if input_box is not None:
            box = input_box[i]
            show_box(box, plt.gca())
        if (input_point is not None) and (input_label is not None): 
            show_points(input_point, input_label, plt.gca())
        
        print(f"Score: {score:.3f}")
        plt.axis('off')
        plt.savefig(filename+'_'+str(i)+'.png',bbox_inches='tight',pad_inches=-0.1)
        plt.close()

def show_res_multi(masks, scores, input_point, input_label, input_box, filename, image):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask, plt.gca(), random_color=True)
    for box in input_box:
        show_box(box, plt.gca())
    for score in scores:
        print(f"Score: {score:.3f}")
    plt.axis('off')
    plt.savefig(filename +'.png',bbox_inches='tight',pad_inches=-0.1)
    plt.close()

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)
    
    
image = cv2.imread(r"F:\Post-doc CERVELLON Alice - RAPETTI Abel (21-22) (J. CORMIER)\2- Micro\5-AMSP2\8h1320C-AQ-24h-1100C-WQ\AMSP2_11_1320C_8h_1100C_24h_B10_D_5.0k_crop_1024-1024.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10,10))
plt.imshow(image)
plt.axis('on')
plt.show()    
    

sam_checkpoint = r"C:\Users\abelr\AppData\Roaming\SAM_path\sam_hq_vit_h.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)



import cv2
import matplotlib.pyplot as plt
import numpy as np

# Listes pour stocker les coordonnées et les valeurs de i
coords = []
values_i = []

# Fonction appelée lors du clic de la souris
def onclick(event):
    if event.xdata is not None and event.ydata is not None:
        x = int(event.xdata)
        y = int(event.ydata)
        
        # Déterminer la valeur de i en fonction du bouton de la souris
        if event.button == 1:  # Bouton gauche
            i = 0
        elif event.button == 3:  # Bouton droit
            i = 1
        else:
            print("Bouton non reconnu. Utilisez le clic gauche pour 0 et le clic droit pour 1.")
            return
        
        coords.append([x, y])
        values_i.append(i)
        print(f'Coordonnées cliquées: ({x}, {y}), Valeur de i: {i}')

# Afficher l'image avec Matplotlib
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(image)
ax.axis('on')

# Connecter l'événement de clic à la fonction onclick
cid = fig.canvas.mpl_connect('button_press_event', onclick)

# Afficher la fenêtre interactive
plt.show()

# Convertir les listes en numpy.array
coords_array = np.array(coords)
values_i_array = np.array(values_i)

# Afficher les numpy.array des coordonnées et des valeurs de i
print("Coordonnées finales (numpy.array) :")
print(coords_array)
print("Valeurs de i associées (numpy.array) :")
print(values_i_array)




predictor = SamPredictor(sam)
predictor.set_image(image)
# input_point = np.array([[450, 325], [375, 325], [900, 575], [250, 375], [250, 675], [400, 900], [1025, 800]])
# input_label = np.array([1, 1, 0, 1, 1, 0, 1])
input_point = coords_array
input_label = values_i_array
plt.figure(figsize=(10,10))
plt.imshow(image)
show_points(input_point, input_label, plt.gca())
plt.axis('on')
plt.show()


mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.8,
    stability_score_thresh=0.9,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,  # Requires open-cv to run post-processing
)
masks = mask_generator.generate(image)
len(masks)

plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show()


mask_generator2 = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.9,
    stability_score_thresh=0.9,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,  # Requires open-cv to run post-processing
)
masks2 = mask_generator2.generate(image, multimask_output=False)
len(masks2)

plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks2)
plt.axis('off')
plt.show()

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)
masks.shape  # (number_of_masks) x H x W
for i, (mask, score) in enumerate(zip(masks, scores)):
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_mask(mask, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.show()  


