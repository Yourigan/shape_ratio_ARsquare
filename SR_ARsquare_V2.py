# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 20:15:26 2024

@author: abelr
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# *************************** Function N°1 **************************************


def find_enclosing_quadrilateral(image_path):
    largest_contour, binary_img_otsu = GetContour(image_path)
    box_points, largest_contour, binary_img_otsu = find_enclosing_quadrilateral_from_contour(largest_contour, binary_img_otsu)
    return box_points, largest_contour, binary_img_otsu

def GetContour(image_path):
    # Load the image
    img = Image.open(image_path).convert("L")
    
    # Convert the image to binary using Otsu's method
    _, binary_img_otsu = cv2.threshold(np.array(img), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Extract contours from the binary image
    contours, _ = cv2.findContours(binary_img_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    return largest_contour, binary_img_otsu

def find_enclosing_quadrilateral_from_contour(largest_contour, binary_img_otsu):
    
    # Get the minimum area bounding quadrilateral (external to the particle)
    rect = cv2.minAreaRect(largest_contour)
    box_points = cv2.boxPoints(rect)
    box_points = np.intp(box_points)

    return box_points, largest_contour, binary_img_otsu


# Load the image
image_path = r"F:\Post-doc CERVELLON Alice - RAPETTI Abel (21-22) (J. CORMIER)\13- papiers\03-misfit\python\shape_ratio_ARsquare\images\example2_upscale_bis.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
box_points, largest_contour, binary_img_otsu = find_enclosing_quadrilateral(image_path)
box_points
largest_contour
binary_img_otsu

# *************************** Function N°2 **************************************
# Function to calculate the closest point on a line segment from a given point
def closest_point_on_segment(p, v, w):
    # Calculate the projection of the point on the segment
    l2 = np.sum((w - v) ** 2)
    if l2 == 0:
        return v
    t = np.dot(p - v, w - v) / l2
    t = max(0, min(1, t))
    projection = v + t * (w - v)
    return projection


# *************************** Function N°3 **************************************
# Calculate and plot distances between particle contour and the enclosing quadrilateral
def plot_distances_between_particle_and_quadrilateral(largest_contour, box_points, image_shape):
    distances_image = np.zeros(image_shape)

    # For each point on the particle contour
    for point in largest_contour:
        point = point[0]  # Extract the coordinates of the point

        # Find the closest point on the quadrilateral
        min_distance = float('inf')
        closest_point = None

        # Iterate over each edge of the quadrilateral
        for i in range(len(box_points)):
            v = box_points[i]
            w = box_points[(i + 1) % len(box_points)]  # Next point in the quadrilateral
            projection = closest_point_on_segment(point, v, w)
            distance = np.linalg.norm(point - projection)

            # Track the closest distance
            if distance < min_distance:
                min_distance = distance
                closest_point = projection

        # Draw the distance line
        cv2.line(distances_image, tuple(point), tuple(closest_point.astype(int)), (255), 1)

    # Plot the result
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(distances_image, cmap="gray")
    ax.set_title("Distances Between Particle and Quadrilateral")
    plt.show()

# *************************** Function N°4 **************************************
def particle_average_size(Largest_Contour, percent):
    # Calculate the area and perimeter of the particle
    particle_area = cv2.contourArea(Largest_Contour)
    particle_perimeter = cv2.arcLength(Largest_Contour, True)

    # Calculate the average size (based on area and perimeter)
    ParticleSize = percent * (particle_area + particle_perimeter) / 2

    return ParticleSize

# *************************** Function N°5 **************************************
# Function to find points where the distance is less than 2
def find_close_points(largest_contour, box_points, min_len):
    close_points = []

    # For each point on the particle contour
    for point in largest_contour:
        point = point[0]  # Extract the coordinates of the point

        # Find the closest point on the quadrilateral
        min_distance = float('inf')

        # Iterate over each edge of the quadrilateral
        for i in range(len(box_points)):
            v = box_points[i]
            w = box_points[(i + 1) % len(box_points)]  # Next point in the quadrilateral
            projection = closest_point_on_segment(point, v, w)
            distance = np.linalg.norm(point - projection)

            # Track the closest distance
            if distance < min_distance:
                min_distance = distance

        # Check if the distance is less than min_len
        if min_distance < min_len:
            close_points.append(point)

    return close_points

# *************************** Function N°6 **************************************

# Fonction pour calculer la distance euclidienne entre deux points
def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# *************************** Function N°7 **************************************

# Fonction pour obtenir les deux points les plus éloignés
def points_les_plus_eloignes(points):
    max_distance = 0
    point_1 = None
    point_2 = None
    
    # Parcourir toutes les paires de points
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dist = distance(points[i], points[j])
            if dist > max_distance:
                max_distance = dist
                point_1 = points[i]
                point_2 = points[j]
    
    return point_1, point_2

# *************************** Function N°8 **************************************

# Function to divide points into four groups based on proximity to each side of the quadrilateral
def divide_points_by_quadrilateral_edges(close_points, box_points):
    subsets = [[] for _ in range(4)]  # Create 4 empty lists for the four edges

    # For each point, find the closest edge of the quadrilateral and assign it to the corresponding subset
    for point in close_points:
        min_distance = float('inf')
        # print(min_distance)
        closest_edge = None
        # print("flute")

        # Iterate over each edge of the quadrilateral
        for i in range(len(box_points)):
            v = box_points[i]
            w = box_points[(i+1) % len(box_points)]  # Next point in the quadrilateral
            projection = closest_point_on_segment(point, v, w)
            distance = np.linalg.norm(point - projection)
            if i == 1:
                i = 1
                # print(distance)
                # print(min_distance)

            # Track the closest edge
            if distance < min_distance:
                # print("merde")
                min_distance = distance
                # print(min_distance)
                closest_edge = i

        # Assign the point to the corresponding edge subset
        subsets[closest_edge].append(point)

    return subsets


# *************************** Function N°9 **************************************

# Function to get the first and last point from each subset
def get_first_and_last_points(subsets):
    first_and_last_points = []
    
    for i in range(len(subsets)):
        point1, point2 = points_les_plus_eloignes(subsets[i])
        first_and_last_points.append((point1, point2))

    # for subset in subsets:
    #     if len(subset) > 1:
    #         first_and_last_points.append((subset[0], subset[-1]))

    return first_and_last_points

# *************************** Function N°10 **************************************

# Function to visualize with points in red
def visualize_full_info_red_points(image_shape, first_and_last_points, largest_contour, box_points):
    visual_image = np.zeros(image_shape)

    # Draw the particle contour in gray
    cv2.drawContours(visual_image, [largest_contour], 0, (100), 2)

    # Draw the quadrilateral in white
    cv2.drawContours(visual_image, [box_points], 0, (155), 2)

    # Draw the first and last points in red
    for (first_point, last_point) in first_and_last_points:
        cv2.circle(visual_image, tuple(first_point), 4, (255), 2)  # First point in red
        cv2.circle(visual_image, tuple(last_point), 4, (255), 2)   # Last point in red

    # Plot the result
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.imshow(visual_image, cmap="gray")
    ax.set_title("Contour, Quadrilateral, and First/Last Points in Red")
    
    # Save the figure in a variable
    canvas = FigureCanvas(fig)
    canvas.draw()

    # Convert the canvas to an array
    image_from_plot = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.show()
    plt.close(fig)  # Close the figure to prevent it from displaying

    return image_from_plot
    
# *************************** Function N°11 **************************************

# # Function to calculate and visualize segment lengths between first and last points
# def visualize_segments_with_lengths(image_shape, first_and_last_points, largest_contour, box_points):
#     visual_image = np.zeros(image_shape)

#     # Draw the particle contour in gray
#     cv2.drawContours(visual_image, [largest_contour], 0, (150), 1)

#     # Draw the quadrilateral in white
#     cv2.drawContours(visual_image, [box_points], 0, (255), 1)

#     # Draw the first and last points in red and measure distances
#     for (first_point, last_point) in first_and_last_points:
#         # Calculate the Euclidean distance between the first and last points
#         distance = np.linalg.norm(first_point - last_point)

#         # Draw a line between the first and last points
#         cv2.line(visual_image, tuple(first_point), tuple(last_point), (255), 1)

#         # Display the distance at the midpoint of the line
#         midpoint = (first_point + last_point) // 2
#         cv2.putText(visual_image, f"{distance:.2f}", tuple(midpoint), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255), 2)

#         # Draw the first and last points
#         cv2.circle(visual_image, tuple(first_point), 4, (255), -1)  # First point in red
#         cv2.circle(visual_image, tuple(last_point), 4, (255), -1)   # Last point in red
        
#     for i in range(len(box_points)):
#         pt1 = box_points[i]
#         pt2 = box_points[(i + 1) % len(box_points)]  # Next point in the quadrilateral

#         # Calculate the Euclidean distance between points of the quadrilateral
#         distance = np.linalg.norm(pt1 - pt2)

#         # Draw a line between the points
#         cv2.line(visual_image, tuple(pt1), tuple(pt2), (255), 1)

#         # Display the distance at the midpoint of the line
#         midpoint = (pt1 + pt2) // 1
#         cv2.putText(visual_image, f"{distance:.2f}", tuple(midpoint), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255), 2)



#     # Plot the result
#     fig, ax = plt.subplots(figsize=(6, 6))
#     ax.imshow(visual_image, cmap="gray")
#     ax.set_title("Segment Lengths Between First and Last Points")
#     plt.show()

# *************************** Function N°12 **************************************
def CalculateA(first_and_last_points):
    A = []
    A
    # Draw the first and last points in red and measure distances
    for (first_point, last_point) in first_and_last_points:
        # Calculate the Euclidean distance between the first and last points
        D = np.linalg.norm(first_point - last_point)
        A = A + [D]
    tuple(first_and_last_points[0][0])

    # Calculate the average between the first and third points, and between the second and fourth points
    average_1_3 = round((A[0] + A[2]) / 2,1)
    average_2_4 = round((A[1] + A[3]) / 2,1)

    # Return the calculated averages
    average_1_3, average_2_4

    a = round((average_1_3 + average_2_4) / 2,1)
    
        
    return a, average_1_3, average_2_4, A

# *************************** Function N°13 **************************************
def CalculateB(box_points):
    B = []
    for i in range(len(box_points)):
            pt1 = box_points[i]
            pt2 = box_points[(i + 1) % len(box_points)]  # Next point in the quadrilateral

            # Calculate the Euclidean distance between points of the quadrilateral
            D = np.linalg.norm(pt1 - pt2)
            B = B + [D]
    # Calculate the average between the first and third points, and between the second and fourth points
    Baverage_1_3 = round((B[0] + B[2]) / 2,1)
    Baverage_2_4 = round((B[1] + B[3]) / 2,1)

    # Return the calculated averages
    Baverage_1_3, Baverage_2_4

    b = round((Baverage_1_3 + Baverage_2_4) / 2,1)
    return b, Baverage_1_3, Baverage_2_4, B

# *************************** Function N°14 **************************************
def CalculateShapeFactor(a,b):
    #Calculate the shape factor as defined by VanSluytmann 2012
    ShapeFactor = round(a/b,2)
    return ShapeFactor, a, b


def CalculateShapeFactorFromImage(image_path, percent):
    # Test the function with the provided image
    box_points, largest_contour, binary_img_otsu = find_enclosing_quadrilateral(image_path)
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

# # Load the image
# image_path = r"F:\Post-doc CERVELLON Alice - RAPETTI Abel (21-22) (J. CORMIER)\13- papiers\03-misfit\python\images\example1.png"
# image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# # Test the function with the provided image
# fig, box_points, largest_contour, binary_img_otsu = find_enclosing_quadrilateral(image_path)
# plt.show()  # Display the result

# # Now rerun the function to plot distances
# plot_distances_between_particle_and_quadrilateral(largest_contour, box_points, binary_img_otsu.shape)

# percent = 0.0001

# min_len = particle_average_size(largest_contour, percent)
# min_len


# # Find and plot points where the distance is less than x
# close_points = find_close_points(largest_contour, box_points, min_len)
# close_points
# # Create an image to display the close points
# close_points_image = np.zeros_like(binary_img_otsu)

# # Draw the close points in white
# for point in close_points:
#     cv2.circle(close_points_image, tuple(point), 1, (255), -1)

# # Plot the result
# fig, ax = plt.subplots(figsize=(6, 6))
# ax.imshow(close_points_image, cmap="gray")
# ax.set_title("Points with Distance < 2")
# plt.show()

# # Divide the close points into four subsets based on proximity to the edges
# subsets = divide_points_by_quadrilateral_edges(close_points, box_points)

# # Get the first and last points for each subset
# first_and_last_points = get_first_and_last_points(subsets)

# # Visualize the particle contour, quadrilateral, and first/last points in red
# visualize_full_info_red_points(binary_img_otsu.shape, first_and_last_points, largest_contour, box_points)

# # # Visualize the segments and their lengths
# # visualize_segments_with_lengths(binary_img_otsu.shape, first_and_last_points, largest_contour, box_points)

# # get a
# a, average_1_3, average_2_4, A = CalculateA(first_and_last_points)

# #get b
# b, Baverage_1_3, Baverage_2_4, B = CalculateB(box_points)

# #get shape factor
# ShapeFactor, a, b = CalculateShapeFactor(a,b)
# print(ShapeFactor, a, b)


# CalculateShapeFactorFromImage(image_path, percent)
