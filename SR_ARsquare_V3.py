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

# Load the image
image_path = r"F:\Post-doc CERVELLON Alice - RAPETTI Abel (21-22) (J. CORMIER)\13- papiers\03-misfit\python\shape_ratio_ARsquare\images\example2_upscale_bis.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


#******************************************** Fonction N°1*******************************************
# # Call the updated function to plot points instead of lines
# find_enclosing_quadrilateral_with_points(image_path)

# Function to interpolate points between two sets of points (contour and quadrilateral)
def interpolate_points(points, num_interpolations=5):
    interpolated_points = []
    for i in range(len(points)):
        pt1 = points[i]
        pt2 = points[(i + 1) % len(points)]  # Next point in the loop

        # Interpolate between pt1 and pt2
        for t in np.linspace(0, 1, num_interpolations, endpoint=False):
            interpolated_point = (1 - t) * pt1 + t * pt2
            interpolated_points.append(interpolated_point)

    return np.array(interpolated_points)

#******************************************** Fonction N°2*******************************************
# Updated function to plot interpolated points for both the contour and the quadrilateral - les points interpolés prennent la place des autres
def find_enclosing_quadrilateral_with_interpolation(image_path,num_interpolations):
    # Load the image
    img = Image.open(image_path).convert("L")
    
    # Convert the image to binary using Otsu's method
    _, binary_img_otsu = cv2.threshold(np.array(img), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Extract contours from the binary image
    contours, _ = cv2.findContours(binary_img_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the minimum area bounding quadrilateral (external to the particle)
    rect = cv2.minAreaRect(largest_contour)
    box_points = cv2.boxPoints(rect)
    box_points = np.intp(box_points)

    # Interpolate points for the contour and quadrilateral
    interpolated_contour_points = interpolate_points(largest_contour[:, 0], num_interpolations)
    interpolated_quad_points = interpolate_points(box_points, num_interpolations)
    
    box_points = interpolated_quad_points
    largest_contour = interpolated_contour_points

    # Create the plot
    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot the interpolated points of the contour in green
    ax.plot(interpolated_contour_points[:, 0], interpolated_contour_points[:, 1], 'go', markersize=2, label='Interpolated Contour')

    # Plot the interpolated points of the enclosing quadrilateral in red
    ax.plot(interpolated_quad_points[:, 0], interpolated_quad_points[:, 1], 'ro', markersize=3, label='Interpolated Quadrilateral')

    ax.set_title("Interpolated Points: Enclosing Quadrilateral vs Particle Contour")
    ax.set_aspect('equal')
    plt.gca().invert_yaxis()  # Invert Y axis to match image coordinates
    plt.legend()
    plt.show()
    return box_points, largest_contour, binary_img_otsu


#******************************************** Fonction N°3*******************************************
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


#******************************************** Fonction N°4 *******************************************
# Calculate and plot distances between particle contour and the enclosing quadrilateral
def plot_distances_between_particle_and_quadrilateral(largest_contour, box_points, image_shape):
    distances_image = np.zeros(image_shape)

    # For each point on the particle contour
    for point in largest_contour:
        point = point  # Extract the coordinates of the point

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


#******************************************** Fonction N°5 *******************************************
def particle_average_size(Largest_Contour, percent):
    # Calculate the area and perimeter of the particle
    particle_area = cv2.contourArea(Largest_Contour)
    particle_perimeter = cv2.arcLength(Largest_Contour, True)

    # Calculate the average size (based on area and perimeter)
    ParticleSize = percent * (particle_area + particle_perimeter) / 2

    return ParticleSize


#******************************************** Fonction N°6 *******************************************
# Function to find points where the distance is less than 2
def find_close_points(largest_contour, box_points, min_len):
    close_points = []

    # For each point on the particle contour
    for point in largest_contour:
        point = point # Extract the coordinates of the point

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

#******************************************** Fonction N°7 *******************************************
# Fonction pour calculer la distance euclidienne entre deux points
def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


#******************************************** Fonction N°8 *******************************************
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

#******************************************** Fonction N°9 *******************************************
# Function to divide points into four groups based on proximity to each side of the quadrilateral
def divide_points_by_quadrilateral_edges(close_points, box_points):
    subsets = [[] for _ in range(len(box_points))]  # Create 4 empty lists for the four edges

    # For each point, find the closest edge of the quadrilateral and assign it to the corresponding subset
    for point in close_points:
        min_distance = float('inf')
        # print(min_distance)
        closest_edge = None


        # Iterate over each edge of the quadrilateral
        for i in range(len(box_points)):
            v = box_points[i]
            w = box_points[(i+1) % len(box_points)]  # Next point in the quadrilateral
            projection = closest_point_on_segment(point, v, w)
            distance = np.linalg.norm(point - projection)
            if i == 1:
                i = 1


            # Track the closest edge
            if distance < min_distance:

                min_distance = distance
                closest_edge = i

        # Assign the point to the corresponding edge subset
        subsets[closest_edge].append(point)

    return subsets

#******************************************** Fonction N°10 *******************************************
# Function to get the first and last point from each subset
def get_first_and_last_points(subsets):
    first_and_last_points = []

    for subset in subsets:
        if len(subset) > 1:
            first_and_last_points.append((subset[0], subset[-1]))

    return first_and_last_points


#******************************************** Fonction N°11 *******************************************
# Function to visualize with points in red
def visualize_full_info_red_points(image_shape, first_and_last_points, largest_contour, box_points):
    visual_image = np.zeros(image_shape)

    # Draw the particle contour in gray
    cv2.drawContours(visual_image, [largest_contour], 0, (100), 2)

    # Draw the quadrilateral in white
    cv2.drawContours(visual_image, [box_points], 0, (155), 2)



    # Draw the first and last points in red
    for (first_point, last_point) in first_and_last_points:
        cv2.circle(visual_image, tuple(first_point), 4, (255), 4)  # First point in red
        cv2.circle(visual_image, tuple(last_point), 4, (255), 4)   # Last point in red
    
    # Plot the result
    fig, ax = plt.subplots(figsize=(6, 6))
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
    # # Plot the result
    # fig, ax = plt.subplots(figsize=(6, 6))
    # ax.imshow(visual_image, cmap="gray")
    # ax.set_title("Contour, Quadrilateral, and First/Last Points in Red")
    # plt.show()



#******************************************** Fonction N°12 *******************************************
# Function to calculate and visualize segment lengths between first and last points
def visualize_segments_with_lengths(image_shape, first_and_last_points, largest_contour, box_points_4_points):
    visual_image = np.zeros(image_shape)

    # Draw the particle contour in gray
    cv2.drawContours(visual_image, [largest_contour], 0, (150), 1)

    # Draw the quadrilateral in white
    cv2.drawContours(visual_image, [box_points_4_points], 0, (255), 1)

    # Draw the first and last points in red and measure distances
    for (first_point, last_point) in first_and_last_points:
        # Calculate the Euclidean distance between the first and last points
        distance = np.linalg.norm(first_point - last_point)

        # Draw a line between the first and last points
        cv2.line(visual_image, tuple(first_point), tuple(last_point), (255), 1)

        # Display the distance at the midpoint of the line
        midpoint = (first_point + last_point) // 2
        cv2.putText(visual_image, f"{distance:.2f}", tuple(midpoint), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255), 2)

        # Draw the first and last points
        cv2.circle(visual_image, tuple(first_point), 4, (255), -1)  # First point in red
        cv2.circle(visual_image, tuple(last_point), 4, (255), -1)   # Last point in red
        
    for i in range(len(box_points_4_points)):
        pt1 = box_points_4_points[i]
        pt2 = box_points_4_points[(i + 1) % len(box_points_4_points)]  # Next point in the quadrilateral

        # Calculate the Euclidean distance between points of the quadrilateral
        distance = np.linalg.norm(pt1 - pt2)

        # Draw a line between the points
        cv2.line(visual_image, tuple(pt1), tuple(pt2), (255), 1)

        # Display the distance at the midpoint of the line
        midpoint = (pt1 + pt2) // 1
        cv2.putText(visual_image, f"{distance:.2f}", tuple(midpoint), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255), 2)



    # Plot the result
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(visual_image, cmap="gray")
    ax.set_title("Segment Lengths Between First and Last Points")
    plt.show()



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


def CalculateShapeFactorFromImage(image_path, percent, num_interpolations):
    # Test the function with the provided image
    box_points, largest_contour, binary_img_otsu = find_enclosing_quadrilateral_with_interpolation(image_path, num_interpolations)
    
    box_points = box_points.astype(int)
    largest_contour = largest_contour.astype(int)
    # Extract contours from the binary image
    contours, _ = cv2.findContours(binary_img_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Get the largest contour
    largest_contour2 = max(contours, key=cv2.contourArea)
    # Get the minimum area bounding quadrilateral (external to the particle)
    rect = cv2.minAreaRect(largest_contour2)
    box_points_4_points = cv2.boxPoints(rect)
    box_points_4_points = np.intp(box_points_4_points)
    
    
    #Calculate the min_len
    min_len = particle_average_size(largest_contour, percent)
    #get the close_points
    close_points = find_close_points(largest_contour, box_points, min_len)
    #get the subsets
    subsets = divide_points_by_quadrilateral_edges(close_points, box_points_4_points)
    # Get the first and last points for each subset
    first_and_last_points = get_first_and_last_points(subsets)
    fig = visualize_full_info_red_points(binary_img_otsu.shape, first_and_last_points, largest_contour, box_points_4_points)
    # get a
    a, average_1_3, average_2_4, A = CalculateA(first_and_last_points)
    #get b
    b, Baverage_1_3, Baverage_2_4, B = CalculateB(box_points_4_points)
    #get shape factor
    ShapeFactor, a, b = CalculateShapeFactor(a,b)
    
    return ShapeFactor, a, b, fig



CalculateShapeFactorFromImage(image_path, 0.0001, 10)

# Call the updated function to plot interpolated points
box_points, largest_contour, binary_img_otsu = find_enclosing_quadrilateral_with_interpolation(image_path, num_interpolations=10)


# # Test the function with the provided image
# fig, box_points = find_enclosing_quadrilateral(image_path)
plt.show()  # Display the result

box_points
box_points = box_points.astype(int)
box_points
type(largest_contour)
largest_contour
largest_contour = largest_contour.astype(int)
largest_contour


# Now rerun the function to plot distances
plot_distances_between_particle_and_quadrilateral(largest_contour, box_points, binary_img_otsu.shape)

percent = 0.0001

ParticleSize = particle_average_size(largest_contour, percent)
ParticleSize
# Find and plot points where the distance is less than x
close_points = find_close_points(largest_contour, box_points, ParticleSize)

# Create an image to display the close points
close_points_image = np.zeros_like(binary_img_otsu)

# Draw the close points in white
for point in close_points:
    cv2.circle(close_points_image, tuple(point), 1, (255), -1)

# Plot the result
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(close_points_image, cmap="gray")
ax.set_title("Points with Distance < 2")
plt.show()


# Re-define the necessary variables from the previous steps to avoid errors
# This block is needed to rerun the required variables

# Convert the image to binary using Otsu's method
_, binary_img_otsu = cv2.threshold(np.array(image), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Extract contours from the binary image
contours, _ = cv2.findContours(binary_img_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get the largest contour
largest_contour2 = max(contours, key=cv2.contourArea)


# Get the minimum area bounding quadrilateral (external to the particle)
rect = cv2.minAreaRect(largest_contour2)
box_points_4_points = cv2.boxPoints(rect)
box_points_4_points = np.intp(box_points_4_points)
box_points_4_points

# ParticleSize = particle_average_size(largest_contour2)
# ParticleSize
# # Find and plot points where the distance is less than x
# close_points = find_close_points(largest_contour2, box_points_4_points, 0.0001*ParticleSize)
# close_points

# # Create an image to display the close points
# close_points_image = np.zeros_like(binary_img_otsu)

# Draw the close points in white
for point in close_points:
    # print(point)
    cv2.circle(close_points_image, tuple(point), 1, (255), -1)

# Plot the result
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(close_points_image, cmap="gray")
ax.set_title("Points with Distance < 2")
plt.show()

# Divide the close points into four subsets based on proximity to the edges
subsets = divide_points_by_quadrilateral_edges(close_points, box_points_4_points)

# Get the first and last points for each subset
first_and_last_points = get_first_and_last_points(subsets)
# Visualize the particle contour, quadrilateral, and first/last points in red
visualize_full_info_red_points(binary_img_otsu.shape, first_and_last_points, largest_contour, box_points)

# Display the first and last points
first_and_last_points

# Visualize the segments and their lengths
visualize_segments_with_lengths(binary_img_otsu.shape, first_and_last_points, largest_contour, box_points_4_points)

first_and_last_points

A = []
A
# Draw the first and last points in red and measure distances
for (first_point, last_point) in first_and_last_points:
    # Calculate the Euclidean distance between the first and last points
    D = np.linalg.norm(first_point - last_point)
    A = A + [D]
    
distance
A

tuple(first_and_last_points[0][0])

# Calculate the average between the first and third points, and between the second and fourth points
average_1_3 = round((A[0] + A[2]) / 2,1)
average_2_4 = round((A[1] + A[3]) / 2,1)

# Return the calculated averages
average_1_3, average_2_4

a = round((average_1_3 + average_2_4) / 2,1)
a, average_1_3, average_2_4

box_points

B = []
B
# # Draw the first and last points in red and measure distances
# for (first_point, last_point) in box_points:
#     # Calculate the Euclidean distance between the first and last points
#     distance = np.linalg.norm(first_point - last_point)
#     print(distance)
#     B = B + [distance]
    
for i in range(len(box_points_4_points)):
        pt1 = box_points_4_points[i]
        pt2 = box_points_4_points[(i + 1) % len(box_points_4_points)]  # Next point in the quadrilateral

        # Calculate the Euclidean distance between points of the quadrilateral
        D = np.linalg.norm(pt1 - pt2)
        B = B + [D]
    
B

# Calculate the average between the first and third points, and between the second and fourth points
Baverage_1_3 = round((B[0] + B[2]) / 2,1)
Baverage_2_4 = round((B[1] + B[3]) / 2,1)

# Return the calculated averages
Baverage_1_3, Baverage_2_4

b = round((Baverage_1_3 + Baverage_2_4) / 2,1)
b, Baverage_1_3, Baverage_2_4

ShapeFactor = round(a/b,2)
print(ShapeFactor,a,b)
