# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 14:01:28 2024

@author: abelr and alexandrer
"""

from PIL import Image
import numpy as np
import scipy.ndimage as ndi
from skimage import measure, filters
import matplotlib.pyplot as plt


# Load the uploaded image
image_path_1 = r"F:\Post-doc CERVELLON Alice - RAPETTI Abel (21-22) (J. CORMIER)\13- papiers\03-misfit\python\images\example1.png"
image_1 = Image.open(image_path_1)

# Convert the image to grayscale
gray_image_1 = image_1.convert("L")

# Convert the grayscale image to a numpy array
image_array_1 = np.array(gray_image_1)

# Apply a threshold to isolate the particle from the background (assuming white background)
threshold_value = 200
binary_image_1 = np.where(image_array_1 < threshold_value, 1, 0)

# Find the bounding box of the particle using this thresholded binary image
non_white_pixels_1 = np.argwhere(binary_image_1 > 0)

# Get the bounding box coordinates
min_row_1, min_col_1 = non_white_pixels_1.min(axis=0)
max_row_1, max_col_1 = non_white_pixels_1.max(axis=0)

# Calculate B1, B2 and B for the first image
B1_image_1 = max_col_1 - min_col_1  # Width of the bounding box
B2_image_1 = max_row_1 - min_row_1  # Height of the bounding box
B_image_1 = (B1_image_1 + B2_image_1) / 2  # Average of B1 and B2

B1_image_1, B2_image_1, B_image_1

image_1.show()


# Load the latest image
image_path = image_path_1
image = Image.open(image_path)

# # Convert the image to grayscale
# gray_image = image.convert("L")

# # Convert the grayscale image to a numpy array
# image_array = np.array(gray_image)

# # Apply a threshold to isolate the particle from the background
# threshold_value = 200
# binary_image = np.where(image_array < threshold_value, 1, 0)

# # Calculate the gradient (tangent) of the binary image
# gradient_x = ndi.sobel(binary_image, axis=1)
# gradient_y = ndi.sobel(binary_image, axis=0)

# # Calculate the magnitude of the gradient (norm of the vector)
# gradient_magnitude = np.hypot(gradient_x, gradient_y)

# # Threshold the gradient to find where the tangent is close to zero
# zero_tangent = (gradient_magnitude < 1e-5).astype(int)

# # Find the points where the tangent is zero (these will correspond to the red cross points)
# coordinates_of_zeros = np.argwhere(zero_tangent > 0)

# # Determine the bounding box of the purple rectangle from these points
# min_row, min_col = coordinates_of_zeros.min(axis=0)
# max_row, max_col = coordinates_of_zeros.max(axis=0)

# # Calculate A1 and A2 (dimensions of the purple rectangle)
# A1 = max_col - min_col  # Width of the purple rectangle
# A2 = max_row - min_row  # Height of the purple rectangle

# A1, A2



# Load the newly uploaded image
image_path_2 = image_path_1
image_2 = Image.open(image_path_2)

# # Convert the image to grayscale
# gray_image_2 = image_2.convert("L")

# # Convert the grayscale image to a numpy array
# image_array_2 = np.array(gray_image_2)

# # Apply a threshold to extract the particle (using Otsu's method to find the optimal threshold)
# threshold_value_2 = filters.threshold_otsu(image_array_2)
# binary_image_2 = image_array_2 < threshold_value_2

# # Find contours of the particle
# contours = measure.find_contours(binary_image_2, level=0.8)

# # Find the largest contour, which corresponds to the outer boundary of the particle
# largest_contour = max(contours, key=len)

# # Now, we will calculate the tangents along the contour and identify the points where tangents might approximate to zero
# # Calculate the gradient (difference) of points along the contour
# gradient = np.gradient(largest_contour, axis=0)
# tangents = np.arctan2(gradient[:, 1], gradient[:, 0])

# # Find indices where the tangent approximately changes direction (potential inflection points)
# tangent_diff = np.diff(tangents)
# zero_crossings = np.where(np.sign(tangent_diff[:-1]) != np.sign(tangent_diff[1:]))[0]

# # Step 1: Calculate the centroid (center) of the particle based on its contour
# centroid = np.mean(largest_contour, axis=0)

# # Step 2: Calculate distances from each point on the contour to the centroid
# distances_to_centroid = np.linalg.norm(largest_contour - centroid, axis=1)

# # Step 4: Select the 8 most distant points from the centroid
# sorted_indices = np.argsort(distances_to_centroid)[-8:]
# key_points_new = largest_contour[sorted_indices]
# key_points_new


# # Separate into A1 and A2 based on points
# A1 = np.linalg.norm(key_points_new[0] - key_points_new[4])  # Opposite points in key_points
# A2 = np.linalg.norm(key_points_new[2] - key_points_new[6])

# # Display the points and rectangle on the image
# fig, ax = plt.subplots()
# ax.imshow(binary_image_2, cmap='gray')
# ax.plot(largest_contour[:, 1], largest_contour[:, 0], '-b', label='Contour')

# # Plot key points (red crosses)
# ax.plot(key_points_new[:, 1], key_points_new[:, 0], 'xr', label='Key points')

# # Display rectangle dimensions
# plt.title(f"A1 = {A1:.2f}, A2 = {A2:.2f}")
# plt.legend()
# plt.show()

# A1, A2


# Load the newly uploaded image (original step 1 image)
image_path_step1 = image_path_1
image_step1 = Image.open(image_path_step1)

# # Convert the image to grayscale
gray_image_step1 = image_step1.convert("L")

# Convert the grayscale image to a numpy array
image_array_step1 = np.array(gray_image_step1)

# Apply a threshold to extract the particle (using Otsu's method to find the optimal threshold)
threshold_value_step1 = filters.threshold_otsu(image_array_step1)
binary_image_step1 = image_array_step1 < threshold_value_step1

# # Find contours of the particle
contours_step1 = measure.find_contours(binary_image_step1, level=0.8)

# # Find the largest contour, which corresponds to the outer boundary of the particle
largest_contour_step1 = max(contours_step1, key=len)

# # Calculate the gradient (difference) of points along the contour
gradient_step1 = np.gradient(largest_contour_step1, axis=0)
tangents_step1 = np.arctan2(gradient_step1[:, 1], gradient_step1[:, 0])

# # Identify the points where the tangents change direction
# tangent_diff_step1 = np.diff(tangents_step1)
# zero_crossings_step1 = np.where(np.sign(tangent_diff_step1[:-1]) != np.sign(tangent_diff_step1[1:]))[0]

# # Step 1: Calculate the centroid (center) of the particle based on its contour
# centroid_step1 = np.mean(largest_contour_step1, axis=0)

# # Step 2: Calculate distances from each point on the contour to the centroid
# distances_to_centroid_step1 = np.linalg.norm(largest_contour_step1 - centroid_step1, axis=1)

# # Step 3: Select the 8 most distant points from the centroid
# sorted_indices_step1 = np.argsort(distances_to_centroid_step1)[-8:]
# key_points_step1 = largest_contour_step1[sorted_indices_step1]

# # Plot the original image with key points marked
# fig, ax = plt.subplots()
# ax.imshow(binary_image_step1, cmap='gray')
# ax.plot(largest_contour_step1[:, 1], largest_contour_step1[:, 0], '-b', label='Contour')

# # Plot key points (red crosses)
# ax.plot(key_points_step1[:, 1], key_points_step1[:, 0], 'xr', markersize=10, label='Key points')

# # Annotate the points with their index
# for i, (y, x) in enumerate(key_points_step1):
#     ax.text(x, y, f'{i+1}', color='yellow', fontsize=12, ha='center')

# plt.title("Identified key points on Step 1")
# plt.legend()
# plt.show()

# # Return the key points coordinates
# key_points_step1


# Load the newly uploaded image
image_path_new = image_path_1
image_new = Image.open(image_path_new)

# Convert the image to grayscale
gray_image_new = image_new.convert("L")

# Convert the grayscale image to a numpy array
image_array_new = np.array(gray_image_new)

# Apply a threshold to extract the particle (using Otsu's method to find the optimal threshold)
threshold_value_new = filters.threshold_otsu(image_array_new)
binary_image_new = image_array_new < threshold_value_new

# Find contours of the particle
contours_new = measure.find_contours(binary_image_new, level=0.8)

# Find the largest contour, which corresponds to the outer boundary of the particle
largest_contour_new = max(contours_new, key=len)

# Step 1: Calculate the centroid (center) of the particle based on its contour
centroid_new = np.mean(largest_contour_new, axis=0)

# Step 2: Calculate distances from each point on the contour to the centroid
distances_to_centroid_new = np.linalg.norm(largest_contour_new - centroid_new, axis=1)

# Split the contour points into four quadrants based on the centroid
top_left_quadrant_new = []
top_right_quadrant_new = []
bottom_left_quadrant_new = []
bottom_right_quadrant_new = []

# Iterate through the contour points and assign them to quadrants
for point in largest_contour_new:
    if point[0] < centroid_new[0] and point[1] < centroid_new[1]:
        top_left_quadrant_new.append(point)
    elif point[0] < centroid_new[0] and point[1] >= centroid_new[1]:
        top_right_quadrant_new.append(point)
    elif point[0] >= centroid_new[0] and point[1] < centroid_new[1]:
        bottom_left_quadrant_new.append(point)
    elif point[0] >= centroid_new[0] and point[1] >= centroid_new[1]:
        bottom_right_quadrant_new.append(point)

# # Convert lists to numpy arrays for easier processing
# top_left_quadrant_new = np.array(top_left_quadrant_new)
# top_right_quadrant_new = np.array(top_right_quadrant_new)
# bottom_left_quadrant_new = np.array(bottom_left_quadrant_new)
# bottom_right_quadrant_new = np.array(bottom_right_quadrant_new)






# # Calculate the window size as 2% of the number of points in the contour
# window_size_percentage = int(len(tangents_step1) * 0.005)

# # Calculate the gradient (difference) of points along the contour
# gradient_step1 = np.gradient(largest_contour_step1, axis=0)
# tangents_step1 = np.arctan2(gradient_step1[:, 1], gradient_step1[:, 0])

# # Ensure the window size is at least 1
# window_size_percentage = max(window_size_percentage, 1)

# # Apply a moving average (floating average) to smooth the tangents using the 2% window size
# smoothed_tangents_percentage = np.convolve(tangents_step1, np.ones(window_size_percentage)/window_size_percentage, mode='same')

# # Plot the smoothed tangents along the contour
# plt.figure(figsize=(10, 6))
# plt.plot(smoothed_tangents_percentage, label=f"Smoothed Tangents (Moving Average, 2% window)", color='orange')
# plt.title(f"Smoothed Tangents along the Contour of the Particle (Moving Average of {window_size_percentage} points)")
# plt.xlabel("Contour Index")
# plt.ylabel("Tangent (radians)")
# plt.legend()
# plt.grid(True)
# plt.show()



# Load the newly uploaded image
image_path_restart = image_path_1
image_restart = Image.open(image_path_restart)

# Convert the image to grayscale
gray_image_restart = image_restart.convert("L")

# Convert the grayscale image to a numpy array
image_array_restart = np.array(gray_image_restart)

# Apply a threshold to extract the particle (using Otsu's method to find the optimal threshold)
threshold_value_restart = filters.threshold_otsu(image_array_restart)
binary_image_restart = image_array_restart < threshold_value_restart

# Find the bounding box of the particle (smallest rectangle that encloses the particle)
non_white_pixels_restart = np.argwhere(binary_image_restart > 0)
min_row_restart, min_col_restart = non_white_pixels_restart.min(axis=0)
max_row_restart, max_col_restart = non_white_pixels_restart.max(axis=0)

# Plot the original image with the bounding box drawn
fig, ax = plt.subplots()
ax.imshow(binary_image_restart, cmap='gray')
ax.plot([min_col_restart, max_col_restart, max_col_restart, min_col_restart, min_col_restart],
        [min_row_restart, min_row_restart, max_row_restart, max_row_restart, min_row_restart], color='red', label='Bounding Box')

plt.title("Bounding Box around the Particle")
plt.legend()
plt.show()

# Calculate B1 (width) and B2 (height) of the bounding box
B1_restart = max_col_restart - min_col_restart  # Width of the bounding box
B2_restart = max_row_restart - min_row_restart  # Height of the bounding box

# Calculate the average B
B_restart = (B1_restart + B2_restart) / 2

B1_restart, B2_restart, B_restart





# # Convert the contour of the particle to polar coordinates with respect to the center of the bounding box
center_x = (min_col_restart + max_col_restart) / 2
center_y = (min_row_restart + max_row_restart) / 2

# # Calculate the polar coordinates (angles and distances from the center)
angles = np.arctan2(largest_contour_new[:, 0] - center_y, largest_contour_new[:, 1] - center_x)
distances_from_center = np.sqrt((largest_contour_new[:, 0] - center_y)**2 + (largest_contour_new[:, 1] - center_x)**2)

# # Calculate the distances between the contour and the rectangle edges
# # Distance to the left/right sides of the rectangle
# distance_to_vertical_edges = np.abs(largest_contour_new[:, 1] - min_col_restart)
# distance_to_horizontal_edges = np.abs(largest_contour_new[:, 0] - min_row_restart)

# # Select the minimum distance to the rectangle (either to vertical or horizontal edges)
# distances_to_rectangle = np.minimum(distance_to_vertical_edges, distance_to_horizontal_edges)

# Sort the angles and corresponding distances for proper plotting
sorted_indices = np.argsort(angles)
sorted_angles = angles[sorted_indices]
# sorted_distances_to_rectangle = distances_to_rectangle[sorted_indices]

# # Plot the distance to the rectangle as a function of the angle
# plt.figure(figsize=(10, 6))
# plt.plot(sorted_angles, sorted_distances_to_rectangle, label="Distance to Rectangle", color='orange')
# plt.title("Distance between the Contour of the Particle and the Rectangle")
# plt.xlabel("Angle (radians)")
# plt.ylabel("Distance (pixels)")
# plt.legend()
# plt.grid(True)
# plt.show()


# Calculate the distances between the contour of the particle and the bounding box edges
# Distances to the four edges of the bounding box
distance_to_left_edge = np.abs(largest_contour_new[:, 1] - min_col_restart)
distance_to_right_edge = np.abs(largest_contour_new[:, 1] - max_col_restart)
distance_to_top_edge = np.abs(largest_contour_new[:, 0] - min_row_restart)
distance_to_bottom_edge = np.abs(largest_contour_new[:, 0] - max_row_restart)

# For each point on the contour, find the minimum distance to the bounding box edges
distances_to_bounding_box = np.minimum.reduce([
    distance_to_left_edge,
    distance_to_right_edge,
    distance_to_top_edge,
    distance_to_bottom_edge
])

# Sort the angles and corresponding distances for proper plotting
sorted_distances_to_bounding_box = distances_to_bounding_box[sorted_indices]

# Plot the distance to the bounding box as a function of the angle
plt.figure(figsize=(10, 6))
plt.plot(sorted_angles, sorted_distances_to_bounding_box, label="Distance to Bounding Box", color='blue')
plt.title("Distance between the Contour of the Particle and the Bounding Box")
plt.xlabel("Angle (radians)")
plt.ylabel("Distance (pixels)")
plt.legend()
plt.grid(True)
plt.show()

# Calculate the derivative of the distance function (numerical differentiation) again
derivative_of_distances = np.gradient(sorted_distances_to_bounding_box, sorted_angles)

# Calculate the distance of each point on the contour from the center of the bounding box
distances_from_center = np.sqrt((largest_contour_new[:, 0] - center_y) ** 2 + (largest_contour_new[:, 1] - center_x) ** 2)

# Calculate the average radius (mean of all distances)
average_radius = np.mean(distances_from_center)

average_radius

# Identify points where the derivative is near zero and the distance is less than 1 pixel
# zero_derivative_threshold = 0.1 # Threshold to consider the derivative as zero
near_zero_distance_threshold = 0.05*average_radius  # Threshold for distance
# flat_points_indices = np.where((np.abs(derivative_of_distances) <= zero_derivative_threshold) &
#                                (sorted_distances_to_bounding_box <= near_zero_distance_threshold))[0]

flat_points_indices = np.where((sorted_distances_to_bounding_box <= near_zero_distance_threshold))[0]

# Plot the distance to the bounding box and mark the points where the derivative is near zero and distance is small
plt.figure(figsize=(10, 6))
plt.plot(sorted_angles, sorted_distances_to_bounding_box, label="Distance to Bounding Box", color='blue')
plt.plot(sorted_angles[flat_points_indices], sorted_distances_to_bounding_box[flat_points_indices], 
         'ro', label="Zero Derivative & Low Distance Points", markersize=8)

plt.title("Distance to Bounding Box with Points where Derivative is Zero and Distance < 1")
plt.xlabel("Angle (radians)")
plt.ylabel("Distance (pixels)")
plt.legend()
plt.grid(True)
plt.show()

# Output the angles and distances of the selected points
flat_points_angles = sorted_angles[flat_points_indices]
flat_points_distances = sorted_distances_to_bounding_box[flat_points_indices]

flat_points_angles, flat_points_distances


# Identify the start and end points of the 5 segments where the points are near-zero
segment_transitions = np.where(np.diff(flat_points_indices) > 1)[0]

# The start points are the beginning of each segment, and the end points are the last in the previous set
start_points_indices = np.hstack(([0], segment_transitions + 1))
end_points_indices = np.hstack((segment_transitions, [len(flat_points_indices) - 1]))

# Select the corresponding start and end points
start_points_angles = flat_points_angles[start_points_indices]
start_points_distances = flat_points_distances[start_points_indices]

end_points_angles = flat_points_angles[end_points_indices]
end_points_distances = flat_points_distances[end_points_indices]

# Combine the results for start and end points
start_end_points = list(zip(start_points_angles, start_points_distances, end_points_angles, end_points_distances))

# Output the start and end angles and distances for the 5 segments
start_end_points

# Find the distances between the center of the bounding box and the contour at the selected angles
def get_distance_at_angle(angle, contour, center_x, center_y):
    # Find the point on the contour that has the closest angle to the given one
    contour_angles = np.arctan2(contour[:, 0] - center_y, contour[:, 1] - center_x)
    index_closest = np.argmin(np.abs(contour_angles - angle))
    closest_point = contour[index_closest]
    distance = np.sqrt((closest_point[0] - center_y) ** 2 + (closest_point[1] - center_x) ** 2)
    return closest_point, distance

# Get the positions of the start and end points based on the correct contour distances
start_points_cartesian = np.array([get_distance_at_angle(angle, largest_contour_new, center_x, center_y)[0]
                                   for angle in start_points_angles])
end_points_cartesian = np.array([get_distance_at_angle(angle, largest_contour_new, center_x, center_y)[0]
                                 for angle in end_points_angles])

# Plot the particle with the bounding box and the start/end points marked correctly
fig, ax = plt.subplots()
ax.imshow(binary_image_restart, cmap='gray')
ax.plot(largest_contour_new[:, 1], largest_contour_new[:, 0], '-b', label='Particle Contour')

# Plot bounding box
ax.plot([min_col_restart, max_col_restart, max_col_restart, min_col_restart, min_col_restart],
        [min_row_restart, min_row_restart, max_row_restart, max_row_restart, min_row_restart], color='red', label='Bounding Box')

# Plot the corrected start and end points on the particle
ax.plot(start_points_cartesian[:, 1], start_points_cartesian[:, 0], 'go', label="Start Points", markersize=8)
ax.plot(end_points_cartesian[:, 1], end_points_cartesian[:, 0], 'ro', label="End Points", markersize=8)

plt.title("Corrected Start and End Points on the Particle")
plt.legend()
plt.show()

# Remove the first and last points from start_end_points
trimmed_start_points_angles = start_points_angles[1:]
trimmed_end_points_angles = end_points_angles[0:-1]

# Get the positions of the trimmed start and end points based on the correct contour distances
trimmed_start_points_cartesian = np.array([get_distance_at_angle(angle, largest_contour_new, center_x, center_y)[0]
                                           for angle in trimmed_start_points_angles])
trimmed_end_points_cartesian = np.array([get_distance_at_angle(angle, largest_contour_new, center_x, center_y)[0]
                                         for angle in trimmed_end_points_angles])

# Plot the particle with the bounding box and the trimmed start/end points marked correctly
fig, ax = plt.subplots()
ax.imshow(binary_image_restart, cmap='gray')
ax.plot(largest_contour_new[:, 1], largest_contour_new[:, 0], '-b', label='Particle Contour')

# Plot bounding box
ax.plot([min_col_restart, max_col_restart, max_col_restart, min_col_restart, min_col_restart],
        [min_row_restart, min_row_restart, max_row_restart, max_row_restart, min_row_restart], color='red', label='Bounding Box')

# Plot the corrected trimmed start and end points on the particle
ax.plot(trimmed_start_points_cartesian[:, 1], trimmed_start_points_cartesian[:, 0], 'go', label="Trimmed Start Points", markersize=8)
ax.plot(trimmed_end_points_cartesian[:, 1], trimmed_end_points_cartesian[:, 0], 'ro', label="Trimmed End Points", markersize=8)

plt.title("Trimmed Start and End Points on the Particle")
plt.legend()
plt.show()

# Combine the results for start and end points
trimmed_start_end_points = list(zip(trimmed_start_points_angles, trimmed_end_points_angles))

# Output the start and end angles and distances for the 5 segments
trimmed_start_end_points
