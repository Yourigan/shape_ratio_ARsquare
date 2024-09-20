# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 14:01:28 2024

@author: abelr and alexandrer
"""



from PIL import Image
import numpy as np
#import scipy.ndimage as ndi
from skimage import measure, filters
import matplotlib.pyplot as plt
import cv2

# Load the uploaded image
image_path = r"F:\Post-doc CERVELLON Alice - RAPETTI Abel (21-22) (J. CORMIER)\13- papiers\03-misfit\python\images\example1.png"
image = Image.open(image_path)

def MakeBinary(image_path):
    #Take an image and make binary, needs an image path
    image = Image.open(image_path)
    # Convert the image to grayscale
    gray_image = image.convert("L")
    # Convert the grayscale image to a numpy array
    image_array = np.array(gray_image)
    # Apply a threshold to isolate the particle from the background (assuming white background)
    threshold_value = filters.threshold_otsu(image_array)
    BinaryImage = np.where(image_array < threshold_value, 1, 0)
    return BinaryImage

def GetB(BinaryImage):
    #Take a binary image and gives the bounding box and the B associated (VanSluytmann 2012)
    binary_image = BinaryImage
    # Find the bounding box of the particle using this thresholded binary image
    non_white_pixels_1 = np.argwhere(binary_image > 0)
    # Get the bounding box coordinates
    min_row, min_col = non_white_pixels_1.min(axis=0)
    max_row, max_col = non_white_pixels_1.max(axis=0)
    # Calculate B1, B2 and B for the first image
    B1 = max_col - min_col  # Width of the bounding box
    B2 = max_row - min_row  # Height of the bounding box
    B = (B1 + B2) / 2  # Average of B1 and B2
    return B, B1, B2, min_row, min_col, max_row, max_col

def GetContour(BinaryImage):
    #Take a binary image and returns the Contour
    # Find contours of the particle
    Contours1 = measure.find_contours(BinaryImage, level=0.8)
    # # Find the largest contour, which corresponds to the outer boundary of the particle
    Contours = max(Contours1, key=len)
    return Contours

def DCBB(Contours, min_row, min_col, max_row, max_col):
    #Take a Contours and returns Distance between the Contour of the particle and the Bounding Box
    # Convert the contour of the particle to polar coordinates with respect to the center of the bounding box
    center_x = (min_col + max_col) / 2
    center_y = (min_row + max_row) / 2
    # Calculate the moments of the contour to find the center
    # moments = cv2.moments(Contours)
    # if moments['m00'] != 0:
    #         center_x = int(moments['m10'] / moments['m00'])
    #         center_y = int(moments['m01'] / moments['m00'])
    # else:
    #     center_x, center_y = 0, 0  # If the contour area is zero, center defaults to (0, 0)

    # # Calculate the polar coordinates (angles and distances from the center)
    angles = np.arctan2(largest_contour_new[:, 0] - center_y, largest_contour_new[:, 1] - center_x)

    # Sort the angles and corresponding distances for proper plotting
    sorted_indices = np.argsort(angles)
    sorted_angles = angles[sorted_indices]
    # sorted_distances_to_rectangle = distances_to_rectangle[sorted_indices]

    # Calculate the distances between the contour of the particle and the bounding box edges
    # Distances to the four edges of the bounding box
    distance_to_left_edge = np.abs(largest_contour_new[:, 1] - min_col)
    distance_to_right_edge = np.abs(largest_contour_new[:, 1] - max_col)
    distance_to_top_edge = np.abs(largest_contour_new[:, 0] - min_row)
    distance_to_bottom_edge = np.abs(largest_contour_new[:, 0] - max_row)

    # For each point on the contour, find the minimum distance to the bounding box edges
    distances_to_bounding_box = np.minimum.reduce([
        distance_to_left_edge,
        distance_to_right_edge,
        distance_to_top_edge,
        distance_to_bottom_edge
    ])

    # Sort the angles and corresponding distances for proper plotting
    sorted_distances_to_bounding_box = distances_to_bounding_box[sorted_indices]
    return sorted_distances_to_bounding_box, sorted_angles

GetB(MakeBinary(image_path))[3]
     
sorted_distances_to_bounding_box=DCBB(GetContour(MakeBinary(image_path)),GetB(MakeBinary(image_path))[3],GetB(MakeBinary(image_path))[4],GetB(MakeBinary(image_path))[5],GetB(MakeBinary(image_path))[6])[0]
sorted_angles=DCBB(GetContour(MakeBinary(image_path)),GetB(MakeBinary(image_path))[3],GetB(MakeBinary(image_path))[4],GetB(MakeBinary(image_path))[5],GetB(MakeBinary(image_path))[6])[1]

# Plot the distance to the bounding box as a function of the angle
plt.figure(figsize=(10, 6))
plt.plot(sorted_angles, sorted_distances_to_bounding_box, label="Distance to Bounding Box", color='blue')
plt.title("Distance between the Contour of the Particle and the Bounding Box")
plt.xlabel("Angle (radians)")
plt.ylabel("Distance (pixels)")
plt.legend()
plt.grid(True)
plt.show()



def find_enclosing_quadrilateral(image_path):
    # Load the image
    img = Image.open(image_path).convert("L")
    
    # Convert the image to binary using Otsu's method
    _, binary_img_otsu = cv2.threshold(np.array(img), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Extract contours from the binary image
    contours, _ = cv2.findContours(binary_img_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Get the minimum area bounding quadrilateral (external to the particle)
        rect = cv2.minAreaRect(largest_contour)
        box_points = cv2.boxPoints(rect)
        box_points = np.int0(box_points)

        # Create an empty image to draw both the original contour and the quadrilateral
        combined_image = np.zeros_like(binary_img_otsu)

        # Draw the original contour in green
        cv2.drawContours(combined_image, [largest_contour], 0, (150), 1)

        # Draw the enclosing quadrilateral in white
        cv2.drawContours(combined_image, [box_points], 0, (255), 1)

        # Create the plot
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(combined_image, cmap="gray")
        ax.set_title("Enclosing Quadrilateral vs Particle")
        
        # Return the figure object
        return fig
    else:
        print("No contours found.")
        return None


find_enclosing_quadrilateral(image_path)



 
BinaryImage = MakeBinary(image_path)
GetB(BinaryImage)


GetB(image_path)

# Convert the image to grayscale
gray_image = image.convert("L")

# Convert the grayscale image to a numpy array
image_array = np.array(gray_image)

# Apply a threshold to isolate the particle from the background (assuming white background)
threshold_value = filters.threshold_otsu(image_array)
binary_image = np.where(image_array < threshold_value, 1, 0)

# Find the bounding box of the particle using this thresholded binary image
non_white_pixels_1 = np.argwhere(binary_image > 0)

# Get the bounding box coordinates
min_row, min_col = non_white_pixels_1.min(axis=0)
max_row, max_col = non_white_pixels_1.max(axis=0)

# Calculate B1, B2 and B for the first image
B1_image = max_col - min_col  # Width of the bounding box
B2_image = max_row - min_row  # Height of the bounding box
B_image = (B1_image + B2_image) / 2  # Average of B1 and B2

B1_image, B2_image, B_image


# # Find contours of the particle
contours_step1 = measure.find_contours(binary_image, level=0.8)

# # Find the largest contour, which corresponds to the outer boundary of the particle
largest_contour_step1 = max(contours_step1, key=len)

# # Calculate the gradient (difference) of points along the contour
gradient_step1 = np.gradient(largest_contour_step1, axis=0)
tangents_step1 = np.arctan2(gradient_step1[:, 1], gradient_step1[:, 0])


# Find contours of the particle
contours_new = measure.find_contours(binary_image, level=0.8)

# Find the largest contour, which corresponds to the outer boundary of the particle
largest_contour_new = max(contours_new, key=len)

# Step 1: Calculate the centroid (center) of the particle based on its contour
centroid_new = np.mean(largest_contour_new, axis=0)

# Step 2: Calculate distances from each point on the contour to the centroid
distances_to_centroid_new = np.linalg.norm(largest_contour_new - centroid_new, axis=1)


# Convert the image to grayscale
gray_image_restart = image.convert("L")

# Plot the original image with the bounding box drawn
fig, ax = plt.subplots()
ax.imshow(binary_image, cmap='gray')
ax.plot([min_col, max_col, max_col, min_col, min_col],
        [min_row, min_row, max_row, max_row, min_row], color='red', label='Bounding Box')

plt.title("Bounding Box around the Particle")
plt.legend()
plt.show()




# # Convert the contour of the particle to polar coordinates with respect to the center of the bounding box
center_x = (min_col + max_col) / 2
center_y = (min_row + max_row) / 2

# # Calculate the polar coordinates (angles and distances from the center)
angles = np.arctan2(largest_contour_new[:, 0] - center_y, largest_contour_new[:, 1] - center_x)
distances_from_center = np.sqrt((largest_contour_new[:, 0] - center_y)**2 + (largest_contour_new[:, 1] - center_x)**2)


# Sort the angles and corresponding distances for proper plotting
sorted_indices = np.argsort(angles)
sorted_angles = angles[sorted_indices]
# sorted_distances_to_rectangle = distances_to_rectangle[sorted_indices]


# Calculate the distances between the contour of the particle and the bounding box edges
# Distances to the four edges of the bounding box
distance_to_left_edge = np.abs(largest_contour_new[:, 1] - min_col)
distance_to_right_edge = np.abs(largest_contour_new[:, 1] - max_col)
distance_to_top_edge = np.abs(largest_contour_new[:, 0] - min_row)
distance_to_bottom_edge = np.abs(largest_contour_new[:, 0] - max_row)

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
