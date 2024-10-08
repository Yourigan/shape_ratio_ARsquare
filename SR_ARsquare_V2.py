# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 20:15:26 2024

Listes de fonctions utiles pour récupérer le facteur de forme d'une image
List of usefull fonction to calculate the Shape Factor of Precipitates of an image

J.S. Van Sluytman, T.M. Pollock,
Optimal precipitate shapes in nickel-base γ–γ′ alloys,
Acta Materialia,
Volume 60, Issue 4,
2012,
Pages 1771-1783,
ISSN 1359-6454,
https://doi.org/10.1016/j.actamat.2011.12.008.

@author: abelr
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
# from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


# Function to remove border touching objects
def remove_border_objects(image_path):
    """
    Removes objects from a binary image that are touching the borders. 
    Black on white image.
    
    Args:
        image_path (str): The file path to the input image. The image should 
        be in a format readable by OpenCV.
    
    Returns:
        numpy.ndarray: The processed image where objects touching the borders 
        have been removed. The output is a binary image with white objects 
        (255) on a black background (0).
    
    Description:
        This function loads an image from the given path, converts it to 
        grayscale, and then applies Otsu's thresholding to create a binary 
        image. It then detects all external contours and checks if any of the 
        contours touch the borders of the image. If they do, those contours 
        are filled (removed). Finally, the resulting image is inverted 
        (white objects on a black background) and returned.
    
    Steps:
        1. Load the image in grayscale.
        2. Apply Otsu's thresholding to convert it to a binary image.
        3. Find external contours in the binary image.
        4. Create a mask where all objects touching the border are removed.
        5. Apply the mask to the binary image.
        6. Invert the image and return the result.
    
    Example usage:
        result_image = remove_border_objects("path_to_image.png")
    """
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Invert colors to have objects in white on black background
    _, binary_img_otsu = cv2.threshold(np.array(image), 0, 255, 
                                       cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


    # Extract contours from the binary image
    contours, _ = cv2.findContours(binary_img_otsu, cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty mask
    mask = np.ones(image.shape, dtype=np.uint8) * 255

    # Remove contours touching the border
    height, width = binary_img_otsu.shape
    for contour in contours:
        for point in contour:
            if point[0][0] == 0 or point[0][0] == width - 1 or point[0][1] == 0 or point[0][1] == height - 1:
                cv2.drawContours(mask, [contour], -1, 0, thickness=cv2.FILLED)
                break

    # Apply the mask to the original binary image
    result = cv2.bitwise_and(binary_img_otsu, binary_img_otsu, mask=mask)

    # Invert the colors to get the final image
    final_image = cv2.bitwise_not(result)

    return final_image


# *************************** Function N°1 **************************************
# 
# 
def get_contour(image_path):
    """
    Extracts the largest contour from a binary version of the input image.

    Args:
        image_path (str): The file path to the image.

    Returns:
        tuple: A tuple containing:
            - largest_contour (numpy.ndarray): The largest contour found in 
            the image.
            - binary_img_otsu (numpy.ndarray): The binary image after applying 
            Otsu's thresholding.

    Description:
        This function loads an image, converts it to grayscale, and applies 
        Otsu's thresholding 
        to obtain a binary image. It then finds all the external contours in 
        the image, and returns 
        the largest one based on contour area.
    """
    # Load the image
    # img = Image.open(image_path).convert("L")
    img = remove_border_objects(image_path)
    
    # Convert the image to binary using Otsu's method
    _, binary_img_otsu = cv2.threshold(np.array(img), 0, 255, 
                                       cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Extract contours from the binary image
    contours, _ = cv2.findContours(binary_img_otsu, cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_SIMPLE)

    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    return largest_contour, binary_img_otsu


# *************************** Function N°2 **************************************
def find_enclosing_quadrilateral_from_contour(largest_contour, binary_img_otsu):
    """
    Finds the minimum area enclosing quadrilateral (bounding box) around the 
    largest contour.
    
    Args:
        largest_contour (numpy.ndarray): The largest contour extracted from 
        the image.
        binary_img_otsu (numpy.ndarray): The binary image used for contour 
        extraction.
    
    Returns:
        tuple: A tuple containing:
            - box_points (numpy.ndarray): The coordinates of the four vertices 
            of the enclosing quadrilateral.
            - largest_contour (numpy.ndarray): The largest contour passed into 
            the function.
            - binary_img_otsu (numpy.ndarray): The binary image used for 
            processing.
    
    Description:
        This function computes the smallest possible quadrilateral 
        (minimum bounding box) that encloses 
        the provided contour. It uses OpenCV's `minAreaRect` and `boxPoints` 
        functions to get the four 
        corner points of this quadrilateral.
        """
    # Get the minimum area bounding quadrilateral (external to the particle)
    rect = cv2.minAreaRect(largest_contour)
    
    # Get the coordinates of the bounding quadrilateral
    box_points = cv2.boxPoints(rect)
    box_points = np.intp(box_points)

    return box_points, largest_contour, binary_img_otsu

# *************************** Function N°3 **************************************
def find_enclosing_quadrilateral(image_path):
    """
    Detects the largest contour and finds the minimum area enclosing 
    quadrilateral for an image.
    
    Args:
        image_path (str): The file path to the image.
    
    Returns:
        tuple: A tuple containing:
            - box_points (numpy.ndarray): The coordinates of the four vertices 
            of the enclosing quadrilateral.
            - largest_contour (numpy.ndarray): The largest contour found in the 
            image.
            - binary_img_otsu (numpy.ndarray): The binary image after applying 
            Otsu's thresholding.
    
    Description:
        This function combines `GetContour` and 
        `find_enclosing_quadrilateral_from_contour`. 
        It first detects the largest contour in the input image using Otsu's 
        thresholding and 
        then computes the minimum area enclosing quadrilateral around 
        that contour.
    """
    # Get the minimum area bounding quadrilateral (external to the particle )
    largest_contour, binary_img_otsu = get_contour(image_path)
    box_points, largest_contour, binary_img_otsu = find_enclosing_quadrilateral_from_contour(largest_contour, binary_img_otsu)
    return box_points, largest_contour, binary_img_otsu


# *************************** Function N°4 **************************************
# Function to calculate the closest point on a line segment from a given point
def closest_point_on_segment(p, v, w):
    """
    Calculates the closest point on a line segment to a given point of the 
    particle.
    
    Args:
        p (numpy.ndarray): The coordinates of the point (as a 2D or 3D vector) 
        for which the closest point on the segment is sought.
        v (numpy.ndarray): The coordinates of the first endpoint of the line 
        segment.
        w (numpy.ndarray): The coordinates of the second endpoint of the line 
        segment.
    
    Returns:
        numpy.ndarray: The coordinates of the closest point on the line segment
        to the point `p`.
    
    Description:
        This function computes the closest point on a line segment defined by 
        two endpoints `v` and `w` to a given point `p`. 
        If the projection of `p` onto the infinite line defined by `v` and `w` 
        falls outside the segment, 
        the closest point is constrained to be either `v` or `w`. Otherwise, 
        the closest point is calculated 
        using the projection formula. The result is the point on the segment 
        that minimizes the distance to `p`.
        
    Notes:
        - The function first checks if the segment length is zero 
        i.e., `v` and `w` are the same point), 
          in which case it returns `v`.
        - The parameter `t` represents the position of the projected point
        along the line segment, 
          constrained between 0 and 1.
    """
    # Calculate the projection of the point on the segment
    l2 = np.sum((w - v) ** 2)
    if l2 == 0:
        return v
    t = np.dot(p - v, w - v) / l2
    t = max(0, min(1, t))
    projection = v + t * (w - v)
    return projection


# # *************************** Function N°4 **************************************
# # Calculate and plot distances between particle contour and the enclosing quadrilateral
# def plot_distances_between_particle_and_quadrilateral(largest_contour, box_points, image_shape):
#     distances_image = np.zeros(image_shape)

#     # For each point on the particle contour
#     for point in largest_contour:
#         point = point[0]  # Extract the coordinates of the point

#         # Find the closest point on the quadrilateral
#         min_distance = float('inf')
#         closest_point = None

#         # Iterate over each edge of the quadrilateral
#         for i in range(len(box_points)):
#             v = box_points[i]
#             w = box_points[(i + 1) % len(box_points)]  # Next point in the quadrilateral
#             projection = closest_point_on_segment(point, v, w)
#             distance = np.linalg.norm(point - projection)

#             # Track the closest distance
#             if distance < min_distance:
#                 min_distance = distance
#                 closest_point = projection

#         # Draw the distance line
#         cv2.line(distances_image, tuple(point), tuple(closest_point.astype(int)), (255), 1)

#     # Plot the result
#     fig, ax = plt.subplots(figsize=(6, 6))
#     ax.imshow(distances_image, cmap="gray")
#     ax.set_title("Distances Between Particle and Quadrilateral")
#     plt.show()

# *************************** Function N°4 **************************************
def define_min_length(Largest_Contour, percent):
    """
    Calculates the minimum length of a particle based on its contour area and 
    a given percentage factor.
    This minimal length will be used to set up the points that are taken into 
    account for the calculation of A (see Van Sluytman shape factor definition)

    Args:
        Largest_Contour (numpy.ndarray): The contour of the particle 
        (as an array of points) used to calculate its area.
        percent (float): A percentage factor used to scale the calculated 
        length from the particle's area.

    Returns:
        float: The minimum length derived from the particle's area, scaled by 
        the percentage factor.

    Description:
        This function computes the minimum length for a particle based on its 
        contour area. It first calculates 
        the area of the particle using OpenCV's `cv2.contourArea` function. 
        Then, it uses the square root of 
        the particle's area, multiplied by the given percentage `percent`, 
        to determine a proportional length.
        
    Notes:
        - The contour of the particle should be provided as an array of points,
        as obtained from contour 
          detection methods like `cv2.findContours`.
        - The `percent` argument serves as a scaling factor to adjust the 
        length based on the particle's area.
    """
    # Calculate the area and perimeter of the particle
    particle_area = cv2.contourArea(Largest_Contour)
    # particle_perimeter = cv2.arcLength(Largest_Contour, True)

    # Calculate the minimum length (based on area and perimeter)
    # min_length = percent * (particle_area + particle_perimeter) / 2
    min_length = percent * math.sqrt(particle_area)

    return min_length

# *************************** Function N°5 **************************************
def find_close_points(largest_contour, box_points, min_len):
    """
    Finds points on the contour of a particle where the distance to the 
    enclosing quadrilateral is less than a given threshold.

    Args:
        largest_contour (numpy.ndarray): The largest contour of the particle, 
            represented as an array of points.
        box_points (numpy.ndarray): The vertices of the enclosing quadrilateral, 
            typically obtained from a minimum area bounding box.
        min_len (float): The distance threshold below which points are considered 
            "close" to the quadrilateral.

    Returns:
        list: A list of points from the contour that are within `min_len` 
            distance from the quadrilateral.

    Description:
        This function iterates through each point in the particle's contour 
        and checks its distance to the nearest edge of the enclosing 
        quadrilateral. For each contour point, the closest point on the 
        quadrilateral is computed using the `closest_point_on_segment` 
        function. If the distance between the contour point and the 
        quadrilateral is less than the specified threshold `min_len`, 
        the point is added to the list of close points.
        
    Notes:
        - The function compares the contour points to each edge of the 
          quadrilateral to find the closest projection point.
        - If the distance between a contour point and the quadrilateral is 
          less than `min_len`, it is classified as "close."
        - The `largest_contour` input should be in the format produced by 
          `cv2.findContours` (i.e., an array of point arrays).
    """
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
def distance_two_points(p1, p2):
    """
    Calculates the Euclidean distance between two points in 2D space.

    Args:
        p1 (tuple or list): The coordinates of the first point as (x, y).
        p2 (tuple or list): The coordinates of the second point as (x, y).

    Returns:
        float: The Euclidean distance between points `p1` and `p2`.

    Description:
        This function computes the straight-line distance between two 
        points using the Euclidean distance formula in 2D. It takes two 
        points as inputs, each represented as an (x, y) coordinate pair.
    """    
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# *************************** Function N°7 **************************************

# Fonction pour obtenir les deux points les plus éloignés
def points_les_plus_eloignes(points):
    """
    Finds the two points that are farthest from each other in a given list 
    of points.

    Args:
        points (list of tuples or lists): A list of points, each represented 
            by an (x, y) coordinate.

    Returns:
        tuple: A tuple containing the two points (point_1, point_2) that 
            are the farthest apart.

    Description:
        This function iterates over all pairs of points in the input list 
        and calculates the Euclidean distance between each pair. It keeps 
        track of the maximum distance and returns the two points that have 
        the largest distance between them.
        
    Notes:
        - The function uses `distance_two_points` to calculate the distance 
          between each pair of points.
        - The input list `points` must contain at least two points.
    """
    max_distance = 0
    point_1 = None
    point_2 = None
    
    # Parcourir toutes les paires de points
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dist = distance_two_points(points[i], points[j])
            if dist > max_distance:
                max_distance = dist
                point_1 = points[i]
                point_2 = points[j]
    
    return point_1, point_2

# *************************** Function N°8 **************************************

# Function to divide points into four groups based on proximity to each side of the quadrilateral
def divide_points_by_quadrilateral_edges(close_points, box_points):
    """
    Divides points into four groups based on their proximity to each side of 
    the quadrilateral.

    Args:
        close_points (list of numpy.ndarray): A list of points that are close 
            to the quadrilateral edges.
        box_points (numpy.ndarray): The vertices of the quadrilateral, 
            typically obtained from a minimum area bounding box.

    Returns:
        list: A list containing four subsets of points, where each subset 
            corresponds to the points that are closest to a specific edge 
            of the quadrilateral.

    Description:
        This function divides a set of points into four groups based on 
        their proximity to each edge of a quadrilateral. For each point, 
        the function finds the closest edge by calculating the minimum 
        distance to each edge of the quadrilateral using the 
        `closest_point_on_segment` function. The point is then assigned 
        to the subset corresponding to the closest edge.

    Notes:
        - The function iterates over each edge of the quadrilateral to find 
          the closest one for each point.
        - The input `close_points` is a list of points (usually from a 
          contour) that need to be grouped based on proximity to the edges 
          of the quadrilateral.
        - The output is a list of four subsets, where each subset contains 
          the points that are closest to one of the four edges.
    """
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
    """
    Retrieves the first and last points (the most distant points) from each 
    subset of points.

    Args:
        subsets (list of lists): A list of point subsets, where each subset 
            contains points that are grouped based on some criteria (e.g., 
            proximity to quadrilateral edges).

    Returns:
        list: A list of tuples, where each tuple contains two points 
            (point1, point2) representing the most distant points in each 
            subset.

    Description:
        This function processes each subset of points and finds the two 
        points that are the farthest from each other within that subset 
        using the `points_les_plus_eloignes` function. It returns a list 
        of tuples, with each tuple containing the first and last points 
        (the most distant pair) for each subset.
        
    Notes:
        - The function assumes that each subset contains at least two points.
        - The `points_les_plus_eloignes` function is used to find the most 
          distant points within each subset.
        - The output is a list of tuples, where each tuple contains the 
          two most distant points from the corresponding subset.
    """
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
    """
    Visualizes the particle contour, the enclosing quadrilateral, and the 
    first and last points in red.

    Args:
        image_shape (tuple): The shape of the output image (height, width).
        first_and_last_points (list of tuples): A list of tuples containing 
            the first and last points (the most distant points) for each 
            subset.
        largest_contour (numpy.ndarray): The largest contour of the particle.
        box_points (numpy.ndarray): The vertices of the enclosing 
            quadrilateral, typically obtained from a minimum area bounding box.

    Returns:
        numpy.ndarray: The visualized image array with the contour, 
            quadrilateral, and first/last points highlighted.

    Description:
        This function creates an image visualization where:
        - The particle's contour is drawn in gray.
        - The quadrilateral surrounding the contour is drawn in white.
        - The first and last points (the most distant points) from each 
          subset are drawn in red.
        
        The image is plotted using `matplotlib` and displayed, with the final 
        result returned as a NumPy array.

    Notes:
        - The function uses OpenCV to draw contours and circles for the 
          particle, quadrilateral, and points.
        - The image is plotted in grayscale, with red highlights for the 
          first and last points.
        - The figure is closed after plotting to prevent further display.
    """
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
    fig, ax = plt.subplots(figsize=(15, 15))
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
    """
    Calculate the average of distances between opposite points in pairs.
    This function is designed for the calculation of the A parameter of the 
    Van Sluytman Shape Factor. It calculates the size of the A box based 
    on the distances between consecutive points of the quadrilateral and 
    assimilates it to a square.

    Parameters
    ----------
    first_and_last_points : list of tuple of numpy arrays
        A list containing tuples of two points, representing the first
        and last points in each pair.

    Returns
    -------
    a : float
        The average of the two opposite side averages (1-3 and 2-4).
    average_1_3 : float
        The average of the distances between points 1 and 3.
    average_2_4 : float
        The average of the distances between points 2 and 4.
    A : list of float
        A list of the Euclidean distances between the first and last points
        of each pair.
    """
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

    """
    Calculate the average of distances between opposite points in a quadrilateral.
    This function is designed for the calculation of the B parameter of the 
    Van Sluytman Shape Factor. It calculates the size of the B box based 
    on the distances between consecutive points of the quadrilateral and 
    assimilates it to a square.

    Parameters
    ----------
    box_points : list of numpy arrays
        A list of four points (numpy arrays) representing the vertices
        of a quadrilateral.

    Returns
    -------
    b : float
        The average of the two opposite side averages (1-3 and 2-4).
    Baverage_1_3 : float
        The average of the distances between points 1 and 3.
    Baverage_2_4 : float
        The average of the distances between points 2 and 4.
    B : list of float
        A list of the Euclidean distances between consecutive points.
    """

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


# # *************************** Function N°14 **************************************
def CalculateShapeFactorFromImage(image_path, percent):
    """
    Calculates the shape factor from an image based on the Van Sluytman 
    Shape Factor (2012). The function processes an image to find the 
    contour of a particle, calculate parameters `a` and `b`, and determine 
    the shape factor.

    Args:
        image_path (str): The file path to the input image containing the 
            particle or object to be analyzed.
        percent (float): The percentage used to define the minimum length 
            for finding close points on the contour.

    Returns:
        tuple: A tuple containing:
            - ShapeFactor (float): The calculated shape factor, which is the 
              ratio `a/b` based on the bounding quadrilateral of the particle.
            - a (float): The value of the A parameter, calculated from the 
              distances between the first and last points in each subset.
            - b (float): The value of the B parameter, calculated from the 
              distances between consecutive points on the quadrilateral.
            - fig (numpy.ndarray): The visual representation of the 
              contour, quadrilateral, and first/last points.

    Description:
        This function processes the provided image to extract the largest 
        contour of a particle, defines the minimum length to find close 
        points on the contour, and divides those points into subsets based 
        on their proximity to the edges of the bounding quadrilateral. It 
        calculates the A and B parameters from the distances between points 
        and computes the shape factor as `a / b`, following the method 
        defined by Van Sluytman (2012). A visual representation of the 
        results is also generated, showing the contour, quadrilateral, 
        and highlighted first/last points.

    Steps:
        - The image is processed to find the largest contour and 
          the enclosing quadrilateral.
        - Close points on the contour are identified based on a 
          distance threshold.
        - These points are divided into subsets corresponding to the 
          edges of the quadrilateral.
        - The first and last points from each subset are used to 
          calculate the A parameter.
        - The B parameter is calculated from the distances between 
          consecutive points on the quadrilateral.
        - The shape factor is computed as the ratio `a / b`.
        - A visual output is created, displaying the relevant features.

    Notes:
        - The function assumes the input image contains a clear object 
          with distinguishable contours.
        - The visualization is returned as a NumPy array representing the 
          generated figure.
    """
    # Test the function with the provided image
    box_points, largest_contour, binary_img_otsu = find_enclosing_quadrilateral(image_path)
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
    ShapeFactor = round(a/b,2) #CalculateShapeFactor(a,b)
    
    return ShapeFactor, a, b, fig

