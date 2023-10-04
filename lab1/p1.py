import numpy as np
import matplotlib.pyplot as plt
import cv2

import plotData as pd
import line2DFittingSVD as l2d


def compute_p(k_matrix: np.matrix, rot_matrix: np.matrix, trasl_matrix: np.matrix) -> np.matrix:
    """
    Compute the projection matrix.
    """
    t_matrix = pd.ensamble_T(rot_matrix, trasl_matrix)
    t_matrix_inverted = np.linalg.inv(t_matrix)
    canon_matrix = np.matrix([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0]])
    p_matrix = k_matrix @ canon_matrix @ t_matrix_inverted
    return p_matrix

def compute_3d_to_pixels(p_matrix: np.matrix, point_3d: np.array) -> np.matrix:
    """
    Compute the projection of 3D points to pixels.
    """
    # reshape the 3D point to a 3x1 matrix and add a 1 to the end
    if point_3d.shape != (4,):
        point_3d = np.append(point_3d, 1)
    point_3d = np.reshape(point_3d, (4, 1))
    points_2d = p_matrix @ point_3d
    points_2d = points_2d / points_2d[2]
    return points_2d    # returns a 3x1 matrix

def plot_points_on_images(x_set, image_name):
    img1 = cv2.cvtColor(cv2.imread(image_name), cv2.COLOR_BGR2RGB)

    plt.figure(1)
    plt.imshow(img1)
    plt.plot(x_set[0, :], x_set[1, :],'+r', markersize=15)
    pd.plotLabeledImagePoints(x_set, ['a','b','c','d','e'], 'r', (20,-20)) # For plotting with labels (choose one of the both options)
    #pd.plotNumberedImagePoints(x_set, 'r', (20,25)) # For plotting with numbers (choose one of the both options)
    plt.title('Image 1')
    plt.draw()  # We update the figure display

    compute_and_plot_lines(x_set)

    print('Click in the image to continue...')
    plt.waitforbuttonpress()
    # clear the figure
    plt.clf()


def compute_and_plot_lines(x_set):
    # homogeneous coordinates of the line that passes through a and b
    point_a = [x_set[0,0], x_set[1,0], 1]
    point_b = [x_set[0,1], x_set[1,1], 1]
    line_ab = np.cross(point_a, point_b)
    line_ab = line_ab / line_ab[2]
    l2d.drawLine(line_ab, 'g-', 1)
    # homogeneous coordinates of the line that passes through c and d
    point_c = [x_set[0,2], x_set[1,2], 1]
    point_d = [x_set[0,3], x_set[1,3], 1]
    line_cd = np.cross(point_c, point_d)
    line_cd = line_cd / line_cd[2]
    l2d.drawLine(line_cd, 'g-', 1)
    # calculate and plot intersection of the two lines
    intersection = np.cross(line_ab, line_cd)
    intersection = intersection / intersection[2]
    intersection = np.array([[intersection[0]], [intersection[1]]])
    plt.plot(intersection[0, :], intersection[1, :],'+r', markersize=15)
    pd.plotLabeledImagePoints(intersection, ['p_12'], 'r', (20,-20)) # For plotting with labels (choose one of the both options)

def calculate_distance_point_plane(point, plane):
    distance = (plane[0]*point[0] + plane[1]*point[1] + plane[2]*point[2] + plane[3])/(np.sqrt(plane[0]**2 + plane[1]**2 + plane[2]**2))
    return distance

if __name__ == "__main__":
    k = np.loadtxt("K.txt")
    r_1 = np.loadtxt("R_w_c1.txt")
    r_2 = np.loadtxt("R_w_c2.txt")
    t_1 = np.loadtxt("t_w_c1.txt")
    t_2 = np.loadtxt("t_w_c2.txt")
    p_1 = compute_p(k, r_1, t_1)
    print(p_1)
    p_2 = compute_p(k, r_2, t_2)
    print(p_2)
    # declare the 3D points
    x_a = np.array([3.44, 0.80, 0.82])
    x_b = np.array([4.20, 0.80, 0.82])
    x_c = np.array([4.20, 0.60, 0.82])
    x_d = np.array([3.55, 0.60, 0.82])
    x_e = np.array([-0.01, 2.6, 1.21])
    x_1 = np.empty((3,0), dtype=float)
    x_2 = np.empty((3,0), dtype=float)
    # calculate the 2D points
    for point_3d in [x_a, x_b, x_c, x_d, x_e]:
        # x_1 and x_2 are matrices with every column being a 2D point in the form of [[x, y, 1], [x, y, 1], ...]
        x_1 = np.hstack((x_1, compute_3d_to_pixels(p_1, point_3d)))
        x_2 = np.hstack((x_2, compute_3d_to_pixels(p_2, point_3d)))
    # remove the last row of the matrices
    x_1 = x_1[0:2, :]
    x_2 = x_2[0:2, :]
    print("The 2D points are: ")
    print(x_1)
    print(x_2)

    plot_points_on_images(x_1, "Image1.jpg")
    plot_points_on_images(x_2, "Image2.jpg")
    plt.close()

    line_ab = np.cross(x_a, x_b)
    line_ab_inf = np.append(line_ab, 0)
    point_inf = compute_3d_to_pixels(p_1, line_ab_inf)
    # adapt the point so it can work with plot_points_on_images
    point_inf = np.array([point_inf[0:2, :]])
    plot_points_on_images(point_inf, "Image1.jpg")




    # planes in homogeneous coordinates
    # Compute the equation of the 3D plane π defined by the points x_a, x_b, x_c using SVD
    x_a_normalized = np.append(x_a, 1)
    x_b_normalized = np.append(x_b, 1)
    x_c_normalized = np.append(x_c, 1)

    x = np.vstack((x_a_normalized, x_b_normalized, x_c_normalized))
    print(x)
    u, s, vh = np.linalg.svd(x)
    plane = vh[-1, :]   # my solution is the last row of the svd
    print("The plane is: ")
    print(plane)

    # Compute the distance of the points to the plane π
    print("The distance of the points to the plane are: ")
    print(calculate_distance_point_plane(x_a, plane))
    print(calculate_distance_point_plane(x_b, plane))
    print(calculate_distance_point_plane(x_c, plane))
    print(calculate_distance_point_plane(x_d, plane))
    print(calculate_distance_point_plane(x_e, plane))



