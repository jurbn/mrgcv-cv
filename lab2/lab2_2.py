"""
This script completes the tasks given on section 2: "Fundamental matrix and Structure from Motion" from lab 2
Subject: Computer Vision, MRGCV
Author: Jorge Urbón, 777295
Date: 03/10/2023
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import plotData

def draw_epipolar_line(point: list[tuple], F: np.ndarray, img: np.ndarray, color: str = 'g'):
    """
    Draws the epipolar line on image 2 given a clicked point on image 1 and the fundamental matrix F.
    :param point: clicked point on image 1
    :param F: fundamental matrix
    :param img: the line will be drawn on
    :param color: color of the epipolar line
    :return: None
    """
    epipolar_line = F @ np.append(point[0], 1)
    # plot the epipolar line
    # define the line from the corresponding points at the left and right borders of the image
    line = [(-epipolar_line[2] - epipolar_line[0] * 0) / epipolar_line[1],              # y = (-c - ax) / b
            (-epipolar_line[2] - epipolar_line[0] * img.shape[1]) / epipolar_line[1]]   # y = (-c - ax) / b
    plt.plot([0, img.shape[1]], line, color)

def get_point_on_image(img: np.ndarray, title: str = 'Image1') -> list[tuple]:
    """
    Gets a point clicked on the image.
    :param img: the image to click on
    :return: the clicked point
    """
    plt.close('all')
    plt.figure(1)
    plt.imshow(img)
    plt.title(title)
    plt.draw()
    plt.waitforbuttonpress()
    point = plt.ginput(1)
    return point

def svd_triangulation(p1_matrix: np.matrix, p2_matrix: np.matrix, x1_data: np.matrix, x2_data: np.matrix) -> np.matrix:
    """Triangulates the 3D point from two 2D points and the projection matri using SVD.

    Args:
    p1_matrix (np.matrix): The projection matrix of the first camera.
    p2_matrix (np.matrix): The projection matrix of the second camera.
    x1_data (np.matrix): The 2D points in the first image.
    x2_data (np.matrix): The 2D points in the second image.

    Returns:
    np.matrix: The 3D coordinates of the triangulated points.
    """
    # create an empty matrix: 4 rows, 0 columns
    x_3d = np.empty((4, 0))
    for i in range(x1_data.shape[1]):   # for each point
        point1 = np.append(x1_data[:, i], 1) # the point is now a 3D point
        point2 = np.append(x2_data[:, i], 1)
        # now we calculate the 3D position of the point using SVD
        ## calculate the equation system
        equation = np.empty([4, 4])
        # for each row of the equation system (equation on page 6 lecture 6)
        for j in range(4):
            equation[j, 0] = p1_matrix[2, j] * point1[0] - p1_matrix[0, j]
            equation[j, 1] = p1_matrix[2, j] * point1[1] - p1_matrix[1, j]
            equation[j, 2] = p2_matrix[2, j] * point2[0] - p2_matrix[0, j]
            equation[j, 3] = p2_matrix[2, j] * point2[1] - p2_matrix[1, j]
        ## obtain the 3d point using svd to solve the system
        _, _, vh = np.linalg.svd(equation.T)    # u, s, vh are the three matrices of the svd, we only need vh
        # the 3D point is the last column of vh
        point_3d = vh[-1]
        # now we normalize the point
        point_3d = point_3d / point_3d[-1]
        # add to the matrix (points go in columns, so we transpose)
        x_3d = np.append(x_3d, point_3d.reshape(4, 1), axis=1)      
    return x_3d

# FIXME: this function is not working
def calculate_r_t(u: np.matrix, s: np.matrix, vh: np.matrix) -> tuple[np.matrix, np.matrix]:
    """
    Calculates the rotation and translation matrices from the SVD of the essential matrix.
    :param u: the first matrix of the SVD
    :param s: the second matrix of the SVD
    :param vh: the third matrix of the SVD
    :return: the rotation and translation matrices
    """
    # Construct the W matrix
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    # Calculate the four possible solutions for R
    R_positive_t = u @ W @ vh.T
    R_negative_t = u @ W.T @ vh.T
    R_positive_neg_t = u @ W @ vh.T
    R_negative_neg_t = u @ W.T @ vh.T
    # Calculate the determinants of the rotation matrices
    det_R_positive_t = np.linalg.det(R_positive_t)
    det_R_negative_t = np.linalg.det(R_negative_t)
    det_R_positive_neg_t = np.linalg.det(R_positive_neg_t)
    det_R_negative_neg_t = np.linalg.det(R_negative_neg_t)
    # Check which solution satisfies the constraints
    if det_R_positive_t > 0 and np.isclose(det_R_positive_t, 1.0):
        R = R_positive_t
    elif det_R_negative_t > 0 and np.isclose(det_R_negative_t, 1.0):
        R = R_negative_t
    elif det_R_positive_neg_t > 0 and np.isclose(det_R_positive_neg_t, 1.0):
        R = R_positive_neg_t
    elif det_R_negative_neg_t > 0 and np.isclose(det_R_negative_neg_t, 1.0):
        R = R_negative_neg_t
    else:
        # None of the solutions satisfy the constraints, handle this case as needed
        raise ValueError("No valid rotation matrix found.")
    t = u[:, 2]
    return R, t

if __name__ == '__main__':
    # load the images
    img1 = cv2.cvtColor(cv2.imread('image1.png'), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread('image2.png'), cv2.COLOR_BGR2RGB)
    ####################################
    # 2.1 Epipolar lines visualization #
    ####################################
    # Implement a function for representing an epipolar line in image 2 given a clicked point on image 1.
    # For checking your code you can use the testing fundamental matrix F_21_test.txt provided in the support files of
    # the practice.
    # first, load the fundamental matrix
    F = np.loadtxt('F_21_test.txt')
    # show image 1 and wait for a click
    plt.close('all')
    plt.figure(1)
    plt.imshow(img1)
    plt.title('Image 1')
    plt.draw()
    plt.waitforbuttonpress()
    point = plt.ginput(1)
    # plot the clicked point on image 1 and the epipolar line on image 2
    plt.plot(point[0][0], point[0][1], 'rx', markersize=10)
    # plotData.plotNumberedImagePoints(x2, 'r', (10,0))
    plt.figure(2)
    plt.imshow(img2)
    plt.title('Image 2')
    # calculate the epipolar line
    draw_epipolar_line(point, F, img2, 'g')
    plt.draw()
    plt.waitforbuttonpress()

    #####################################
    # 2.2 Fundamental matrix definition #
    #####################################
    # load the transformation matrices
    T_w_c1 = np.loadtxt('T_w_c1.txt')
    T_w_c2 = np.loadtxt('T_w_c2.txt')
    # given that F = K^(-T)*E*K^(-1), we need E to calculate F
    # E is the essential matrix, defined as E = [t] x R (cross product of traslation vector and rotation matrix)
    # and we can find both matrices in the resulting transformation matrix
    T_2_1 = np.linalg.inv(T_w_c2) @ T_w_c1
    # t is the traslation vector and has the form:
    #      [[0, -tz, ty]
    # [t] = [tz, 0, -tx]
    #       [-ty, tx, 0]]
    t = np.matrix([ [0, -T_2_1[2, 3], T_2_1[1, 3]],
                    [T_2_1[2, 3], 0, -T_2_1[0, 3]],
                    [-T_2_1[1, 3], T_2_1[0, 3], 0]])
    # R is the rotation matrix and has the form:
    #      [[R11, R12, R13]
    # [R] = [R21, R22, R23]
    #       [R31, R32, R33]]
    R = T_2_1[0:3, 0:3]
    # calculate E = [t] x R
    E = t @ R
    # calculate F = K^(-T)*E*K^(-1)
    K_c = np.loadtxt('K_c.txt')
    F_2_1 = np.linalg.inv(K_c.T) @ E @ np.linalg.inv(K_c) # this is a np.matrix, we need a np.ndarray
    F_2_1 = np.asarray(F_2_1)
    # we repeat the process of 2.1 but with both F matrices
    point = get_point_on_image(img1)
    # plot the clicked point on image 1 and the epipolar line on image 2
    plt.plot(point[0][0], point[0][1], 'rx', markersize=10)
    plt.figure(2)
    plt.imshow(img2)
    plt.title('Image 2')
    # calculate the epipolar line
    draw_epipolar_line(point, F, img2, 'g')
    draw_epipolar_line(point, F_2_1, img2, 'b')
    plt.draw()
    plt.waitforbuttonpress()
    # CONCLUSSION:
    # As the epipolar lines are different, we can estimate that the ground truth poses T_w_c1 and T_w_c2 are not correct
    # due to the loaded F being more accurate than F_2_1.
    
    ######################################################################
    # 2.3 Fundamental matrix linear estimation with eight point solution #
    ######################################################################
    # load the points
    x1 = np.loadtxt('x1Data.txt')
    x2 = np.loadtxt('x2Data.txt')
    # to obtain F from the matches, we will use SVD to solve the system of equations
    # the system of equations consists on eight equations that relate the points in both images:
    # x2.T @ F @ x1 = 0
    equations = np.empty([9, 8])
    # first, keep 8 random samples of the matches
    random_samples = np.random.randint(0, x1.shape[1], 8)
    x1 = x1[:, random_samples]
    x2 = x2[:, random_samples]
    for i in range(8):
        equations[:, i] = [x1[0, i] * x2[0, i], x1[1, i] * x2[0, i], x2[0, i],
                           x1[0, i] * x2[1, i], x1[1, i] * x2[1, i], x2[1, i],
                           x1[0, i], x1[1, i], 1]
    # now we calculate the svd
    _, _, vh = np.linalg.svd(equations.T)   # u, s, vh are the three matrices of the svd, we only need vh
    F_2_1 = vh[-1, :].reshape(3, 3)
    point = get_point_on_image(img1)
    # plot the clicked point on image 1 and the epipolar line on image 2
    plt.plot(point[0][0], point[0][1], 'rx', markersize=10)
    plt.figure(2)
    plt.imshow(img2)
    plt.title('Image 2')
    # calculate the epipolar line
    draw_epipolar_line(point, F, img2, 'g')
    draw_epipolar_line(point, F_2_1, img2, 'b')
    plt.draw()
    plt.waitforbuttonpress()
    # CONCLUSSION:
    # The results are pretty poor, this is because using 8 random mathes is not enough to estimate F. We would need to
    # use RANSAC to obtain a better estimation.

    ######################################
    # 2.4 Pose estimation from two views #
    ######################################
    # Given F, we can obtain the essential matrix E = K^(-T)*F*K^(-1) and with it, the transformation matrix T_2_1.
    # Then, we can obtain the 3D points using triangulation.
    # first, obtain E
    E = K_c.T @ F @ K_c # we use the given F to ensure a better estimation
    # obtain T_2_1 from E, knowing that E = [t] x R we can obtain R and t and then T_2_1
    # first, solve the SVD and obtain the 4 possible solutions
    u, s, vh = np.linalg.svd(E)
    # 4 solutions: (R+90, t), (R+90, -t), (R-90, t), (R-90, -t)
    # we need to find the correct one
    R, t = calculate_r_t(u, s, vh)
    # ensamble T from the correct R and t
    T_2_1 = plotData.ensamble_T(R, t)

    # triangulate the points
    # first, we need the projection matrices of both cameras
    # we can obtain them from the transformation matrices
    canon_matrix = np.matrix([  [1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 1, 0]])
    P_1 = K_c @ canon_matrix    # asuming the first camera is in the origin
    P_2 = K_c @ T_2_1
    # now we triangulate the points
    # all code from here is copied from lab2_1.py
    x_3d = svd_triangulation(P_1, P_2, x1, x2)
    # plot the points
    # Plot the triangulated points
    fig3D = plt.figure(3)
    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plotData.drawRefSystem(ax, np.eye(4, 4), '-', 'W')
    plotData.drawRefSystem(ax, canon_matrix, '-', 'C1')
    plotData.drawRefSystem(ax, T_2_1, '-', 'C2')
    ax.scatter(x_3d[0, :], x_3d[1, :], x_3d[2, :], marker='.')
    plotData.plotNumbered3DPoints(ax, x_3d, 'r', (0.1, 0.1, 0.1))
    plt.show()


