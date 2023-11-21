"""
This file contains the exercises for lab 4 of the course Computer Vision for the Masters Degree in Robotics, Graphics
and Computer Vision at the University of Zaragoza.
Authors:
        - Daniel Capuz  (760306)
        - Jorge Urbón   (777295)
"""

import numpy as np
import cv2 as cv
import scipy.linalg as scAlg
import scipy.optimize as scOptim

from matplotlib import pyplot as plt

from utils.draw_functions import drawRefSystem, plotNumbered3DPoints, plot2Ddistances
from utils.matrix_operations import extractPose, crossMatrixInv, crossMatrix
from utils.bundle_adjustment import resBundleProjection, Parametrice_Pose, ObtainPose

if __name__ == "__main__":
    # load everything from resources folder
    x1 = np.loadtxt("resources/x1Data.txt")
    x2 = np.loadtxt("resources/x2Data.txt")
    x3 = np.loadtxt("resources/x3Data.txt")
    T_w_c1 = np.loadtxt("resources/T_w_c1.txt")
    # T_w_c2 = np.loadtxt("resources/T_w_c2.txt")
    # T_w_c3 = np.loadtxt("resources/T_w_c3.txt")
    # x_3d = np.loadtxt("resources/X_w.txt")
    K_c = np.loadtxt("resources/K_c.txt")
    F_21 = np.loadtxt("resources/F_21.txt")
    image_1 = cv.imread("resources/image1.png")
    assert image_1 is not None, "Image 1 not found"
    image_2 = cv.imread("resources/image2.png")
    assert image_2 is not None, "Image 2 not found"
    image_3 = cv.imread("resources/image3.png")
    assert image_3 is not None, "Image 3 not found"

    # calculate T_w_c2 and x_3d for the new world reference (without using the given ground truth)
    E = K_c.T @ F_21 @ K_c
    # compute camera pose and 3d points from the matches and camera 1 position
    x_3d, T_w_c2 = extractPose(E, K_c, K_c, np.linalg.inv(T_w_c1), x1, x2)     # note that this wont be the same as the ground truth
    # T_w_c2 = np.linalg.inv(T_c2_w)

    # we can plot the original 3d points and cameras
    fig3D = plt.figure(3)
    ax = plt.axes(projection="3d", adjustable="box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    drawRefSystem(ax, np.eye(4, 4), "-", "W")
    drawRefSystem(ax, T_w_c1, "-", "C1")
    drawRefSystem(ax, T_w_c2, "-", "C2")

    ax.scatter(x_3d[0, :], x_3d[1, :], x_3d[2, :], marker=".")
    # plotNumbered3DPoints(
    #     ax, x_3d, "r", (0.1, 0.1, 0.1)
    # )  # For plotting with numbers (choose one of the both options)

    # Matplotlib does not correctly manage the axis('equal')
    xFakeBoundingBox = np.linspace(0, 4, 2)
    yFakeBoundingBox = np.linspace(0, 4, 2)
    zFakeBoundingBox = np.linspace(0, 4, 2)
    plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, "w.")
    print("Close the figure to continue. Left button for orbit, right button for zoom.")
    plt.show()

    #########################################
    # 2.1: Bundle adjustment from two views #
    #########################################
    # we'll use resBundleProjection function to compute the residuals for each point
    # now we use bundle adjustment to refine the 3d points and the camera pose
    T_c1_c2 = np.linalg.inv(T_w_c1) @ T_w_c2
    theta_rot, tras = Parametrice_Pose(T_c1_c2)
    x_3d_c1 = np.linalg.inv(T_w_c1) @ x_3d
    Op = np.hstack([np.array(theta_rot).flatten(), np.array(tras).flatten(), x_3d_c1[0:3,:].flatten()])
    
    # we use least squares to optimize the parameters given the residuals:
    # res = resBundleProjection(Op, x1, x2, K_c, x_3d.shape[1])
    print("Optimizing...")
    OpOptim = scOptim.least_squares(resBundleProjection, Op, args=(x1, x2, K_c, x_3d.shape[1]))
    print("Optimization finished")
    # OpOptim includes the optimized parameters for theta_rot, [theta_tras, phi_tras] and the 3d points
    OpOptim = OpOptim.x # this is the optimized parameters vector (original OpOptim is a class)
    
    T_c1_c2_optim = ObtainPose(OpOptim[0:3],OpOptim[3],OpOptim[4])
    T_w_c2_optim = T_w_c1 @ T_c1_c2_optim   # THIS IS THE OPTIMIZED C2 POSE
    x_3d_c1_optim = OpOptim[5:].reshape((3, x_3d.shape[1]))
    x_3d_c1_optim = np.vstack([x_3d_c1_optim, np.ones((1, x_3d.shape[1]))]) # add a 1 for the homogeneous coordinates
    x_3d_optim = T_w_c1 @ x_3d_c1_optim     # THIS IS THE OPTIMIZED 3D POINTS

    # we can plot the optimized 3d points and cameras
    fig3D = plt.figure(3)
    ax = plt.axes(projection="3d", adjustable="box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    drawRefSystem(ax, np.eye(4, 4), "-", "W")
    drawRefSystem(ax, T_w_c1, "-", "C1")
    drawRefSystem(ax, T_w_c2_optim, "-", "C2_optim")
    drawRefSystem(ax, T_w_c2, "-", "C2")

    ax.scatter(x_3d_optim[0, :], x_3d_optim[1, :], x_3d_optim[2, :], marker=".")
    ax.scatter(x_3d[0, :], x_3d[1, :], x_3d[2, :], marker=".", c="r")
    xFakeBoundingBox = np.linspace(0, 4, 2)
    yFakeBoundingBox = np.linspace(0, 4, 2)
    zFakeBoundingBox = np.linspace(0, 4, 2)
    plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, "w.")
    print("Close the figure to continue. Left button for orbit, right button for zoom.")
    plt.show()

    # now let's get the projections of this optimized 3d points in the other cameras
    x1_optim = K_c @ np.eye(3, 4) @ np.linalg.inv(T_w_c1) @ x_3d_optim
    x2_optim = K_c @ np.eye(3, 4) @ np.linalg.inv(T_w_c2) @ x_3d_optim
    x1_optim = x1_optim / x1_optim[2, :]
    x2_optim = x2_optim / x2_optim[2, :]
    plot2Ddistances(5,image_1,x1,x1_optim,"Image 1")
    plot2Ddistances(6,image_2,x2,x2_optim,"Image 2")

    ######################################################
    # 3. Perspective-N-Point pose estimation of camera 3 #
    ######################################################
    # we'll use the function cv.solvePnP to estimate the pose of camera 3
    # knowing the 3d points and the 2d points in the image
    # we'll use the optimized 3d points and the 2d points in image 3
    imagePoints = np.ascontiguousarray(x3[0:2, :].T).reshape((x3.shape[1], 1, 2))
    objectPoints = np.ascontiguousarray(x_3d_c1_optim[0:3,:].T).reshape((x3.shape[1], 1, 3))
    distCoeffs = np.zeros((4, 1))
    retval, rvec, tvec = cv.solvePnP(objectPoints, imagePoints, K_c, distCoeffs, flags=cv.SOLVEPNP_EPNP)

    T_c3_c1 = np.hstack([scAlg.expm(crossMatrix(rvec)), tvec])
    T_c3_c1 = np.vstack([T_c3_c1, [0, 0, 0, 1]])
    T_c1_c3 = np.linalg.inv(T_c3_c1)
    T_w_c3 = T_w_c1 @ T_c1_c3

    # now we can plot cam 3 too!
    fig3D = plt.figure(3)
    ax = plt.axes(projection="3d", adjustable="box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    drawRefSystem(ax, np.eye(4, 4), "-", "W")
    drawRefSystem(ax, T_w_c1, "-", "C1")
    drawRefSystem(ax, T_w_c2_optim, "-", "C2_optim")
    drawRefSystem(ax, T_w_c3, "-", "C3")

    ax.scatter(x_3d_optim[0, :], x_3d_optim[1, :], x_3d_optim[2, :], marker=".")
    xFakeBoundingBox = np.linspace(0, 4, 2)
    yFakeBoundingBox = np.linspace(0, 4, 2)
    zFakeBoundingBox = np.linspace(0, 4, 2)
    plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, "w.")
    print("Close the figure to continue. Left button for orbit, right button for zoom.")
    plt.show()

    # now plot the 2D points on image3 and the projections of the 3D points
    x3_optim = K_c @ np.eye(3, 4) @ np.linalg.inv(T_w_c3) @ x_3d_optim
    x3_optim /= x3_optim[2, :]

    plot2Ddistances(7, image_3, x3, x3_optim, "Image 3")
    
    #####################################
    # 4. Bundle adjustment from 3 views #
    #####################################
    # generate the Op vector for this adjustment
    T_c1_c2 = np.linalg.inv(T_w_c1) @ T_w_c2
    trasl_c1_c2 = T_c1_c2[0:3, 3]
    scale_c2 = np.linalg.norm(trasl_c1_c2)

    theta_rot2, tras2 = Parametrice_Pose(T_c1_c2)
    theta_rot3, tras3 = Parametrice_Pose(T_c1_c3)
    theta_rot3 = np.hstack([theta_rot2, theta_rot3])
    tras3 = T_c1_c3[0:3, 3]
    tras3 = np.hstack([tras2, tras3])
    Op3 = np.hstack([np.array(theta_rot3).flatten(), np.array(tras3).flatten(), x_3d_c1_optim[0:3,:].flatten()])
    # we use least squares to optimize the parameters given the residuals:
    # res = resBundleProjection(Op, x1, x2, K_c, x_3d.shape[1])
    x2_x3 = np.hstack([x2, x3])
    print("Optimizing...")
    OpOptim3 = scOptim.least_squares(resBundleProjection, Op3, args=(x1, x2_x3, K_c, x_3d_c1.shape[1], 3))
    print("Optimization finished")
    # OpOptim includes the optimized parameters for theta_rot, [theta_tras, phi_tras] and the 3d points
    OpOptim3 = OpOptim3.x # this is the optimized parameters vector (original OpOptim is a class)
    
    T_c1_c2_optim3 = ObtainPose(OpOptim3[0:3], OpOptim3[6], OpOptim3[7])
    T_w_c2_optim3 = T_w_c1 @ T_c1_c2_optim3   # THIS IS THE OPTIMIZED C2 POSE

    T_c1_c3_optim3 = ObtainPose(OpOptim3[3:6], OpOptim3[8], OpOptim3[9])
    T_w_c3_optim3 = T_w_c1 @ T_c1_c3_optim3   # THIS IS THE OPTIMIZED C3 POSE

    # now we need to apply the scale factor to the traslations
    T_c1_c2_optim3[0:3, 3] *= scale_c2
    T_c1_c3_optim3[0:3, 3] *= scale_c2

    # we need to get the 3D points from the Op vector
    x_3d_c1_optim3 = OpOptim3[11:].reshape((3, x_3d_c1.shape[1]))
    x_3d_c1_optim3 *= scale_c2

    x_3d_c1_optim3 = np.vstack([x_3d_c1_optim3, np.ones((1, x_3d_c1.shape[1]))]) # add a 1 for the homogeneous coordinates

    x_3d_optim3 = T_w_c1 @ x_3d_c1_optim3     # THIS IS THE OPTIMIZED 3D POINTS

    # we can plot the optimized 3d points and cameras
    fig3D = plt.figure(3)
    ax = plt.axes(projection="3d", adjustable="box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    drawRefSystem(ax, np.eye(4, 4), "-", "W")
    drawRefSystem(ax, T_w_c1, "-", "C1")
    drawRefSystem(ax, T_w_c2, "-", "C2")
    drawRefSystem(ax, T_w_c2_optim, "-", "C2_optim")
    drawRefSystem(ax, T_w_c2_optim3, "-", "C2_optim3")
    drawRefSystem(ax, T_w_c3, "-", "C3")
    drawRefSystem(ax, T_w_c3_optim3, "-", "C3_optim3")

    ax.scatter(x_3d_optim3[0, :], x_3d_optim3[1, :], x_3d_optim3[2, :], marker=".", c="g")
    ax.scatter(x_3d[0, :], x_3d[1, :], x_3d[2, :], marker=".", c="r")
    ax.scatter(x_3d_optim[0, :], x_3d_optim[1, :], x_3d_optim[2, :], marker=".")

    xFakeBoundingBox = np.linspace(0, 4, 2)
    yFakeBoundingBox = np.linspace(0, 4, 2)
    zFakeBoundingBox = np.linspace(0, 4, 2)
    plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, "w.")
    print("Close the figure to continue. Left button for orbit, right button for zoom.")
    plt.show()

    # now let's get the projections of this optimized 3d points in the other cameras
    x1_optim3 = K_c @ np.eye(3, 4) @ np.linalg.inv(T_w_c1) @ x_3d_optim3
    x2_optim3 = K_c @ np.eye(3, 4) @ np.linalg.inv(T_w_c2_optim3) @ x_3d_optim3
    x3_optim3 = K_c @ np.eye(3, 4) @ np.linalg.inv(T_w_c3_optim3) @ x_3d_optim3
    x1_optim3 /= x1_optim3[2, :]
    x2_optim3 /= x2_optim3[2, :]
    x3_optim3 /= x3_optim3[2, :]

    plot2Ddistances(8,image_1,x1,x1_optim3,"Image 1")
    plot2Ddistances(9,image_2,x2,x2_optim3,"Image 2")
    plot2Ddistances(10,image_3,x3,x3_optim3,"Image 3")


