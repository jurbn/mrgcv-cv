import cv2
import numpy as np
import scipy.optimize as scOptim

from matplotlib import pyplot as plt

from utils.draw_functions import *
from utils.bundle_adjustment import Parametrice_Pose, ObtainPose

def resBundleProjection(Op, x_data, T_wc1, T_wc2, K_1, K_2, D1_k_array, D2_k_array, nPairs=2):
    """
    Input:
        Op: vector of parameters (theta_rot, tras, x_3d)
        x_data: list of matches for every image
        T_wc1: pose of the first camera in the world frame (center of the pair)
        T_wc2: pose of the second camera in the world frame (center of the pair)
        K_1: camera calibration matrix of the first camera
        K_2: camera calibration matrix of the second camera
        D1_k_array: distortion coefficients of the first camera
        D2_k_array: distortion coefficients of the second camera
        nPairs: number of pairs of cameras
    Output:
        res: residuals
    """
    posStartX = 5 * (nPairs - 1)    # you pass 5 params for each pair of cameras and dont pass the first pair
    x_3d = Op[posStartX:]     # get the 3d points
    x_3d = x_3d.reshape([3, int(x_3d.shape[0]/3)])    # reshape the 3d points
    x_3d = np.vstack([x_3d, np.ones([1, x_3d.shape[1]])])    # add the 1s to the 3d points

    u_1_array = np.empty([3, x_3d.shape[0]])
    u_2_array = np.empty([3, x_3d.shape[0]])
    T_c1w = np.linalg.inv(T_wc1)    # cam left respect to center
    T_c2w = np.linalg.inv(T_wc2)    # cam right respect to center
    for i in range(x_3d.shape[0]):
        x_3d = x_3d[i, :]
        x_3d_1 = T_c1w @ x_3d.T
        x_3d_2 = T_c2w @ x_3d.T
        u_1 = kannala_forward_model(x_3d_1, K_1, D1_k_array)
        u_2 = kannala_forward_model(x_3d_2, K_2, D2_k_array)
        u_1_array[:, i] = u_1
        u_2_array[:, i] = u_2
    

    res = []
    for j in range(x_3d.shape[0]):
        res.append(x_data[0, j] - u_1_array[0, j])
        res.append(x_data[1, j] - u_1_array[1, j])
        
        res.append(x_data[0, j+x_3d.shape[0]] - u_2_array[0, j])
        res.append(x_data[1, j+x_3d.shape[0]] - u_2_array[1, j])
    
    for i in range(nPairs-1):
        theta_rot = Op[i*5:i*5+3]
        tras = Op[i*5+3:i*5+5]
        T_wAwB = ObtainPose(theta_rot, tras[0], tras[1])    # traslation between pair i and original pair
        T_wA_c1B = np.linalg.inv(T_wAwB) @ T_wc1
        T_wA_c2B = np.linalg.inv(T_wAwB) @ T_wc2
        for j in range(x_3d.shape[0]):
            x_3d = x_3d[j, :]
            x_3d_1 = T_wA_c1B @ x_3d.T
            x_3d_2 = T_wA_c2B @ x_3d.T
            u_1 = kannala_forward_model(x_3d_1, K_1, D1_k_array)
            u_2 = kannala_forward_model(x_3d_2, K_2, D2_k_array)
            u_1_array[:, j] = u_1
            u_2_array[:, j] = u_2

            res.append(x_data[0, j+2*i*x_3d.shape[0]+2*x_3d.shape[0]] - u_1_array[0, j])
            res.append(x_data[1, j+2*i*x_3d.shape[0]+2*x_3d.shape[0]] - u_1_array[1, j])
            
            res.append(x_data[0, j+2*i*x_3d.shape[0]+2*x_3d.shape[0]+x_3d.shape[0]] - u_2_array[0, j])
            res.append(x_data[1, j+2*i*x_3d.shape[0]+2*x_3d.shape[0]+x_3d.shape[0]] - u_2_array[1, j])
    return res


def kannala_triangularization(x1, x2, K_1, K_2, D1_k_array, D2_k_array, T_1_2, T_w_2 = np.eye(4)):
    """
    This function implements the triangulation algorithm based on planes.
    Inputs:
        x1: points in camera 1
        x2: points in camera 2
    Outputs:
        p: points in 3D
    """
    assert x1.shape[1] == x2.shape[1]   # to ensure both x1 and x2 have the same number of points
    n_points = x1.shape[1]
    v_1 = np.empty([3, n_points])
    v_2 = np.empty([3, n_points])
    for i in range(n_points):
        # append to the np array
        point_1 = x1[:, i]
        v_1[:, i] = kannala_backward_model(point_1, K_1, D1_k_array)
        point_2 = x2[:, i]
        v_2[:, i] = kannala_backward_model(point_2, K_2, D2_k_array)
    
    x_3d_array = np.empty([4, n_points])
    for i in range(n_points):
        ray_1 = v_1[:, i]
        ray_2 = v_2[:, i]
        plane_sym_1 = np.array([-ray_1[1], ray_1[0], 0, 0])
        plane_perp_1 = np.array([-ray_1[2]*ray_1[0], -ray_1[2]*ray_1[1], ray_1[0]**2 + ray_1[1]**2, 0])
        plane_sym_2 = np.array([-ray_2[1], ray_2[0], 0, 0])
        plane_perp_2 = np.array([-ray_2[2]*ray_2[0], -ray_2[2]*ray_2[1], ray_2[0]**2 + ray_2[1]**2, 0])
        
        plane_sym_1_2 = T_1_2.T @ plane_sym_1
        plane_perp_1_2 = T_1_2.T @ plane_perp_1
        A = np.array([plane_sym_1_2.T, plane_perp_1_2.T, plane_sym_2.T, plane_perp_2.T])
        u, s, vh = np.linalg.svd(A)
        # ensure rank 3 for A
        S = np.zeros([4, 4])
        S[0, 0] = s[0]
        S[1, 1] = s[1]
        S[2, 2] = s[2]
        A = u @ S @ vh
        u, s, vh = np.linalg.svd(A)
        # now we can get the 3d point
        x_3d_cam2 = vh[-1, :]
        # now bring the point to the world frame
        x_3d = T_w_2 @ x_3d_cam2
        x_3d /= x_3d[3]
        x_3d_array[:, i] = x_3d
    return x_3d_array

def kannala_forward_model(x_3d, K_c, D):
    """
    This function implements the Kannala-Brandt projection model.
    Inputs:
        x_3d: 3d points in the camera frame
        K_c: camera calibration matrix
        D: distortion coefficients
    Outputs:
        u: 2d coordinates on the image
    """
    x_3d = np.array(x_3d, dtype=np.float64)
    x_3d /= x_3d[3] # sanity check
    phi = np.arctan2(x_3d[1], x_3d[0])
    radius = np.sqrt(x_3d[0]**2 + x_3d[1]**2)
    theta = np.arctan2(radius, x_3d[2])
    d = theta + D[0]*theta**3 + D[1]*theta**5 + D[2]*theta**7 + D[3]*theta**9;
    u = K_c @ np.array([d*np.cos(phi), d*np.sin(phi), 1])
    u /= u[2]
    return u

def kannala_backward_model(u, K_c, D):
    """
    This function implements the Kannala-Brandt unprojection model.
    Inputs:
        u: 2d coordinates on the image
        K_c: camera calibration matrix
        D: distortion coefficients
    Outputs:
        v: ray in the camera frame
    """
    u = np.array(u, dtype=np.float64)
    u /= u[2]   # sanity check
    x_1 = np.linalg.inv(K_c) @ u
    d = np.sqrt((x_1[0]**2 + x_1[1]**2)/(x_1[2]**2))
    phi = np.arctan2(x_1[1], x_1[0])
    theta_solutions = np.roots([D[3], 0, D[2], 0, D[1], 0, D[0], 0, 1, -d])
    # get the real root
    for theta_value in theta_solutions:
        if theta_value.imag == 0:
            theta = theta_value.real
            break
    v = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
    return v

if __name__ == "__main__":
    ################################
    # import all the provided data #
    ################################
    D1_k_array = np.loadtxt("res/D1_k_array.txt")
    D2_k_array = np.loadtxt("res/D2_k_array.txt")
    K_1 = np.loadtxt("res/K_1.txt")
    K_2 = np.loadtxt("res/K_2.txt")
    T_leftRight = np.loadtxt("res/T_leftRight.txt")
    T_wAwB_gt = np.loadtxt("res/T_wAwB_gt.txt")
    T_wAwB_seed = np.loadtxt("res/T_wAwB_seed.txt")
    T_wc1 = np.loadtxt("res/T_wc1.txt")
    T_wc2 = np.loadtxt("res/T_wc2.txt")
    x1 = np.loadtxt("res/x1.txt")
    x2 = np.loadtxt("res/x2.txt")
    x3 = np.loadtxt("res/x3.txt")
    x4 = np.loadtxt("res/x4.txt")
    fisheye1_frameA = cv2.imread("res/fisheye1_frameA.png")
    fisheye1_frameB = cv2.imread("res/fisheye1_frameB.png")
    fisheye2_frameA = cv2.imread("res/fisheye2_frameA.png")
    fisheye2_frameB = cv2.imread("res/fisheye2_frameB.png")
    # check if the images have been loaded correctly
    assert fisheye1_frameA is not None
    assert fisheye1_frameB is not None
    assert fisheye2_frameA is not None
    assert fisheye2_frameB is not None

    ######################################################################
    # 2.1 Implement the Kannala-Brandt projection and unprojection model #
    ######################################################################
    x_1 = np.array([3, 2, 10, 1], dtype=np.float64)
    x_2 = np.array([-5, 6, 7, 1], dtype=np.float64)
    x_3 = np.array([1, 5, 14, 1], dtype=np.float64)
    u_1 = kannala_forward_model(x_1, K_1, D1_k_array)
    u_2 = kannala_forward_model(x_2, K_1, D1_k_array)
    u_3 = kannala_forward_model(x_3, K_1, D1_k_array)
    print("u_1: ", u_1)
    print("u_2: ", u_2)
    print("u_3: ", u_3)

    v_1 = kannala_backward_model(u_1, K_1, D1_k_array)
    v_2 = kannala_backward_model(u_2, K_1, D1_k_array)
    v_3 = kannala_backward_model(u_3, K_1, D1_k_array)
    print("v_1: ", v_1)
    print("v_2: ", v_2)
    print("v_3: ", v_3)

    # Check if the results are correct
    print(f"v_1 scaled: {v_1/v_1[2] * 10}")
    print(f"v_2 scaled: {v_2/v_2[2] * 7}")
    print(f"v_3 scaled: {v_3/v_3[2] * 14}")

    ###################################################################################################################
    # 2.2 Implement the triangulation algorithm based on planes and compute the 3D points by triangulation for pose A #
    ###################################################################################################################
    x_3d_array = kannala_triangularization(x1, x2, K_1, K_2, D1_k_array, D2_k_array, T_leftRight, T_wc2)

    #Plot the 3D 
    fig3D = plt.figure(1)
    ax = plt.axes(projection="3d", adjustable="box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    drawRefSystem(ax, T_wc1, "-", "L")
    drawRefSystem(ax, T_wc2, "-", "R")
    ax.scatter(x_3d_array[0, :], x_3d_array[1, :], x_3d_array[2, :], marker=".")     
    # Matplotlib does not correctly manage the axis('equal')
    xFakeBoundingBox = np.linspace(0, 4, 2)
    yFakeBoundingBox = np.linspace(0, 4, 2)
    zFakeBoundingBox = np.linspace(0, 4, 2)
    plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, "w.")
    print('Click in the image to continue...')
    plt.show()
    
    # now plot them on the 2d images
    # point3d to 2d via forward model
    u_1_array = np.empty([3, x_3d_array.shape[1]])
    u_2_array = np.empty([3, x_3d_array.shape[1]])
    x_3d_array = x_3d_array.T
    T_c1w = np.linalg.inv(T_wc1)
    T_c2w = np.linalg.inv(T_wc2)
    for i in range(x_3d_array.shape[0]):
        x_3d = x_3d_array[i, :]
        x_3d_1 = T_c1w @ x_3d.T
        x_3d_2 = T_c2w @ x_3d.T
        u_1 = kannala_forward_model(x_3d_1, K_1, D1_k_array)
        u_2 = kannala_forward_model(x_3d_2, K_2, D2_k_array)
        u_1_array[:, i] = u_1
        u_2_array[:, i] = u_2

    # plot the points on the images
    fig1 = plt.figure(2)
    plt.imshow(fisheye1_frameA)
    plt.scatter(u_1_array[0, :], u_1_array[1, :], marker=".", c="r")
    plt.title("Fisheye1 FrameA")
    fig2 = plt.figure(3)
    plt.imshow(fisheye2_frameA)
    plt.scatter(u_2_array[0, :], u_2_array[1, :], marker=".", c="r")
    plt.title("Fisheye2 FrameA")
    plt.show()

    ##########################################################################
    # 3. Bundle adjustment using calibrated stereo with fish-eyes (optional) #
    ##########################################################################
    theta_rot, tras = Parametrice_Pose(T_wAwB_seed)
    #We make the Op list
    Op = np.hstack([np.array(theta_rot).flatten(), np.array(tras).flatten(), x_3d_array[0:3,:].flatten()])
    x_data = np.hstack([x1, x2, x3, x4])

    OpOptim = scOptim.least_squares(resBundleProjection, Op, args=(x_data, T_wc1, T_wc2, K_1, K_2, D1_k_array, D2_k_array, 2))

    T_wAwB_optim = ObtainPose(OpOptim.x[0:3], OpOptim.x[3], OpOptim.x[4])
    print("T_wAwB_optim: ", T_wAwB_optim)
    print("T_wAwB_gt: ", T_wAwB_gt)
