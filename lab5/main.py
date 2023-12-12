import cv2
import numpy as np
import scipy.optimize as scOptim

from matplotlib import pyplot as plt

from utils.draw_functions import *
from utils.bundle_adjustment import Parametrice_Pose, ObtainPose

def resBundleProjection(Op, x_data, T_wc1, T_wc2, K_1, K_2, D1_k_array, D2_k_array, nPairs=2):
    """
    Input:
        Op: parameters to optimize
        x_data: 2d points
        T_wc1: camera 1 pose in the world frame
        T_wc2: camera 2 pose in the world frame
        K_1: camera 1 calibration matrix
        K_2: camera 2 calibration matrix
        D1_k_array: camera 1 distortion coefficients
        D2_k_array: camera 2 distortion coefficients
        nPairs: number of pairs of cameras
    Output:
        res: residuals
    """
    posStartX = 6 * (nPairs - 1)    # you pass 5 params for each pair of cameras and dont pass the first pair
    
    nPoints = int((Op.shape[0] - posStartX)/3)   # get the number of 3d points
    x_3dp = Op[posStartX:].reshape((3, nPoints))    # get the 3d points
    
    
    #x_3dp = x_3dp.T   # reshape the 3d points
    x_3dp = np.vstack([x_3dp, np.ones((1, nPoints))])    # add the 1s to the 3d points
       # transpose the 3d points
    
    u_1_array = np.empty([3, nPoints])
    u_2_array = np.empty([3, nPoints])
    T_c1w = np.linalg.inv(T_wc1)    # cam left respect to center
    T_c2w = np.linalg.inv(T_wc2)    # cam right respect to center
    for i in range(nPoints):
        x_3d = x_3dp[:,i]                       
        x_3d_1 = T_c1w @ x_3d.T
        x_3d_2 = T_c2w @ x_3d.T
        u_1 = kannala_forward_model(x_3d_1, K_1, D1_k_array)        
        u_2 = kannala_forward_model(x_3d_2, K_2, D2_k_array)
        u_1_array[:, i] = u_1
        u_2_array[:, i] = u_2  
    
    res = []
    for j in range(nPoints):        
        res.append(x_data[0, j] - u_1_array[0, j])        
        res.append(x_data[1, j] - u_1_array[1, j])        
        res.append(x_data[0, j+nPoints] - u_2_array[0, j])
        res.append(x_data[1, j+nPoints] - u_2_array[1, j])
        
    for i in range(nPairs-1):
        theta_rot = Op[i*6:i*6+3]
        tras = Op[i*6+3:i*6+6]
        T_wAwB = ObtainPose(theta_rot, tras[0], tras[1])    # traslation between pair i and original pair
        T_wAwB[0:3,3] = Op[i*6+3:i*6+6]        
        for j in range(nPoints):            
            x_3d = x_3dp[:, j]
            x_3d_B = T_wAwB @ x_3d.T
            x_3d_1B = T_c1w @ x_3d_B.T           
            x_3d_2B = T_c2w @ x_3d_B.T  
            u_1 = kannala_forward_model(x_3d_1B, K_1, D1_k_array)
            u_2 = kannala_forward_model(x_3d_2B, K_2, D2_k_array)
                 
            res.append(x_data[0, j+2*i*nPoints+2*nPoints] - u_1[0])
            res.append(x_data[1, j+2*i*nPoints+2*nPoints] - u_1[1])            
            res.append(x_data[0, j+2*i*nPoints+2*nPoints+nPoints] - u_2[0])
            res.append(x_data[1, j+2*i*nPoints+2*nPoints+nPoints] - u_2[1])
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

def projectPoints(x_3d,T_c1w,K_1,D1_K_array,n):
    """
    This function projects the 3D points to the 2D image plane.
    Inputs:
        x_3d: 3d points in the camera frame
        T_c1w: camera pose in the world frame
        K_1: camera calibration matrix
        D1_K_array: distortion coefficients
        n: number of points
    Outputs:
        u_1_op: 2d coordinates on the image
    """
    u_1_op = np.empty([3, n])    
    for i in range(n):
        x_3d_point = x_3d[:,i]        
        x_3d_1 = T_c1w @ x_3d_point.T        
        u_1 = kannala_forward_model(x_3d_1, K_1, D1_k_array)
        u_1_op[:, i] = u_1 
        
    return u_1_op

def plot2DImages(u_1,x1,image,imageName,figNum):
    """
    This function plots the 2D images.
    Inputs:
        u_1: 2d coordinates on the image
        x1: 2d coordinates on the image
        image: image
        imageName: image name
        figNum: figure number
    """
    fig1 = plt.figure(figNum)
    plt.imshow(image)
    plt.scatter(u_1[0, :], u_1[1, :], marker=".", c="r")
    plt.scatter(x1[0, :], x1[1, :], marker=".", c="b")
    plt.title(imageName)


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
    npoints = x_3d_array.shape[1]
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
    T_c1w = np.linalg.inv(T_wc1)
    T_c2w = np.linalg.inv(T_wc2)
    u_1 = projectPoints(x_3d_array,T_c1w,K_1,D1_k_array,npoints)
    u_2 = projectPoints(x_3d_array,T_c2w,K_2,D2_k_array,npoints)
    x_3d_optimB = np.empty([4,npoints])
    for i in range(npoints):
        x_3d = x_3d_array[:,i]
        x_3d_B = T_wAwB_gt @ x_3d.T
        x_3d_optimB [:,i]=x_3d_B
    u_3 = projectPoints(x_3d_optimB,T_c1w,K_1,D1_k_array,npoints)
    u_4 = projectPoints(x_3d_optimB,T_c2w,K_2,D2_k_array,npoints)
   
    # plot the points on the images   
    plot2DImages(u_1,x1,fisheye1_frameA,"Fisheye1 FrameA",2) 
    plot2DImages(u_2,x2,fisheye2_frameA,"Fisheye2 FrameA",3)
    plot2DImages(u_3,x3,fisheye1_frameB,"Fisheye1 FrameB",4)
    plot2DImages(u_4,x4,fisheye2_frameB,"Fisheye2 FrameB",5)
    plt.show()

    ##########################################################################
    # 3. Bundle adjustment using calibrated stereo with fish-eyes (optional) #
    ##########################################################################
    theta_rot, tras = Parametrice_Pose(T_wAwB_seed)
    tras = T_wAwB_seed[0:3,3]
    #We make the Op list
    Op = np.hstack([np.array(theta_rot).flatten(), np.array(tras).flatten(), x_3d_array[0:3,:].flatten()])
    x_data = np.hstack([x1, x2, x3, x4])    
    
    OpOptim = scOptim.least_squares(resBundleProjection, Op, args=(x_data, T_wc1, T_wc2, K_1, K_2, D1_k_array, D2_k_array, 2))
    T_wAwB_optim = ObtainPose(OpOptim.x[0:3], OpOptim.x[3], OpOptim.x[4])
    T_wAwB_optim[0:3,3] = OpOptim.x[3:6]
    x_3d_optim = OpOptim.x[6:].reshape((3, int(OpOptim.x[6:].shape[0]/3)))
    x_3d_optim = np.vstack([x_3d_optim, np.ones((1, x_3d_optim.shape[1]))])
    
    
    u_1_op = projectPoints(x_3d_optim,T_c1w,K_1,D1_k_array,npoints)
    u_2_op = projectPoints(x_3d_optim,T_c2w,K_2,D2_k_array,npoints)
    x_3d_optimB = np.empty([4, npoints])
    for i in range(npoints):
        x_3d = x_3d_optim[:,i]
        x_3d_B = T_wAwB_optim @ x_3d.T
        x_3d_optimB [:,i]=x_3d_B
    u_3_op = projectPoints(x_3d_optimB,T_c1w,K_1,D1_k_array,npoints)
    u_4_op = projectPoints(x_3d_optimB,T_c2w,K_2,D2_k_array,npoints)   
    
    plot2DImages(u_1_op,x1,fisheye1_frameA,"Fisheye1 FrameA",6)
    plot2DImages(u_2_op,x2,fisheye2_frameA,"Fisheye2 FrameA",7)
    plot2DImages(u_3_op,x3,fisheye1_frameB,"Fisheye1 FrameB",8)
    plot2DImages(u_4_op,x4,fisheye2_frameB,"Fisheye2 FrameB",9)   

    
    fig3D = plt.figure(10)
    ax = plt.axes(projection="3d", adjustable="box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    drawRefSystem(ax, T_wc1, "-", "LA")    
    drawRefSystem(ax, T_wc2, "-", "RA")  
    drawRefSystem(ax, T_wAwB_optim@T_wc1, "-", "LB")
    drawRefSystem(ax, T_wAwB_optim@T_wc2, "-", "RB") 
    ax.scatter(x_3d_optim[0, :], x_3d_optim[1, :], x_3d_optim[2, :], marker=".")    
    # Matplotlib does not correctly manage the axis('equal')
    xFakeBoundingBox = np.linspace(0, 4, 2)
    yFakeBoundingBox = np.linspace(0, 4, 2)
    zFakeBoundingBox = np.linspace(0, 4, 2)
    plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, "w.")
    print('Click in the image to continue...')
    plt.show()