import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import scipy.linalg as scAlg
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as scOptim

from utils.draw_functions import *
from utils.matrix_operations import *
from utils.bundle_adjustment import *

def resBundleProjection(Op, x1Data, xCData, K_c, nPoints,nCameras=2):
    """
     -input:
     Op: Optimization parameters: this must include a paramtrization for T_21 (reference 1 seen from reference 2) in a
        proper way and for X1 (3D points in ref 1)
     x1Data: (3xnPoints) 2D points on image 1 (homogeneous coordinates)
     x2Data: (3xnPoints) 2D points on image 2 (homogeneous coordinates)
     K_c: (3x3) Intrinsic calibration matrix
     nPoints: Number of points
     -output:
     res: residuals from the error between the 2D matched points and the projected points from the 3D points (2 
        equations/residuals per 2D point)
    """    
    posStartX = (nCameras-1)*5
    # we need to get the rotation and traslation from the Op vector
    T_wc1 = np.eye(4)

    # we need to get the 3D points from the Op vector
    X_C1 = Op[posStartX:].reshape((3, nPoints))        
    # add a 1 for the homogeneous coordinates
    X_C1 = np.vstack([X_C1, np.ones((1, nPoints))])
    x1_p = K_c @ np.eye(3, 4) @ T_wc1 @ X_C1
    x1_p /= x1_p[2, :]

    for i in range(nCameras-1):
        posTheta_tras = (nCameras-1)*3+i*2 #jump all 3 rotations and the 2 translations of before cameras
        posPhi_tras = (nCameras-1)*3+i*2+1
        theta_rot = Op[i*3:(i*3+3)]
        theta_tras = Op[posTheta_tras]
        phi_tras = Op[posPhi_tras]

        # we need to get the rotation and traslation from the Op vector
        T_c1c = ObtainPose(theta_rot,theta_tras,phi_tras)
        xC_p = K_c @ np.eye(3, 4) @ np.linalg.inv(T_c1c) @ X_C1 
        xC_p /= xC_p[2, :]
        res = []        
        for j in range(nPoints):            
            res.append(x1Data[0, j] - x1_p[0, j])
            res.append(x1Data[1, j] - x1_p[1, j])
            res.append(xCData[0, j+nPoints*i] - xC_p[0, j])
            res.append(xCData[1, j+nPoints*i] - xC_p[1, j])
        res = np.array(res).flatten()
    return res

def resBundleProjectionScale(Op, x1Data, x2Data, K_c, nPoints, TScale,nCameras=2):
    """
     -input:
     Op: Optimization parameters: this must include a paramtrization for T_21 (reference 1 seen from reference 2) in a
        proper way and for X1 (3D points in ref 1)
     x1Data: (3xnPoints) 2D points on image 1 (homogeneous coordinates)
     x2Data: (3xnPoints) 2D points on image 2 (homogeneous coordinates)
     K_c: (3x3) Intrinsic calibration matrix
     nPoints: Number of points
     -output:
     res: residuals from the error between the 2D matched points and the projected points from the 3D points (2 
        equations/residuals per 2D point)
    """  
    posStartX = (nCameras-2)*5  
    # we need to get the 3D points from the Op vector
    X_C1 = Op[posStartX:].reshape((3, nPoints))
            
    # add a 1 for the homogeneous coordinates
    X_C1 = np.vstack([X_C1, np.ones((1, nPoints))]) 

    # we need to get the rotation and traslation from the Op vector
    T_wc1 = np.eye(4)
    x1_p = K_c @ np.eye(3, 4) @ T_wc1 @ X_C1
    x1_p /= x1_p[2, :]
    
    for i in range(nCameras-1):
        posStartRot = (nCameras-2)*3*(i-1)
        posEndRot = (nCameras-2)*3*(i-1)+3
        posTheta_tras = (nCameras-2)*3+(i-1)*2 #jump all 3 rotations and the 2 translations of before cameras
        posPhi_tras = (nCameras-2)*3+(i-1)*2+1   

        if(i==0):             
            T_c1c = TScale
        else:   
            theta_rot = Op[posStartRot:posEndRot]                    
            theta_tras = Op[posTheta_tras]
            phi_tras = Op[posPhi_tras] 
            T_c1c = ObtainPose(theta_rot,theta_tras,phi_tras)            
        
        xC_p = K_c @ np.eye(3, 4) @ np.linalg.inv(T_c1c) @ X_C1        
        xC_p /= xC_p[2, :]
        res = []        
        for j in range(nPoints):            
            res.append(x1Data[0, j] - x1_p[0, j])
            res.append(x1Data[1, j] - x1_p[1, j])
            res.append(x2Data[0, j+nPoints*i] - xC_p[0, j])
            res.append(x2Data[1, j+nPoints*i] - xC_p[1, j])
        res = np.array(res).flatten()
    return res

def Parametrice_Pose(T):
    """
    -input:
    T: Transformation matrix
    -output:List rot with the 3 angles and list tras with the 2 angles
    """
    R = T[0:3, 0:3]
    theta_rot = crossMatrixInv(scAlg.logm(R))

    #two angles that describe the traslation T_C1_C3 (spherical coordinates)
    theta_tras = np.arccos(T[2,2])
    phi_tras = np.arccos(T[2,0]/np.sin(theta_tras))   # slides page 46 chapter 6
    tras = [theta_tras, phi_tras]
    return theta_rot,tras

def ObtainPose(theta_rot,theta_tras,phi_tras):
    """
    -input:theta_rot,theta_tras,phi_tras
    -output: Pose matrix
    """
        
    R = scAlg.expm(crossMatrix(theta_rot))
    Tras = np.array([np.sin(theta_tras)*np.cos(phi_tras),
                  np.sin(theta_tras)*np.sin(phi_tras),
                  np.cos(theta_tras)])
    T = np.hstack([R, Tras.reshape((3, 1))])
    T = np.vstack([T, [0, 0, 0, 1]])
    return T

if __name__ == "__main__":
    # load all the points and T
    x1 = np.loadtxt("resources/x1Data.txt")
    x2 = np.loadtxt("resources/x2Data.txt")
    x3 = np.loadtxt("resources/x3Data.txt")
    T_w_c1 = np.loadtxt("resources/T_w_c1.txt")
    T_w_c2 = np.loadtxt("resources/T_w_c2.txt")
    T_w_c3 = np.loadtxt("resources/T_w_c3.txt")
    x_3d = np.loadtxt("resources/X_w.txt")
    K_c = np.loadtxt("resources/K_c.txt")
    F_21 = np.loadtxt("resources/F_21.txt")
    image_1 = cv.imread("resources/image1.png")
    assert image_1 is not None, "Image 1 not found"
    image_2 = cv.imread("resources/image2.png")
    assert image_2 is not None, "Image 2 not found"
    image_3 = cv.imread("resources/image3.png")
    assert image_3 is not None, "Image 3 not found"

    # get a computed T_w_c2 from the points
    T_12 = np.linalg.inv(T_w_c1) @ T_w_c2
    # calculate E
    T_wc1 = T_w_c1
    #T_wc1 = np.eye(4)
    E = K_c.T @ F_21 @ K_c 
    X_w_comp, T_w_c2_comp = ExtractPose(E, K_c, K_c, np.linalg.inv(T_wc1), x1, x2) 
  
    #Plot the 3D 
    fig3D = plt.figure(1)
    ax = plt.axes(projection="3d", adjustable="box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    drawRefSystem(ax, T_wc1, "-", "C1")    
    drawRefSystem(ax, T_w_c2_comp, "-", "C2") 
    ax.scatter(X_w_comp[0, :], X_w_comp[1, :], X_w_comp[2, :], marker=".") 
    #ax.scatter(x_3d[0, :], x_3d[1, :], x_3d[2, :], marker=".")
    # Matplotlib does not correctly manage the axis('equal')
    xFakeBoundingBox = np.linspace(0, 4, 2)
    yFakeBoundingBox = np.linspace(0, 4, 2)
    zFakeBoundingBox = np.linspace(0, 4, 2)
    plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, "w.")
    plt.draw()
    print('Click in the image to continue...')
    plt.waitforbuttonpress()
    

    # 2_1 #
        
    # we will use the computed R and t to get the other T_wc's
    T_c1c2 = np.linalg.inv(T_wc1)@T_w_c2_comp

    #Project the 3D points to the 2D image
    x1_p = K_c @ np.eye(3, 4) @ np.linalg.inv(T_wc1) @ X_w_comp
    x2_p = K_c @ np.eye(3, 4) @ np.linalg.inv(T_w_c2_comp) @ X_w_comp
    x1_p /= x1_p[2, :]
    x2_p /= x2_p[2, :]

    # Plot the 2D points
    plot2Ddistances(2,image_1,x1,x1_p,"Image 1")
    plot2Ddistances(3,image_2,x2,x2_p,"Image 2")
   
    # We make the OP list with 3 angles that describe the rotatation R_C1_C2, 
    # two angles that describe the traslation T_C1_C2 and the 3D points
    theta_rot,tras = Parametrice_Pose(T_c1c2) 
    
    X_C1 = np.linalg.inv(T_wc1)@X_w_comp
    #We make the Op list
    Op = np.hstack([np.array(theta_rot).flatten(), np.array(tras).flatten(), X_C1[0:3,:].flatten()])
   
    #We do the optimization
    OpOptim = scOptim.least_squares(resBundleProjection, Op, args=(x1, x2, K_c, X_w_comp.shape[1]))    
   
    #From the optimization we get the OpOptim vector
    OpOptim = OpOptim.x 

    #We get the rotation and traslation from the OpOptim vector
    T_C1c2_optim = ObtainPose(OpOptim[0:3],OpOptim[3],OpOptim[4])
    T_wC2_optim = T_wc1@T_C1c2_optim 

    # we need to get the 3D points from the Op vector
    X_C1_op = OpOptim[5:].reshape((3, X_w_comp.shape[1]))

    # add a 1 for the homogeneous coordinates
    X_C1_op = np.vstack([X_C1_op, np.ones((1, X_w_comp.shape[1]))])
    X_w_op = T_wc1@X_C1_op 

    
    #Plot the 3D
    fig3D = plt.figure(4)
    ax = plt.axes(projection="3d", adjustable="box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    drawRefSystem(ax, T_wc1, "-", "C1")
    #drawRefSystem(ax, T_w_c2, "-", "C2_truth")
    drawRefSystem(ax, T_w_c2_comp, "-", "C2")
    drawRefSystem(ax, T_wC2_optim, "-", "C2_optim")
    ax.scatter(X_w_comp[0, :], X_w_comp[1, :], X_w_comp[2, :], marker=".")    
    ax.scatter(X_w_op[0, :],X_w_op[1, :], X_w_op[2, :], marker=".")  
    #ax.scatter(x_3d[0, :], x_3d[1, :], x_3d[2, :], marker=".")
    # Matplotlib does not correctly manage the axis('equal')
    xFakeBoundingBox = np.linspace(0, 4, 2)
    yFakeBoundingBox = np.linspace(0, 4, 2)
    zFakeBoundingBox = np.linspace(0, 4, 2)
    plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, "w.")
    plt.draw()
    print('Click in the image to continue...')
    plt.waitforbuttonpress()
       
    
    
    #Proyect the optim 3D points to the 2D image 
    x1_p = K_c @ np.eye(3, 4) @ np.linalg.inv(T_wc1) @ X_w_op
    x2_p = K_c @ np.eye(3, 4) @ np.linalg.inv(T_wC2_optim) @ X_w_op
    x1_p /= x1_p[2, :]
    x2_p /= x2_p[2, :]

    #Plot the 2D points
    plot2Ddistances(5,image_1,x1,x1_p,"Image 1")
    plot2Ddistances(6,image_2,x2,x2_p,"Image 2")
    
    

    #3
    #define imagePoints
    imagePoints = np.ascontiguousarray(x3[0:2, :].T).reshape((x3.shape[1], 1, 2))    

    #define objectPoints
    objectPoints = np.ascontiguousarray(X_C1_op[0:3,:].T).reshape((x3.shape[1], 1, 3))
 
    #define distortion coefficients
    distCoeffs = np.zeros((4, 1))

    #Pose estimation of camera 3
    retval, rvec, tvec = cv.solvePnP(objectPoints, imagePoints, K_c, 0,flags=cv.SOLVEPNP_EPNP)
    
    #Rotation     
    R3 = scAlg.expm(crossMatrix(rvec))

    #Optain pose
    T_C3C1 = np.hstack([R3, tvec])
    T_C3C1 = np.vstack([T_C3C1, [0, 0, 0, 1]])
    T_C1C3 = np.linalg.inv(T_C3C1)
    T_wC3 = T_wc1@T_C1C3

    #Plot the 3D
    fig3D = plt.figure(7)
    ax = plt.axes(projection="3d", adjustable="box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    drawRefSystem(ax, T_wc1, "-", "C1")
    drawRefSystem(ax, T_wC2_optim, "-", "C2_optim")
    drawRefSystem(ax, T_wC3, "-", "C3")
    ax.scatter(X_w_op[0, :],X_w_op[1, :], X_w_op[2, :], marker=".")  
    #ax.scatter(x_3d[0, :], x_3d[1, :], x_3d[2, :], marker=".")  
    #Matplotlib does not correctly manage the axis('equal')
    xFakeBoundingBox = np.linspace(0, 4, 2)
    yFakeBoundingBox = np.linspace(0, 4, 2)
    zFakeBoundingBox = np.linspace(0, 4, 2)
    plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, "w.")
    plt.draw()
    print('Click in the image to continue...')
    plt.waitforbuttonpress()      

    x1_p = K_c @ np.eye(3, 4) @ np.linalg.inv(T_wc1) @ X_w_op
    x2_p = K_c @ np.eye(3, 4) @ np.linalg.inv(T_wC2_optim) @ X_w_op
    x3_p = K_c @ np.eye(3, 4) @ np.linalg.inv(T_wC3) @ X_w_op
    x1_p /= x1_p[2, :]
    x2_p /= x2_p[2, :]
    x3_p /= x3_p[2, :]

    #Plot the 2D points
    plot2Ddistances(8,image_1,x1,x1_p,"Image 1")
    plot2Ddistances(9,image_2,x2,x2_p,"Image 2")
    plot2Ddistances(10,image_3,x3,x3_p,"Image 3")

    #Bandel Adjustment      
    #Make the Op list
    theta_rot2,tras2 = Parametrice_Pose(T_C1c2_optim)
    theta_rot3,tras3 = Parametrice_Pose(T_C1C3)  
    theta_rot = np.hstack([theta_rot2,theta_rot3])
    tras = np.hstack([tras2,tras3])
    Op = np.hstack([np.array(theta_rot).flatten(), np.array(tras).flatten(), X_C1_op[0:3,:].flatten()])
    
    #We do the optimization    
    x2_3 = np.hstack([x2,x3])
    OpOptim3 = scOptim.least_squares(resBundleProjection, Op, args=(x1, x2_3, K_c, X_C1.shape[1],3)) 
    OpOptim3 = OpOptim3.x 

    #We get the rotation and traslation from the OpOptim vector
    T_C1c2_optim3 = ObtainPose(OpOptim3[0:3],OpOptim3[6],OpOptim3[7])
    T_wC2_optim3 = T_wc1@T_C1c2_optim3 
    T_C1c3_optim = ObtainPose(OpOptim3[3:6],OpOptim3[8],OpOptim3[9])
    T_wC3_optim = T_wc1@T_C1c3_optim
   
    # we need to get the 3D points from the Op vector
    X_C1_op3 = OpOptim3[10:].reshape((3, X_C1.shape[1]))
    # add a 1 for the homogeneous coordinates
    X_C1_op3 = np.vstack([X_C1_op3, np.ones((1, X_C1.shape[1]))])
    X_w_op3 = T_wc1@X_C1_op3  

    fig3D = plt.figure(11)
    ax = plt.axes(projection="3d", adjustable="box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    drawRefSystem(ax, T_wc1, "-", "C1")    
    #drawRefSystem(ax, T_wC2_optim, "-", "C2")
    drawRefSystem(ax, T_wC2_optim3, "-", "C2_optim3")
    drawRefSystem(ax, T_wC3, "-", "C3")
    drawRefSystem(ax, T_wC3_optim, "-", "C3_optim")
    ax.scatter(X_w_op[0, :],X_w_op[1, :], X_w_op[2, :], marker=".")    
    ax.scatter(X_w_op3[0, :],X_w_op3[1, :], X_w_op3[2, :], marker=".") 
    #ax.scatter(x_3d[0, :], x_3d[1, :], x_3d[2, :], marker=".")    
    # Matplotlib does not correctly manage the axis('equal')
    xFakeBoundingBox = np.linspace(0, 4, 2)
    yFakeBoundingBox = np.linspace(0, 4, 2)
    zFakeBoundingBox = np.linspace(0, 4, 2)
    plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, "w.")    
    plt.draw()
    print('Click in the image to continue...')
    plt.waitforbuttonpress() 

    # #Proyect the optim 3D points to the 2D image 
    x1_p = K_c @ np.eye(3, 4) @ np.linalg.inv(T_wc1) @ X_w_op3
    x2_p = K_c @ np.eye(3, 4) @ np.linalg.inv(T_wC2_optim3) @ X_w_op3
    x3_p = K_c @ np.eye(3, 4) @ np.linalg.inv(T_wC3_optim) @ X_w_op3
    x1_p /= x1_p[2, :]
    x2_p /= x2_p[2, :]
    x3_p /= x3_p[2, :]

    # # #Plot the 2D points
    plot2Ddistances(12,image_1,x1,x1_p,"Image 1")
    plot2Ddistances(13,image_2,x2,x2_p,"Image 2")
    plot2Ddistances(14,image_3,x3,x3_p,"Image 3")

    #Make the op list with the scale
    theta_rot3,tras3 = Parametrice_Pose(T_C1C3) 
    Op = np.hstack([np.array(theta_rot3).flatten(), np.array(tras3).flatten(), X_C1_op[0:3,:].flatten()])
    
    #We do the optimization    
    x2_3 = np.hstack([x2,x3]) 
    OpOptim3S = scOptim.least_squares(resBundleProjectionScale, Op, args=(x1, x2_3, K_c, X_C1.shape[1],T_12,3)) 
    OpOptim3S = OpOptim3S.x 
    
    #We get the rotation and traslation from the OpOptim vector
    T_C1c3_optimS = ObtainPose(OpOptim3S[0:3],OpOptim3S[3],OpOptim3S[4])
    T_wC3_optimS = T_wc1@T_C1c3_optimS
   
    # we need to get the 3D points from the Op vector
    X_C1_op3S = OpOptim3S[5:].reshape((3, X_C1.shape[1]))
    # add a 1 for the homogeneous coordinates
    X_C1_op3S = np.vstack([X_C1_op3S, np.ones((1, X_C1.shape[1]))])
    X_w_op3S = T_wc1@X_C1_op3S  

    fig3D = plt.figure(15)
    ax = plt.axes(projection="3d", adjustable="box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    drawRefSystem(ax, T_wc1, "-", "C1")
    #drawRefSystem(ax, T_w_c2, "-", "C2") 
    drawRefSystem(ax, T_wC2_optim, "-", "C2_optim")      
    drawRefSystem(ax, T_wC3, "-", "C3")
    #drawRefSystem(ax, T_wC3_optim, "-", "C3_optim")
    drawRefSystem(ax, T_wC3_optimS, "-", "C3_optimS")
    # ax.scatter(X_w_op[0, :],X_w_op[1, :], X_w_op[2, :], marker=".")    
    ax.scatter(X_w_op3[0, :],X_w_op3[1, :], X_w_op3[2, :], marker=".") 
    ax.scatter(X_w_op3S[0, :],X_w_op3S[1, :], X_w_op3S[2, :], marker=".") 
    ax.scatter(x_3d[0, :], x_3d[1, :], x_3d[2, :], marker=".")  
    # Matplotlib does not correctly manage the axis('equal')
    xFakeBoundingBox = np.linspace(0, 4, 2)
    yFakeBoundingBox = np.linspace(0, 4, 2)
    zFakeBoundingBox = np.linspace(0, 4, 2)
    plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, "w.")    
    plt.draw()
    print('Click in the image to continue...')
    plt.waitforbuttonpress() 


    # #Proyect the optim 3D points to the 2D image 
    x1_p = K_c @ np.eye(3, 4) @ np.linalg.inv(T_wc1) @ X_w_op3S
    x2_p = K_c @ np.eye(3, 4) @ np.linalg.inv(T_w_c2) @ X_w_op3S
    x3_p = K_c @ np.eye(3, 4) @ np.linalg.inv(T_wC3_optimS) @ X_w_op3S
    x1_p /= x1_p[2, :]
    x2_p /= x2_p[2, :]
    x3_p /= x3_p[2, :]

    # #Plot the 2D points
    plot2Ddistances(16,image_1,x1,x1_p,"Image 1")
    plot2Ddistances(17,image_2,x2,x2_p,"Image 2")
    
    plt.figure(18)
    plt.imshow(image_3, cmap="gray", vmin=0, vmax=255)
    plotResidual(x3, x3_p, "k-")
    plt.plot(x3_p[0, :], x3_p[1, :], "bo")
    plt.plot(x3[0, :], x3[1, :], "rx")
    plotNumberedImagePoints(x3[0:2, :], "r", 4)
    plt.title("Image 3")
    plt.draw()
    plt.show()

