import numpy as np
import scipy.linalg as scAlg

from utils.matrix_operations import crossMatrix, crossMatrixInv

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
    posStartX = 5+(nCameras-2)*6
    # we need to get the rotation and traslation from the Op vector
    T_wc1 = np.eye(4)

    # we need to get the 3D points from the Op vector
    X_C1 = Op[posStartX:].reshape((3, nPoints))        
    # add a 1 for the homogeneous coordinates
    X_C1 = np.vstack([X_C1, np.ones((1, nPoints))])
    x1_p = K_c @ np.eye(3, 4) @ T_wc1 @ X_C1
    x1_p /= x1_p[2, :]

    res = []
    for j in range(nPoints):            
        res.append(x1Data[0, j] - x1_p[0, j])
        res.append(x1Data[1, j] - x1_p[1, j])

    posEndRot = 0
    EndTras = 0
    for i in range(nCameras-1):
        posStartRot = 0 + posEndRot
        posEndRot = posStartRot+3

        if(i==0): 
            theta_tras = Op[(nCameras-1)*3]
            phi_tras = Op[(nCameras-1)*3+1]
            Rot = Op[posStartRot:posEndRot] 
            T_c1c = ObtainPose(Rot,theta_tras,phi_tras)            

        else:
            StartTras = (nCameras-1)*3+2+(i-1)*3
            EndTras = StartTras+3
            Tras = Op[StartTras:EndTras]
            Rot = Op[posStartRot:posEndRot]    
            R = scAlg.expm(crossMatrix(Rot))
            T_c1c = np.hstack([R, Tras.reshape((3, 1))])
            T_c1c = np.vstack([T_c1c, [0, 0, 0, 1]])
                
        xC_p = K_c @ np.eye(3, 4) @ np.linalg.inv(T_c1c) @ X_C1 
        xC_p /= xC_p[2, :]
        res1 = []  
        for j in range(nPoints):
            res1.append(xCData[0, j+nPoints*i] - xC_p[0, j])
            res1.append(xCData[1, j+nPoints*i] - xC_p[1, j])
        res = np.hstack([res,res1])
        res = np.array(res).flatten()
    return res