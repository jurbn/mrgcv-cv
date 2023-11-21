import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import scipy.linalg as scAlg
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as scOptim


def plotResidual(x,xProjected,strStyle):
    """
        Plot the residual between an image point and an estimation based on a projection model.
         -input:
             x: Image points.
             xProjected: Projected points.
             strStyle: Line style.
         -output: None
         """

    for k in range(x.shape[1]):
        plt.plot([x[0, k], xProjected[0, k]], [x[1, k], xProjected[1, k]], strStyle)

def plotNumberedImagePoints(x,strColor,offset):
    """
        Plot indexes of points on a 2D image.
         -input:
             x: Points coordinates.
             strColor: Color of the text.
             offset: Offset from the point to the text.
         -output: None
         """
    for k in range(x.shape[1]):
        plt.text(x[0, k]+offset, x[1, k]+offset, str(k), color=strColor)

def plotNumbered3DPoints(ax, X,strColor, offset):
    """
        Plot indexes of points on a 3D plot.
         -input:
             ax: axis handle
             X: Points coordinates.
             strColor: Color of the text.
             offset: Offset from the point to the text.
         -output: None
         """
    for k in range(X.shape[1]):
        ax.text(X[0, k]+offset, X[1, k]+offset, X[2,k]+offset, str(k), color=strColor)


def crossMatrixInv(M):
    x = [M[2, 1], M[0, 2], M[1, 0]]
    return x


def crossMatrix(x):
    M = np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]], dtype="object")
    return M


def draw3DLine(ax, xIni, xEnd, strStyle, lColor, lWidth):
    """
    Draw a segment in a 3D plot
    -input:
        ax: axis handle
        xIni: Initial 3D point.
        xEnd: Final 3D point.
        strStyle: Line style.
        lColor: Line color.
        lWidth: Line width.
    """
    ax.plot([np.squeeze(xIni[0]), np.squeeze(xEnd[0])], [np.squeeze(xIni[1]), np.squeeze(xEnd[1])], [np.squeeze(xIni[2]), np.squeeze(xEnd[2])],
            strStyle, color=lColor, linewidth=lWidth)

def drawRefSystem(ax, T_w_c, strStyle, nameStr):
    """
        Draw a reference system in a 3D plot: Red for X axis, Green for Y axis, and Blue for Z axis
    -input:
        ax: axis handle
        T_w_c: (4x4 matrix) Reference system C seen from W.
        strStyle: lines style.
        nameStr: Name of the reference system.
    """
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 0:1], strStyle, 'r', 1)
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 1:2], strStyle, 'g', 1)
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 2:3], strStyle, 'b', 1)
    ax.text(np.squeeze( T_w_c[0, 3]+0.1), np.squeeze( T_w_c[1, 3]+0.1), np.squeeze( T_w_c[2, 3]+0.1), nameStr)

def plot2Ddistances(i,image,x1,x1_p,title):
    plt.figure(i)
    plt.imshow(image, cmap="gray", vmin=0, vmax=255)
    plotResidual(x1, x1_p, "k-")
    plt.plot(x1_p[0, :], x1_p[1, :], "bo")
    plt.plot(x1[0, :], x1[1, :], "rx")
    plotNumberedImagePoints(x1[0:2, :], "r", 4)
    plt.title(title)
    plt.draw()
    print('Click in the image to continue...')
    plt.waitforbuttonpress() 
    return 0


def ExtractPose(E,K1,K2,T_C1_W,x1,x2):
    '''Inputs:
       E: Essential matrix
       K1: Camera calibration matrix of camera 1
       K2: Camera calibration matrix of camera 2
       T_C1_W: Transformation matrix from camera 1 to world frame
       x1: points in camera 1
       x2: points in camera 2

       Returns: point in 3D format array [4,n] and the pose of the camera 2 in the 
       world frame'''
    Best_vote = 0
    u, s, vh = np.linalg.svd(E)    
    W = np.array([[0, -1, 0, ], [1, 0, 0], [0, 0, 1]])
    
    #Get posible rotations
    R_90 = u@W@vh  
    if (np.linalg.det(R_90)<0):
        R_90 = -R_90
        
    R_n90 = u@W.T@vh
    if (np.linalg.det(R_n90)<0):    
        R_n90 = -R_n90
      
    #get translations 
    A = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])  
    P1 = K1@A@T_C1_W 
    #Choose the correct answer and calculate P1 and P2
    for i in range(4):
        Pose = np.eye(4,4)  
        t = u[:,-1] 
        Rot = R_90 
        vote = 0            
        if(i%2 == 0):
            t = -(t)            
        if(i>1):
            Rot = R_n90
        Pose[:3,:3] = Rot
        Pose[:3,3] = t                            
        Pose = Pose@T_C1_W
        P2 = K2@A@Pose  
        p = np.empty([x1.shape[1],4])        
        for k in range(x1.shape[1]):
            p[:][k] = (triangulation(P1,P2,x1[:,k],x2[:,k]))
        p = p.T 
        p = T_C1_W@p
        p = normalize3Dpoints(p)        
        Pose = T_C1_W@Pose
        for k in range(x1.shape[1]):   
            #if a point is in front both cameras votes +1                     
            if(p[2,k] > 0 and p[2,k] > Pose[2,3]):
                    vote = vote + 1
        
        if (vote>Best_vote):                 
            Best_vote = vote
            Bestp = np.linalg.inv(T_C1_W)@p
            BestPose = np.linalg.inv(T_C1_W)@Pose
            BestPose = np.linalg.inv(BestPose)
           
    return Bestp, BestPose

def triangulation(P1,P2,x1,x2):
    Ec = np.empty([4,4])
    for i in range(4):
        Ec[i][0] = P1[2][i]*x1[0]-P1[0][i]
        Ec[i][1] = P1[2][i]*x1[1]-P1[1][i]
        Ec[i][2] = P2[2][i]*x2[0]-P2[0][i]
        Ec[i][3] = P2[2][i]*x2[1]-P2[1][i]
    u, s, vh = np.linalg.svd(Ec.T)
    p = vh[-1, :]   
    p = normalize3Dpoints(p)     
    return p

def normalize3Dpoints(A):
    for i in range(4):
        A[i] = A[i]/A[3]
        
    return A
