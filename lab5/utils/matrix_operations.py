import numpy as np
import numpy as np

def crossMatrixInv(M):
    x = [M[2, 1], M[0, 2], M[1, 0]]
    return x

def crossMatrix(x):
    M = np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]], dtype="object")
    return M

def normalize3Dpoints(A):
    for i in range(4):
        A[i] = A[i]/A[3]
        
    return A

def triangulation(P1, P2, x1, x2):
    Ec = np.empty([4, 4])
    for i in range(4):
        Ec[i][0] = P1[2][i] * x1[0] - P1[0][i]
        Ec[i][1] = P1[2][i] * x1[1] - P1[1][i]
        Ec[i][2] = P2[2][i] * x2[0] - P2[0][i]
        Ec[i][3] = P2[2][i] * x2[1] - P2[1][i]
    u, s, vh = np.linalg.svd(Ec.T)
    p = vh[-1, :]   
    p = normalize3Dpoints(p)  # Implement normalize3Dpoints function
    return p

def extractPose(E,K1,K2,T_C1_W,x1,x2):
    """
    Inputs:
        E: Essential matrix
        K1: Camera calibration matrix of camera 1
        K2: Camera calibration matrix of camera 2
        T_C1_W: Transformation matrix from camera 1 to world frame
        x1: points in camera 1
        x2: points in camera 2

    Returns: point in 3D format array [4,n] and the pose of the camera 2 in the 
        world frame
    """
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


def indexMatrixToMatchesList(matchesList):
    """
    Convert a numpy matrix of index in a list of DMatch OpenCv matches.
     -input:
         matchesList: nMatches x 3 --> [[indexDesc1,indexDesc2,descriptorDistance],...]]
     -output:
        dMatchesList: list of n DMatch object
     """
    dMatchesList = []
    for row in matchesList:
        dMatchesList.append(cv.DMatch(_queryIdx=row[0].astype('int'), _trainIdx=row[1].astype('int'), _distance=row[2]))
    return dMatchesList


def matchesListToIndexMatrix(dMatchesList):
    """
    Convert a list of DMatch OpenCv matches into a numpy matrix of index.

     -input:
         dMatchesList: list of n DMatch object
     -output:
        matchesList: nMatches x 3 --> [[indexDesc1,indexDesc2,descriptorDistance],...]]
     """
    matchesList = []
    for k in range(len(dMatchesList)):
        matchesList.append([np.int(dMatchesList[k].queryIdx), np.int(dMatchesList[k].trainIdx), dMatchesList[k].distance])
    return matchesList