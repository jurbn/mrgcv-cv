#####################################################################################
#
# MRGCV Unizar - Computer vision - Laboratory 3
#
# Title: Bundle Adjustment and Multiview Geometry
#
# Date: 26 October 2020
#
#####################################################################################
#
# Authors: Jesus Bermudez, Richard Elvira, Jose Lamarca, JMM Montiel
#
# Version: 1.0
#
#####################################################################################

from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import numpy as np
import cv2
import scipy.linalg as scAlg
import csv
import scipy as sc
import scipy.optimize as scOptim
import scipy.io as sio

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
        dMatchesList.append(cv2.DMatch(_queryIdx=row[0].astype('int'), _trainIdx=row[1].astype('int'), _distance=row[2]))
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

if __name__ == '__main__':
    np.set_printoptions(precision=4,linewidth=1024,suppress=True)


    # Load ground truth
    T_wc1 = np.loadtxt('lab4/T_w_c1.txt')
    T_wc2 = np.loadtxt('lab4/T_w_c2.txt')
    T_wc3 = np.loadtxt('lab4/T_w_c3.txt')
    K_c = np.loadtxt('lab4/K_c.txt')
    X_w = np.loadtxt('lab4/X_w.txt')

    x1Data = np.loadtxt('lab4/x1Data.txt')
    x2Data = np.loadtxt('lab4/x2Data.txt')
    x3Data = np.loadtxt('lab4/x3Data.txt')


    #Plot the 3D cameras and the 3D points
    fig3D = plt.figure(1)

    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    drawRefSystem(ax, np.eye(4, 4), '-', 'W')
    drawRefSystem(ax, T_wc1, '-', 'C1')
    drawRefSystem(ax, T_wc2, '-', 'C2')
    drawRefSystem(ax, T_wc3, '-', 'C3')

    ax.scatter(X_w[0, :], X_w[1, :], X_w[2, :], marker='.')
    plotNumbered3DPoints(ax, X_w, 'r', 0.1)

    #Matplotlib does not correctly manage the axis('equal')
    xFakeBoundingBox = np.linspace(0, 4, 2)
    yFakeBoundingBox = np.linspace(0, 4, 2)
    zFakeBoundingBox = np.linspace(0, 4, 2)
    plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, 'w.')
    print('Close the figure to continue. Left button for orbit, right button for zoom.')
    plt.show()


    #Read the images
    path_image_1 = 'lab4/image1.png'
    path_image_2 = 'lab4/image2.png'
    path_image_3 = 'lab4/image3.png'
    image_pers_1 = cv2.imread(path_image_1)
    image_pers_2 = cv2.imread(path_image_2)
    image_pers_3 = cv2.imread(path_image_3)


    # Construct the matches
    kpCv1 = []
    kpCv2 = []
    kpCv3 = []
    for kPoint in range(x1Data.shape[1]):
        kpCv1.append(cv2.KeyPoint(x1Data[0, kPoint], x1Data[1, kPoint],1))
        kpCv2.append(cv2.KeyPoint(x2Data[0, kPoint], x2Data[1, kPoint],1))
        kpCv3.append(cv2.KeyPoint(x3Data[0, kPoint], x3Data[1, kPoint],1))

    matchesList12 = np.hstack((np.reshape(np.arange(0, x1Data.shape[1]),(x2Data.shape[1],1)),
                                        np.reshape(np.arange(0, x1Data.shape[1]), (x1Data.shape[1], 1)),np.ones((x1Data.shape[1],1))))

    matchesList13 = matchesList12
    dMatchesList12 = indexMatrixToMatchesList(matchesList12)
    dMatchesList13 = indexMatrixToMatchesList(matchesList13)

    imgMatched12 = cv2.drawMatches(image_pers_1, kpCv1, image_pers_2, kpCv2, dMatchesList12,
                                   None,
                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS and cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    imgMatched13 = cv2.drawMatches(image_pers_1, kpCv1, image_pers_3, kpCv3, dMatchesList13,
                                   None,
                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS and cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    plt.figure(2)
    plt.imshow(imgMatched12)
    plt.title("{} matches between views 1 and 2".format(len(dMatchesList12)))
    plt.draw()

    plt.figure(3)
    plt.imshow(imgMatched13)
    plt.title("{} matches between views 1 and 3".format(len(dMatchesList13)))
    print('Close the figures to continue.')
    plt.show()

    # Project the points
    x1_p = K_c @ np.eye(3, 4) @ np.linalg.inv(T_wc1) @ X_w
    x2_p = K_c @ np.eye(3, 4) @ np.linalg.inv(T_wc2) @ X_w
    x3_p = K_c @ np.eye(3, 4) @ np.linalg.inv(T_wc3) @ X_w
    x1_p /= x1_p[2, :]
    x2_p /= x2_p[2, :]
    x3_p /= x3_p[2, :]


    # Plot the 2D points
    plt.figure(4)
    plt.imshow(image_pers_1, cmap='gray', vmin=0, vmax=255)
    plotResidual(x1Data, x1_p, 'k-')
    plt.plot(x1_p[0, :], x1_p[1, :], 'bo')
    plt.plot(x1Data[0, :], x1Data[1, :], 'rx')
    plotNumberedImagePoints(x1Data[0:2, :], 'r', 4)
    plt.title('Image 1')
    plt.draw()

    plt.figure(5)
    plt.imshow(image_pers_2, cmap='gray', vmin=0, vmax=255)
    plotResidual(x2Data, x2_p, 'k-')
    plt.plot(x2_p[0, :], x2_p[1, :], 'bo')
    plt.plot(x2Data[0, :], x2Data[1, :], 'rx')
    plotNumberedImagePoints(x2Data[0:2, :], 'r', 4)
    plt.title('Image 2')
    plt.draw()

    plt.figure(6)
    plt.imshow(image_pers_3, cmap='gray', vmin=0, vmax=255)
    plotResidual(x3Data, x3_p, 'k-')
    plt.plot(x3_p[0, :], x3_p[1, :], 'bo')
    plt.plot(x3Data[0, :], x3Data[1, :], 'rx')
    plotNumberedImagePoints(x3Data[0:2, :], 'r', 4)
    plt.title('Image 3')
    print('Close the figures to continue.')
    plt.show()
