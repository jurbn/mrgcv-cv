#####################################################################################
#
# MRGCV Unizar - Computer vision - Laboratory 2
#
# Title: Line fitting with SVD
#
# Date: 15 September 2022
#
#####################################################################################
#
# Authors: Jesus Bermudez, Richard Elvira, Jose Lamarca, JMM Montiel
#
# Version: 1.0
#
#####################################################################################

import matplotlib.pyplot as plt
import numpy as np
import random
import scipy.linalg as scAlg

def drawLine(l,strFormat,lWidth):
    """
    Draw a line
    -input:
      l: image line in homogenous coordinates
      strFormat: line format
      lWidth: line width
    -output: None
    """
    # p_l_y is the intersection of the line with the axis Y (x=0)
    p_l_y = np.array([0, -l[2] / l[1]])
    # p_l_x is the intersection point of the line with the axis X (y=0)
    p_l_x = np.array([-l[2] / l[0], 0])
    # Draw the line segment p_l_x to  p_l_y
    plt.plot([p_l_y[0], p_l_x[0]], [p_l_y[1], p_l_x[1]], strFormat, linewidth=lWidth)


if __name__ == '__main__':
    np.set_printoptions(precision=4,linewidth=1024,suppress=True)

    # This is the ground truth
    l_GT = np.array([[2], [1], [-1500]])

    plt.figure(1)
    plt.plot([-100, 1800], [0, 0], '--k', linewidth=1)
    plt.plot([0, 0], [-100, 1800], '--k', linewidth=1)
    # Draw the line segment p_l_x to  p_l_y
    drawLine(l_GT, 'g-', 1)
    plt.draw()
    plt.axis('equal')

    print('Click in the image to continue...')
    plt.waitforbuttonpress()

    # Generating points lying on the line but adding perpendicular Gaussian noise
    l_GTNorm = l_GT/np.sqrt(np.sum(l_GT[0:2]**2, axis=0)) #Normalized the line with respect to the normal norm

    x_l0 = np.vstack((-l_GTNorm[0:2]*l_GTNorm[2],1))  #The closest point of the line to the origin
    plt.plot([0, x_l0[0]], [0, x_l0[1]], '-r')
    plt.draw()

    # mu = np.arange(-1000, 1000, 250)
    # noiseSigma = 15 #Standard deviation
    # xGT = x_l0 + np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]]) @ (l_GTNorm * mu)
    # x = xGT + np.diag([1, 1, 0]) @ np.random.normal(0, noiseSigma, (3, len(mu)))


    xGT = np.loadtxt('x2DGTLineFittingSVD.txt')
    x = np.loadtxt('x2DLineFittingSVD.txt')
    plt.plot(xGT[0, :], xGT[1, :], 'b.')
    plt.plot(x[0, :], x[1, :], 'rx')
    plt.draw()
    print('Click in the image to continue...')
    plt.waitforbuttonpress()

    # Fit the least squares solution of inliers using svd
    u, s, vh = np.linalg.svd(x.T)
    l_ls = vh[-1, :]  # solution!

    drawLine(l_ls, 'r--', 1)
    plt.draw()
    plt.waitforbuttonpress()

    # Project the points on the line using SVD
    s[2] = 0  # If all the points are lying on the line s[2] = 0, therefore we impose it
    xProjectedOnTheLine = (u @ scAlg.diagsvd(s, u.shape[0], vh.shape[0]) @ vh).T
    xProjectedOnTheLine /= xProjectedOnTheLine[2, :]

    plt.plot(xProjectedOnTheLine[0,:], xProjectedOnTheLine[1, :], 'bx')
    plt.show()
    print('End')