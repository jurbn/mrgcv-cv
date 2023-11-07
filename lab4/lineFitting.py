#####################################################################################
#
# MRGCV Unizar - Computer vision - Laboratory 3
#
# Title: Line fitting with least squares optimization
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

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as scOptim

def resLineFitting(Op,xData):
    """
    Residual function for least squares method
    -input:
      Op: vector containing the model parameters to optimize (in this case the line). This description should be minimal
      xData: the measurements of our model whose residual we want to calculate
    -output:
      res: vector of residuals to compute the loss
    """
    theta = Op[0]
    d = Op[1]
    l_model = np.vstack((np.cos(theta),np.sin(theta), -d))
    res = (l_model.T @ xData).squeeze() # Since the norm is unitary the distance is easier
    res = res.flatten()
    return res

def drawLine(l,strFormat,lWidth):
    """
    Draw a line
    -input:
      l: image line in homogenous coordinates
      strFormat: line format
      lWidth: line width
    -output: None
    """
    assert len(l.shape) ==1
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
    drawLine(l_GT.reshape((3)), 'g-', 1)
    plt.draw()
    plt.axis('equal')

    print('Click in the image to continue...')
    plt.waitforbuttonpress()

    # Generating points lying on the line but adding perpendicular Gaussian noise
    l_GTNorm = l_GT/np.sqrt(np.sum(l_GT[0:2]**2, axis=0)) #Normalized the line with respect to the normal norm

    x_l0 = np.vstack((-l_GTNorm[0:2]*l_GTNorm[2],1))  #The closest point of the line to the origin
    plt.plot([0, x_l0[0,0]], [0, x_l0[1,0]], '-r') # Fixed Warning in this line. This line 
    plt.draw()

    mu = np.arange(-1000, 1000, 100)
    inliersSigma = 30 #Standard deviation of inliers
    xInliersGT = x_l0 + np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]]) @ (l_GTNorm * mu) + np.diag([1, 1, 0]) @ np.random.normal(0, inliersSigma, (3, len(mu)))
    nInliers = len(mu)

    # Generating uniformly random points as outliers
    nOutliers = 2
    xOutliersGT = np.diag([1, 1, 0]) @ (np.random.rand(3, nOutliers)*3000-1500) + np.array([[0], [0], [1]])

    plt.plot(xInliersGT[0, :], xInliersGT[1, :], 'rx')
    plt.plot(xOutliersGT[0, :], xOutliersGT[1, :], 'bo')
    plt.draw()
    print('Click in the image to continue...')
    plt.waitforbuttonpress()

    xData = np.hstack((xInliersGT, xOutliersGT))
    xData = xData[:, np.random.permutation(xData.shape[1])] # Shuffle the points

    thetaGT = np.arctan2(l_GTNorm[1], l_GTNorm[0])
    dIniGT = -l_GTNorm[2]

    #Op = [thetaGT.squeeze(), dIni.dIniGT()]
    Op = [0, 0]
    res = resLineFitting(Op, xData)

    # Optimization with L2 norm and Levenberg-Marquardt
    OpOptim = scOptim.least_squares(resLineFitting, Op, args=(xData,), method='lm')

    # Optimization with Huber norm and Trust Region Reflective algorithm
    OpOptimHuber = scOptim.least_squares(resLineFitting, Op, args=(xData,), method='trf', jac='3-point', loss='huber')

    #Args could be more than one variable but always in a tuple e.j. args=(Data1,Data2,Data3)

    l_model_op = np.vstack((np.cos(OpOptim.x[0]), np.sin(OpOptim.x[0]), -OpOptim.x[1]))
    l_model_op_h = np.vstack((np.cos(OpOptimHuber.x[0]), np.sin(OpOptimHuber.x[0]), -OpOptimHuber.x[1]))

    drawLine(l_model_op.reshape((3)), 'r--', 1)
    drawLine(l_model_op_h.reshape((3)), 'b--', 1)
    plt.draw()
    plt.waitforbuttonpress()


    print('End')