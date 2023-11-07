#####################################################################################
#
# MRGCV Unizar - Computer vision - Laboratory 2
#
# Title: Line RANSAC fitting
#
# Date: 28 September 2020
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

    mu = np.arange(-1000, 1000, 100)
    inliersSigma = 10 #Standard deviation of inliers
    xInliersGT = x_l0 + np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]]) @ (l_GTNorm * mu) + np.diag([1, 1, 0]) @ np.random.normal(0, inliersSigma, (3, len(mu)))
    nInliers = len(mu)

    # Generating uniformly random points as outliers
    nOutliers = 5
    xOutliersGT = np.diag([1, 1, 0]) @ (np.random.rand(3, 5)*3000-1500) + np.array([[0], [0], [1]])

    plt.plot(xInliersGT[0, :], xInliersGT[1, :], 'rx')
    plt.plot(xOutliersGT[0, :], xOutliersGT[1, :], 'bo')
    plt.draw()
    print('Click in the image to continue...')
    plt.waitforbuttonpress()

    x = np.hstack((xInliersGT, xOutliersGT))
    x = x[:, np.random.permutation(x.shape[1])] # Shuffle the points

    # parameters of random sample selection
    spFrac = nOutliers/nInliers  # spurious fraction
    P = 0.999  # probability of selecting at least one sample without spurious
    pMinSet = 2  # number of points needed to compute the fundamental matrix
    thresholdFactor = 1.96  # a point is spurious if abs(r/s)>factor Threshold

    # number m of random samples
    nAttempts = np.round(np.log(1 - P) / np.log(1 - np.power((1 - spFrac), pMinSet)))
    nAttempts = nAttempts.astype(int)
    print('nAttempts = ' + str(nAttempts))

    nElements = x.shape[1]

    RANSACThreshold = 3*inliersSigma
    nVotesMax = 0
    rng = np.random.default_rng()
    for kAttempt in range(nAttempts):

        # Compute the minimal set defining your model
        xSubSel = rng.choice(x.T, size=pMinSet, replace=False)
        l_model = np.reshape(np.cross(xSubSel[0], xSubSel[1]), (3, 1))

        normalNorm = np.sqrt(np.sum(l_model[0:2]**2, axis=0))

        l_model /= normalNorm
        # Computing the distance from the points to the model
        res = l_model.T @ x #Since I already have normalized the line with respect the normal the dot product gives the distance

        votes = np.abs(res) < RANSACThreshold  #votes
        nVotes = np.sum(votes) # Number of votes

        if nVotes > nVotesMax:
            nVotesMax = nVotes
            votesMax = votes
            l_mostVoted = l_model


    drawLine(l_mostVoted, 'b-', 1)
    plt.draw()
    plt.waitforbuttonpress()


    # Filter the outliers and fit the line
    iVoted = np.squeeze(np.argwhere(np.squeeze(votesMax)))
    xInliers = x[:, iVoted]

    plt.plot(xInliers[0, :], xInliers[1, :], 'y*')
    plt.draw()
    plt.waitforbuttonpress()

    # Fit the least squares solution of inliers using svd
    u, s, vh = np.linalg.svd(xInliers.T)
    l_ls = vh[-1, :]

    drawLine(l_ls, 'r--', 1)
    plt.draw()
    plt.waitforbuttonpress()

    # Project the points on the line using SVD
    s[2] = 0
    xInliersProjectedOnTheLine = (u @ scAlg.diagsvd(s, u.shape[0], vh.shape[0]) @ vh).T
    xInliersProjectedOnTheLine /= xInliersProjectedOnTheLine[2, :]

    plt.plot(xInliersProjectedOnTheLine[0,:], xInliersProjectedOnTheLine[1, :], 'bx')
    plt.draw()
    plt.waitforbuttonpress()
    print('End')