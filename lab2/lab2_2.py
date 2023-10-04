"""
This script completes the tasks given on section 2: "Fundamental matrix and Structure from Motion" from lab 2
Subject: Computer Vision, MRGCV
Author: Jorge Urbón, 777295
Date: 03/10/2023
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import plotData

if __name__ == '__main__':
    # load the images
    img1 = cv2.cvtColor(cv2.imread('image1.png'), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread('image2.png'), cv2.COLOR_BGR2RGB)
    # load the points
    x1 = np.loadtxt('x1Data.txt')
    x2 = np.loadtxt('x2Data.txt')
    # Implement a function for representing an epipolar line in image 2 given a clicked point on image 1. 
    # For checking your code you can use the testing fundamental matrix F_21_test.txt provided in the support files of
    # the practice.
    # load the fundamental matrix
    F = np.loadtxt('F_21_test.txt')
    # show image 1 and wait for a click
    plt.close('all')
    plt.figure(1)
    plt.imshow(img1)
    plt.title('Image 1')
    plt.draw()
    plt.waitforbuttonpress()
    point = plt.ginput(1)
    # plot the clicked point on image 1 and the epipolar line on image 2
    plt.plot(point[0][0], point[0][1], 'rx', markersize=10)
    plotData.plotNumberedImagePoints(x2, 'r', (10,0))
    plt.figure(2)
    plt.imshow(img2)
    plt.title('Image 2')
    # calculate the epipolar line
    epipolar_line = F @ np.append(point[0], 1)
    # plot the epipolar line
    # define the line from the corresponding points at the left and right borders of the image
    line = [(-epipolar_line[2] - epipolar_line[0] * 0) / epipolar_line[1],              # y = (-c - ax) / b
            (-epipolar_line[2] - epipolar_line[0] * img2.shape[1]) / epipolar_line[1]]  # y = (-c - ax) / b
    plt.plot([0, img2.shape[1]], line, 'g')
    plt.draw()
    plt.waitforbuttonpress()

    # 2.2
    # load the poses
    T_w_c1 = np.loadtxt('T_w_c1.txt')
    T_w_c2 = np.loadtxt('T_w_c2.txt')
    # compute the fundamental matrix given the poses
    F = np.linalg.inv(T_w_c2) @ T_w_c1