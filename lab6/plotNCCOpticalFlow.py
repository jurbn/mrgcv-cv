"""
This is exercise 2.1
"""

#####################################################################################
#
# MRGCV Unizar - Computer vision - Laboratory 4
#
# Title: Optical Flow
#
# Date: 22 November 2020
#
#####################################################################################
#
# Authors: Jose Lamarca, Jesus Bermudez, Richard Elvira, JMM Montiel
#
# Version: 1.0
#
#####################################################################################

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from NCCTemplate import seed_estimation_NCC_single_point


def read_image(filename: str, ):
    """
    Read image using opencv converting from BGR to RGB
    :param filename: name of the image
    :return: np matrix with the image
    """
    img = cv.imread(filename)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img


def read_flo_file(filename, verbose=False):
    """
    Read from .flo optical flow file (Middlebury format)
    :param flow_file: name of the flow file
    :return: optical flow data in matrix

    adapted from https://github.com/liruoteng/OpticalFlowToolkit/

    """
    f = open(filename, 'rb')
    magic = np.fromfile(f, np.float32, count=1)
    data2d = None

    if 202021.25 != magic:
        raise TypeError('Magic number incorrect. Invalid .flo file')
    else:
        w = np.fromfile(f, np.int32, count=1)
        h = np.fromfile(f, np.int32, count=1)
        if verbose:
            print("Reading %d x %d flow file in .flo format" % (h, w))
        data2d = np.fromfile(f, np.float32, count=int(2 * w * h))
        # reshape data into 3D array (columns, rows, channels)
        data2d = np.resize(data2d, (h[0], w[0], 2))
    f.close()
    return data2d


def draw_hsv(flow, scale):
    """
    Draw optical flow data (Middlebury format)
    :param flow: optical flow data in matrix
    :return: scale: scale for representing the optical flow
    adapted from https://github.com/npinto/opencv/blob/master/samples/python2/opt_flow.py
    """
    h, w = flow.shape[:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx * fx + fy * fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = ang * (180 / np.pi / 2)
    hsv[..., 1] = 255
    hsv[..., 2] = np.minimum(v * scale, 255)
    rgb = cv.cvtColor(hsv, cv.COLOR_HSV2RGB)
    return rgb


def generate_wheel(size):
    """
     Generate wheel optical flow for visualizing colors
     :param size: size of the image
     :return: flow: optical flow for visualizing colors
     """
    rMax = size / 2
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    u = x - size / 2
    v = y - size / 2
    r = np.sqrt(u ** 2 + v ** 2)
    u[r > rMax] = 0
    v[r > rMax] = 0
    flow = np.dstack((u, v))

    return flow


if __name__ == '__main__':
    np.set_printoptions(precision=4, linewidth=1024, suppress=True)
    unknownFlowThresh = 1e9

    flow_12 = read_flo_file("flow10.flo", verbose=True)
    binUnknownFlow = flow_12 > unknownFlowThresh

    img1 = read_image("frame10.png")
    img2 = read_image("frame11.png")

    img1_gray = cv.cvtColor(img1, cv.COLOR_RGB2GRAY)
    img2_gray = cv.cvtColor(img2, cv.COLOR_RGB2GRAY)


    # Adding random noise to the gt optical flow for plotting example
    # flow_est = flow_12 * np.bitwise_not(binUnknownFlow) + np.random.rand(flow_12.shape[0], flow_12.shape[1], flow_12.shape[2]) * 1.2 - 0.6


    # List of sparse points
    points_selected = np.loadtxt('points_selected.txt')
    points_selected = points_selected.astype(int)
    # we will calculate the flow by using our function on NCCTemplate.py
    template_size_half = 5
    searching_area_size: int = 15
    flow_est = np.zeros((points_selected.shape))
    for k in range(0,points_selected.shape[0]):
        i_flow, j_flow = seed_estimation_NCC_single_point(img1_gray, img2_gray, points_selected[k,1], points_selected[k,0], template_size_half, searching_area_size)
        flow_est[k,:] = np.hstack((j_flow,i_flow))
    print(flow_est)
    ## Sparse optical flow
    print(points_selected[:, 1].astype(int))
    print(points_selected[:, 0].astype(int))
    flow_gt = flow_12[points_selected[:, 1].astype(int), points_selected[:, 0].astype(int)].astype(float)
    flow_est_sparse = flow_est # flow_est[points_selected[:, 1].astype(int), points_selected[:, 0].astype(int)]
    flow_est_sparse_norm = np.sqrt(np.sum(flow_est_sparse ** 2, axis=1))
    error_sparse = flow_est_sparse - flow_gt
    error_sparse_norm = np.sqrt(np.sum(error_sparse ** 2, axis=1))


    # Plot results for sparse optical flow
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(img1)
    axs[0].plot(points_selected[:, 0], points_selected[:, 1], '+r', markersize=15)
    for k in range(points_selected.shape[0]):
        axs[0].text(points_selected[k, 0] + 5, points_selected[k, 1] + 5, '{:.2f}'.format(flow_est_sparse_norm[k]), color='r')
    axs[0].quiver(points_selected[:, 0], points_selected[:, 1], flow_est_sparse[:, 0], flow_est_sparse[:, 1], color='b', angles='xy', scale_units='xy', scale=0.05)
    axs[0].title.set_text('Optical flow NCC')
    axs[1].imshow(img1)
    axs[1].plot(points_selected[:, 0], points_selected[:, 1], '+r', markersize=15)
    for k in range(points_selected.shape[0]):
        axs[1].text(points_selected[k, 0] + 5, points_selected[k, 1] + 5, '{:.2f}'.format(error_sparse_norm[k]),
                    color='r')
    axs[1].quiver(points_selected[:, 0], points_selected[:, 1], error_sparse[:, 0], error_sparse[:, 1], color='b',
               angles='xy', scale_units='xy', scale=0.05)

    axs[1].title.set_text('Error with respect to GT')
    plt.show()