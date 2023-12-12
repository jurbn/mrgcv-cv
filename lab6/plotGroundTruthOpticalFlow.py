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


    # Adding random noise to the gt optical flow for plotting example
    flow_est = flow_12 * np.bitwise_not(binUnknownFlow) + np.random.rand(flow_12.shape[0], flow_12.shape[1], flow_12.shape[2]) * 1.2 - 0.6


    # List of sparse points
    points_selected = np.loadtxt('points_selected.txt')

    ## Sparse optical flow
    flow_gt = flow_12[points_selected[:, 1].astype(int), points_selected[:, 0].astype(int)].astype(float)
    flow_est_sparse = flow_est[points_selected[:, 1].astype(int), points_selected[:, 0].astype(int)]
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
    axs[0].title.set_text('Optical flow')
    axs[1].imshow(img1)
    axs[1].plot(points_selected[:, 0], points_selected[:, 1], '+r', markersize=15)
    for k in range(points_selected.shape[0]):
        axs[1].text(points_selected[k, 0] + 5, points_selected[k, 1] + 5, '{:.2f}'.format(error_sparse_norm[k]),
                    color='r')
    axs[1].quiver(points_selected[:, 0], points_selected[:, 1], error_sparse[:, 0], error_sparse[:, 1], color='b',
               angles='xy', scale_units='xy', scale=0.05)

    axs[1].title.set_text('Error with respect to GT')
    plt.show()


    ## Dense optical flow
    flow_error = flow_est - flow_12
    flow_error[binUnknownFlow] = 0
    error_norm = np.sqrt(np.sum(flow_error ** 2, axis=-1))


    # Plot results for dense optical flow
    scale = 40
    wheelFlow = generate_wheel(256)
    fig, axs = plt.subplots(2, 3)
    axs[0, 0].imshow(img1)
    axs[0, 0].title.set_text('image 1')
    axs[1, 0].imshow(img2)
    axs[1, 0].title.set_text('image 2')
    axs[0, 1].imshow(draw_hsv(flow_12 * np.bitwise_not(binUnknownFlow), scale))
    axs[0, 1].title.set_text('Optical flow ground truth')
    axs[1, 1].imshow(draw_hsv(flow_est, scale))
    axs[1, 1].title.set_text('LK estimated optical flow ')
    axs[0, 2].imshow(error_norm, cmap='jet')
    axs[0, 2].title.set_text('Optical flow error norm')
    axs[1, 2].imshow(draw_hsv(wheelFlow, 3))
    axs[1, 2].title.set_text('Color legend')
    axs[1, 2].set_axis_off()
    fig.subplots_adjust(hspace=0.5)
    plt.show()
