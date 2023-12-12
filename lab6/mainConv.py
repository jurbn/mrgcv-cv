#####################################################################################
#
# MRGCV Unizar - Computer vision - Laboratory 6
#
# Title: Convolution example
#
# Date: 22 November 2020
#
#####################################################################################
#
# Authors: Jesus Bermudez, Jose Lamarca,  Richard Elvira, JMM Montiel
#
# Version: 1.0
#
#####################################################################################


import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def convolutionFor(imgChannel, x, y, kernel):
    """
    Apply a convolution kernel to a pixel (no padding considerations)

    :param imgChannel: image input channel
    :param x: x coordinate of point to apply the convolution
    :param y: y coordinate of point to apply the convolution
    :param kernel: convolution kernel
    :return: convolution value at the pixel x,y
    """
    kernelHeightHalfSize = np.int(np.floor(kernel.shape[0] / 2))  # the size of the kernel must be odd
    kernelWidthHalfSize = np.int(np.floor(kernel.shape[1] / 2))  # the size of the kernel must be odd
    conv = 0
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            conv += imgChannel[y - kernelHeightHalfSize + i, x - kernelWidthHalfSize + j] * kernel[
                kernel.shape[0] - i - 1, kernel.shape[1] - j - 1]

    return conv


def convolutionRanges(imgChannel, x, y, kernel):
    """
    Apply a convolution kernel to a pixel (no padding considerations)

    :param imgChannel: image input channel
    :param x: x coordinate of point to apply the convolution
    :param y: y coordinate of point to apply the convolution
    :param kernel: convolution kernel
    :return: convolution value at the pixel x,y
    """
    iImgIni = np.int(y - np.floor(kernel.shape[0] / 2))  # the size of the kernel must be odd
    iImgEnd = np.int(y + np.floor(kernel.shape[0] / 2))  # the size of the kernel must be odd
    jImgIni = np.int(x - np.floor(kernel.shape[1] / 2))  # the size of the kernel must be odd
    jImgEnd = np.int(x + np.floor(kernel.shape[1] / 2))  # the size of the kernel must be odd
    imgPatch = imgChannel[iImgIni:iImgEnd + 1, jImgIni:jImgEnd + 1]
    conv = np.sum(imgPatch * kernel[::-1, ::-1])  #::-1 means inverse indexing.
    return conv


def convolutionPoints(imgChannel, x, y, kernel):
    """
    Apply a convolution kernel to a pixel (no padding considerations)

    NOTE: This implementation is not particularly fast for this task but
    it can hint you in other applications that can take advantage of this
    way of programming.

    :param imgChannel: image input channel
    :param x: x coordinate of point to apply the convolution
    :param y: y coordinate of point to apply the convolution
    :param kernel: convolution kernel
    :return: convolution value at the pixel x,y
    """

    iImgIni = np.int(y - np.floor(kernel.shape[0] / 2))  # the size of the kernel must be odd
    iImgEnd = np.int(y + np.floor(kernel.shape[0] / 2))  # the size of the kernel must be odd
    jImgIni = np.int(x - np.floor(kernel.shape[1] / 2))  # the size of the kernel must be odd
    jImgEnd = np.int(x + np.floor(kernel.shape[1] / 2))  # the size of the kernel must be odd

    xImgGrid, yImgGrid = np.meshgrid(np.arange(jImgIni, jImgEnd + 1), np.arange(iImgIni, iImgEnd + 1))
    xKernelGrid, yKenelGrid = np.meshgrid(np.arange(kernel.shape[1], 0, -1) - 1, np.arange(kernel.shape[0], 0, -1) - 1)
    conv = np.sum(imgChannel[yImgGrid.flatten().astype(int), xImgGrid.flatten().astype(int)] * kernel[
        yKenelGrid.flatten().astype(int), xKernelGrid.flatten().astype(int)])
    # NOTE: We are obtaining iImgIni and iImgEnd etc again so this doesn't make sense, but understanding what it is doing
    # may help you in other implementations
    # The astype here is also redundant this is just for showing two different ways of int conversion
    return conv


if __name__ == '__main__':
    np.set_printoptions(precision=4, linewidth=1024, suppress=True)

    img = cv.imread('frame10.png')
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    x = 256
    y = 376
    kernel = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    convFor = convolutionFor(imgGray, x, y, kernel)
    print(convFor)

    convRanges = convolutionRanges(imgGray, x, y, kernel)
    print(convRanges)

    convPoints = convolutionPoints(imgGray, x, y, kernel)
    print(convPoints)
