#####################################################################################
#
# MRGCV Unizar - Computer vision - Laboratory 4
#
# Title: Interpolation functions
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

def numerical_gradient(img_int: np.array, point: np.array)->np.array:
    """
    https://es.wikipedia.org/wiki/Interpolaci%C3%B3n_bilineal
    :param img:image to interpolate
    :param point: point subpixel
    :return: 
    """

    a = np.zeros((point.shape[0], 2), dtype= np.float)
    filter = np.array([-1, 0, 1], dtype=np.float)
    point_int = point.astype(np.int)
    img = img_int.astype(np.float)

    for i in range(0,point.shape[0]):
        py = img[point_int[i,0]-1:point_int[i,0]+2,point_int[i,1]].astype(np.float)
        px = img[point_int[i,0],point_int[i,1]-1:point_int[i,1]+2].astype(np.float)
        a[i, 0] = 1/2*np.dot(filter,px)
        a[i, 1] = 1/2*np.dot(filter,py)

    return a

def int_bilineal(img: np.array, point: np.array)->np.array:
    """
    https://es.wikipedia.org/wiki/Interpolaci%C3%B3n_bilineal
    Vq = scipy.ndimage.map_coordinates(img.astype(np.float), [point[:, 0].ravel(), point[:, 1].ravel()], order=1, mode='nearest').reshape((point.shape[0],))

    :param img:image to interpolate
    :param point: point subpixel
    :return: 
    """
    A = np.zeros((point.shape[0], 2, 2), dtype= np.float)
    point_lu = point.astype(np.int)
    point_ru = np.copy(point_lu)
    point_ru[:,1] = point_ru[:,1] + 1
    point_ld = np.copy(point_lu)
    point_ld[:, 0] = point_ld[:, 0] + 1
    point_rd = np.copy(point_lu)
    point_rd[:, 0] = point_rd[:, 0] + 1
    point_rd[:, 1] = point_rd[:, 1] + 1

    A[:, 0, 0] = img[point_lu[:,0],point_lu[:,1]]
    A[:, 0, 1] = img[point_ru[:,0],point_ru[:,1]]
    A[:, 1, 0] = img[point_ld[:,0],point_ld[:,1]]
    A[:, 1, 1] = img[point_rd[:,0],point_rd[:,1]]
    l_u = np.zeros((point.shape[0],1,2),dtype= np.float)
    l_u[:, 0, 0] = -((point[:,0]-point_lu[:,0])-1)
    l_u[:, 0, 1] = point[:,0]-point_lu[:,0]

    r_u = np.zeros((point.shape[0],2,1),dtype= np.float)
    r_u[:, 0, 0] = -((point[:,1]-point_lu[:,1])-1)
    r_u[:, 1, 0] = point[:, 1]-point_lu[:,1]
    grays = l_u @ A @ r_u

    return grays.reshape((point.shape[0],))