"""
This is exercise 2.2
"""
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


from NCCTemplate import seed_estimation_NCC_single_point
from interpolationFunctions import numerical_gradient, int_bilineal
from plotNCCOpticalFlow import read_flo_file


def lucas_kanade(img0_gray, img1_gray, og_flow: np.array, points_selected: np.array, patch_half_size: int = 5):
    """
    This function implements the Lucas Kanade method for optical flow estimation.
    https://en.wikipedia.org/wiki/Lucas%E2%80%93Kanade_method
    :param img0_gray: first image in grayscale
    :param img1_gray: second image in grayscale
    :param og_flow: initial flow estimation
    :param points_selected: points to track
    :param patch_half_size: half size of the patch to track
    :return: refined flow estimation
    """
    # for each point, extract the patch around it
    # i0 is an array of patches (each patch is a 11x11 array)
    i0 = np.zeros((points_selected.shape[0], patch_half_size*2+1, patch_half_size*2+1))
    i1 = np.zeros((points_selected.shape[0], patch_half_size*2+1, patch_half_size*2+1))
    for i in range(0, points_selected.shape[0]):
        patch = img0_gray[points_selected[i, 1] - patch_half_size:points_selected[i, 1] + patch_half_size + 1,
             points_selected[i, 0] - patch_half_size:points_selected[i, 0] + patch_half_size + 1]
        i0[i, :, :] = patch
    # for each point, compute the partial derivatives
    # A = np.zeros((points_selected.shape[0], 2), dtype=float)
    A = numerical_gradient(img1_gray, points_selected)
    # iterate until u_increment is small enough
    u = og_flow
    u_increment = np.ones(u[0].shape)
    epsilon = 0.0001
    while (np.sqrt(np.sum(u_increment ** 2))) >= epsilon:
        # from current motion u, compute I1(x+u), i.e. the warped patch 0 on image 1.
        # use bilinear interpolation
        new_points = points_selected + u
        # i1 = int_bilineal(img1_gray, new_points)
        for i in range(0, points_selected.shape[0]):
            patch = int_bilineal(img1_gray, np.array([new_points[i]]))
            i1[i, :, :] = patch
        # compute the error between the patches
        error = i1 - i0
        # compute the vector b from the error between the patches
        b = np.zeros((points_selected.shape[0], 2), dtype=float)
        for i in range(0, points_selected.shape[0]):
            b[i, 0] = np.sum(error[i, :, :] * A[i, 0])
            b[i, 1] = np.sum(error[i, :, :] * A[i, 1])
        # compute the increment of u by solving A * u_increment = b
        u_increment = np.zeros((points_selected.shape[0], 2), dtype=float)
        for i in range(0, points_selected.shape[0]):
            u_increment[i, 0] = b[i, 0] / A[i, 0]
            u_increment[i, 1] = b[i, 1] / A[i, 1]
        # u_increment = np.linalg.solve(A, b)
        # update u
        u = u + u_increment
        print(f'New flow: {u}')
    return u



def read_image(filename: str, ):
    """
    Read image using opencv converting from BGR to RGB
    :param filename: name of the image
    :return: np matrix with the image
    """
    img = cv.imread(filename)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img

if __name__ == "__main__":
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
    print(f'The estimated flow is: {flow_est}')

    # now apply lucas kanade method to refine the flow solution we obtained via ncc
    flow_refined = lucas_kanade(img1_gray, img2_gray, flow_est, points_selected, template_size_half)
    print(flow_refined)

    flow_12 = read_flo_file("flow10.flo", verbose=True)
    flow_gt = flow_12[points_selected[:, 1].astype(int), points_selected[:, 0].astype(int)].astype(float)
    flow_est_sparse = flow_est # flow_est[points_selected[:, 1].astype(int), points_selected[:, 0].astype(int)]
    flow_est_sparse_norm = np.sqrt(np.sum(flow_est_sparse ** 2, axis=1))
    error_est_sparse = flow_est_sparse - flow_gt
    error_est_sparse_norm = np.sqrt(np.sum(error_est_sparse ** 2, axis=1))

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
        axs[1].text(points_selected[k, 0] + 5, points_selected[k, 1] + 5, '{:.2f}'.format(error_est_sparse_norm[k]),
                    color='r')
    axs[1].quiver(points_selected[:, 0], points_selected[:, 1], error_est_sparse[:, 0], error_est_sparse[:, 1], color='b',
               angles='xy', scale_units='xy', scale=0.05)

    axs[1].title.set_text('Error of the estimated flow with respect to GT')
    plt.show()

    flow_refined_sparse = flow_refined # flow_refined[points_selected[:, 1].astype(int), points_selected[:, 0].astype(int)]
    flow_refined_sparse_norm = np.sqrt(np.sum(flow_refined_sparse ** 2, axis=1))
    error_refined_sparse = flow_refined_sparse - flow_gt
    error_refined_sparse_norm = np.sqrt(np.sum(error_refined_sparse ** 2, axis=1))

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(img1)
    axs[0].plot(points_selected[:, 0], points_selected[:, 1], '+r', markersize=15)
    for k in range(points_selected.shape[0]):
        axs[0].text(points_selected[k, 0] + 5, points_selected[k, 1] + 5, '{:.2f}'.format(flow_refined_sparse_norm[k]), color='r')
    axs[0].quiver(points_selected[:, 0], points_selected[:, 1], flow_refined_sparse[:, 0], flow_refined_sparse[:, 1], color='b', angles='xy', scale_units='xy', scale=0.05)
    axs[0].title.set_text('Optical flow NCC')
    axs[1].imshow(img1)
    axs[1].plot(points_selected[:, 0], points_selected[:, 1], '+r', markersize=15)
    for k in range(points_selected.shape[0]):
        axs[1].text(points_selected[k, 0] + 5, points_selected[k, 1] + 5, '{:.2f}'.format(error_refined_sparse_norm[k]),
                    color='r')
    axs[1].quiver(points_selected[:, 0], points_selected[:, 1], error_refined_sparse[:, 0], error_refined_sparse[:, 1], color='b',
               angles='xy', scale_units='xy', scale=0.05)

    axs[1].title.set_text('Error of the refined flow with respect to GT')
    plt.show()