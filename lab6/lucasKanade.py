"""
This is exercise 2.2
"""
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


from NCCTemplate import seed_estimation_NCC_single_point
from interpolationFunctions import numerical_gradient, int_bilineal
from plotNCCOpticalFlow import read_flo_file


def lucas_kanade(img0_gray, img1_gray, og_flow: np.array, points_selected: np.array, patch_half_size: int = 2):
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
    # patches_0_coord is an array of patches (each patch is a 11x11 array)
    epsilon = 0.0001
    patch_size = (patch_half_size*2+1)*(patch_half_size*2+1)
    refined_flow = np.zeros(og_flow.shape, dtype=float)

    # for each point, compute the flow
    for i in range(0, points_selected.shape[0]):
        print(f'Point {i}:\n{points_selected[i]}')

        # first, we need a patch on the first image
        center_pixel = points_selected[i, :]
        x_min = center_pixel[0] - patch_half_size
        y_min = center_pixel[1] - patch_half_size
        patch_0_coord = np.zeros((patch_half_size*2+1, patch_half_size*2+1, 2), dtype=int)
        for j in range(0, patch_half_size*2+1):
            for k in range(0, patch_half_size*2+1):
                patch_0_coord[j, k, 0] = x_min + j
                patch_0_coord[j, k, 1] = y_min + k
        patch_0_coord = patch_0_coord.reshape(patch_size, 2)
        patch_0_coord_swap = [[point_y, point_x] for point_x, point_y in patch_0_coord]
        patch_0_coord_swap = np.array(patch_0_coord_swap, dtype=float)
        patch_0 = int_bilineal(img0_gray, patch_0_coord_swap)

        delta_u = np.ones(og_flow[0].shape)
        # we will calculate the flow for every point between the `points_selected`
        u = og_flow[i]
        # we need to adapt the patch to `numrical_gradient` function
        patch_0_coord_swap = [[point_y, point_x] for point_x, point_y in patch_0_coord]
        patch_0_coord_swap = np.array(patch_0_coord_swap, dtype=float)
        gradients = numerical_gradient(img0_gray, patch_0_coord_swap)    # array of Ix and Iy for each pixel in the patch
        # restore the patch_0_coord to its original shape
        A = np.zeros((2, 2), dtype=float)
        b = np.zeros((2, 1), dtype=float)
        A[0, 0] = np.sum(gradients[:, 0] ** 2)
        A[0, 1] = np.sum(gradients[:, 0] * gradients[:, 1])
        A[1, 0] = np.sum(gradients[:, 0] * gradients[:, 1])
        A[1, 1] = np.sum(gradients[:, 1] ** 2)
        print(f'A:\n{A}')
        # check A is invertible
        assert np.linalg.det(A) != 0

        while (np.sqrt(np.sum(delta_u ** 2))) >= epsilon:
            # compute patch_1_coord by using the current motion u and the patch_0_coord
            # apply the motion to the patch_0_coord
            patch_1_coord = patch_0_coord + u
            patch_1_coord_swap = [[point_y, point_x] for point_x, point_y in patch_1_coord]
            patch_1_coord_swap = np.array(patch_1_coord_swap, dtype=float)
            patch_1 = int_bilineal(img1_gray, patch_1_coord_swap)
            # compute the error between the patches
            error = patch_1 - patch_0
            # compute the vector b from the error between the patches and the gradients
            print(f'gradients:\n{gradients[0]}')
            # for each pixel in the error, we need to multiply it by the gradient of the pixel
            b[0] = np.sum(error * gradients[:, 0])
            b[1] = np.sum(error * gradients[:, 1])
            b = -b
            print(f'b:\n{b}')
            # compute the increment of u by solving A * delta_u = b
            delta_u = np.linalg.solve(A, b)
            delta_u = delta_u.reshape((2,))
            print(f'delta_u:\n{delta_u}')
            # update u
            u = u + delta_u
            print(f'u:\n{u}')
            # assert False
        refined_flow[i] = u
    return refined_flow



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
    print(f'The estimated flow is:\n{flow_est}')

    # now apply lucas kanade method to refine the flow solution we obtained via ncc
    flow_refined = lucas_kanade(img1_gray, img2_gray, flow_est, points_selected, template_size_half)
    print(flow_refined)

    flow_12 = read_flo_file("flow10.flo", verbose=True)
    flow_gt = flow_12[points_selected[:, 1].astype(int), points_selected[:, 0].astype(int)].astype(float)
    flow_est_sparse = flow_est # flow_est[points_selected[:, 1].astype(int), points_selected[:, 0].astype(int)]
    flow_est_sparse_norm = np.sqrt(np.sum(flow_est_sparse ** 2, axis=1))
    error_est_sparse = flow_est_sparse - flow_gt
    error_est_sparse_norm = np.sqrt(np.sum(error_est_sparse ** 2, axis=1))

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(img1)
    axs[0, 0].plot(points_selected[:, 0], points_selected[:, 1], '+r', markersize=15)
    for k in range(points_selected.shape[0]):
        axs[0, 0].text(points_selected[k, 0] + 5, points_selected[k, 1] + 5, '{:.2f}'.format(flow_est_sparse_norm[k]), color='r')
    axs[0, 0].quiver(points_selected[:, 0], points_selected[:, 1], flow_est_sparse[:, 0], flow_est_sparse[:, 1], color='b', angles='xy', scale_units='xy', scale=0.05)
    axs[0, 0].title.set_text('Optical flow NCC')
    axs[0, 1].imshow(img1)
    axs[0, 1].plot(points_selected[:, 0], points_selected[:, 1], '+r', markersize=15)
    for k in range(points_selected.shape[0]):
        axs[0, 1].text(points_selected[k, 0] + 5, points_selected[k, 1] + 5, '{:.2f}'.format(error_est_sparse_norm[k]),
                    color='r')
    axs[0, 1].quiver(points_selected[:, 0], points_selected[:, 1], error_est_sparse[:, 0], error_est_sparse[:, 1], color='b',
               angles='xy', scale_units='xy', scale=0.05)

    axs[0, 1].title.set_text('Error of the estimated flow with respect to GT')
    # plt.show()

    flow_refined_sparse = flow_refined # flow_refined[points_selected[:, 1].astype(int), points_selected[:, 0].astype(int)]
    flow_refined_sparse_norm = np.sqrt(np.sum(flow_refined_sparse ** 2, axis=1))
    error_refined_sparse = flow_refined_sparse - flow_gt
    error_refined_sparse_norm = np.sqrt(np.sum(error_refined_sparse ** 2, axis=1))

    # fig, axs = plt.subplots(1, 2)
    axs[1, 0].imshow(img1)
    axs[1, 0].plot(points_selected[:, 0], points_selected[:, 1], '+r', markersize=15)
    for k in range(points_selected.shape[0]):
        axs[1, 0].text(points_selected[k, 0] + 5, points_selected[k, 1] + 5, '{:.2f}'.format(flow_refined_sparse_norm[k]), color='r')
    axs[1, 0].quiver(points_selected[:, 0], points_selected[:, 1], flow_refined_sparse[:, 0], flow_refined_sparse[:, 1], color='b', angles='xy', scale_units='xy', scale=0.05)
    axs[1, 0].title.set_text('Optical flow Lucas Kanade')
    axs[1, 1].imshow(img1)
    axs[1, 1].plot(points_selected[:, 0], points_selected[:, 1], '+r', markersize=15)
    for k in range(points_selected.shape[0]):
        axs[1, 1].text(points_selected[k, 0] + 5, points_selected[k, 1] + 5, '{:.2f}'.format(error_refined_sparse_norm[k]),
                    color='r')
    axs[1, 1].quiver(points_selected[:, 0], points_selected[:, 1], error_refined_sparse[:, 0], error_refined_sparse[:, 1], color='b',
               angles='xy', scale_units='xy', scale=0.05)

    axs[1, 1].title.set_text('Error of the refined flow with respect to GT')
    plt.show()