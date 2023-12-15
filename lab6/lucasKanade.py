"""
This is exercise 2.2
"""
import numpy as np
import cv2 as cv


from NCCTemplate import seed_estimation_NCC_single_point

def lucas_kanade(img0_gray, img1_gray, og_flow: np.array, points_selected: np.array, patch_half_size: int = 5):
    epsilon = 0.01
    u = og_flow
    u_increment = np.ones(u[0].shape)
    i0 = np.zeros((points_selected.shape[0]))   # THERE ARE THE PATCHES 0 ON IMAGE 1
    for i in range(0, points_selected.shape[0]):
        point = points_selected[i]
        patch = img1_gray[point[0] - patch_half_size:point[0] + patch_half_size + 1,
                    point[1] - patch_half_size:point[1] + patch_half_size + 1 ]
        i0 = np.vstack((i0, patch))
    i1 = np.zeros((points_selected.shape[0]))
    while (np.sqrt(np.sum(u_increment ** 2))) >= epsilon:
        # 1. from current motion u, compute i1(xi+u)
        # this is the warped patch 0 on image 1, use linear interpolation for this warping
        for i in range(0, points_selected.shape[0]):
            patch = i0[i]
            flow = u[i]
            # warp the patch to the new point
            patch = cv.remap(img1_gray, patch, flow, cv.INTER_LINEAR)   #FIXME: check if this is correct
            i1 = np.vstack((i1, patch))
        # 2. compute the error between patches (ei = i1(xi+u)-i0(xi)=it)
        error = i1 - img0_gray
        # 3. compute  the vector b from the error between patches and the gradients
        b = np.zeros((points_selected.shape[0]))
        for i in range(0, points_selected.shape[0]):
            point = points_selected[i]
            patch = error[point[0] - patch_half_size:point[0] + patch_half_size + 1,
                    point[1] - patch_half_size:point[1] + patch_half_size + 1]
            b = np.vstack((b, patch))
        # 4. solve A delta u = b
        # 5. update u = u + delta u
        # 6. repeat until delta u is small enough


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
    print(flow_est)

    # now apply lucas kanade method to refine the flow solution we obtained via ncc
    refined_flow = lucas_kanade(flow_est)