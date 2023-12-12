"""
This is exercise 2.2
"""
import numpy as np
import cv2 as cv


from NCCTemplate import seed_estimation_NCC_single_point

def lucas_kanade(og_flow):
    epsilon = 0.01
    u = og_flow
    u_increment = np.ones(u[0].shape)
    while (np.sqrt(np.sum(u_increment ** 2))) >= epsilon:
        # 1. from current motion u, compute i1(xi+u)
        # 2. compute the error between patches (ei = i1(xi+u)-i0(xi)=it)
        # 3. compu


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