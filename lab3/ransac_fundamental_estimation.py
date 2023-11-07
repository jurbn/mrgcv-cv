import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from utils.ransac import RANSAC
from utils.keypoints import load_superglue_matches, compute_SIFT_matches

def compute_epipolar_lines(F_matrix, inlier_matches):
    epipolar_lines_2_1 = []
    for match in inlier_matches:
        x0 = np.array([match[0][0], match[0][1], 1])
        x1 = np.array([match[1][0], match[1][1], 1])
        epipolar_line = np.dot(F_matrix, x1)
        epipolar_lines_2_1.append(epipolar_line)
    epipolar_lines_1_2 = []
    for match in inlier_matches:
        x0 = np.array([match[0][0], match[0][1], 1])
        x1 = np.array([match[1][0], match[1][1], 1])
        epipolar_line = np.dot(F_matrix.T, x0)
        epipolar_lines_1_2.append(epipolar_line)
    return epipolar_lines_2_1, epipolar_lines_1_2

def plot_epipolar_lines(image_1, image_2, epipolar_lines_2_1, epipolar_lines_1_2):    
    # now plot the lines
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(image_1)
    axs[1].imshow(image_2)
    for i in range(len(epipolar_lines_2_1)):
        line = epipolar_lines_2_1[i]
        # get the x and y coordinates for the line
        x = np.array([0, image_1.shape[1]])
        y = (-line[2] - line[0] * x) / line[1]
        axs[0].plot(x, y, color='r')
    for i in range(len(epipolar_lines_1_2)):
        line = epipolar_lines_1_2[i]
        # get the x and y coordinates for the line
        x = np.array([0, image_2.shape[1]])
        y = (-line[2] - line[0] * x) / line[1]
        axs[1].plot(x, y, color='r')
    plt.show()

if __name__ == "__main__":
    # load the images
    image_1 = cv.imread("lab3/image1.png")
    assert image_1 is not None and image_1.any(), "Image 1 not found"
    image_2 = cv.imread("lab3/image2.png")
    assert image_2 is not None and image_2.any(), "Image 2 not found"

    print(f'Computing Fundamental matrix for Superglue matches')
    match_pairs, keypoints_0, keypoints_1 = load_superglue_matches('lab3/image1_image2_matches.npz')

    ransac = RANSAC()
    best_F, best_inlier_count, inlier_matches = ransac.compute_fundamental_matrix(match_pairs)
    print(f"Best fundamental matrix:\n{best_F}")
    print(f"Number of inliers: {best_inlier_count}/{len(match_pairs)}")
    # now that I have a fundamental matrix, let's draw the epipolar lines
    # first get the epipolar lines for every inlier
    ep_lines_1, ep_lines_2 = compute_epipolar_lines(best_F, inlier_matches)
    plot_epipolar_lines(image_1, image_2, ep_lines_1, ep_lines_2)


    # do the same but for SIFT matches
    print(f'Computing Fundamental matrix for SIFT matches')
    match_pairs, keypoints_0, keypoints_1 = compute_SIFT_matches(image_1, image_2)
    best_F, best_inlier_count, inlier_matches = ransac.compute_fundamental_matrix(match_pairs)
    print(f"Best fundamental matrix:\n{best_F}")
    print(f"Number of inliers: {best_inlier_count}/{len(match_pairs)}")
    # now that I have a fundamental matrix, let's draw the epipolar lines
    # first get the epipolar lines for every inlier
    ep_lines_1, ep_lines_2 = compute_epipolar_lines(best_F, inlier_matches)
    plot_epipolar_lines(image_1, image_2, ep_lines_1, ep_lines_2)
