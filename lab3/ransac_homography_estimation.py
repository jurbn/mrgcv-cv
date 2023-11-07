"""
This script performs homography estimation using RANSAC algorithm. It loads two images and their corresponding keypoints
and matches from a .npz file. Then, it randomly selects 4 matches and estimates the homography using these matches. It
then checks the number of inliers for this homography and repeats the process for a number of iterations. The homography
with the highest number of inliers is returned as the best homography. Finally, the script plots the 4 random matches on
the images and waits for a button press to continue.
"""
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

from utils.ransac import RANSAC
from utils.keypoints import load_superglue_matches, compute_SIFT_matches

if __name__ == "__main__":
    # load the images
    image_1 = cv.imread("lab3/image1.png")
    assert image_1 is not None and image_1.any(), "Image 1 not found"
    image_2 = cv.imread("lab3/image2.png")
    assert image_2 is not None and image_2.any(), "Image 2 not found"

    print(f'Computing Homography for Superglue matches')
    match_pairs, keypoints_0, keypoints_1 = load_superglue_matches('lab3/image1_image2_matches.npz')

    ransac = RANSAC()
    best_H, best_inlier_count, inlier_matches = ransac.compute_homography(match_pairs)
    print(f"Best homography:\n{best_H}")
    print(f"Number of inliers: {best_inlier_count}/{len(match_pairs)}")
    # now that we have the Homography, we can draw the plane using the inlier matches
    # first, get the coordinates of the inlier matches 
    # remember each match is a pair of coordinates (x0, x1), so we need to add a 1 to normalize them
    x0 = [match[0] for match in inlier_matches]
    x1 = [match[1] for match in inlier_matches]
    x0 = np.asarray(x0).T
    x1 = np.asarray(x1).T
    # now, compute the plane
    plane = np.dot(best_H, x1)
    # normalize the plane
    plane = plane / plane[2]
    # now plot the plane
    fig, ax = plt.subplots()
    ax.imshow(image_2)
    ax.scatter(plane[0], plane[1], color='r')
    plt.show()


    # do the same but dor SIFT matches
    print(f'Computing Homography for SIFT matches')
    match_pairs, keypoints_0, keypoints_1 = compute_SIFT_matches(image_1, image_2)
    best_H, best_inlier_count, inlier_matches = ransac.compute_homography(match_pairs)
    print(f"Best homography:\n{best_H}")
    print(f"Number of inliers: {best_inlier_count}/{len(match_pairs)}")

