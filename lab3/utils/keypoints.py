import cv2 as cv
import numpy as np

def load_superglue_matches(file='lab3/superglue_matches.npz'):
    matches_dict = np.load("lab3/image1_image2_matches.npz")
    keypoints_0 = matches_dict["keypoints0"]
    keypoints_1 = matches_dict["keypoints1"]
    matches = matches_dict["matches"]
    match_confidence = matches_dict["match_confidence"]

    # get the match pairs and remove the -1
    match_pairs = np.empty([len(matches)], dtype=object)
    for i in range(len(matches)):
        if matches[i] == -1:
            match_pairs[i] = None
        else:
            match_pairs[i] = [keypoints_0[i], keypoints_1[matches[i]]]
    # remove the None elements
    match_pairs = [
        np.array(match_pairs[i])
        for i in range(len(match_pairs))
        if match_pairs[i] is not None
    ]
    return match_pairs, keypoints_0, keypoints_1

def matchWith2NDRR(desc1, desc2, kp1, kp2, distRatio, minDist):
    """
    Nearest Neighbours Matching algorithm checking the Distance Ratio.
    A match is accepted only if its distance is less than distRatio times
    the distance to the second match.
    -input:
        desc1: descriptors from image 1 nDesc x 128
        desc2: descriptors from image 2 nDesc x 128
        distRatio:
    -output:
       matches: nMatches x 3 --> [[indexDesc1,indexDesc2,descriptorDistance],...]]
    """
    matches = []
    nDesc1 = desc1.shape[0]
    for kDesc1 in range(nDesc1):
        dist = np.sqrt(np.sum((desc2 - desc1[kDesc1, :]) ** 2, axis=1))
        indexSort = np.argsort(dist)
        dist1 = dist[indexSort[0]]
        dist2 = dist[indexSort[1]]
        if (dist1 < distRatio * dist2) and (dist1 < minDist): 
            # if (dist1 < distRatio * dist2), get the pixel of the match on image 1 and image 2
            # and add them to the matches list
            match_coord_image_1 = kp1[kDesc1].pt
            # cast to to numpy array float32
            match_coord_image_1 = np.array(match_coord_image_1, dtype=np.float32)
            match_coord_image_2 = kp2[indexSort[0]].pt
            match_coord_image_2 = np.array(match_coord_image_2, dtype=np.float32)
            matches.append([match_coord_image_1, match_coord_image_2])
    return matches

def get_SIFT_keypoints(image):
    sift = cv.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    print(f'I found a total of {len(keypoints)} keypoints')
    return keypoints, descriptors

def compute_SIFT_matches(image_1, image_2):
    # get the keypoints and descriptors for both images
    keypoints_0, descriptors_0 = get_SIFT_keypoints(image_1)
    keypoints_1, descriptors_1 = get_SIFT_keypoints(image_2)
    # match via KNN
    matches = matchWith2NDRR(descriptors_0, descriptors_1, keypoints_0, keypoints_1, 0.6, 150)
    print(f'I found a total of {len(matches)} matches')
    # # get the match pairs
    # match_pairs = np.empty([len(matches)], dtype=object)
    # for i in range(len(matches)):
    #     match_pairs[i] = [keypoints_0[matches[i].queryIdx].pt, keypoints_1[matches[i].trainIdx].pt]
    return matches, keypoints_0, keypoints_1