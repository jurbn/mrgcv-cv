import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def indexMatrixToMatchesList(matchesList):
    """
     -input:
         matchesList: nMatches x 3 --> [[indexDesc1,indexDesc2,descriptorDistance],...]]
     -output:
        dMatchesList: list of n DMatch object
     """
    dMatchesList = []
    for row in matchesList:
        dMatchesList.append(cv.DMatch(_queryIdx=row[0], _trainIdx=row[1], _distance=row[2]))
    return dMatchesList

def matchesListToIndexMatrix(dMatchesList):
    """
     -input:
         dMatchesList: list of n DMatch object
     -output:
        matchesList: nMatches x 3 --> [[indexDesc1,indexDesc2,descriptorDistance],...]]
     """
    matchesList = []
    for k in range(len(dMatchesList)):
        matchesList.append([int(dMatchesList[k].queryIdx), int(dMatchesList[k].trainIdx), dMatchesList[k].distance])
    return matchesList

def matchWith2NDRR(desc1, desc2, distRatio, minDist):
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
            # added distRatio to the condition
            matches.append([kDesc1, indexSort[0], dist1])
    return matches

def ComputeH(image_1,image_2,match_pairs,RANSAC_pixel_threshold,RANSAC_confidence,RANSAC_inlier_ratio):
    number_of_iterations = np.log(1 - RANSAC_confidence) / np.log(1 - RANSAC_inlier_ratio**4)
    number_of_iterations = int(number_of_iterations)
    print(f'The number of iterations is {number_of_iterations}')

    print(f'The number of matches is {len(match_pairs)}')
    best_H = None
    best_inlier_count = 0
    best_matches = None

    for i in range(number_of_iterations):
        # get a sample of 4 matches
        random_numbers = np.random.randint(0, len(match_pairs), 4)
        matches_sample = [match_pairs[random_number] for random_number in random_numbers]
        x0 = [match[0] for match in matches_sample]
        x1 = [match[1] for match in matches_sample]
        x0 = np.asarray(x0).T
        x1 = np.asarray(x1).T
        # given this 4 matches, get the homography
        Ec = np.empty([8, 9])
        for j in range(4):
            Ec[2*j,:] = [x0[0,j], x0[1,j], 1,
                        0, 0, 0,
                        -x1[0,j]*x0[0,j], -x1[0,j]*x0[1,j],-x1[0,j]]    
                            
            Ec[2*j+1,:] = [0, 0, 0,
                        x0[0,j], x0[1,j], 1,
                        -x1[1,j]*x0[0,j], -x1[1,j]*x0[1,j],-x1[1,j]]
        u, s, vh = np.linalg.svd(Ec)
        H = vh[-1, :]    
        H = H.reshape(3,3)
        # let the pixels vote for the homography
        inlier_count = 0
        x0_m = np.empty([0,2])        
        x1_m = np.empty([0,2])
        for match in match_pairs:
            #make array x0_m and x1_m empty
            x0p = match[0]
            x1p = match[1]
            x0p = np.append(x0p, 1)
            x1p = np.append(x1p, 1)
            x1p_hat = H @ x0p
            x1p_hat = x1p_hat / x1p_hat[2] 
            x0p_hat = np.linalg.inv(H) @ x1p
            x0p_hat = x0p_hat / x0p_hat[2] 

                   
            # check if the pixel is in the inlier set
            if ((np.linalg.norm(x1p - x1p_hat) < RANSAC_pixel_threshold) and (np.linalg.norm(x0p - x0p_hat) < RANSAC_pixel_threshold)):
                inlier_count += 1
                #add the match to the inlier set
                x0_m = np.append(x0_m, [match[0]], axis=0)
                x1_m = np.append(x1_m, [match[1]], axis=0)                           
              
                

        if inlier_count > best_inlier_count:
            print('New best H found')
            print(f'Inlier count: {inlier_count}')
            best_inlier_count = inlier_count
            best_H = H
            best_matches = matches_sample
            #make x0_m and x1_m homogeneous
            x0_m = x0_m.T
            x0_m = np.append(x0_m, np.ones((1, len(x0_m[0]))), axis=0)
            x1_m = x1_m.T
            x1_m = np.append(x1_m, np.ones((1, len(x1_m[0]))), axis=0)
            #copy the matches from x0_m and x1_m to x0_m_H and x1_m_H 
            x0_m_H = x0_m.copy()
            x1_m_H = x1_m.copy()
                        
            #parse x0 and x1 to cv.KeyPoint
            x0_H = [cv.KeyPoint(x0[0,i], x0[1,i], 1) for i in range(4)]
            x1_H = [cv.KeyPoint(x1[0,i], x1[1,i], 1) for i in range(4)]

            #make the matcheslist with x0 and x1 using indexMatrixToMatchesList and indexMatrixToMatchesList
            matchesList = []
            for k in range(4):
                matchesList.append([int(k), int(k), 0]) 
            dMatchesList = indexMatrixToMatchesList(matchesList)
            
            
            #plot the 4 random matches on the images with their inliners
            imgMatched = cv.drawMatches(image_1, x0_H, image_2, x1_H, dMatchesList[:100],
                                 None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS and cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            plt.figure(1)
            plt.clf()
            plt.imshow(imgMatched, cmap='gray', vmin=0, vmax=255)
            #plot x0_m in the first image and x1_m in the second image
            plt.plot(x0_m_H[0][:], x0_m_H[1][:], 'rx')
            plt.plot(x1_m_H[0][:]+750, x1_m_H[1][:], 'bx') 
            plt.draw
            plt.draw()
            plt.waitforbuttonpress()

    
    print(f'The best H is {best_H}')
    print(f'The number of inliers is {best_inlier_count}')
    
    imgMatched = cv.drawMatches(image_1, x0_H, image_2, x1_H, dMatchesList[:100],
                            None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS and cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.figure(2)
    plt.imshow(imgMatched, cmap='gray', vmin=0, vmax=255)
    #plot x0_m in the first image and x1_m in the second image
    plt.plot(x0_m_H[0][:], x0_m_H[1][:], 'rx')
    plt.plot(x1_m_H[0][:]+750, x1_m_H[1][:], 'bx') 
    plt.draw
    plt.draw()
    plt.waitforbuttonpress()

    return best_H

if __name__ == "__main__":
    np.set_printoptions(precision=4, linewidth=1024, suppress=True)
    # load the images
    image_1 = cv.imread('lab3/image1.png')
    image_2 = cv.imread('lab3/image2.png')
    # import the pairs .npz file
    matches_dict = np.load('lab3/image1_image2_matches.npz')
    keypoints_0 = matches_dict['keypoints0']
    keypoints_1 = matches_dict['keypoints1']
    matches = matches_dict['matches']
    match_confidence = matches_dict['match_confidence']

    
    # get the match pairs and remove the -1
    match_pairs = np.empty([len(matches)], dtype=object)
    for i in range(len(matches)):
        if matches[i] == -1:
            match_pairs[i] = None
        else:
            match_pairs[i] = [keypoints_0[i], keypoints_1[matches[i]]]
    
      
    match_pairs = [np.array(match_pairs[i]) for i in range(len(match_pairs)) if match_pairs[i] is not None]
      
     
    RANSAC_inlier_ratio = 0.7
    RANSAC_confidence = 0.999
    RANSAC_pixel_threshold = 2
    
    #H = ComputeH(image_1,image_2,match_pairs,RANSAC_pixel_threshold,RANSAC_confidence,RANSAC_inlier_ratio)
    
    #SIFT
    sift = cv.SIFT_create(nfeatures=0, nOctaveLayers = 5, contrastThreshold = 0.02, edgeThreshold = 20, sigma = 0.5)
    keypoints_sift_1, descriptors_1 = sift.detectAndCompute(image_1, None)
    keypoints_sift_2, descriptors_2 = sift.detectAndCompute(image_2, None)
    

    distRatio = 0.7
    minDist = 500
    matchesList = matchWith2NDRR(descriptors_1, descriptors_2, distRatio, minDist)
    dMatchesList = indexMatrixToMatchesList(matchesList)
    dMatchesList = sorted(dMatchesList, key=lambda x: x.distance)
    
    # Conversion from DMatches to Python list
    matchesList = matchesListToIndexMatrix(dMatchesList)

    # Matched points in numpy from list of DMatches
    srcPts = np.float32([keypoints_sift_1[m.queryIdx].pt for m in dMatchesList]).reshape(len(dMatchesList), 2)
    dstPts = np.float32([keypoints_sift_2[m.trainIdx].pt for m in dMatchesList]).reshape(len(dMatchesList), 2)

    #Make the list of match pairs 
    match_pairs_sift = np.empty([len(matchesList)], dtype=object)
    print(srcPts[0])
    for i in range(len(matchesList)):
        match_pairs_sift[i] = [srcPts[i], dstPts[i]] 
    H = ComputeH(image_1,image_2,match_pairs_sift,RANSAC_pixel_threshold,RANSAC_confidence,RANSAC_inlier_ratio)
    

        

