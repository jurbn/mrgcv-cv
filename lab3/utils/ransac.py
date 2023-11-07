import numpy as np
import cv2 as cv

class RANSAC:
    def __init__(self, inlier_ratio=0.5, confidence=0.999, pixel_threshold=2.0):
        self.inlier_ratio = inlier_ratio
        self.confidence = confidence
        self.pixel_threshold = pixel_threshold
        print(f"RANSAC object created with inlier ratio {inlier_ratio}, confidence {confidence} and pixel threshold {pixel_threshold}")

    def compute_homography(self, match_pairs):
        number_of_iterations = np.log(1 - self.confidence) / np.log(1 - self.inlier_ratio**4)
        number_of_iterations = int(number_of_iterations)
        print(f'The number of iterations to compute the homography is {number_of_iterations}')
        best_inlier_count = 0
        best_H = None
        best_matches = None
        for _ in range(number_of_iterations):
            # get a sample of 4 matches
            random_numbers = np.random.randint(0, len(match_pairs), 4)
            matches_sample = [
                match_pairs[random_number] for random_number in random_numbers
            ]
            x0 = [match[0] for match in matches_sample]
            x1 = [match[1] for match in matches_sample]
            x0 = np.asarray(x0).T
            x1 = np.asarray(x1).T
            # given this 4 matches, get the homography
            Ec = np.empty([8, 9])
            for j in range(4):
                Ec[2 * j, :] = [
                    x0[0, j],
                    x0[1, j],
                    1,
                    0,
                    0,
                    0,
                    -x1[0, j] * x0[0, j],
                    -x1[0, j] * x0[1, j],
                    -x1[0, j],
                ]
                Ec[2 * j + 1, :] = [
                    0,
                    0,
                    0,
                    x0[0, j],
                    x0[1, j],
                    1,
                    -x1[1, j] * x0[0, j],
                    -x1[1, j] * x0[1, j],
                    -x1[1, j],
                ]
            _, _, vh = np.linalg.svd(Ec)
            H = vh[-1, :]
            H = H.reshape(3, 3)
            # let the pixels vote for the homography
            inlier_count = 0
            temp_inlier_matches = []
            for match in match_pairs:
                x0p = match[0]
                x1p = match[1]
                x0p = np.append(x0p, 1)
                x1p = np.append(x1p, 1)
                x1p_hat = H @ x0p
                x1p_hat = x1p_hat / x1p_hat[2]
                # check if the pixel is in the inlier set
                if np.linalg.norm(x1p - x1p_hat) < self.pixel_threshold:
                    inlier_count += 1
                    temp_inlier_matches.append(match)
            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_H = H
                best_matches = matches_sample
                inlier_matches = temp_inlier_matches
        return best_H, best_inlier_count, inlier_matches

    def compute_fundamental_matrix(self, match_pairs):
        number_of_iterations = np.log(1 - self.confidence) / np.log(1 - self.inlier_ratio**8)
        number_of_iterations = int(number_of_iterations)
        print(f'The number of iterations to compute the fundamental matrix is {number_of_iterations}')
        best_F = None
        best_inlier_count = 0
        best_matches = None

        for i in range(number_of_iterations):
            # get a sample of 8 matches
            random_numbers = np.random.randint(0, len(match_pairs), 8)
            matches_sample = [match_pairs[random_number] for random_number in random_numbers]
            x0 = [match[0] for match in matches_sample]
            x1 = [match[1] for match in matches_sample]
            x0 = np.asarray(x0).T
            x1 = np.asarray(x1).T
            # given this 4 matches, get the homography
            Ec = np.empty([9,8])
            for i in range(8):
                Ec[:,i] = [x0[0][i]*x1[0][i], x0[1][i]*x1[0][i], x1[0][i],
                        x0[0][i]*x1[1][i], x0[1][i]*x1[1][i], x1[1][i],
                            x0[0][i], x0[1][i], 1] 
            u, s, vh = np.linalg.svd(Ec.T)    
            F = vh[-1,:]    
            F = F.reshape(3,3)
            u,s,vh=np.linalg.svd(F)    
            S = np.zeros([3,3])    
            S[0,0] = s[0]
            S[1,1] = s[1]    
            F = u@S@vh
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
                l1 = F@x0p
                l1 = l1 / np.linalg.norm(l1[:2])  
                l2 = F@x1p
                l2 = l2 / np.linalg.norm(l2[:2])
                #calculate the distance between the line l and x1p points
                distance1 = np.abs(np.dot(l1, x1p)/np.sqrt(l1[0]**2 + l1[1]**2))
                distance2 = np.abs(np.dot(l2, x0p)/np.sqrt(l2[0]**2 + l2[1]**2))

                # check if the pixel is in the inlier set
                if (distance1 < self.pixel_threshold and distance2 < self.pixel_threshold):
                    inlier_count += 1
                    #add the match to the inlier set
                    x0_m = np.append(x0_m, [match[0]], axis=0)
                    x1_m = np.append(x1_m, [match[1]], axis=0) 
            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_F = F
                best_matches = matches_sample
                
                #make the matcheslist with x0 and x1 using indexMatrixToMatchesList and indexMatrixToMatchesList
                matchesList = []
                for k in range(8):
                    matchesList.append([int(k), int(k), 0])
        return best_F, best_inlier_count, best_matches
