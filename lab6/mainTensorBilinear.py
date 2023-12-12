#####################################################################################
#
# MRGCV Unizar - Computer vision - Laboratory 6
#
# Title: Bilinear map example
#
# Date: 22 November 2020
#
#####################################################################################
#
# Authors: Jesus Bermudez, Jose Lamarca,  Richard Elvira, JMM Montiel
#
# Version: 1.0
#
#####################################################################################

import numpy as np

if __name__ == '__main__':
    np.set_printoptions(precision=4, linewidth=1024, suppress=True)

    n = 3
    m = 10

    x = np.random.rand(m, n)
    y = np.random.rand(n, m)
    A_t = np.random.rand(m, n, n)
    b = np.zeros((m, 1))

    # Computing the bilinear map using a for loop
    for k in range(m):
        b[k] = x[k:k + 1, :] @ A_t[k, :, :] @ y[:, k:k+1]


    # Computing the bilinear map using tensors
    x_t = np.reshape(x, (m, 1, n))
    y_t = np.reshape(y.T, (m, n, 1))

    b_t = x_t @ A_t @ y_t

    print(b_t.flatten())
    print(b.flatten())

    error = b_t.flatten() - b.flatten()

    print(np.sum(error))
    print('End')