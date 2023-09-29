from numpy import argsort, array, diag, sign, sort, sqrt, zeros
from numpy.linalg import eig


class SVDReducer:
    """
    Singular-value decomposition.
    https://en.wikipedia.org/wiki/Singular_value_decomposition
    """

    def __init__(self, vectors):
        """
        Accept vector space and preprocess.
        """
        self.D = []

        for label, vector in vectors.values():
            self.D.append(vector)

        self.D = array(self.D)


    def reduce_features(self, K):
        """
        Reduce number of features in input vector space to K using truncated SVD.
        This function would be a lot simpler if we could just use numpy.linalg.svd,
        but the project report does not explicitly provide permission, so we are
        implementing it without it.
        """
        DT_D = self.D.T @ self.D
        D_DT = self.D @ self.D.T

        Lambda_DT_D, V = eig(DT_D)
        Lambda_D_DT, U = eig(D_DT)

        # Eigen values and vectors returned from np.linalg.eig aren't sorted. We want
        # to ensure largest Eigen values come first.
        i_1 = argsort(Lambda_DT_D)[::-1]
        i_2 = argsort(Lambda_D_DT)[::-1]

        V = V[:, i_1]
        U = U[:, i_2]

        Lambda = sqrt(sort(Lambda_DT_D)[::-1])

        # Lambda is a 1d vector representing the diagonal of a matrix. We need to
        # reconstruct the matrix in-order to perform matrix multiplication.
        S = zeros((self.D.shape[0], self.D.shape[1]))
        S[:self.D.shape[1], :self.D.shape[1]] = diag(Lambda)


        # D @ V and U @ S should be equal, however, since we are separately finding
        # Eigen vectors for DT_D and D_DT, they can be out of sync, since the negative
        # of an Eigen vector is also an Eigen vector. We check to see if signs differ,
        # and then updates V's sign to compensate.
        same_sign = sign((self.D @ V)[0] * (U @ S)[0])
        V = V * same_sign.reshape(1, -1)

        # Gives back dataset.
        # D_latent = U[:,:K] @ S[0:K,:K] @ V.T[:K,:]

        return U[:,:K] @ S[:K,:K]