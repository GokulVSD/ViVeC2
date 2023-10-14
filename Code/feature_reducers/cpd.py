import numpy as np
import numpy.linalg as la
from sklearn.metrics.pairwise import paired_distances
from scipy.spatial.distance import euclidean
from utils.dataset_utils import initialize_dataset
from utils.vector_utils import get_representative_vectors_for_labels

MAX_ITERATIONS = 100

class CPDecomposition:
    def __init__(self, feature_vectors, K):
        self.K = K
        self.tensor = self.create_feature_format(feature_vectors)

    def create_feature_format(self, feature_vectors):
        # For feature space of every image, find distance from cluster centers

        labels = initialize_dataset().categories
        result = []
        for i, feature_item in enumerate(feature_vectors.items()): #loop over all images
            # find distance from every center in find_centers(feature_vectors)
            distance = []
            _, feature_tuple = feature_item
            current_label, current_feature_vector = feature_tuple

            # distance = len(labels) x len(feature vector) array
            for label in labels:
                if label == current_label:
                    distance.append(current_feature_vector)
                else:
                    distance.append(np.zeros(len(current_feature_vector)))   
            result.append(distance)
        print(np.array(result).shape)
        return np.array(result)


    # def create_feature_format1(self, feature_vectors):
    #     # For feature space of every image, find distance from cluster centers
    #     centers = self.find_centers(feature_vectors)
    #     result = []
    #     for i in range(len(feature_vectors)): #loop over all images
    #         # find distance from every center in find_centers(feature_vectors)
    #         distance = []
    #         if i % 2 == 0:
    #             # distance = len(labels) x len(feature vector) array
    #             for label in centers:
    #                 distance.append(np.abs(feature_vectors[i][1] - centers[label]))
    #             result.append(distance)
    #     return np.array(result)


    def get_similarity_matrix(self, feature_vectors):
        """
        Get the similarity matrix. Some techniques do not expose the similarity matrix,
        for those, we just reduce the features and return.
        """
        return self.reduce_features(feature_vectors)


    def reduce_features(self, _ = []):
        config = "tensorly"
        if config == "tensorlearn":
            import tensorlearn as tl
            print("Running ALS using tensorlearn for finding CP decomposition...")
            weights, factors = tl.cp_als_rand_init(self.tensor, self.K, MAX_ITERATIONS)
            tensor_hat=tl.cp_to_tensor(weights, factors)
            error=tensor_hat-self.tensor
            error_ratio=tl.tensor_frobenius_norm(error)/tl.tensor_frobenius_norm(self.tensor)
            recovery_rate = 1 - error_ratio
            print(f'The recovery rate is {recovery_rate:2.0%}')
        else:
            import tensorly as tl
            from tensorly.decomposition import parafac
            print("Running ALS using tensorly for finding CP decomposition...")
            decomposition, errors = parafac(self.tensor, self.K, return_errors=True, init='random', tol=1e-4, n_iter_max=MAX_ITERATIONS, normalize_factors=True)

            weights, factors =  decomposition #tl.parafac2_tensor.apply_parafac2_projections(decomposition)
            # reconstruction_error = la.norm(est_tensor - self.tensor)
            # recovery_rate = 1 - reconstruction_error/la.norm(self.tensor)
            # print(f'The recovery rate is {recovery_rate:2.0%}')
        self.factors = factors
        return factors[0]
