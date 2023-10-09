import numpy as np
import numpy.linalg as la
MAX_ITERATIONS = 100

class CPDecomposition:
    def __init__(self, feature_vectors, K):
        self.K = K
        self.tensor = self.create_feature_format(feature_vectors)

    def create_feature_format(self, feature_vectors):
        # For feature space of every image, find distance from cluster centers
        def find_labels(feature_vectors):
            centers = {}
            for i in range(len(feature_vectors)): #loop over all images
                if i % 2 == 0:
                    if feature_vectors[i][0] not in centers:
                        centers[feature_vectors[i][0]] = True
            return centers

        labels = find_labels(feature_vectors)
        result = []
        for i in range(len(feature_vectors)): #loop over all images
            # find distance from every center in find_centers(feature_vectors)
            distance = []
            if i % 2 == 0:
                # distance = len(labels) x len(feature vector) array
                for label in labels:
                    if label == feature_vectors[i][0]:
                        distance.append(feature_vectors[i][1])
                    else:
                        distance.append(np.zeros(len(feature_vectors[i][1])))
                result.append(distance)
        return np.array(result)

    # def find_centers(self, feature_vectors):
    #         cluster_centers = {}
    #         for i in range(len(feature_vectors)):
    #             if i % 2 == 0:
    #                 if feature_vectors[i][0] in cluster_centers:
    #                     cluster_centers[feature_vectors[i][0]].append(feature_vectors[i][1])
    #                 else:
    #                     cluster_centers[feature_vectors[i][0]] = []
    #         for label in cluster_centers:
    #             cluster_centers[label] = np.average(cluster_centers[label], axis=0)
    #         return cluster_centers

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
        config = "tensorlearn"
        if config == "tensorlearn":
            import tensorlearn as tl
            print("Running ALS for finding CP decomposition...")
            weights, factors = tl.cp_als_rand_init(self.tensor, self.K, MAX_ITERATIONS)
            tensor_hat=tl.cp_to_tensor(weights, factors)
            error=tensor_hat-self.tensor
            error_ratio=tl.tensor_frobenius_norm(error)/tl.tensor_frobenius_norm(self.tensor)
            recovery_rate = 1 - error_ratio
            print(f'The recovery rate is {recovery_rate:2.0%}')
        else:
            import tensorly as tl
            from tensorly.decomposition import parafac2
            print("Running ALS for finding CP decomposition...")
            # best_err = np.inf
            # decomposition = None

            # for run in range(10):
            #     print(f'Training model {run}...')
            #     trial_decomposition, trial_errs = parafac2(self.tensor, self.K, return_errors=True, tol=1e-8, n_iter_max=500, random_state=run)
            #     print(f'Number of iterations: {len(trial_errs)}')
            #     print(f'Final error: {trial_errs[-1]}')
            #     if best_err > trial_errs[-1]:
            #         best_err = trial_errs[-1]
            #         errors = trial_errs
            #         decomposition = trial_decomposition
            #     print('-------------------------------')
            #     print(f'Best model error: {best_err}')

            decomposition, errors = parafac2(self.tensor, self.K, return_errors=True, tol=1e-4, n_iter_max=MAX_ITERATIONS)
            est_tensor = tl.parafac2_tensor.parafac2_to_tensor(decomposition)
            weights, factors = tl.parafac2_tensor.apply_parafac2_projections(decomposition)
            reconstruction_error = la.norm(est_tensor - self.tensor)
            recovery_rate = 1 - reconstruction_error/la.norm(self.tensor)
            print(f'The recovery rate is {recovery_rate:2.0%}')
        return factors[0]
