from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import MinMaxScaler
from utils.vector_utils import feature_vectors_to_np_vectors


class LDAReducer:
    """
    Latent Dirichlet Allocation reducer.
    https://towardsdatascience.com/dimensionality-reduction-with-latent-dirichlet-allocation-8d73c586738c
    """

    def __init__(self, feature_vectors, K):
        """
        Accept vector space and target dimensions, and preprocess.
        """
        D = feature_vectors_to_np_vectors(feature_vectors)
        self.scaler = MinMaxScaler().fit(D)

        self.lda = LatentDirichletAllocation(n_components = K).fit(self.scaler.transform(D))


    def reduce_features(self, feature_vectors):
        """
        Reduce number of features in input vector space to K.
        This uses the trained LDA model during __init__.
        """

        D = feature_vectors_to_np_vectors(feature_vectors)

        return self.lda.transform(self.scaler.transform(D))