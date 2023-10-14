import numpy as np
from scipy.spatial.distance import cosine
from utils.vector_utils import get_representative_vectors_for_labels
from utils.distance_utils import cosine_similarity

class LabelLabelSimilarity:
    def __init__(self, feature_vectors, all_labels):
        self.feature_vectors = feature_vectors
        self.all_labels = all_labels

    def get_matrix(self):
        centroids = get_representative_vectors_for_labels(self.feature_vectors, self.all_labels, 1)
        result = {}
        for i, label_centroid_tuple in enumerate(centroids.items()):
            current_label, current_centroid = label_centroid_tuple
            arr = np.zeros(shape=(len(self.all_labels)))
            for j, label in enumerate(self.all_labels):
                # find similarity between current_centroid and centroids[label] and store in array
                arr[j] = cosine_similarity(current_centroid, centroids[label])
            result[i] = (current_label, arr)
        return result

