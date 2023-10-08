import numpy as np
from scipy.spatial.distance import correlation, cosine
from utils.distance_utils import find_centroids

class LabelLabelSimilarity:
    def __init__(self, feature_vectors):
        self.feature_vectors = feature_vectors
        
    def get_matrix(self):
        centroids = find_centroids(self.feature_vectors)
        label_list = centroids.keys()
        result = {}
        for i, item in enumerate(centroids.items()):
            current_label, current_centroid = item
            arr = np.zeros(shape=(len(label_list)))
            for j, label in enumerate(label_list):
                # find similarity between current_centroid and centroids[label] and store in array
                arr[j] = max(1 - cosine(current_centroid, centroids[label]), 0)
            result[i] = (current_label, arr)
        return result

