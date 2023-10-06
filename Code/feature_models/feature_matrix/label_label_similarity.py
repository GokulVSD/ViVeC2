import numpy as np
from scipy.spatial.distance import correlation, cosine
from utils.distance_utils import find_centroids

class LabelLabelSimilarity:
    def __init__(self, feature_vectors):
        self.feature_vectors = feature_vectors
        
def get_matrix(feature_vectors):
    centroids = find_centroids(feature_vectors)
    label_list = centroids.keys()
    result = np.zeros(shape=(len(label_list), (len(label_list))))
    for i, item in enumerate(centroids.items()):
        current_label, current_centroid = item
        for j, label in enumerate(label_list):
            # find similarity between current_centroid and centroids[label] and store in array
            result[i][j] = max(1 - cosine(current_centroid, centroids[label]), 0)
    return label_list, result


        
            

        