import numpy as np
from scipy.spatial.distance import cosine

class ImageImageSimilarity:
    def __init__(self, feature_vectors):
        self.feature_vectors = feature_vectors

    def get_matrix(self):
        result = {}
        image_id_list = self.feature_vectors.keys()
        for i, item in enumerate(self.feature_vectors.items()):
            ref_image_id, (ref_image_label, ref_feature_vec) = item
            #arr = np.zeros(shape=(len(image_id_list)), dtype=np.float16)
            arr = np.zeros(shape=(len(image_id_list)))
            for idx, image_id in enumerate(image_id_list):
                feature_vec = self.feature_vectors[image_id][1]
                arr[idx] = max(1 - cosine(ref_feature_vec, feature_vec), 0)
            result[i] = (ref_image_id, arr)
        return result
