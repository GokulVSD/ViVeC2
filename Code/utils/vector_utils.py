from numpy import array
from sklearn.cluster import KMeans


def feature_vectors_to_np_vectors(feature_vectors):
    vectors = []

    for label, vector in feature_vectors.values():
        vectors.append(vector)

    return array(vectors)


def get_latent_feature_vectors(feature_vectors, reducer):
    latent_vectors = reducer.reduce_features(feature_vectors)

    latent_feature_vectors = {}
    for i, feature_item in enumerate(feature_vectors.items()):
        img_id, feature_tuple = feature_item
        label, _ = feature_tuple
        latent_feature_vectors[img_id] = (label, latent_vectors[i])

    return latent_feature_vectors


def flatten_feature_vectors(feature_vectors):
    """
    Input Format - dictionary: {img_id: (label, vector), ... }
    Output Format - list: [[img_id, label, vector], ... ]
    """
    flattened_feature_vectors = []
    for img_id, feature_tuple in feature_vectors.items():
        label, vector = feature_tuple
        flattened_feature_vectors.append([img_id, label, vector])

    return flattened_feature_vectors


def get_representative_vectors_using_kmeans(vectors, K):
    """
    Clusters the provided vectors into K clusters, and returns the list of K cluster
    centers.
    """
    cluster_centers = []
    kmeans = KMeans(n_clusters=K, random_state=42, n_init="auto").fit(array(vectors))
    for cluster_center in kmeans.cluster_centers_:
        cluster_centers.append(cluster_center)

    return cluster_centers