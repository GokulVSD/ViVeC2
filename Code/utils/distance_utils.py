from scipy.spatial.distance import cityblock, correlation, cosine, euclidean
import numpy as np

def cosine_similarity(vector_a, vector_b):
    return max(1 - cosine(vector_a, vector_b), 0)

FEATURE_SPACE_DISTANCE_MAP = {
    # Manhattan distance (City-block distance): This distance can be imagined as the length needed
    # to move between two points in a grid where you can only move up, down, left or right. This
    # measure is sensitive to differences along each dimension of the feature vectors, capturing
    # the total absolute difference between corresponding moments, which makes it great for
    # assessing differences in color distribution in terms of absolute deviations. Since it
    # does not square the differences like Euclidean distance, it is less susceptible outliers
    # when many dimensions differ.
    "color": cityblock,
    # Correlation similarity: This measures the linear relationship between HOG feature vectors,
    # and performs well with features that have a strong linear correlation or dependency in
    # their gradient orientation distributions, taking into account directionality.
    "hog": correlation,
    # Cosine similarity: We use Cosine since the output of ResNet layers are
    # normalized, reducing the importance of magnitude. It is good at discerning
    # semantic or structural similarity rather than their absolute feature values.
    # It also works well with sparse feature spaces such as those found in CNNs.
    "avgpool": cosine,
    "layer3": cosine,
    "fc": cosine,
    "resnet": cosine,
    "label_label_similarity": cosine_similarity,
    "task_9": cityblock,
    "task_10": euclidean,
}


def get_distance_fn(feature_space):
    """
    Retrieve the distance function chosen for the specific feature space.
    """
    return FEATURE_SPACE_DISTANCE_MAP[feature_space]


def top_k_distance_ranker(k, query_vector, feature_vectors, distance_fn):
    return all_distance_ranker(query_vector, feature_vectors, distance_fn)[:k]


def top_k_min_distance_ranker(k, query_vectors, feature_vectors, distance_fn):
    return min_distance_ranker(query_vectors, feature_vectors, distance_fn)[:k]


def top_k_unique_label_distance_ranker(k, query_vector, feature_vectors, distance_fn):
    """
    Same as top k distance ranker but only includes the distance to the closest entry
    in feature vectors for a particular label.
    """
    return all_unique_label_distance_ranker(query_vector, feature_vectors, distance_fn)[:k]


def all_unique_label_distance_ranker(query_vector, feature_vectors, distance_fn):
    """
    Same as all distance ranker but only includes the distance to the closest entry
    in feature vectors for a particular label.
    """
    distances = all_distance_ranker(query_vector, feature_vectors, distance_fn)

    unique_label_distances = []
    found_labels = set()

    for dist_tuple in distances:
        dist, img_id, label = dist_tuple

        if label not in found_labels:
            found_labels.add(label)
            unique_label_distances.append(dist_tuple)

    return unique_label_distances


def all_distance_ranker(query_vector, feature_vectors, distance_fn):
    """
    Computes distance of query vector to all vectors in feature_vectors,
    returns sorted by distance (distance, img_id, label)
    """
    distances = []

    for img_id, feature_tuple in feature_vectors.items():
        label, feature_vector = feature_tuple
        distances.append((distance_fn(query_vector, feature_vector), img_id, label))

    distances = sorted(distances)

    return distances


def min_distance_ranker(query_vectors, feature_vectors, distance_fn):
    """
    Computes distance of each vector in feature_vectors to each vector
    in query_vectors. Includes only the smallest distance to a query
    vector in the result (one query vector is selected for every
    feature_vector, which happens to have the min distance to it).
    Returns sorted by distance (distance, img_id, label)
    """
    distances = []

    for img_id, feature_tuple in feature_vectors.items():
        label, feature_vector = feature_tuple

        query_distances = []
        for query_vector in query_vectors:
            query_distances.append((distance_fn(query_vector, feature_vector), img_id, label))

        query_distances = sorted(query_distances)

        distances.append(query_distances[0])

    distances = sorted(distances)

    return distances