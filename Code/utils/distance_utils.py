from scipy.spatial.distance import cityblock, correlation, cosine

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
    "resnet_output": cosine,
}


def get_distance_fn(feature_space):
    """
    Retrieve the distance function chosen for the specific feature space.
    """
    return FEATURE_SPACE_DISTANCE_MAP[feature_space]


def top_k_distance_ranker(k, query_vector, feature_vectors, distance_fn):

    distances = []

    for img_id, feature_tuple in feature_vectors.items():
        label, feature_vector = feature_tuple
        distances.append((distance_fn(query_vector, feature_vector), img_id, label))

    distances = sorted(distances)

    return distances[:k]