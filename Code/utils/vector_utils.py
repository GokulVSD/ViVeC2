from numpy import array


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