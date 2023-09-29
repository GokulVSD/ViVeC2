# TODO Place your vector averages / medians calculation logic, etc here.


def latent_features_to_vectors_dict(latent_features, feature_vectors):
    latent_vectors = {}

    for i, img_id in enumerate(feature_vectors.keys()):
        label, _ = feature_vectors[img_id]
        latent_vectors[img_id] = (label, latent_features[i])

    return latent_vectors