def get_valid_tasks(*args, **kwargs):
    dataset_name = kwargs["dataset_name"]
    valid_tasks = {
        1: "Feature Descriptor Extraction for a given Input Image ID",
        2: "Feature Descriptor Extraction for complete {} Dataset".format(dataset_name),
        3: "To Find 'k' Closest Images based on each Feature Descriptor for a given Input Image ID"
    }
    return valid_tasks
