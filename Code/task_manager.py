def get_phase_one_tasks(*args, **kwargs):
    dataset_name = kwargs["dataset_name"] if "dataset_name" in kwargs else "CalTech101"
    valid_tasks = {
        1: "Feature Descriptor Extraction for a given Input Image ID",
        2: "Feature Descriptor Extraction for complete {} Dataset".format(dataset_name),
        3: "To Find 'k' Closest Images based on each Feature Descriptor for a given Input Image ID"
    }
    return valid_tasks


