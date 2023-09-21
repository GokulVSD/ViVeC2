from distance_measures import get_distance_measures
from gen_utils import *
from data_utils import save_image, show_images
from directory_manager import create_directory, delete_directory
import os

# Get Available Distance Measures
distance_measures = get_distance_measures()


# Read Feature Descriptors Database
def read_database_file(file_name=None):
    file_name = "caltech101_dataset_feature_descriptors.pkl" if file_name is None else file_name
    database = read_pickle_file(file_name)
    return database


# Retrieves the top "k" matching images based on the Input Image, Feature Descriptor, and Distance Measure used
def retrieve_top_k_matching_images(database, image_id, k, fd, measure):
    try:
        distances = []
        v1 = database[image_id][fd]
        # Calculate Distance Scores for all Images (except the same Image in the Dataset)
        for idx in database:
            if idx == image_id:
                continue
            v2 = database[idx][fd]
            distances.append([measure(v1, v2), idx])
        # Sort the Distances in Ascending Order & Filter according to the "k" value
        distances.sort()
        matching_image_ids = [distances[i] for i in range(0, min(k, len(distances)), 1)]
    except Exception as e:  # Throws an Exception when Input Image is missing in the Feature Descriptor Database
        print("Invalid Image ID Input! Input Image only has 1 Color Channel and cannot be processed!", e)
        matching_image_ids = None
    return matching_image_ids


# Comparison based on Feature Descriptor: Color Moments
def color_moments_comparison(database, image_id, k=5):
    feature_descriptor = "color_moments"
    measure = distance_measures["cosine"]  # Cosine-Distance selected as Distance Measure
    results = retrieve_top_k_matching_images(database, image_id, k, feature_descriptor, measure)
    return results


# Comparison based on Feature Descriptor: Histogram of Oriented Gradients
def hog_comparison(database, image_id, k=5):
    feature_descriptor = "histogram_oriented_gradients"
    measure = distance_measures["cosine"]  # Cosine Distance selected as Distance Measure
    results = retrieve_top_k_matching_images(database, image_id, k, feature_descriptor, measure)
    return results


# Comparison based on Feature Descriptor: ResNet50 L3 Layer
def resnet_l3_comparison(database, image_id, k=5):
    feature_descriptor = "resnet_l3"
    measure = distance_measures["cosine"]  # Cosine Distance selected as Distance Measure
    results = retrieve_top_k_matching_images(database, image_id, k, feature_descriptor, measure)
    return results


# Comparison based on Feature Descriptor: ResNet50 Average Pooling Layer
def resnet_avg_pool_comparison(database, image_id, k=5):
    feature_descriptor = "resnet_avg-pool"
    measure = distance_measures["cosine"]  # Cosine Distance selected as Distance Measure
    results = retrieve_top_k_matching_images(database, image_id, k, feature_descriptor, measure)
    return results


# Comparison based on Feature Descriptor: ResNet50 Fully-Connected Layer
def resnet_fc_comparison(database, image_id, k=5):
    feature_descriptor = "resnet_fc"
    measure = distance_measures["cosine"]  # Cosine Distance selected as Distance Measure
    results = retrieve_top_k_matching_images(database, image_id, k, feature_descriptor, measure)
    return results


# Stores the Top "K" Matches onto the Outputs Folder
# Storage Format: Outputs -> folder per Input Image ID -> folder for each Feature Descriptor -> Top "k" Images
def display_results(dataset, image_id, results, outputs_dir):
    image_dir = os.path.join(outputs_dir, str(image_id))
    delete_directory(image_dir)
    create_directory(image_dir)

    img, _ = dataset[image_id]
    img_path = os.path.join(image_dir, str(image_id) + ".png")
    img_title = "Original Image - Image ID: {}".format(str(image_id))
    save_image(img, img_path, img_title)

    for fd in results:
        fd_dir = os.path.join(image_dir, str(fd))
        create_directory(fd_dir)
        res_json = {}
        # res_json_path = os.path.join(fd_dir, "results_metadata.json")
        print("Comparison for Feature Descriptor:", fd)
        for i in range(0, len(results[fd]), 1):
            dist_score, res_id = results[fd][i]
            rank = "Rank {}".format(i + 1)
            res_img, _ = dataset[res_id]
            res_img_path = os.path.join(fd_dir, rank.replace(" ", "_") + ".png")
            res_img_title = fd + " " + rank + " Image ID: {} Distance Score: {}".format(res_id, round(dist_score, 3))
            print(res_img_title)
            res_json[rank] = {
                "image_id": res_id,
                "dist_score": dist_score
            }
            save_image(res_img, res_img_path, res_img_title)
        print()
    show_images()
    print("Stored Results for Task-3 for Image:", image_id)


# List of Feature Descriptor Comparator Functions
def get_fd_comparators():
    fd_comparators = {
        "color_moments": color_moments_comparison,
        "histogram_oriented_gradients": hog_comparison,
        "resnet_l3": resnet_l3_comparison,
        "resnet_avg-pool": resnet_avg_pool_comparison,
        "resnet_fc": resnet_fc_comparison,
    }
    return fd_comparators


# Retrieves and Stores the Top "k" Comparisons for an Input Image for Each Feature Descriptor
def make_comparisons_for_image(dataset, image_id, k=5, outputs_dir=None):
    print("Retrieving Feature Descriptors & respective Comparators for Dataset")
    fd_comparators = get_fd_comparators()
    fd_database = read_database_file()
    img, _ = dataset[image_id]
    img_fd = fd_database[image_id]
    final_results = {}
    for fd in img_fd:
        results = fd_comparators[fd](fd_database, image_id, k)
        final_results[fd] = results
    display_results(dataset, image_id, final_results, outputs_dir)
    try:
        pass
    except Exception as e:
        print("Invalid Image ID Input! Input Image only has 1 Color Channel and cannot be processed!", e)
