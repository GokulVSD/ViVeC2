from gen_utils import *
from data_utils import prepare_dataset, prepare_model, display_images, show_images
from directory_manager import create_directory
from task_manager import get_phase_one_tasks
from user_input_manager import retrieve_task_number, retrieve_input_image_id, retrieve_k_value, display_warning
from feature_descriptors import retrieve_img_color_moments, retrieve_img_hog, retrieve_resnet_outputs
from image_comparator import make_comparisons_for_image
import os


# Code for Task 1 - visualization is automatically set to True (can be configured)
def run_task1(dataset, resnet_model, configured_image_id=None, visualize=True):
    last_dataset_idx = len(dataset) - 1
    # Use a Pre-Configured Image ID if configuration not provided
    if configured_image_id is None:
        configured_image_id = retrieve_input_image_id(last_dataset_idx)
    else:
        pass

    print("Running Task-1 for Image ID:", configured_image_id)

    # Load Image
    img, label = dataset[configured_image_id]

    # Display Image if Visualize attribute is set to True
    if visualize:
        display_images(img)

    # Check Channel Details for Image
    if tuple(img.size())[0] != 3:
        print("Only 1 Color Channel Found for Image:", configured_image_id, img.size())
        image_fd = None
        return image_fd

    # Retrieve Feature Descriptors for Image
    image_fd = {
        "color_moments": retrieve_img_color_moments(img, visualize),
        "histogram_oriented_gradients": retrieve_img_hog(img, visualize)
    }
    l3, avg_pool, fc = retrieve_resnet_outputs(resnet_model, img, visualize)
    image_fd["resnet_l3"] = l3
    image_fd["resnet_avg-pool"] = avg_pool
    image_fd["resnet_fc"] = fc

    if visualize:
        show_images()

    return image_fd


# Code for Task 2 - visualization is automatically set to False (can be configured)
def run_task2(dataset, resnet_model, visualize=False):
    print("Running Task-2")
    dataset_size = len(dataset)
    print("Dataset Size:", dataset_size)

    # Create a Dictionary (JSON Structure) to store Feature Descriptors for all Images in Dataset
    dataset_feature_descriptors = {}
    single_channel_count = 0  # Keep track of # of Single Channel Images

    # Iterate through Dataset
    for image_id in range(0, dataset_size, 1):
        try:
            image_fd = run_task1(dataset, resnet_model,
                                 configured_image_id=image_id, visualize=visualize)
            if image_fd is None:
                single_channel_count = single_channel_count + 1
                continue
            dataset_feature_descriptors[image_id] = image_fd
        except Exception as e:
            print("Exception Encountered for Image ID:", image_id, e)
        print()

    print("Single Channel Count:", single_channel_count)
    file_name = "dataset_feature_descriptors_2.pkl"
    write_pickle_file(dataset_feature_descriptors, file_name=file_name)


def run_task3(dataset, outputs_dir):
    last_dataset_idx = len(dataset) - 1
    image_id = retrieve_input_image_id(last_dataset_idx)
    k = retrieve_k_value()
    print("Running Task-3 for Image ID:", image_id, " k =", k)
    make_comparisons_for_image(dataset, image_id, k, outputs_dir)


def main():
    # Initialize File Paths for Project Phase #1
    # Code Directory
    code_dir = os.getcwd()
    parent_dir = os.path.dirname(code_dir)

    # Datasets Directory
    datasets_dir = os.path.join(code_dir, "Datasets")
    create_directory(datasets_dir)

    # Outputs Directory
    outputs_dir = os.path.join(parent_dir, "Outputs")
    create_directory(outputs_dir)

    # Download Dataset & Model
    dataset_name, model_name = "Caltech101", "ResNet50"
    dataset = prepare_dataset(dataset_name, datasets_dir)
    resnet_model = prepare_model(model_name=model_name)

    # Get Valid Tasks for Project Phase 1
    valid_tasks = get_phase_one_tasks(dataset_name=dataset_name)

    # Retrieve and Run Selected Task
    task_number = retrieve_task_number(valid_tasks)
    if task_number == 1:
        run_task1(dataset, resnet_model)
    elif task_number == 2:
        warning = "Running Task 2 is Time-Consuming and might override" + \
                  "the Feature Descriptors of the entire {} Dataset.".format(dataset_name)
        if display_warning(warning):
            run_task2(dataset, resnet_model)
    else:
        run_task3(dataset, outputs_dir)


if __name__ == '__main__':
    main()
