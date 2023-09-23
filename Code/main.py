from data_utils import prepare_dataset, prepare_model
from directory_manager import create_directory
from phase_manager import phase_manager
import os


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

    # Pack Directories
    directories = {
        "parent_dir": parent_dir,
        "code_dir": code_dir,
        "datasets_dir": datasets_dir,
        "outputs_dir": outputs_dir
    }

    # Download Dataset & Model
    dataset_name, model_name = "Caltech101", "ResNet50"
    dataset = prepare_dataset(dataset_name, datasets_dir)
    resnet_model = prepare_model(model_name=model_name)

    data_info = {
        "dataset": dataset,
        "dataset_name": dataset_name,
        "model": resnet_model,
        "model_name": model_name
    }

    print()
    phase_manager(directories=directories, data=data_info)


if __name__ == '__main__':
    main()
