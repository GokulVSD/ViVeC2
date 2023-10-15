from .dataset_utils import initialize_dataset, get_image_with_label
from os import path
from PIL.Image import open

INPUT_PATH = path.join(path.dirname(path.dirname( __file__ )), 'input')


def get_query_image(image_input):
    """
    Handle user input for query image, either IMAGE_ID or an image file.
    Returns PIL image or raises an exception.
    """
    if image_input.isnumeric():

        IMG_ID = int(image_input)
        dataset = initialize_dataset()

        if IMG_ID < 0 or IMG_ID >= len(dataset):
            raise Exception("Image ID out of range for Caltech101.")

        return get_image_with_label(dataset, IMG_ID)[0]

    else:

        return open(path.join(INPUT_PATH, image_input))


def check_label_is_valid(label):
    """
    Checks to see that the provided label is present in Caltech101.
    """
    print("Input label: ", label)

    dataset = initialize_dataset()
    valid_labels = dataset.categories
    if label.isnumeric():
        try:
            label = valid_labels[int(label)]
        except Exception as e:
            raise Exception("Input Label ID is out of bounds.")
    elif label not in valid_labels:
        raise Exception("Label does not exist in Caltech101.")
    else:
        pass

    print("Output label:", label, "\n\n")
    return label
