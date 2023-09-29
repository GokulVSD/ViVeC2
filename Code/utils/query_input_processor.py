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