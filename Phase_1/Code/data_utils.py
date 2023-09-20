import torch
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights
from matplotlib import pyplot as plt
from IPython.display import display


def get_device_type():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)


def get_transforms():
    avl_transforms = {
        "pil_transform": transforms.ToPILImage(),
        "tensor_transform": transforms.PILToTensor(),
        "gray_transform": transforms.Grayscale(),
    }
    return avl_transforms


def resize_image(img, img_size):
    avl_transforms = get_transforms()
    pil_img = avl_transforms["pil_transform"](img)
    resized_img = pil_img.resize(img_size)
    resized_img = avl_transforms["tensor_transform"](resized_img)
    return resized_img


def convert_to_pil_image(img):
    avl_transforms = get_transforms()
    pil_img = avl_transforms["pil_transform"](img)
    return pil_img


def convert_to_gray_scale(img):
    avl_transforms = get_transforms()
    gray_img = avl_transforms["gray_transform"](img)
    return gray_img


def display_images(img, title_text=""):
    pil_img = convert_to_pil_image(img)
    plt.imshow(pil_img)
    if len(title_text) > 0:
        plt.title(title_text)


def save_image(img, path, title=""):
    display_images(img, title)
    plt.savefig(path)


def show_images():
    plt.show()


def download_dataset(dataset_file_path, dataset_name, transforms=None):
    if str(dataset_name).lower() == "caltech101":
        dataset = datasets.Caltech101(dataset_file_path, download=True, transform=transforms)
    else:
        print("Invalid Dataset Entered!!")
        dataset = None
    return dataset


def download_model(model_name):
    if str(model_name).lower() == "resnet50":
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
    else:
        print("Invalid Dataset Entered!!")
        model = None
    return model


def prepare_dataset(dataset_name, datasets_path):
    avl_transforms = get_transforms()
    transforms_list = [avl_transforms["tensor_transform"], ]
    composed_transforms = transforms.Compose(transforms_list)
    dataset = download_dataset(datasets_path, dataset_name, transforms=composed_transforms)
    return dataset


def prepare_model(model_name):
    model = download_model(model_name)
    return model




