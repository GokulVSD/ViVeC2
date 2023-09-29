from torchvision.transforms import transforms


def convert_to_rgb(pil_img):
    """
    Convert image that isn't RGB to RGB.
    """
    if pil_img.mode != 'RGB':
        return pil_img.convert('RGB')
    return pil_img


def partition_to_grid(pil_img, num_rows, num_cols, hor_pixels, ver_pixels):
    """
    Partition input image into patches. It is expected that the image should
    be partitionable with the provided parameters.
    """

    img_tensor = transforms.Compose([transforms.PILToTensor()])(pil_img)

    grid = []

    for i in range(num_rows):
        grid.append([])
        for j in range(num_cols):
            grid[-1].append(
                img_tensor[:, ver_pixels*i : ver_pixels*(i+1), hor_pixels*j : hor_pixels*(j+1)]
            )

    return grid