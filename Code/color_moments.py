from data_utils import resize_image
import numpy as np


def cms_preprocess(img):
    # Resize Input Image as per Specification
    img_size = (300, 100)
    final_img = resize_image(img, img_size)
    print("Original Image Size: {} Color-Moments Resized Image Size: {}".format(img.size(), final_img.size()))
    return final_img


def calculate_grid_cms(grid):
    grid_size = grid.shape
    total_pixels = (grid_size[0] * grid_size[1])
    mean = np.mean(grid)
    std = np.std(grid)
    skew = np.cbrt((np.sum(np.power(grid - mean, 3))) / total_pixels)
    # print("Mean = {} Standard Deviation = {} Skew = {}".format(mean, std, skew))
    return mean, std, skew


def calculate_channel_cms(img_channel):
    channel_shape = img_channel.shape
    total_grids = (10, 10)
    grid_size = (channel_shape[0] // total_grids[0], channel_shape[1] // total_grids[1])
    channel_cms = []
    for i in range(0, channel_shape[0], grid_size[0]):
        for j in range(0, channel_shape[1], grid_size[1]):
            grid = img_channel[i: i + grid_size[0], j: j + grid_size[1]]
            grid_mean, grid_std, grid_skew = calculate_grid_cms(grid)
            channel_cms += [grid_mean, grid_std, grid_skew]
    return channel_cms


def retrieve_img_color_moments(img, visualize=False):
    img = cms_preprocess(img)
    img = np.array(img)
    img_color_moments = []
    if img.shape[0] == 3:
        for channel in range(3):
            channel_color_moments = calculate_channel_cms(img[channel])
            img_color_moments += channel_color_moments
        img_color_moments = np.array(img_color_moments)
        if visualize:
            print("Color Moments Feature Descriptor:", img_color_moments)
        print("Color Moments Feature Descriptor Shape:", img_color_moments.shape)
    else:
        print("Color Moments Retrieval Error - 3 Color Channels Not Found!")
        img_color_moments = None
    return img_color_moments
