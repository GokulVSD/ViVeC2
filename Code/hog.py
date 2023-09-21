from data_utils import convert_to_gray_scale, resize_image
import numpy as np


def get_sobel_mask():
    sobel_mask = [-1, 0, 1]
    return sobel_mask


def hog_preprocess(img):
    # Convert Input Image to Gray-Scale & Resize as per Specification
    img_size = (300, 100)
    gray_img = convert_to_gray_scale(img)
    final_img = resize_image(gray_img, img_size)
    print_str = "Original Image Size: {} Histogram of Oriented Gradients Resized Image Size: {}"
    print(print_str.format(img.size(), final_img.size()))
    return final_img


def calculate_gradients(grid, grid_size, i, j, mask=None):
    mask = get_sobel_mask() if mask is None else mask
    px = grid[i, j]
    left_px = 0 if j == 0 else grid[i, j - 1]
    right_px = 0 if j == (grid_size[1] - 1) else grid[i, j + 1]
    top_px = 0 if i == 0 else grid[i - 1, j]
    bot_px = 0 if i == (grid_size[0] - 1) else grid[i + 1, j]
    gx = np.dot([left_px, px, right_px], mask)
    gy = np.dot([top_px, px, bot_px], mask)
    return gx, gy


def calculate_grid_hog(grid):
    bin_angle = 40
    total_bins = 360 // bin_angle
    weighted_bins = [0] * total_bins
    grid_size = grid.shape
    mask = [-1, 0, 1]
    for i in range(0, grid_size[0], 1):
        for j in range(0, grid_size[1], 1):
            gx, gy = calculate_gradients(grid, grid_size, i, j, mask)
            magnitude = np.sqrt(np.power(gx, 2) + np.power(gy, 2))
            angle = (np.arctan2(gy, gx) * (360 / (2 * np.pi)) + 360) % 360
            prev_bin = int(angle // bin_angle) % total_bins
            next_bin = (prev_bin + 1) % total_bins
            prev_bin_angle, next_bin_angle = prev_bin * bin_angle, next_bin * bin_angle
            prev_bin_magnitude = ((next_bin_angle - angle) / bin_angle) * magnitude
            next_bin_magnitude = ((angle - prev_bin_angle) / bin_angle) * magnitude
            weighted_bins[prev_bin] += prev_bin_magnitude
            weighted_bins[next_bin] += next_bin_magnitude

    # print("Weighted Bins for Grid:", weighted_bins)
    return weighted_bins


def retrieve_img_hog(img, visualize=False):
    img = hog_preprocess(img)
    img = np.array(img)[0]
    img_shape = img.shape
    total_grids = (10, 10)
    grid_size = (img_shape[0] // total_grids[0], img_shape[1] // total_grids[1])
    img_hog = []
    for i in range(0, img_shape[0], grid_size[0]):
        for j in range(0, img_shape[1], grid_size[1]):
            grid = img[i: i + grid_size[0], j: j + grid_size[1]]
            img_hog += calculate_grid_hog(grid)

    img_hog = np.array(img_hog)
    if visualize:
        print("Histogram of Oriented Gradients Feature Descriptor:", img_hog)
    print("Histogram of Oriented Gradients Feature Descriptor Shape:", img_hog.shape)
    return img_hog
