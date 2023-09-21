from color_moments import retrieve_img_color_moments
from hog import retrieve_img_hog
from resnet_fd import retrieve_resnet_outputs


def get_color_moments(img, visualize=False):
    color_moments = retrieve_img_color_moments(img, visualize)
    return color_moments


def get_histogram_oriented_gradients(img, visualize=False):
    hog = retrieve_img_hog(img, visualize)
    return hog


def get_resnet_outputs(resnet_model, img, visualize=False):
    l3_out, avg_pool_out, fc_out = retrieve_resnet_outputs(resnet_model, img, visualize)
    return l3_out, avg_pool_out, fc_out
