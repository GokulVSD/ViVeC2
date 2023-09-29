from data_utils import resize_image
import numpy as np
import torch


def resnet_preprocess(resnet_model, img):
    # Resize Input Image as per Specification
    img_size = (224, 224)
    final_img = resize_image(img, img_size)
    final_img = final_img.float().unsqueeze(0)
    print("Original Image Size: {} ResNet50 Resized Input Image Size: {}".format(img.size(), final_img.size()))
    return resnet_model, final_img


def make_resnet_hooks(resnet_model):
    # Create ResNet50 Hooks as Per Specification
    hooks = {}

    def layer3_hook(module, input, output):
        hooks["l3"] = output.squeeze()

    def avgpool_hook(module, input, output):
        hooks["avg_pool"] = output.squeeze()

    def fc_hook(module, input, output):
        hooks["fc"] = output.squeeze()

    resnet_model.layer3.register_forward_hook(layer3_hook)
    resnet_model.avgpool.register_forward_hook(avgpool_hook)
    resnet_model.fc.register_forward_hook(fc_hook)
    return hooks


def l3_post_process(data):
    data = np.array(data)
    data_shape = data.shape
    final_data = [np.mean(data[i]) for i in range(0, data_shape[0], 1)]
    return np.array(final_data)


def avg_pool_post_process(data):
    data = np.array(data)
    data_shape = data.shape
    final_data = [float((data[i] + data[i + 1]) / 2) for i in range(0, data_shape[0], 2)]
    return np.array(final_data)


def fc_post_process(data):
    return np.array(data)


def resnet_post_process(hook_data):
    l3_out = l3_post_process(hook_data["l3"])
    avg_pool_out = avg_pool_post_process(hook_data["avg_pool"])
    fc_out = fc_post_process(hook_data["fc"])
    return l3_out, avg_pool_out, fc_out


def retrieve_resnet_outputs(resnet_model, img, visualize=False):
    resnet_model, img = resnet_preprocess(resnet_model, img)
    hooks = make_resnet_hooks(resnet_model)
    with torch.no_grad():
        resnet_model(img)
    l3_out, avg_pool_out, fc_out = resnet_post_process(hooks)

    if visualize:
        print("ResNet50 L3 Layer Feature Descriptor:", l3_out)
        print("ResNet50 L3 Layer Feature Descriptor Shape:", l3_out.shape)
        print("ResNet50 Average-Pooling Layer Feature Descriptor:", avg_pool_out)
        print("ResNet50 Average-Pooling Layer Feature Descriptor Shape:", avg_pool_out.shape)
        print("ResNet50 FC Layer Feature Descriptor:", fc_out)
        print("ResNet50 FC Layer Feature Descriptor Shape:", fc_out.shape)
    else:
        print("ResNet50 L3 Layer Feature Descriptor Shape:", l3_out.shape)
        print("ResNet50 Average-Pooling Layer Feature Descriptor Shape:", avg_pool_out.shape)
        print("ResNet50 FC Layer Feature Descriptor Shape:", fc_out.shape)

    return l3_out, avg_pool_out, fc_out





