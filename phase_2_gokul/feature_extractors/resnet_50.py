import torch
from torchvision.models import resnet50
from torchvision.models.resnet import ResNet50_Weights

# ResNet-50 Pretrained with default weights.
# https://pytorch.org/vision/master/models/generated/torchvision.models.resnet50.html

def get_resnet50_feature_vectors(img_tensor):

    # Use default pre-trained weights (ImageNet1K_V2).
    model = resnet50(weights=ResNet50_Weights.DEFAULT)

    global layer_outputs
    layer_outputs = {}

    # Return a hook that modifies layer_outputs variable.
    def get_hook_fn(layer_name):

        def named_hook(module, input, output):
            global layer_outputs
            layer_outputs[layer_name] = output

        return named_hook

    hooks = {}
    layers_to_hook = ['avgpool', 'layer3', 'fc']

    for name, module in model.named_children():
        if name in layers_to_hook:
            hooks[name] = module.register_forward_hook(get_hook_fn(name))

    # Set ResNet-50 to evaluation mode.
    model.eval()

    # Don't compute gradient, we are only inferencing.
    with torch.no_grad():
        output = model(img_tensor.unsqueeze(dim=0))

    # Detach hooks.
    for hook in hooks.values():
        hook.remove()

    for layer_name, out_tensor in layer_outputs.items():
        if layer_name == 'avgpool':
            # Shape is 1x2048, we have to average every 2 values sequentially
            # to obtain a 1024 dimension vector. We do this by reshaping to
            # a vector of 2 value vectors, and mean each 2 value vector.
            avgpool_vector = out_tensor.reshape([1024, 2]).mean(dim=1)
        elif layer_name == 'layer3':
            # Shape is 1x1024x14x14, We have to take the mean of each 14x14
            # slice, resulting in a 1024 dimension vector.
            layer3_vector = out_tensor.mean(dim=(2, 3))[0]
        elif layer_name == 'fc':
            # Shape is 1x1000.
            fc1000_vector = out_tensor[0]


    return avgpool_vector, layer3_vector, fc1000_vector