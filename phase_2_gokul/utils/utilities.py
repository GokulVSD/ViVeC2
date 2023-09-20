import os

import torch
from torchvision.transforms import transforms

def partition_to_grid(img, num_rows, num_cols, hor_pixels, ver_pixels):

    # Convert PIL to tensor.
    img_tensor = transforms.Compose([transforms.PILToTensor()])(img)

    grid = []

    for i in range(num_rows):
        grid.append([])
        for j in range(num_cols):
            grid[-1].append(
                img_tensor[:, ver_pixels*i : ver_pixels*(i+1), hor_pixels*j : hor_pixels*(j+1)]
            )

    return grid



def tensor_loader(tensor_name, root):

    db_dir = os.path.join(root, 'tensor_database')

    return torch.load(os.path.join(db_dir, tensor_name))



def top_k_ranker(k, vector, tensor_dict, distance_fn):

    distances = []

    for img_id, tensor_tuple in tensor_dict.items():
        tensor, category = tensor_tuple
        distances.append((distance_fn(vector, tensor), img_id, category))

    distances = sorted(distances)

    return distances[:k]