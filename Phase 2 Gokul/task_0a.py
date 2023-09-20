# – Task 0a: Using pre-trained RESNET50 neural network model, map even numbered (labeled) images in the Caltec101 data set into 5 different feature spaces and store the resulting data vectors:
# ∗ Color moments, CM10x10
# ∗ Histograms of oriented gradients, HOG
# ∗ ResNet-AvgPool-1024
# ∗ ResNet-Layer3-1024
# ∗ ResNet-FC-1000

# WARNING: Running this script will overwrite some of the contents of /tensor_database/


import os

import torch
from torchvision.datasets import Caltech101
from torchvision.transforms import Grayscale, transforms

from feature_extractors.color_moments import get_color_vector
from feature_extractors.HOG import get_hog_vector
from feature_extractors.resnet_50 import get_resnet50_feature_vectors
from utils.utilities import partition_to_grid


print("Running Task 0a.")
print("Downloading dataset if not present.")

dataset = Caltech101(root=os.path.abspath(os.path.join(os.path.dirname( __file__ ))), download=True)

print(dataset, "\n")

color_tensors = {}
hog_tensors = {}
avgpool_tensors = {}
layer3_tensors = {}
fc1000_tensors = {}

for i in range(len(dataset)):
    if i%2 != 0:
        continue
    print(f"Processing image {i+1}/{len(dataset)} \t ({int(100*(i+1)/len(dataset))} %)", end='\r')

    image, category = dataset[i][0], dataset.categories[dataset[i][1]]

    # Convert images that are not RGB to RGB.
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Resize image to 300x100.
    image300x100 = image.resize((300, 100))

    # Partition image into 10x10 grid.
    grid = partition_to_grid(image300x100, num_rows=10, num_cols=10, hor_pixels=30, ver_pixels=10)

    # Compute color moments vector.
    color_tensors[i] = (get_color_vector(grid), category)

    # Convert image to grayscale.
    gs_image = Grayscale()(image300x100)

    # Partition grayscale image into 10x10 grid.
    gs_grid = partition_to_grid(gs_image, num_rows=10, num_cols=10, hor_pixels=30, ver_pixels=10)

    # Compute HOG moments vector.
    hog_tensors[i] = (get_hog_vector(gs_grid), category)

    # Resize original image to 244x244.
    image224x224 = image.resize((224, 224))

    # Convert to float tensor.
    img_tensor = transforms.Compose([transforms.ToTensor()])(image224x224)

    # Extract features from ResNet-50.
    avgpool_vector, layer3_vector, fc1000_vector = get_resnet50_feature_vectors(img_tensor)

    avgpool_tensors[i] = (avgpool_vector, category)
    layer3_tensors[i] = (layer3_vector, category)
    fc1000_tensors[i] = (fc1000_vector, category)


db_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'tensor_database'))

print("\n\nSaving color feature space to '<db_dir>/even_color_tensors.pt")

torch.save(color_tensors, os.path.join(db_dir, 'even_color_tensors.pt'))

print("\nSaving HOG feature space to '<db_dir>/even_hog_tensors.pt'")

torch.save(hog_tensors, os.path.join(db_dir, 'even_hog_tensors.pt'))

print("\nSaving ResNet-50 AvgPool feature space to '<db_dir>/even_avgpool_tensors.pt'")

torch.save(avgpool_tensors, os.path.join(db_dir, 'even_avgpool_tensors.pt'))

print("\nSaving ResNet-50 Layer3 feature space to '<db_dir>/even_layer3_tensors.pt'")

torch.save(layer3_tensors, os.path.join(db_dir, 'even_layer3_tensors.pt'))

print("\nSaving ResNet-50 Fc1000 feature space to '<db_dir>/even_fc1000_tensors.pt'")

torch.save(fc1000_tensors, os.path.join(db_dir, 'even_fc1000_tensors.pt'))

print(
"""
> Feature vectors for all images are stored in a tensor dictionary per descriptor type.
> Stored in binary form, the key is the image ID, val is feature vector.
"""
)
