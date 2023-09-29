# – Task 0a: Using pre-trained RESNET50 neural network model, map even numbered (labeled)
#   images in the Caltec101 data set into 5 different feature spaces and store the resulting data vectors:
# ∗ Color moments, CM10x10
# ∗ Histograms of oriented gradients, HOG
# ∗ ResNet-AvgPool-1024
# ∗ ResNet-Layer3-1024
# ∗ ResNet-FC-1000

# WARNING: Running this script will overwrite some of the contents of Code/database/


from feature_models.color_moments import ColorMomentsExtractor
from feature_models.hog import HOGExtractor
from feature_models.resnet import ResNetExtractor
from utils.database_utils import store
from utils.dataset_utils import initialize_dataset, get_image_with_label

def main():
    """
    Extract feature vectors for even images in Caltech101 and store.
    """
    print("Running Task 0a.", "\n")

    dataset = initialize_dataset()

    print(dataset, "\n")

    color_vectors = {}
    hog_vectors = {}
    avgpool_vectors = {}
    layer3_vectors = {}
    fc_vectors = {}
    resnet_output_vectors = {}

    for img_id in range(len(dataset)):

        # Skip images that do not have an even image ID.
        if img_id % 2 != 0:
            continue

        print(f"Processing image {img_id} \t ({int(100*(img_id+1)/len(dataset))} %)", end='\r')

        image, label = get_image_with_label(dataset, img_id)

        # Compute color moments vector.
        color_vectors[img_id] = (label, ColorMomentsExtractor(image).get_color_vector())

        # Compute HOG moments vector.
        hog_vectors[img_id] = (label, HOGExtractor(image).get_hog_vector())

        # Generate outputs from ResNet50.
        resnet = ResNetExtractor(image)

        # Retrieve vectors from hooked layers and output.
        avgpool_vectors[img_id] = (label, resnet.get_avgpool_vector())

        layer3_vectors[img_id] = (label, resnet.get_layer3_vector())

        fc_vectors[img_id] = (label, resnet.get_fc_vector())

        resnet_output_vectors[img_id] = (label, resnet.get_output_vector())


    # Display samples.
    print("\nSample color moments vector:", "\n")
    print("\nColor vector: ", color_vectors[0], "\nLength: ", len(color_vectors[0][1]))
    print("\nHOG vector: ", hog_vectors[0], "\nLength: ", len(hog_vectors[0][1]))
    print("\nAvgPool vector: ", avgpool_vectors[0], "\nLength: ", len(avgpool_vectors[0][1]))
    print("\nLayer3 vector: ", layer3_vectors[0], "\nLength: ", len(layer3_vectors[0][1]))
    print("\nFC vector: ", fc_vectors[0], "\nLength: ", len(fc_vectors[0][1]))
    print("\nResNet output vector: ", resnet_output_vectors[0], "\nLength: ", len(resnet_output_vectors[0][1]))

    # Save all vectors
    store(color_vectors, "color.pt")
    store(hog_vectors, "hog.pt")
    store(avgpool_vectors, "avgpool.pt")
    store(layer3_vectors, "layer3.pt")
    store(fc_vectors, "fc.pt")
    store(resnet_output_vectors, "resnet_output.pt")

    print(
    """
    > Feature vectors for the images are stored in a tensor dictionary per descriptor type.
    > Stored in binary form, the key is the image ID, val is label + feature vector tuple.
    """
    )


if __name__ == "__main__":
    main()