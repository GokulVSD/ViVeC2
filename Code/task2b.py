import pandas as pd
import numpy as np
import torch
import torchvision
import PIL
from pathlib import Path
from tqdm import tqdm
import scipy
import sklearn.cluster
import math

from utils.query_input_processor import get_query_image
from utils.image_utils import convert_to_rgb
"""
Instructions:
1. Run task_0a.py to generate the vector database if not already generated.
3. You will be prompted to provide input values.


Task 2b: ImplementED a program which, given (a) a query imageID or image file and 
(b) positive integer k, identifies and lists k most likely matching labels, along 
with their scores, under the RESNET50 neural network model.
"""


# Research References:
# https://saturncloud.io/blog/how-to-check-if-pytorch-is-using-the-gpu/
# https://saturncloud.io/blog/how-to-train-a-pytorch-model-on-gpu/
# https://docs.scipy.org/doc/scipy/reference/spatial.distance.html
# https://learnopencv.com/image-classification-using-transfer-learning-in-pytorch/
# https://pytorch.org/docs/master/generated/torch.topk.html
# https://pytorch.org/vision/stable/models.html

# This functions are unique to task 2b
# Using convert_to_rgb, get_query_image from utility folder

# This function gets the device to use as the goal of 
# the code is to use GPU if possible
def getDevice():
    torchDevice = "cpu"
    if torch.cuda.is_available():
        torchDevice = "cuda"

    print("Using Device: ", torchDevice)
    return torchDevice


# This function converts the image to 224x224 as required by
# the ResNet model, then transforms it with the ResNet weight
def transformImage(image, resNetWeight, torchDevice):
    imageCopy = image.resize(size=(224, 224))
    imageTransform = resNetWeight.transforms()(imageCopy).unsqueeze(0)

    return imageTransform


# This function gets the name of the label id given in the database
def getCalTechCategory(caltechDB, label_id):
    return caltechDB.annotation_categories[label_id]


# This function finds the k similar labels
def kSimilarImages_2b_1(imageID, df, caltechDB, k=10):
    # Extract the row that we want to compare with
    checkRow = df[df["ImageID"] == imageID]
    searchDF = df[df["ImageID"] != imageID]

    # distDF = pd.DataFrame()
    df_list = []

    for index, row in searchDF.iterrows():
        # keys: "ImageID", "Label", "ResNet"
        # distance = scipy.spatial.distance.cityblock(checkRow["ResNet"].values[0], row["ResNet"])
        distance = scipy.spatial.distance.cityblock(checkRow["ResNet"].values[0], row["ResNet"])
        df_list.append({"ImageID": row["ImageID"], "Label": row["Label"], "Distance": distance})

    distanceDF = pd.DataFrame(df_list)
    results = distanceDF.sort_values(by="Distance")[0:k].reset_index()
    print("For ImageID: " + str(checkRow["ImageID"].values[0]).strip() + " & Label: " + str(checkRow["Label"].values[0]).strip())
    print(results[["Label", "Distance"]])


def kSimilarImages_2b(imageID, df, caltechDB, k):
    # Clustering and Parameters used from: https://www.analyticsvidhya.com/blog/2021/01/a-simple-guide-to-centroid-based-clustering-with-python-code/

    # Get training set of dataset
    traindf = df[df["ImageID"] % 2 == 0]

    # Extract Row of imageID
    if imageID == 1
    checkRow = df[df["ImageID"] == imageID]

    # Create the KMeans Clusters and get the centroids
    kmeans = sklearn.cluster.KMeans(n_clusters=101, init="k-means++", random_state=0, n_init="auto").fit([list(row.astype(float)) for row in traindf["ResNet"]])
    # kmeans = sklearn.cluster.DBSCAN(eps=3, min_samples=2).fit([list(row.astype(float)) for row in traindf["ResNet"]])
    # print(kmeans.cluster_centers_)
    # print(set(kmeans.labels_))

    # Go through the centroids and find the closest ones
    counter = 0
    df_list = []

    for centroid in kmeans.cluster_centers_:
        # Distance is affecting results
        distance = scipy.spatial.distance.cosine(checkRow["ResNet"].values[0], centroid)
        # if counter != caltechDB[imageID][1]: # I was thinking about comparing the image's label (centroid) to find nearest ones
        #     distance = scipy.spatial.distance.cityblock(kmeans.cluster_centers_[caltechDB[imageID][1]], centroid)
        #     df_list.append({"LabelID": counter, "Label": getCalTechCategory(caltechDB, counter), "Distance": distance})
        df_list.append({"LabelID": counter, "Label": getCalTechCategory(caltechDB, counter), "Distance": distance})
        counter = counter + 1
    
    # Sort the results and get the top k results
    distanceDF = pd.DataFrame(df_list)
    results = distanceDF.sort_values(by="Distance")[0:k].reset_index()[["LabelID", "Label", "Distance"]]

    print(results)

# This function is the code to generate the resnet
# vectors and exports the database so it can be called in k similar labels
# As a note, task2b and k similar label functions are too differet functions so
# I could test the k similar labels without waiting for the resnet model each time to 
# analyze the images.
def task2b(caltechDB):
    # Get Device
    torchDevice = getDevice()

    # Setup Model
    resNetWeight = torchvision.models.ResNet50_Weights.DEFAULT
    resNetModel = torchvision.models.resnet50(progress=True, weights=resNetWeight).to(torchDevice)
    for parameter in resNetModel.parameters():
        parameter.requires_grad = False
    resNetModel.zero_grad(set_to_none=True)
    resNetModel.eval()

    # Create Pandas Dataframe
    df_list = []

    # For Loop in databases
    for i in tqdm(range(0, len(caltechDB)), desc="Extracting Features", ncols=100):
        # for i in range(1580, 1582, 1):
        # Get Image
        image = caltechDB[i][0]

        # Convert Non-RGB to RGB
        if image.mode != "RGB":
            image = convert_to_rgb(image)

        # Setup Image
        imageTransform = transformImage(image, resNetWeight, torchDevice).to(torchDevice)

        # Put image into model and get results
        result = resNetModel(imageTransform).squeeze(0).softmax(0).to(torchDevice)

        # Attach to List
        row_df = {
            "ImageID": i,
            "Label": getCalTechCategory(caltechDB, caltechDB[i][1]),
            "LabelID": caltechDB[i][1],
            "ResNet": result.detach().cpu().numpy(),
        }
        df_list.append(row_df)

    torch.cuda.empty_cache()

    # Convert List to DataFrame for easy accessability
    df = pd.DataFrame(df_list)
    # print(df.head(5))

    df.to_pickle("./testFile_2b")

# # This function gets the reuqired user input for the functions
def userInput():
    IMAGE_INPUT = input(
    """
    Provide one of the following:
    1. An Image ID in the Caltech101 dataset in range [0, 8676].
    2. The name of an image file in /Code/input/ directory (eg: image.jpg).
    """
    )

    K = int(input("Enter K, the number of similar labels to find for ResNet model."))

    image = get_query_image(IMAGE_INPUT)
    print("User inputed image ID / image: ", IMAGE_INPUT)
    display(image)
    return image, k


# # Main code that includes the user input, then calls task2b, then the k similar labels function
def main():
    # Load Database
    caltechDB = torchvision.datasets.Caltech101("./Databases/CaltechDB/", download=True, target_type="category")

    image, k = userInput()
    task2b(caltechDB)
    # Get K Similar Images based on an image
    df = pd.read_pickle("./testFile_2b")
    kSimilarImages_2b(image, df, caltechDB, k)

