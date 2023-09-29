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

# Research References: 
# https://saturncloud.io/blog/how-to-check-if-pytorch-is-using-the-gpu/
# https://saturncloud.io/blog/how-to-train-a-pytorch-model-on-gpu/
# https://docs.scipy.org/doc/scipy/reference/spatial.distance.html
# https://learnopencv.com/image-classification-using-transfer-learning-in-pytorch/ 
# https://pytorch.org/docs/master/generated/torch.topk.html
# https://pytorch.org/vision/stable/models.html

def getDevice():
    torchDevice = "cpu"
    if  torch.cuda.is_available():
        torchDevice = "cuda"

    print("Using Device: ", torchDevice)
    return torchDevice

def transformImage(image, resNetWeight, torchDevice):
    imageCopy = image.resize(size=(224,224))   
    imageTransform = resNetWeight.transforms()(imageCopy).unsqueeze(0)

    return imageTransform    

def getCalTechCategory(caltechDB, label_id):
    return caltechDB.annotation_categories[label_id]

def kSimilarImages_2b_1(imageID, df, caltechDB, k=10):
    # Extract the row that we want to compare with
    checkRow = df[df["ImageID"] == imageID]
    searchDF = df[df["ImageID"] != imageID]

    # distDF = pd.DataFrame()
    df_list = []

    for index, row in searchDF.iterrows():
        # keys: "ImageID", "Label", "ResNet"
        distance = scipy.spatial.distance.cityblock(checkRow["ResNet"].values[0], row["ResNet"])
        df_list.append({"ImageID": row["ImageID"], "Label": row["Label"], "Distance": distance})
    
    distanceDF = pd.DataFrame(df_list)
    results = distanceDF.sort_values(by="Distance")[0:k].reset_index()
    print("For ImageID: " + str(checkRow["ImageID"].values[0]).strip() + " & Label: " + str(checkRow["Label"].values[0]).strip())
    print(results[["Label", "Distance"]])

def kSimilarImages_2b(imageID, df, caltechDB, k=10):
    # Clustering and Parameters used from: https://www.analyticsvidhya.com/blog/2021/01/a-simple-guide-to-centroid-based-clustering-with-python-code/

    # Get training set of dataset
    traindf = df[df["ImageID"] % 2 == 0]

    # Extract Row of imageID
    checkRow = df[df["ImageID"] == imageID]

    # Create the KMeans Clusters and get the centroids
    kmeans = sklearn.cluster.KMeans(n_clusters=101, init="k-means++", random_state=0, n_init="auto").fit([list(row.astype(float)) for row in traindf["ResNet"]])
    # print(kmeans.cluster_centers_)
    # print(set(kmeans.labels_))

    # Go through the centroids and find the closest ones
    counter = 0
    df_list = []
    for centroid in kmeans.cluster_centers_:
        # Distance is affecting results
        distance = scipy.spatial.distance.cosine(checkRow["ResNet"].values[0], centroid)
        df_list.append({"LabelID": counter, "Label": getCalTechCategory(caltechDB, counter), "Distance": distance})
        counter = counter + 1

    # Sort the results and get the top k results
    distanceDF = pd.DataFrame(df_list)
    results = distanceDF.sort_values(by="Distance")[0:k].reset_index()[["LabelID", "Label", "Distance"]]

    print(results)

def task2b():
    # Get Device
    torchDevice = getDevice()

    # Load Database
    caltechDB = torchvision.datasets.Caltech101("./Databases/CaltechDB/", download=True, target_type="category")

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
    for i in tqdm (range(0, len(caltechDB)), desc="Extracting Features", ncols=100):
    # for i in range(1580, 1582, 1):
        # Get Image
        image = caltechDB[i][0]

        # Skip Non-RGB Images
        if image.mode == "RGB":
            # Setup Image
            imageTransform = transformImage(image, resNetWeight, torchDevice).to(torchDevice)

            # Put image into model and get results
            result = resNetModel(imageTransform).squeeze(0).softmax(0).to(torchDevice)

            # Attach to List
            row_df = {"ImageID": i, "Label": getCalTechCategory(caltechDB, caltechDB[i][1]), "LabelID": caltechDB[i][1], "ResNet": result.detach().cpu().numpy()}
            df_list.append(row_df)
        torch.cuda.empty_cache()

    # Convert List to DataFrame for easy accessability
    df = pd.DataFrame(df_list)
    # print(df.head(5))

    df.to_pickle("./testFile_2b")

# task2b()

# Get K Similar Images based on an image
caltechDB = torchvision.datasets.Caltech101("./Databases/CaltechDB/", download=True, target_type="category")
df = pd.read_pickle("./testFile_2b")
kSimilarImages_2b(0, df, caltechDB)


    