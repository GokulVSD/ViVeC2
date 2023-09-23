import pandas as pd
import numpy as np
import torch
import torchvision
import PIL
from pathlib import Path
from tqdm import tqdm
import scipy

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

def kSimilarImages_2b(imageID, df, caltechDB, k=10):
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
    for i in tqdm (range(0, len(caltechDB)), desc="Extracting Database Images' Features...", ncols=120):
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
            row_df = {"ImageID": i, "Label": getCalTechCategory(caltechDB, caltechDB[i][1]), "ResNet": result.detach().cpu().numpy()}
            df_list.append(row_df)
        torch.cuda.empty_cache()

    # Convert List to DataFrame for easy accessability
    df = pd.DataFrame(df_list)
    # print(df.head(5))

    # Get K Similar Images based on an image
    kSimilarImages_2b(0, df, caltechDB)


task2b()

    