from scipy.spatial.distance import cityblock, correlation, cosine
import numpy as np

FEATURE_SPACE_DISTANCE_MAP = {
    # Manhattan distance (City-block distance): This distance can be imagined as the length needed
    # to move between two points in a grid where you can only move up, down, left or right. This
    # measure is sensitive to differences along each dimension of the feature vectors, capturing
    # the total absolute difference between corresponding moments, which makes it great for
    # assessing differences in color distribution in terms of absolute deviations. Since it
    # does not square the differences like Euclidean distance, it is less susceptible outliers
    # when many dimensions differ.
    "color": cityblock,
    # Correlation similarity: This measures the linear relationship between HOG feature vectors,
    # and performs well with features that have a strong linear correlation or dependency in
    # their gradient orientation distributions, taking into account directionality.
    "hog": correlation,
    # Cosine similarity: We use Cosine since the output of ResNet layers are
    # normalized, reducing the importance of magnitude. It is good at discerning
    # semantic or structural similarity rather than their absolute feature values.
    # It also works well with sparse feature spaces such as those found in CNNs.
    "avgpool": cosine,
    "layer3": cosine,
    "fc": cosine,
    "resnet_output": cosine,
}


def get_distance_fn(feature_space):
    """
    Retrieve the distance function chosen for the specific feature space.
    """
    return FEATURE_SPACE_DISTANCE_MAP[feature_space]


def top_k_distance_ranker(k, query_vector, feature_vectors, distance_fn):

    distances = []

    for img_id, feature_tuple in feature_vectors.items():
        label, feature_vector = feature_tuple
        distances.append((distance_fn(query_vector, feature_vector), img_id, label))

    distances = sorted(distances)

    return distances[:k]

def find_centroids(feature_vectors):
    previous_label = ""
    item_count = 0
    total_feature = 0
    result = {}
    for i, feature_item in enumerate(feature_vectors.items()):
        label = feature_item[1][0]
        feature = feature_item[1][1]
        if item_count > 0 and label != previous_label:
            result[previous_label] = total_feature/item_count
            total_feature = feature
            item_count = 1
        else:
            total_feature = np.add(total_feature, feature)
            item_count += 1
        previous_label = label
    return result

# K-Means modified for task 8, 10
def getEachLabelCentroids(df, feature = "ResNet", fileName="task_2b_resnet_centers"):
    centroidLists = []
    # Go through each label id
    for i in tqdm(range(0, len(df["LabelID"].unique())), desc="Finding Clusters", ncols=100):
        # Dictionary to keep track of the best one
        maxK = {"K": 0, "SIL": sys.float_info.min, "Centroids": [], "Interia": sys.float_info.min}

        # Split the training dataset into even images for training (mentioned in ED Discussion Post)
        # then extract the ones with i's label id and convert to format readable to kmeans package
        traindf = df[df["ImageID"] % 2 == 0]
        traindf = traindf[traindf["LabelID"] == i]
        dataList = [list(row.astype(float)) for row in traindf[feature]]

        # Taking too long, so using a counter to stop kinda of like earlystop in Tensorflow
        counter = 0

        # Range goes to length of traindf as when doing a fixed like 500, I got error messages saying not enough
        # samples, so to fix it I use the length of the traindf
        for k in range(2, len(traindf)):
            # Looking at the example graphs, it seems to generally flat line for a long time and 50 makes sure we don't stop too early
            if counter >= 50:
                break
            else:
                # Using kmeans, get the silhouette score
                kmeans = sklearn.cluster.KMeans(n_clusters=k, init="k-means++", random_state=0, n_init="auto").fit(dataList)
                # For silhouette score, used euclidean as it was in the articles and it seemed to give good results
                silhouetteScore = sklearn.metrics.silhouette_score(dataList, kmeans.labels_, metric="euclidean")
                # Trying to find max, these is how we did this in ASU CSE 110
                if silhouetteScore > maxK["SIL"]:
                    maxK["K"] = k
                    maxK["SIL"] = silhouetteScore
                    maxK["Centroids"] = kmeans.cluster_centers_
                    maxK["Interia"] = kmeans.inertia_
                    # print(k, silScore, maxK["SIL"])
                    # Since there is a change i, counter needs to reset to 0
                    counter = 0
                else:
                    # No update, increase counter so we can stop if there is no changes
                    counter = counter + 1
        # print(maxK)
        # Now finally save the best results to the centroids list for label id i
        centroidLists.append({"LabelID": i, "Centroids": maxK["Centroids"], "Interia": maxK["Interia"], "Silhouette": maxK["SIL"]})
        sleep(1)
        gc.collect()

    # Convert to DataFrame and save results, so it doesn't need to be called all the time
    centerDF = pd.DataFrame(centroidLists)
    centerDF.to_pickle("./database/{0}.pickle".format(fileName))
    # centerDF


def kSimilarImages_2b(imageVector, df, caltechDB, k, resNetWeight, resNetModel, torchDevice, fileName="task_2b_resnet_centers"):
    # Load the Dataframe
    centerDF = pd.read_pickle("./database/{0}.pickle".format(fileName))

    df_list = []
    # Go through all label ids' centroids
    for index, row in centerDF.iterrows():
        distList = []
        # For each centroid, get the distance for label id i
        for centroid in row["Centroids"]:
            distList.append(scipy.spatial.distance.cosine(result, centroid))
        # Get the min dist as that is the closest for label id i to the image that we are comparing to
        df_list.append({"LabelID": row["LabelID"], "Label": getCalTechCategory(caltechDB, row["LabelID"]), "Distance": min(distList)})

    # Sort and get the top k labels
    result = pd.DataFrame(df_list).sort_values(by="Distance")[0:k].reset_index()[["LabelID", "Label", "Distance"]]

    return result