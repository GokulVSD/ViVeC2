from utils.database_utils import retrieve
from utils.dataset_utils import initialize_dataset
from utils.database_utils import store
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def format_feature_db(feature_db):
    """
    Reformat feature vector database to utilize SQL like functionality (where clause)
    using Pandas DataFrame
        Input Format - dictionary: {image_id: (label, feature_vector), ... }
        Output Format - list: [[image_id, label, feature_vector], ... ] 
    """
    fmt_feature_db = []
    for k, v in feature_db.items():
        line_items= []
        line_items.append(k)                                    # key is image-id
        line_items.append(v[0])                                 # first element of value is label
        line_items.append(v[1].tolist())                        # second element of value is feature vector
        assert(len(v[1].tolist()) == len(feature_db[0][1]))
        fmt_feature_db.append(line_items)
    return fmt_feature_db


def build_representatve_feature_db(feature_db, label_space, method):

    """
    Select rows from feature database based on labels
    For each label find centroid based on feature vector of 
    images under that label
    """

    # rep_feture_db = []
    rep_feture_db = {}
    fmt_feature_db = format_feature_db(feature_db)
    fmt_feature_df = pd.DataFrame(fmt_feature_db)
    fmt_feature_df.columns = ['image_id', 'label', 'feature_vector']

    # if method == 'mean':
    #     for l in label_space:
    #         category_df = fmt_feature_df.loc[fmt_feature_df['label'] == l]
    #         if category_df.shape[0] > 0:
    #             rep_feature = [l]
    #             fv = category_df['feature_vector'].tolist()
    #             fv_np = np.array(fv)
    #             rfv = np.average(fv_np, axis=0)
    #             rep_feature.append(rfv)
    #             rep_feture_db.append(rep_feature)
    
    if method == 'mean':
        for l in label_space:
            category_df = fmt_feature_df.loc[fmt_feature_df['label'] == l]
            if category_df.shape[0] > 0:
                fv = category_df['feature_vector'].tolist()
                fv_np = np.array(fv)
                rfv = np.average(fv_np, axis=0)
                # rep_feture_db[l]= rfv
                rep_feture_db[l]= [rfv]
    elif method == 'kmeans':
        for l in label_space:
            category_df = fmt_feature_df.loc[fmt_feature_df['label'] == l]
            if category_df.shape[0] > 0:
                cluster_centers = []
                fv = category_df['feature_vector'].tolist()
                fv_np = np.array(fv)

                kmeans = KMeans(n_clusters=1, random_state=0, n_init="auto").fit(fv_np)
                cluster_centers.append(kmeans.cluster_centers_[0])

                kmeans = KMeans(n_clusters=5, random_state=0, n_init="auto").fit(fv_np)
                for c in kmeans.cluster_centers_:
                    cluster_centers.append(c)
                
                rep_feture_db[l]= cluster_centers
    
    return rep_feture_db


def main():

    # Get Caltech101 dataset and retrieve categories in the dataset
    dataset = initialize_dataset()
    label_space = dataset.categories

    # Given Feature Space for the application
    feature_space = ['color', 'hog', 'avgpool', 'layer3', 'fc']

    # Iterate through the feature space to generate rep. feature vector for each lebel and store as dataset
    for feature_name in feature_space:
        feature_vectors = retrieve(f'{feature_name}.pt')
        rep_label_feature = build_representatve_feature_db(feature_vectors, label_space, 'kmeans')
        rep_label_db_name = 'rep_label_' + feature_name + '.pt'
        store(rep_label_feature, rep_label_db_name)

    
    print(
    """
    > Representative feature vectors for the labels are stored in dictionary per label.
    > Stored in binary form, the key is the label, val is representative feature vector.
    """
    )



if __name__ == "__main__":
    main()