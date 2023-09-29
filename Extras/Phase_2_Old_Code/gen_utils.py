import os
import pickle
import json
import pandas as pd
import numpy as np


# Read from a Pickle File
def read_pickle_file(file_name=None):
    file_name = "dataset_feature_descriptors.pkl" if file_name is None else file_name
    with open(file_name, "rb") as f:
        data = pickle.load(f)
    return data


# Write to a Pickle File
def write_pickle_file(data, file_name=None):
    file_name = "dataset_feature_descriptors.pkl" if file_name is None else file_name
    with open(file_name, "wb") as f:
        pickle.dump(data, file=f)


# Read from a JSON File
def read_json_file(file_name):
    with open(file_name, "r", encoding="utf8") as f:
        data = json.load(f)
    return data


# Write to a JSON File
def write_json_file(data, file_name):
    with open(file_name, "w", encoding="utf8") as f:
        data = json.dumps(str(data))
        f.write(data)


def exportPandasChunks(df, split=2, directoryPath="./Databases/pickle_chunks/", compressionType="zstd"):
    count = 1
    for i in range(0, len(df) - 1, (len(df) // split)):
        print("Exporting Chunk from " + str(i) + " to " + str(i + (len(df) // split) + 1))
        df[i : (i + (len(df) // split) + 1)].to_pickle("{0}/chuck{1}.pickle".format(directoryPath, count), compression=compressionType)
        count = count + 1


def importPandasChunks(directoryPath="./Databases/pickle_chunks/", compressionType="zstd"):
    combined_df = pd.DataFrame()
    count = 0
    # Search directory code from: https://www.geeksforgeeks.org/how-to-iterate-over-files-in-directory-using-python/#
    for filename in os.scandir(directoryPath):
        if filename.is_file():
            print(filename.name)
            df_chunk = pd.read_pickle(filename.path, compression=compressionType)
            count = count + len(df_chunk)
            combined_df = pd.concat([combined_df, df_chunk], axis=0)

    print("Chunked Dataframes Rows: " + str(count))
    print("Combined Dataframe Rows: " + str(len(combined_df)))

    return combined_df


def testFunctions():
    # Just for testing: https://stackoverflow.com/questions/32752292/how-to-create-a-dataframe-of-random-integers-with-pandas
    df = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list("ABCD"))
    exportPandasChunks(df)
    combined_df = importPandasChunks()
    print(len(combined_df))
