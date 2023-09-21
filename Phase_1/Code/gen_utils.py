import os
import shutil
import pickle
import json


# Directory Management Utilities
# Directory Creation
def create_directory(directory_path):
    if os.path.exists(directory_path):
        print("Directory {} already Exists!".format(directory_path))
        created_directory = False
    else:
        try:
            os.mkdir(directory_path)
            created_directory = True
        except Exception as e:
            print("Exception in Creating Directory:", e)
            created_directory = False
    return created_directory


# Directory Management Utilities
# Directory Deletion
def delete_directory(directory_path):
    try:
        if os.path.exists(directory_path):
            shutil.rmtree(directory_path)
        else:
            print("Directory Path {} does not exist!".format(directory_path))
    except Exception as e:
        print("Exception in Deleting Directory:", directory_path, e)


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
