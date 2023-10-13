from os import path
from torch import load, save

DATABASE_PATH = path.join(path.dirname(path.dirname( __file__ )), 'database')


def store(obj, filename):
    """
    Stores in /Code/database/ as a pickled .pt file. Tracked by git LFS.
    """
    print("\n", "Saving: ", filename, "\n")

    save(obj, path.join(DATABASE_PATH, filename))


def retrieve(filename):
    """
    Retrieves .pt file in /Code/database/ as object.
    """
    return load(path.join(DATABASE_PATH, filename))