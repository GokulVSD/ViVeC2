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


def pt2df(FEATURE_SPACE):
    data = retrieve(f'latent_{FEATURE_SPACE}.pt')
    return pd.DataFrame(data).T

# Reference: https://stackoverflow.com/questions/13331698/how-to-apply-a-function-to-two-columns-of-pandas-dataframe
def combineRows(a, b):
    return (a, b)

def df2pt(df):
    combineCol = df.apply(lambda x: test(x[0], x[1]), axis=1)
    return combineCol.to_dict()
