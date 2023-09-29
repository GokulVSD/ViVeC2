import numpy as np
from scipy.spatial.distance import cosine, euclidean


# Decorator Function: Checks if Input Vectors are of same Shape for each 1-D Distance Measures
def check_lengths(func):
    def check(v1, v2):
        v1, v2 = np.array(v1), np.array(v2)
        if v1.shape != v2.shape:
            print(func, "Mismatch in Input Vector Shapes!")
            return None
        return func(v1, v2)
    return check


# Manhattan Distance Calculation
@check_lengths
def manhattan_distance(v1, v2):
    v1, v2 = np.array(v1), np.array(v2)
    md = np.sum(np.abs(v1 - v2))
    return md


# Euclidean Distance Calculation
@check_lengths
def euclidean_distance(v1, v2):
    v1, v2 = np.array(v1), np.array(v2)
    ed = euclidean(v1, v2)
    return ed


# Cosine Distance Calculation
@check_lengths
def cosine_distance(v1, v2):
    v1, v2 = np.array(v1), np.array(v2)
    cd = cosine(v1, v2)
    return cd


# Intersection Distance Calculation
@check_lengths
def intersection_distance(v1, v2):
    v1, v2 = np.array(v1), np.array(v2)
    min_values = [min(v1[i], v2[i]) for i in range(0, v1.shape[0], 1)]
    max_values = [max(v1[i], v2[i]) for i in range(0, v1.shape[0], 1)]
    ind = 1 - (sum(min_values) / sum(max_values))
    return ind


def get_distance_measures():
    distance_measures = {
        "manhattan": manhattan_distance,
        "euclidean": euclidean_distance,
        "cosine": cosine_distance,
        "intersection": intersection_distance
    }
    return distance_measures


# Functionality Tests
def main():
    v1 = [1, 2, 1]
    v2 = [1, 2, 2]
    print("MD:", manhattan_distance(v1, v2))
    print("ED:", euclidean_distance(v1, v2))
    print("CD:", cosine_distance(v1, v2))
    print("IND:", intersection_distance(v1, v2))


if __name__ == '__main__':
    main()
