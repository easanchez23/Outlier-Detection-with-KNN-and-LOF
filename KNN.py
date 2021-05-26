from math import sqrt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


def getEucDistance(row, row2):
    distance = float(0.0)
    for i in range(len(row)):
        if set(row) == set(row2):
            for x in range(len(row)): # need to check if our dataset has a index or ouput in the last spot of each row
                row[x] = row[x]*1.000000001
        distance = distance + float((row[i] - row2[i]) * float(row[i] - row2[i]))
    return sqrt(distance)


def getDistances(data):
    distances = np.zeros((data.shape[0], data.shape[0], 3))
    index1 = 0
    for row in data:
        index2 = 0
        for row2 in data:
            if index1 != index2:
                if all(distances[index1][index2]) == 0:
                    temp = getEucDistance(row, row2)
                    distances[index1][index2] = (index1, index2, temp)
                    distances[index2][index1] = (index2, index1, temp)
            index2 = index2 + 1
        index1 = index1 + 1
    for j,i in enumerate(distances):
        distances[j] = i[i[:,2].argsort()]
    return distances  # notice getEucDistance(row, row2) returns 0 when row = row2


def knnOutliers_matrix(data, n_neighbors, detector="mean"):
    """
    Returns indices for datapoints presumed to be outliers.
    Arguments: data, # of nearest neighbors, detector mode
    Returns: list of indicies corresponding to datapoints fed in
    
    LRD: Local reachability density
    LOF: Local Outlier Factor
    """
    
    # list of indicies for outlier datapoints, outliers are those with LOF > 1
    LOF = []

    # getting distances for all pairs of datapoints
    distances = getDistances(data)
    # print(distances)

    distances = np.delete(distances, (0), axis=1)
    k_nearest_matrix = distances[:, :n_neighbors]
    print(k_nearest_matrix)


    # dictionary to hold all distances for each pair of points
    dictionary = {}
    
    for i in k_nearest_matrix:
        for x in i:
            dictionary.update({(x[0], x[1]): x[2]})
            dictionary.update({(x[1], x[0]): x[2]})

    # for every datapoint
    for i in range(len(k_nearest_matrix)):
    # for i in range(0,1):
        holder = []
        column = k_nearest_matrix[i]
        # print(column)
        lrdp = 0
        
        # gets LRD for this datapoint
        for x in range(len(column)):
            temp = int(column[:, 1][x])
            lrdp = lrdp + max(k_nearest_matrix[temp][:, 2][n_neighbors - 1], dictionary.get((i, temp)))
        holder.append((i, (1 / ((1 / n_neighbors) * lrdp))))
        # gets LRD for all of its kNN
        for y in range(len(column)):
            num = int(column[:, 1][y])
            column2 = k_nearest_matrix[num]
            lrdp = 0
            
            # going through "their" neighbors
            for n in range(len(column2)):
                temp = int(column2[:, 1][n])
                lrdp = lrdp + max(k_nearest_matrix[temp][:, 2][n_neighbors - 1], dictionary.get((num, temp)))
            holder.append((num, (1 / ((1 / n_neighbors) * lrdp))))
        

        # calculating LOF for each datapoint
        numerator = 0
        denominator = 0
        for a in holder:
            if a[0] != i:
                numerator = numerator + a[1]
            else:
                denominator = a[1]
        lof = (1 / n_neighbors * (numerator / denominator))
        LOF.append(lof)
    print(LOF)
    return [LOF.index(x) for x in LOF if x > 1]


if __name__ == '__main__':
     data = np.genfromtxt("iris.csv", delimiter=',')
    #import data however you like. Make sure to clean the data by removing any value or index rows or columns.
    # normalizing the data
    for j,i in enumerate(data):
        data[j] = i / np.linalg.norm(i)
    outliers = kd_tree_kNN_scipy(data, 3)
    print(outliers)

