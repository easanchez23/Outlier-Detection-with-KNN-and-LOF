

# kd-tree implementation

# Written by Khalid and Esteban, Matteo gave go-ahead for using scipy.spatial.KDTree
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import scipy.stats as sp


def kd_tree_kNN_scipy(data, n_neighbors):
    # list of indicies for outlier datapoints, outliers are those with LOF > 1
    LOF = []
    
    # p = 2 means euclidiean distance
    tree = KDTree(data)
    # quer = tree.query(data, k=data.shape[0], p=2)
    quer = tree.query(data, k=n_neighbors+1, p=2)

    # first column is 0th closest point, aka itself or another datapoint with same feature values
    query = [quer[0][:,0:n_neighbors+1], quer[1][:,0:n_neighbors+1]]
    for i in range(len(data)):
        holder = []
        column = (query[0][i], query[1][i])
        lrdp = 0
        # gets LRD for this datapoint
        for x in range(len(column[1])):
            temp = column[1][x]
            lrdp = lrdp + max(query[0][temp][n_neighbors], column[0][x]) 
        holder.append((i, (1 / ((1 / n_neighbors) * lrdp))))     
        # gets LRD for all of its kNN
        for y in range(len(column[1])):
            num = int(column[1][y])
            column2 = query[1][num]
            column2_dist = query[0][num]
            # print(column2_dist)
            lrdp = 0

            # going through "their" neighbors
            for n in range(len(column2)):
                temp = int(column2[n])
                lrdp = lrdp + max(query[0][temp][n_neighbors], column2_dist[n])  
            holder.append((num, (1 / ((1 / n_neighbors) * lrdp))))
        # calculating LOF for each datapoint
        numerator = 0
        denominator = 0
        for a in holder:
            #print(a)
            if a[0] != i:
                numerator = numerator + a[1]
            else:
                denominator = a[1]
        lof = (1 / n_neighbors * (numerator / denominator))
        LOF.append(lof)
        # play around with cutoff for each dataset. Generally outliers are greater than 1.
    return [LOF.index(x) for x in LOF if x > np.median(LOF)+(2)*(np.std(LOF))]

if __name__ == '__main__':
    data = np.genfromtxt("iris.csv", delimiter=',')
    #import data however you like. Make sure to clean the data by removing any value or index rows or columns.
    # normalizing the data
    for j,i in enumerate(data):
        data[j] = i / np.linalg.norm(i)
    outliers = kd_tree_kNN_scipy(data, 3)
    print(outliers)



