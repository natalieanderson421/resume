# Natalie Anderson
# CMSC423 Fall 2019
# Input file 'rosalind_ba8c.txt' provided by Rosalind.info

import numpy as np
import math

"""
lloyds.py implements Lloyd's k-means clustering algorithm
for finding clusters in a dataset.

The algorithm first chooses k arbitrary Center points
from the data, and then iteratively performs
"Centers to Clusters" and "Clusters to Centers" steps
until the algorithm converges (no changes between
iterations) and prints the resulting set of k Centers.

Dataset format:
The first line provides integers k (number of center
points to start with) and m.
This line is followed by a set of data points in
m-dimensional space.
"""


# calculates the euclidean distance between points a and b.
# a and b are m-dimensional
# returns sqrt( (b1 - a1)^2 + ... + (bm - am)^2 )
def euclidean(a, b, m):
    dist = 0
    for i in range(m):
        dist += (b[i] - a[i]) ** 2
    return math.sqrt(dist)


# calculates the center of gravity of
# the m-dimensional points in cluster.
def center_of_gravity(cluster, m):
    # cg is the m-dimensional center of gravity
    cg = []
    for i in range(m):
        j = 0
        for c in cluster:
            j += c[i]
        # appends avg of each dimension to cg
        cg.append(j / len(cluster))
    return cg


# boolean function indicating whether
# or not the set of Centers has changed
# between iterations
def converged(old, new):
    return np.array_equal(old, new)


def main():
    # read integers k and m from the file
    file = open("rosalind_ba8c.txt", "r")
    ints = file.readline().rstrip().split(" ")
    k = int(ints[0])
    m = int(ints[1])

    # read data points from the file
    data = []
    ints = file.readline().rstrip()
    while ints:
        ints = ints.split(" ")
        d = []
        for i in ints:
            d.append(float(i))
        data.append(d)
        ints = file.readline().rstrip()

    # select first k data points as
    # the first set of Centers
    curr_centers = []
    for i in range(k):
        curr_centers.append(data[i])
    curr_centers = np.array(curr_centers)
    # old_centers will be used in convergence comparison
    old_centers = np.array([])

    while not converged(old_centers, curr_centers):
        # assign data points to clusters
        clusters = []
        for i in range(k):
            clusters.append([])

        # calculate distance of point to each center
        # and assign to the cluster of the nearest center
        for d in data:
            minimum = 0
            dist = euclidean(d, curr_centers[0], m)
            for i in range(1, k):
                curr_dist = euclidean(d, curr_centers[i], m)
                if curr_dist < dist:
                    dist = curr_dist
                    minimum = i
            clusters[minimum].append(d)

        old_centers = curr_centers

        # centers for next iteration are
        # the center of gravity of each cluster
        cgs = []
        for i in range(k):
            cgs.append(center_of_gravity(clusters[i], m))
        curr_centers = np.array(cgs)

    # print final centers
    for center in curr_centers:
        for i in range(m):
            print(round(center[i], 3), end=" ")
        print("\n")


main()
