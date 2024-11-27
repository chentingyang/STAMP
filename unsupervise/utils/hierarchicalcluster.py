import numpy as np


def euclidean_distance(a, b):
    return np.linalg.norm(a - b)


def hierarchical_clustering(data, min_clusters=1, max_distance=np.inf):
    distances = np.zeros((len(data), len(data)))
    distances.fill(np.inf)
    np.fill_diagonal(distances, 0)

    for i in range(len(data)):
        for j in range(i+1, len(data)):
            distances[i, j] = euclidean_distance(data[i], data[j])
            distances[j, i] = distances[i, j]

    clusters = [[i] for i in range(len(data))]
    while len(clusters) > min_clusters:
        min_distance = np.inf
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                cluster_distance = 0
                for k in clusters[i]:
                    for l in clusters[j]:
                        cluster_distance += distances[k, l]
                cluster_distance /= (len(clusters[i]) * len(clusters[j]))
                if cluster_distance < min_distance:
                    min_distance = cluster_distance
                    merge_indices = (i, j)
        if min_distance > max_distance:
            break
        # print(min_distance)
        i, j = merge_indices
        clusters[i].extend(clusters[j])
        del clusters[j]
    labels = [1] * len(data)
    for i in range(len(clusters)):
        for j in clusters[i]:
            labels[j] = i
    return labels

