import sobol_seq
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict



def  subset_clustering(n_clusters, X, Y, last_n = 5):
    result_x = []
    kmn = KMeans(n_clusters=int(n_clusters), random_state=0)
    kmn.fit(X)
    lables = kmn.labels_
    centers = kmn.cluster_centers_
    min_Y = [np.inf]*n_clusters
    selected_id = [0] * n_clusters
    for c_id, center in enumerate(centers):
        min_dis_x_id = 0
        for x_id, x in enumerate(X):
            if lables[x_id] == c_id:
                if Y[x_id] < min_Y[c_id]:
                    min_Y[c_id] = Y[x_id]
                    selected_id[c_id] = x_id
        result_x.append(X[min_dis_x_id])

    for i in range(last_n):
        if len(X) - (i+1) not in selected_id:
            selected_id.append(len(X) -(i+1))

    return selected_id



def  subset_decomposition(vector_num, X, Y, last_n = 5):
    total_num = X.shape[0]
    Xdim = X.shape[1]

    vector = 2 * sobol_seq.i4_sobol_generate(Xdim, vector_num) - 1

    selected_id = []

    dis = np.empty((vector_num,total_num), dtype=np.float64)
    for i in range(dis.shape[0]):
        dis[i] = np.linalg.norm(vector[i]-X,axis=1)
    dis = dis.T

    cluster_label = np.argmin(dis, axis=1)
    # min_distance_matrix = np.min(dis, axis=1)[:, np.newaxis]

    for i in range(vector_num):
        label = np.argwhere(cluster_label==i)[:,0]
        if len(label) != 0:
            min_label_id = np.argmin(Y[label],axis=0)[0]
            selected_id.append(label[min_label_id])

    for i in range(last_n):
        if len(X) - (i+1) not in selected_id:
            selected_id.append(len(X) -(i+1))

    return selected_id

def  subset_random(num, X, Y, last_n = 5):
    total_num = X.shape[0]
    selected_id = list(np.random.randint(0, total_num, num))

    return selected_id

def local_search(distance_matrix, x_id, Y, distance_threshold):
    sorted_dist_mat = np.argsort(distance_matrix,axis=1)


    if distance_matrix[x_id][sorted_dist_mat[x_id][1]] > distance_threshold:
        return x_id
    else:
        label = np.argwhere(distance_matrix[x_id] <= distance_threshold)[:, 0]
        min_label_id = label[np.argmin(Y[label], axis=0)[0]]
        if min_label_id == x_id:
            return x_id
        else:
            return local_search(distance_matrix, min_label_id, Y, distance_threshold)



def subset_localoptima(subset_number, X, Y,last_n = 5):
    total_num = X.shape[0]
    Xdim = X.shape[1]

    distance_threshold = 0.12*Xdim

    selected_id = []

    dis = np.empty((total_num, total_num), dtype=np.float64)
    for i in range(dis.shape[0]):
        dis[i] = np.linalg.norm(X[i] - X, axis=1)


    for x_id,x in enumerate(X):
        local_id = local_search(dis, x_id, Y,distance_threshold)
        if local_id not  in selected_id:
            selected_id.append(local_id)

    # for i in range(vector_num):
    #     label = np.argwhere(cluster_label == i)[:, 0]
    #     if len(label) != 0:
    #         min_label_id = np.argmin(Y[label], axis=0)[0]
    #         selected_id.append(label[min_label_id])

    for i in range(last_n):
        if len(X) - (i + 1) not in selected_id:
            selected_id.append(len(X) - (i + 1))

    return selected_id
