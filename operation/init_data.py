import numpy as np
from sklearn.cluster import KMeans
import sobol_seq
from pyDOE import lhs

def InitData(Init_method, KB, Init, Xdim, Dty):
    if np.float64 != Dty:
        print('Unsupport data type! shut down')
        return
    if Init_method == 'random':
        train_x = 2 * np.random.random(size=(Init, Xdim)) - 1

    elif Init_method == 'uniform':
        # train_x = 2 * lhs(Xdim, Init) - 1
        train_x = 2 * sobol_seq.i4_sobol_generate(Xdim, Init) - 1

    # elif Init_method == 'delta':
    #     if KB.len == 0 or KB.len == 1:
    #         if np.float64 == Dty:
    #             train_x = 2 * lhs(Xdim, Init) - 1
    #         else:
    #             print('Unsupport data type! shut down')
    #             return
    #     else:
    #         train_x = KB.initial_x[KB.len - 1]
    #         delta = 0.2
    #         train_x  = train_x + delta
    #         train_x[train_x[:,:]>1] = train_x[train_x[:,:]>1] - 2


    elif Init_method == 'grid':
        if KB.len == 0:
            if np.float64 == Dty:
                train_x = 2 * np.random.random(size=(Init, Xdim)) - 1
            else:
                print('Unsupport data type! shut down')
                return
        else:
            train_x = KB.local_optimal[0]
            for i in range(1, KB.len):
                train_x = np.vstack((train_x, KB.local_optimal[i]))
            train_x = np.unique(train_x, axis=0)

            if len(train_x) == Init:
                pass
                # train_x = np.array(train_x, dtype=Dty)
            elif len(train_x) > Init:
                result_x = []
                kmn = KMeans(n_clusters=int(Init), random_state=0)
                kmn.fit(train_x)
                lables = kmn.labels_
                centers = kmn.cluster_centers_
                for c_id,center in enumerate(centers):
                    min_dis = 100
                    min_dis_x_id = 0
                    for x_id, x in enumerate(train_x):
                        if lables[x_id] == c_id:
                            dis = np.linalg.norm(x - center)
                            if dis < min_dis:
                                min_dis = dis
                                min_dis_x_id = x_id
                    result_x.append(train_x[min_dis_x_id])

                train_x = np.array(result_x)
                # train_x = np.concatenate(
                #     (train_x, 2 * np.random.random(size=(Init - len(train_x), Xdim)) - 1))
            else:
                # train_x = np.array(train_x, dtype=Dty)
                train_x = np.concatenate(
                    (train_x, 2 * np.random.random(size=(Init - len(train_x), Xdim)) - 1))
    else:
        raise ValueError

    return train_x