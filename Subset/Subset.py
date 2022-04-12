import logging
import os
import numpy as np
import time
import pickle
import GPyOpt
import GPy

import Plot.Plot as Plot
import Subset.Subset_selection as ss

from GPyOpt.acquisitions.EI import AcquisitionEI
from operation.normalize import Normalize_mean_std, Normalize
from operation.init_data import InitData
from library.sequential import Sequential








def \
        SUBSET(
        Dty=np.float64,
        Plt=False,
        Evol=None,
        Init=None,
        Xdim=None,
        Task=None,
        Acf='EI',
        Seed=None,
        Method=None,
        KB=None,
        Init_method='random',
        Save_mode=1,
        Exper_floder=None,
        subset_method = 'clustering'
):
    np.random.seed(Seed)

    sum_time_fit = 0
    sum_time_acfun = 0
    sum_time_gs = 0
    sum_time_select = 0

    fit_time_list = []
    acf_time_list = []
    select_time_list = []
    iter_time_list = []

    bounds = Task.bounds
    Init_now = Init

    if not os.path.exists('{}/data/{}/{}d/{}/{}'.format(Exper_floder, Method, Xdim, Seed, Task.name)):
        os.makedirs('{}/data/{}/{}d/{}/{}'.format(Exper_floder, Method, Xdim, Seed, Task.name))

    if not os.path.exists('{}/time/{}/{}d/{}/{}'.format(Exper_floder, Method, Xdim, Seed, Task.name)):
        os.makedirs('{}/time/{}/{}d/{}/{}'.format(Exper_floder, Method, Xdim, Seed, Task.name))

    train_x = InitData(Init_method, KB, Init, Xdim, Dty)

    grid_x = None

    train_y = Task.f(train_x)
    train_y = train_y[:, np.newaxis]
    for y in train_y:
        KB.random_y.append(y)


    #Set optimize objective
    objective = GPyOpt.core.task.SingleObjective(Task.f)

    #Set decision space
    task_design_space = []
    for i in range(Xdim):
        var_dic = {'name': f'var_{i}', 'type': 'continuous', 'domain': tuple([Task.bounds[0][i],Task.bounds[1][i]])}
        task_design_space.append(var_dic.copy())
    space = GPyOpt.Design_space(space=task_design_space)

    #Set model
    kernel = GPy.kern.RBF(input_dim=Xdim)
    model = GPyOpt.models.GPModel(kernel=kernel, optimize_restarts=10, verbose=False, optimizer='lbfgsb',exact_feval=False,)
    acquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(space)
    acquisition = AcquisitionEI(model, space,optimizer=acquisition_optimizer)
    evaluator = Sequential(acquisition)

    mean, std = KB.random_mean_std()


    Subset_start_num = 30*Xdim

    if subset_method == 'clustering':
        method = 'SUBSET_clustering'
    elif subset_method == 'decomposition':
        method = 'SUBSET_decomposition'
    elif subset_method == 'random':
        method = 'SUBSET_random'
    else:
        method = 'SUBSET_clustering'


    Archive_num = 0

    time_BO_start = time.time()
    for i in range(Init_now, Evol):
        time_iter_start = time.time()

        train_y_norm = Normalize(train_y)

        if (i >= Subset_start_num):
            if i % (5*Xdim) == 0:
                select_flag = 1
            else:
                select_flag = 0
            if select_flag == 1:
                time_select_start = time.time()
                if subset_method == 'clustering':
                    Archive_num = int(i / (15))
                    subset_id = ss.subset_clustering(Archive_num, train_x, train_y_norm, last_n = Xdim)
                elif subset_method == 'decomposition':
                    Archive_num = int(i / (15))
                    subset_id = ss.subset_decomposition(Archive_num, train_x, train_y_norm, last_n= Xdim)
                elif subset_method == 'random':
                    Archive_num = int(i / (15))
                    subset_id = ss.subset_random(Archive_num, train_x, train_y_norm, last_n= Xdim)
                    model.model['Gaussian_noise.*variance'] = 1.0
                    model.model['rbf.*lengthscale'] = 1.0

                time_select_end = time.time()
                sum_time_select += time_select_end - time_select_start
                X_subset = train_x[subset_id]
                Y_subset = train_y_norm[subset_id]
                select_time_list.append(time_select_end - time_select_start)
            else:
                X_subset = np.vstack((X_subset, [train_x[-1]]))
                Y_subset = np.vstack((Y_subset, [train_y_norm[-1]]))
                subset_id.append(len(train_x) - 1)
                select_time_list.append(0)

        else:
            subset_id = list(range(len(train_x)))
            X_subset = train_x
            Y_subset = train_y_norm
            select_time_list.append(0)

        time_fit_start = time.time()
        model.updateModel(X_subset, Y_subset, None,None)
        time_fit_end = time.time()

        print("[iter{}] FitModel:\t{:.0f}h{:.0f}m{:.1f}s".format(
            i,
            (time_fit_end-time_fit_start)/3600,
            (time_fit_end-time_fit_start) % 3600 / 60,
            (time_fit_end-time_fit_start) % 3600 % 60,))
        sum_time_fit += time_fit_end - time_fit_start

        fit_time_list.append(time_fit_end - time_fit_start)

        # context_manager = ContextManager(space, None)
        time_acf_start = time.time()
        suggested_sample, acq_value = evaluator.compute_batch(None, context_manager=None)
        time_acf_end = time.time()

        sum_time_acfun += time_acf_end - time_acf_start

        print("[iter{}] AcFun:\t\t{:.0f}h{:.0f}m{:.1f}s".format(
            i,
            (time_acf_end-time_acf_start)/3600,
            (time_acf_end-time_acf_start) % 3600 / 60,
            (time_acf_end-time_acf_start) % 3600 % 60,))

        acf_time_list.append(time_acf_end-time_acf_start)

        # suggested_sample, _ = acquisition.optimize()
        suggested_sample = space.zip_inputs(suggested_sample)

        # --- Augment X
        train_x = np.vstack((train_x, suggested_sample))

        # --- Evaluate *f* in X, augment Y and update cost function (if needed)
        Y_new, _ = objective.evaluate(suggested_sample)

        train_y = np.vstack((train_y, Y_new))

        Y_predict = model.predict(suggested_sample)[0][0][0]
        fx_opt = np.min(train_y)


        if Plt:
            if Xdim == 2:
                Plot.plot_contour_subset('{}_bo'.format(i), Task.name, model, Task, acquisition,
                                  subset_id, train_x, train_y, suggested_sample, acq_value,
                                  method, KB, MOGP=False, Seed=Seed, test_size=101,
                                  show=False, dtype=Dty, Exper_floder=Exper_floder)
            elif Xdim == 1:
                Plot.plot_one_dimension_subset('{}_bo'.format(i), Task.name, model, Task,
                                        acquisition, subset_id, train_x, train_y, suggested_sample, acq_value,
                                        method, KB, MOGP=False, Seed=Seed, show=False,
                                        dtype=Dty, Exper_floder=Exper_floder)

        time_iter_end = time.time()
        iter_time_list.append(time_iter_end - time_iter_start)


        logging.info('Target:%s\t Seed:%d\t Iteration:%d\n '
                     'Cand_f:%f\t Best_f:%f\t True_f:%f\n '
                     'Time:%dh%dm%ds\t' %
                     (Task.name, Seed, i,
                      Y_predict,
                      fx_opt,
                      Task.optimal_value,
                      (time.time()-time_iter_start)/3600,
                      (time.time()-time_iter_start) % 3600 / 60,
                      (time.time()-time_iter_start) % 3600 % 60,
                      ))


        KB.current_task_x = train_x
        if i == Subset_start_num - 1:
            time_BO_end = time.time()
            time_subset_start = time.time()

    time_subset_end = time.time()

    logging.info('GridSearch_time:\t%dh%dm%ds\t %d\n'
                 'FitModel_time:\t%dh%dm%ds\t %d\n'
                 'AcFun_time:\t%dh%dm%ds\t %d\n\n' %
                 ((sum_time_gs) / 3600,
                  (sum_time_gs) % 3600 / 60,
                  (sum_time_gs) % 3600 % 60,
                  100 * sum_time_gs / (sum_time_gs + sum_time_fit + sum_time_acfun),
                  (sum_time_fit) / 3600,
                  (sum_time_fit) % 3600 / 60,
                  (sum_time_fit) % 3600 % 60,
                  100 * sum_time_fit / (sum_time_gs + sum_time_fit + sum_time_acfun),
                  (sum_time_acfun) / 3600,
                  (sum_time_acfun) % 3600 / 60,
                  (sum_time_acfun) % 3600 % 60,
                  100 * sum_time_acfun / (sum_time_gs + sum_time_fit + sum_time_acfun)))







    KB.add(Task.name, 'BO', train_x, train_y, grid_x)
    np.savetxt('{}/data/{}/{}d/{}/{}/train_x.txt'.format(Exper_floder, Method, Xdim, Seed, Task.name), train_x)
    np.savetxt('{}/data/{}/{}d/{}/{}/train_y.txt'.format(Exper_floder, Method, Xdim, Seed, Task.name), train_y)
    if Save_mode == 1:
        with open('{}/model/{}d/{}/{}_KB.txt'.format(Exper_floder, Xdim, Method, Seed), 'wb') as f:  # 打开文件
            pickle.dump(KB, f)

    KB.current_task_x = []


    total_time = np.array([sum_time_fit,sum_time_acfun, sum_time_select, time_BO_end-time_BO_start, time_subset_end-time_subset_start])
    np.savetxt('{}/time/{}/{}d/{}/{}/fit_time.txt'.format(Exper_floder, Method, Xdim, Seed, Task.name), np.array(fit_time_list))
    np.savetxt('{}/time/{}/{}d/{}/{}/acf_time.txt'.format(Exper_floder, Method, Xdim, Seed, Task.name), np.array(acf_time_list))
    np.savetxt('{}/time/{}/{}d/{}/{}/select_time.txt'.format(Exper_floder, Method, Xdim, Seed, Task.name),
               np.array(select_time_list))
    np.savetxt('{}/time/{}/{}d/{}/{}/iter_time.txt'.format(Exper_floder, Method, Xdim, Seed, Task.name),
               np.array(iter_time_list))
    np.savetxt('{}/time/{}/{}d/{}/{}/total_time.txt'.format(Exper_floder, Method, Xdim, Seed, Task.name),
               total_time)








