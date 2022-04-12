import logging
import os
import numpy as np
import time
import pickle
import sobol_seq
import GPyOpt
import GPy

import Plot
from GPyOpt.acquisitions.EI import AcquisitionEI

from operation.normalize import Normalize,Normalize_mean_std
from operation.init_data import InitData

from library.sequential import Sequential

from Subset.Sparse_BO import SparseGPBO
def SPARSE(
        Dty=np.float64,
        Plt=False,
        Evol=None,
        Init=None,
        GSN=None,
        Xdim=None,
        Task=None,
        Acf='EI',
        Seed=None,
        Method=None,
        KB=None,
        Init_method='random',
        Save_mode=1,
        Exper_floder=None,
        sparse_method='FITC',
        Save_random=1,
        Gym_mode = 0,
):
    np.random.seed(Seed)

    sum_time_fit = 0
    sum_time_acfun = 0
    sum_time_gs = 0

    fit_time_list = []
    acf_time_list = []
    iter_time_list = []

    bounds = Task.bounds
    Init_now = Init

    sparse_start_num = 30*Xdim

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
    time_start_bo = time.time()
    for i in range(Init_now, sparse_start_num):
        time_iter_start = time.time()

        train_y_norm = Normalize(train_y)

        time_fit_start = time.time()
        model.updateModel(train_x, train_y_norm, None,None)
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


        # if Plt:
        #     if Xdim == 2:
        #         Plot.plot_contour('{}_bo'.format(i), Task.name, model, Task, acquisition,
        #                           train_x, train_y, suggested_sample, acq_value,
        #                           Method, KB, MOGP=False, Seed=Seed, test_size=101,
        #                           show=False, dtype=Dty, Exper_floder=Exper_floder)
        #     elif Xdim == 1:
        #         Plot.plot_one_dimension('{}_bo'.format(i), Task.name, model, Task,
        #                                 acquisition, train_x, train_y, suggested_sample, acq_value,
        #                                 'BO_Sparse', KB, MOGP=False, Seed=Seed, show=False,
        #                                 dtype=Dty, Exper_floder=Exper_floder)
        time_iter_end = time.time()
        logging.info('Target:%s\t Seed:%d\t Iteration:%d\n '
                     'Cand_f:%f\t Best_f:%f\t True_f:%f\n '
                     'Time:%dh%dm%ds\t Loss:%f' %
                     (Task.name, Seed, i,
                      Y_predict,
                      fx_opt,
                      Task.optimal_value,
                      (time.time()-time_iter_start)/3600,
                      (time.time()-time_iter_start) % 3600 / 60,
                      (time.time()-time_iter_start) % 3600 % 60,
                      model.model.log_likelihood()))

        KB.current_task_x = train_x

        iter_time_list.append(time_iter_end - time_iter_start)

    time_end_bo = time.time()

    current_inducing_num = 5*Xdim
    kernel = GPy.kern.RBF(input_dim=Xdim)
    model = SparseGPBO(num_inducing=current_inducing_num, kernel=kernel, optimize_restarts=1, verbose=False,
                       optimizer='lbfgsb', exact_feval=False, sparse_method=sparse_method)
    acquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(space)
    acquisition = AcquisitionEI(model, space, optimizer=acquisition_optimizer)
    evaluator = Sequential(acquisition)
    time_start_sparse = time.time()
    for i in range(sparse_start_num, Evol):
        time_iter_start = time.time()

        train_y_norm = Normalize_mean_std(train_y, mean, std)


        time_fit_start = time.time()
        model.updateModel(train_x, train_y_norm, None,None)
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


        # if Plt:
        #     if Xdim == 2:
        #         Plot.plot_contour('{}_bo'.format(i), Task.name, model, Task, acquisition,
        #                           train_x, train_y, suggested_sample, acq_value,
        #                           Method, KB, MOGP=False, Seed=Seed, test_size=101,
        #                           show=False, dtype=Dty, Exper_floder=Exper_floder)
        #     elif Xdim == 1:
        #         Plot.plot_one_dimension('{}_bo'.format(i), Task.name, model, Task,
        #                                 acquisition, train_x, train_y, suggested_sample, acq_value,
        #                                 'BO_Sparse', KB, MOGP=False, Seed=Seed, show=False,
        #                                 dtype=Dty, Exper_floder=Exper_floder)
        time_iter_end = time.time()
        iter_time_list.append(time_iter_end - time_iter_start)
        logging.info('Target:%s\t Seed:%d\t Iteration:%d\n '
                     'Cand_f:%f\t Best_f:%f\t True_f:%f\n '
                     'Time:%dh%dm%ds\t Loss:%f' %
                     (Task.name, Seed, i,
                      Y_predict,
                      fx_opt,
                      Task.optimal_value,
                      (time.time()-time_iter_start)/3600,
                      (time.time()-time_iter_start) % 3600 / 60,
                      (time.time()-time_iter_start) % 3600 % 60,
                      model.model.log_likelihood()))

        KB.current_task_x = train_x
    time_end_sparse = time.time()


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


    total_time = np.array([sum_time_fit,sum_time_acfun,time_end_bo-time_start_bo, time_end_sparse - time_start_sparse])
    np.savetxt('{}/time/{}/{}d/{}/{}/fit_time.txt'.format(Exper_floder, Method, Xdim, Seed, Task.name), np.array(fit_time_list))
    np.savetxt('{}/time/{}/{}d/{}/{}/acf_time.txt'.format(Exper_floder, Method, Xdim, Seed, Task.name), np.array(acf_time_list))
    np.savetxt('{}/time/{}/{}d/{}/{}/iter_time.txt'.format(Exper_floder, Method, Xdim, Seed, Task.name),
               np.array(iter_time_list))
    np.savetxt('{}/time/{}/{}d/{}/{}/total_time.txt'.format(Exper_floder, Method, Xdim, Seed, Task.name),
               total_time)






