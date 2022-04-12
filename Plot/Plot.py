import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import cm
import warnings
from Problem import Problem
from operation.normalize import Normalize_mean_std, Normalize
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def plot_contour(file_name, title, model, problem, ac_model,
                 train_x, train_y, Ac_candi, Ac_candi_f,
                 method,  KB ,task_mark=1,
                 MOGP: bool = True, Seed: int = 0, test_size=101, show: bool = True,
                 dtype=np.float64, Exper_floder=None):
    # Initialize plots
    f, y_ax = plt.subplots(1, 2, figsize=(16, 6))

    # Test points every 0.02 in [0,1]

    bounds = problem.bounds
    optimizers = problem.optimizers
    xgrid_0, xgrid_1 = np.meshgrid(np.linspace(bounds[0][0], bounds[1][0], test_size, dtype=dtype),
                                      np.linspace(bounds[0][1], bounds[1][1], test_size, dtype=dtype))
    test_x = np.concatenate((xgrid_0.reshape((xgrid_0.shape[0] * xgrid_0.shape[1], 1)),
                        xgrid_1.reshape((xgrid_0.shape[0] * xgrid_0.shape[1], 1))),
                       axis=1)

    # Make predictions - one task at a time
    # We control the task we cae about using the indices
    # The gpytorch.settings.fast_pred_var flag activates LOVE (for fast variances)
    if (MOGP == False):
        observed_pred_y,observed_corv = model.predict(test_x)
        observed_pred_y = observed_pred_y.reshape(xgrid_0.shape)
        observed_corv = observed_corv.reshape(xgrid_0.shape)
    else:
        observed_pred_y, observed_corv = model.predict(test_x, task_mark)
        observed_pred_y = observed_pred_y.reshape(xgrid_0.shape)
        observed_corv = observed_corv.reshape(xgrid_0.shape)

    # Calculate the true value
    test_y = problem.f(test_x)
    test_y = test_y.reshape(xgrid_0.shape)
    if method.split('_')[0] != 'BO':
        mean, std = KB.random_mean_std()
        test_y = Normalize_mean_std(test_y,mean, std)
        train_y_temp = Normalize_mean_std(train_y,mean, std)
    else:
        mean = np.mean(train_y)
        std = np.std(train_y)
        test_y = Normalize_mean_std(test_y, mean, std)
        train_y_temp = Normalize(train_y)



    # test_y = Normalize(test_y)


    # Calculate EI for the problem

    test_ei = ac_model._compute_acq(test_x)
    GRID_Best = np.max(test_ei)
    GRID_BestScore = test_x[np.argmax(test_ei)]

    test_ei = test_ei.reshape(xgrid_0.shape)

    # Define plotting function
    def ax_plot(title, ax, train_y, train_x, test_y, test_x, test_ei, best_ei, test_size, observed_pred_y, observed_corv, Seed, Ac_x, Ac_y):
        # Get lower and upper confidence bounds
        # lower, upper = rand_var.confidence_region()
        # Plot training data as black stars
        # train_mean = observed_pred_y.reshape(xgrid_0.shape)
        ax[0].plot(train_x[:, 0], train_x[:, 1], 'k*')
        # Predictive mean as blue line
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ax[0].contour(
                xgrid_0,
                xgrid_1,
                observed_pred_y,
                cmap=cm.Blues
            )
            min_loc_1 = (int(np.argmin(observed_pred_y) / test_size),
                         np.remainder(np.argmin(observed_pred_y), test_size))
            ax[0].plot(xgrid_0[min_loc_1],
                       xgrid_1[min_loc_1], 'b*')
            # True value as red line
            ax[0].contour(xgrid_0, xgrid_1,
                          test_y, cmap=cm.Reds)
            min_loc_2 = (int(np.argmin(test_y) / test_size),
                         np.remainder(np.argmin(test_y), test_size))
            ax[0].plot(optimizers[:, 0], optimizers[:, 1], 'r*')
            # Shade in confidence
            # ax.fill_between(test_x.squeeze().detach().numpy(), lower.detach().numpy(), upper.detach().numpy(), alpha=0.5)
            # EI in gray line
            ax[0].contour(xgrid_0, xgrid_1,
                          test_ei, cmap=cm.Greens)
            max_loc = (int(np.argmax(test_ei) / test_size),
                       np.remainder(np.argmax(test_ei), test_size))
            ax[0].plot(xgrid_0[max_loc],
                       xgrid_1[max_loc],
                       'g*')
            ax[0].plot(Ac_x[0][0], Ac_x[0][1],
                       color='orange', marker='*', linewidth=0)

        # ax.set_ylim([-3, 3])
        ax[0].legend(['Observed Data', 'Prediction', 'True f(x)', 'EI', 'Candidate'])
        ax[0].set_xlim([bounds[0][0], bounds[1][0]])
        ax[0].set_ylim([bounds[0][1], bounds[1][1]])
        num_sample = train_x.shape[0]
        ax[0].set_title(title + ' at Seed=' + str(Seed) + ' Sample(' + str(num_sample) + ')')

        # plt.subplot(grid[0, 2])
        ax[1].text(1, 1, "Prediction:\n"
                         "x1={:.4}, x2={:.4}, y={:.4}\n"
                         "\n"
                         "True f(x):\n"
                         "x1={:.4}, x2={:.4}, y={:.4}\n"
                         "\n"
                         "EI:\n"
                         "x1={:.4}, x2={:.4}, y={:.4}\n"
                         "\n"
                         "Candidate:\n"
                         "x1={:.4}, x2={:.4}, y={:.4}".format(
            xgrid_0[min_loc_1],
            xgrid_1[min_loc_1],
            np.min(observed_pred_y),
            xgrid_0[min_loc_2],
            xgrid_1[min_loc_2],
            np.min(test_y),
            xgrid_0[max_loc],
            xgrid_1[max_loc],
            np.max(test_ei),
            Ac_x[0][0],
            Ac_x[0][1],
            Ac_y[0][0]
        ), fontsize=12)
        ax[1].axis([0, 10, 0, 10])
        ax[1].axis('off')

    ax_plot(title, y_ax, train_y_temp, train_x, test_y, test_x, test_ei, GRID_BestScore, test_size, observed_pred_y, observed_corv, Seed,
            Ac_candi, Ac_candi_f)
    plt.grid()
    if (show):
        plt.show()

    if not os.path.exists('{}/figs/contour/{}/{}/{}'.format(Exper_floder, method, Seed, title)):
        os.makedirs('{}/figs/contour/{}/{}/{}'.format(Exper_floder, method, Seed, title))

    plt.savefig('{}/figs/contour/{}/{}/{}/{}.png'.format(Exper_floder, method, Seed, title, file_name), format='png')
    plt.close()

def plot_one_dimension(file_name, title, model, problem, ac_model,
                 train_x, train_y, Ac_candi, Ac_candi_f,
                 method, KB, task_mark=1,
                 MOGP: bool = True, Seed: int = 0, show: bool = True,
                dtype=np.float64, Exper_floder=None):
    # Initialize plots
    f, ax = plt.subplots(1, 2, figsize=(16, 6))

    # Test points every 0.02 in [0,1]

    bounds = problem.bounds
    opt_x = problem.optimizers
    opt_val = problem.optimal_value
    test_x = np.arange(-1, 1.05, 0.005, dtype=dtype)



    # Make predictions - one task at a time
    # We control the task we cae about using the indices
    # The gpytorch.settings.fast_pred_var flag activates LOVE (for fast variances)
    if (MOGP == False):
        observed_pred_y,observed_corv = model.predict(test_x[:,np.newaxis])
    else:
        observed_pred_y, observed_corv = model.predict(test_x[:,np.newaxis],task_mark)

    # Calculate the true value
    test_y = problem.f(test_x[:,np.newaxis])
    if method.split('_')[0] != 'BO':
        mean, std = KB.random_mean_std()
        test_y = Normalize_mean_std(test_y,mean, std)
        train_y_temp = Normalize_mean_std(train_y,mean, std)
        # observed_pred_y = Normalize_mean_std(observed_pred_y, mean, std)
    else:
        mean = np.mean(train_y)
        std = np.std(train_y)
        test_y = Normalize_mean_std(test_y,mean, std)
        train_y_temp = Normalize(train_y)
        # observed_pred_y = Normalize_mean_std(observed_pred_y,mean,std)




    # Calculate EI for the problem
    test_ei = ac_model._compute_acq(test_x[:,np.newaxis])
    best_ei_score = np.max(test_ei)
    best_ei_x = test_x[np.argmax(test_ei)]

    # Define plotting function
        # Get lower and upper confidence bounds
        # lower, upper = rand_var.confidence_region()
        # Plot training data as black stars
    pre_mean = observed_pred_y
    pre_best_y = np.min(pre_mean)
    pre_best_x = test_x[np.argmin(pre_mean)]
    pre_up = observed_pred_y + observed_corv
    pre_low = observed_pred_y - observed_corv

    ax[0].plot(test_x, test_y, 'r-', linewidth=1, alpha=1)
    ax[0].plot(test_x, pre_mean[:,0], 'b-', linewidth=1, alpha=1)
    ax[0].plot(test_x, test_ei[:,0], 'g-', linewidth=1, alpha=1)

    ax[0].plot(train_x[:,0], train_y_temp[:,0], marker='*', color='black', linewidth=0)
    # ax[0].plot(Ac_candi[:,0], Ac_candi_f[:,0], marker='*', color='orange', linewidth=0)
    ax[0].plot(Ac_candi[:, 0], 0, marker='*', color='orange', linewidth=0)
    ax[0].plot(best_ei_x, best_ei_score, marker='*', color='green', linewidth=0)
    # ax[0].plot(opt_x[:,0], opt_val, marker='*', color='red', linewidth=0)
    ax[0].plot(opt_x[:, 0], 0, marker='*', color='red', linewidth=0)
    ax[0].plot(pre_best_x, pre_best_y, marker='*', color='blue', linewidth=0)
    ax[0].fill_between(test_x, pre_up[:,0], pre_low[:,0], alpha=0.2, facecolor='blue')



    # ax.set_ylim([-3, 3])
    ax[0].legend(['True f(x)', 'Prediction', 'EI', 'Observed Data', 'Candidate'])
    ax[0].set_xlim([bounds[0][0], bounds[1][0]])
    num_sample = train_x.shape[0]
    ax[0].set_title(title + ' at Seed=' + str(Seed) + ' Sample(' + str(num_sample) + ')')

    # plt.subplot(grid[0, 2])
    ax[1].text(1, 1, "Prediction:\n"
                     "x={:.4}, y={:.4}\n"
                     "\n"
                     "True f(x):\n"
                     "x={:.4}, y={:.4}\n"
                     "\n"
                     "EI:\n"
                     "x={:.4}, y={:.4}\n"
                     "\n"
                     "Candidate:\n"
                     "x={:.4}, y={:.4}".format(
        pre_best_x,
        pre_best_y,
        opt_x[0][0],
        opt_val,
        best_ei_x,
        best_ei_score,
        Ac_candi[0][0],
        Ac_candi_f[0][0]
    ), fontsize=12)

    ax[1].axis([0, 10, 0, 10])
    ax[1].axis('off')

    plt.grid()
    if (show):
        plt.show()

    if not os.path.exists('{}/figs/oneD/{}/{}/{}'.format(Exper_floder, method, Seed, title)):
        os.makedirs('{}/figs/oneD/{}/{}/{}'.format(Exper_floder, method, Seed, title))

    plt.savefig('{}/figs/oneD/{}/{}/{}/{}.png'.format(Exper_floder, method, Seed, title, file_name), format='png')
    plt.close()

def plot_true_contour(obj_fun_list, dim, dtype, Exper_floder=None):
    for i in obj_fun_list:
        obj_fun = Problem.Select_test_fun(fun_name=i, input_dim=dim, Seed=0, dtype=dtype)
        print(i)

        if not os.path.exists('{}/figs/contour/true_f/{}'.format(Exper_floder, obj_fun.name)):
            os.makedirs('{}/figs/contour/true_f/{}'.format(Exper_floder, obj_fun.name))

        save_load = '{}/figs/contour/true_f/{}/'.format(Exper_floder, obj_fun.name)

        x = np.linspace(-1, 1, 101)
        y = np.linspace(-1, 1, 101)
        X, Y = np.meshgrid(x, y)
        all_sample = np.array(np.c_[X.ravel(), Y.ravel()])
        Z_true = obj_fun.f(all_sample)
        Z_true = Z_true[:,np.newaxis]
        Z_true = np.asarray(Z_true)
        Z_true = Z_true.reshape(X.shape)

        optimizers = obj_fun.optimizers

        fig = plt.figure(figsize=(10, 8))
        ax = plt.subplot(111)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])
        a = plt.contourf(X, Y, Z_true, 100, cmap=plt.cm.summer)
        b = plt.contour(X, Y, Z_true, 50, colors='black', linewidths=1, linestyles='solid')
        plt.plot(optimizers[:, 0], optimizers[:, 1], marker='*', linewidth=0, color='white', markersize=10, label="GlobalOpt")
        plt.colorbar(a)
        plt.title(i)
        fig.legend(facecolor='gray')
        plt.draw()
        plt.savefig(save_load+'PF_1.png')
        plt.close()


def plot_convergence_x(target_fun, data_X, data_Y, name, Method, Dim, Seed, INIT, b_x, b_index, b_y, Exper_floder, opt):
    x_min = target_fun.optimizers

    dis = np.sqrt(np.sum((data_X - x_min) ** 2, axis=1))
    b_dis = np.sqrt(np.sum((b_x - x_min) ** 2, axis=1))
    b_index = np.sort(list(set(b_index)))
    b_dis = b_dis[b_index]

    iter = np.array([k for k in range(len(data_Y))])
    plt.figure(figsize=(10, 5))
    plt.suptitle("{}_{}_{}d_{}".format(name, Method, Dim, Seed))
    plt.subplot(1, 2, 1)
    plt.plot(iter, dis, '--*', alpha=0.4, color='blue')
    plt.plot(b_index, b_dis, '*', color='orange')
    plt.axvline(x=INIT, alpha=0.2, color='r')
    plt.xlabel('Iteration')
    plt.ylabel('d(x[n], opt_x)')
    plt.title('Distance between f_opt x\'s')

    plt.subplot(1, 2, 2)
    if name in ['Rosenbrock', 'RotatedHyperEllipsoid']:
        plt.plot(iter, np.log(b_y + 1), '-', color='orange')
        plt.plot(iter, np.log(data_Y + 1), '--*', alpha=0.4, color='blue')
        plt.ylabel('Best log(y+1)')
    else:
        plt.plot(iter, b_y, '-', color='orange')
        plt.plot(iter, data_Y, '--*', alpha=0.4, color='blue')
        plt.ylabel('Best y')
    plt.axvline(x=INIT, alpha=0.2, color='r')
    plt.title('Value of the best selected sample')
    plt.xlabel('Iteration')
    # plt.show()

    plt.savefig('{}/figs/convergence/{}/{}d/{}/{}/result_x.png'.format(Exper_floder, Method, Dim, Seed, name))
    plt.close()


def plot_convergence_y(target_fun, data_X, data_Y, name, Method, Dim, Seed, INIT, b_x, b_index, b_y, Exper_floder, opt):
    x_min = target_fun.optimizers
    y_min = target_fun.f(x_min)

    dis = np.abs(data_Y - y_min)
    b_dis = np.abs(b_y - y_min)
    b_index = np.sort(list(set(b_index)))
    b_dis = b_dis[b_index]

    iter = np.array([k for k in range(len(data_Y))])
    plt.figure(figsize=(10, 5))
    plt.suptitle("{}_{}_{}d_{}".format(name, Method, Dim, Seed))
    plt.subplot(1, 2, 1)
    plt.plot(iter, dis, '--*', alpha=0.4, color='blue')
    plt.plot(b_index, b_dis, '*', color='orange')
    plt.axvline(x=INIT, alpha=0.2, color='r')
    plt.xlabel('Iteration')
    plt.ylabel('d(y[n], opt_y)')
    plt.title('Distance between f_opt y\'s')

    plt.subplot(1, 2, 2)
    if name in ['Rosenbrock', 'RotatedHyperEllipsoid']:
        plt.plot(iter, np.log(b_y + 1), '-', color='orange')
        plt.plot(iter, np.log(data_Y + 1), '--*', alpha=0.4, color='blue')
        plt.ylabel('Best log(y+1)')
    else:
        plt.plot(iter, b_y, '-', color='orange')
        plt.plot(iter, data_Y, '--*', alpha=0.4, color='blue')
        plt.ylabel('Best y')
    plt.axvline(x=INIT, alpha=0.2, color='r')
    plt.title('Value of the best selected sample')
    plt.xlabel('Iteration')
    # plt.show()

    plt.savefig('{}/figs/convergence/{}/{}d/{}/{}/result_y.png'.format(Exper_floder, Method, Dim, Seed, name))
    plt.close()



def plot_contour_subset(file_name, title, model, problem, ac_model, subset_id,
                 train_x, train_y, Ac_candi, Ac_candi_f,
                 method,  KB ,task_mark=1,
                 MOGP: bool = True, Seed: int = 0, test_size=101, show: bool = True,
                 dtype=np.float64, Exper_floder=None):
    # Initialize plots
    f, y_ax = plt.subplots(1, 1, figsize=(16, 6))

    # Test points every 0.02 in [0,1]

    bounds = problem.bounds
    optimizers = problem.optimizers
    xgrid_0, xgrid_1 = np.meshgrid(np.linspace(bounds[0][0], bounds[1][0], test_size, dtype=dtype),
                                      np.linspace(bounds[0][1], bounds[1][1], test_size, dtype=dtype))
    test_x = np.concatenate((xgrid_0.reshape((xgrid_0.shape[0] * xgrid_0.shape[1], 1)),
                        xgrid_1.reshape((xgrid_0.shape[0] * xgrid_0.shape[1], 1))),
                       axis=1)

    # Make predictions - one task at a time
    # We control the task we cae about using the indices
    # The gpytorch.settings.fast_pred_var flag activates LOVE (for fast variances)
    if (MOGP == False):
        observed_pred_y,observed_corv = model.predict(test_x)
        observed_pred_y = observed_pred_y.reshape(xgrid_0.shape)
        observed_corv = observed_corv.reshape(xgrid_0.shape)
    else:
        observed_pred_y, observed_corv = model.predict(test_x, task_mark)
        observed_pred_y = observed_pred_y.reshape(xgrid_0.shape)
        observed_corv = observed_corv.reshape(xgrid_0.shape)

    # Calculate the true value
    test_y = problem.f(test_x)
    test_y = test_y.reshape(xgrid_0.shape)
    mean, std = KB.random_mean_std()
    test_y = Normalize_mean_std(test_y,mean, std)
    train_y_temp = Normalize_mean_std(train_y,mean, std)

    test_ei = ac_model._compute_acq(test_x)
    GRID_Best = np.max(test_ei)
    GRID_BestScore = test_x[np.argmax(test_ei)]

    test_ei = test_ei.reshape(xgrid_0.shape)

    # Define plotting function
    def ax_plot(title, ax, subset_id, train_y, train_x, test_y, test_x, test_ei, best_ei, test_size, observed_pred_y, observed_corv, Seed, Ac_x, Ac_y):
        # Get lower and upper confidence bounds
        # lower, upper = rand_var.confidence_region()
        # Plot training data as black stars
        # train_mean = observed_pred_y.reshape(xgrid_0.shape)
        ax.plot(train_x[:, 0], train_x[:, 1], 'k*', alpha=0.3)
        ax.plot(train_x[subset_id, 0], train_x[subset_id, 1], 'k*', alpha=0.3)
        # Predictive mean as blue line
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ax.contour(
                xgrid_0,
                xgrid_1,
                observed_pred_y,
                cmap=cm.Blues
            )
            min_loc_1 = (int(np.argmin(observed_pred_y) / test_size),
                         np.remainder(np.argmin(observed_pred_y), test_size))
            ax.plot(xgrid_0[min_loc_1],
                       xgrid_1[min_loc_1], 'b*')
            # True value as red line
            ax.contour(xgrid_0, xgrid_1,
                          test_y, cmap=cm.Reds)
            min_loc_2 = (int(np.argmin(test_y) / test_size),
                         np.remainder(np.argmin(test_y), test_size))
            ax.plot(optimizers[:, 0], optimizers[:, 1], 'r*')
            # Shade in confidence
            # ax.fill_between(test_x.squeeze().detach().numpy(), lower.detach().numpy(), upper.detach().numpy(), alpha=0.5)
            # EI in gray line
            ax.contour(xgrid_0, xgrid_1,
                          test_ei, cmap=cm.Greens)
            max_loc = (int(np.argmax(test_ei) / test_size),
                       np.remainder(np.argmax(test_ei), test_size))
            ax.plot(xgrid_0[max_loc],
                       xgrid_1[max_loc],
                       'g*')
            ax.plot(Ac_x[0][0], Ac_x[0][1],
                       color='orange', marker='*', linewidth=0)

        # ax.set_ylim([-3, 3])
        ax.legend(['Observed Data', 'Prediction', 'True f(x)', 'EI', 'Candidate'])
        ax.set_xlim([bounds[0][0], bounds[1][0]])
        ax.set_ylim([bounds[0][1], bounds[1][1]])
        num_sample = train_x.shape[0]
        ax.set_title(title + ' at Seed=' + str(Seed) + ' Sample(' + str(num_sample) + ')')



    ax_plot(title, y_ax, subset_id, train_y_temp, train_x, test_y, test_x, test_ei, GRID_BestScore, test_size, observed_pred_y, observed_corv, Seed,
            Ac_candi, Ac_candi_f)
    plt.grid()
    if (show):
        plt.show()

    if not os.path.exists('{}/figs/contour/{}/{}/{}'.format(Exper_floder, method, Seed, title)):
        os.makedirs('{}/figs/contour/{}/{}/{}'.format(Exper_floder, method, Seed, title))

    plt.savefig('{}/figs/contour/{}/{}/{}/{}.png'.format(Exper_floder, method, Seed, title, file_name), format='png')
    plt.close()



def plot_one_dimension_subset(file_name, title, model, problem, ac_model, subset_id,
                 train_x, train_y, Ac_candi, Ac_candi_f,
                 method, KB, task_mark=1,
                 MOGP: bool = True, Seed: int = 0, show: bool = True,
                dtype=np.float64, Exper_floder=None):
    # Initialize plots
    f, ax = plt.subplots(1, 1, figsize=(16, 6))

    # Test points every 0.02 in [0,1]

    bounds = problem.bounds
    opt_x = problem.optimizers
    opt_val = problem.optimal_value
    test_x = np.arange(-1, 1.05, 0.005, dtype=dtype)



    # Make predictions - one task at a time
    # We control the task we cae about using the indices
    # The gpytorch.settings.fast_pred_var flag activates LOVE (for fast variances)
    if (MOGP == False):
        observed_pred_y,observed_corv = model.predict(test_x[:,np.newaxis])
    else:
        observed_pred_y, observed_corv = model.predict(test_x[:,np.newaxis],task_mark)

    # Calculate the true value
    test_y = problem.f(test_x[:,np.newaxis])
    mean, std = KB.random_mean_std()
    test_y = Normalize_mean_std(test_y,mean, std)
    train_y_temp = Normalize_mean_std(train_y,mean, std)


    # Calculate EI for the problem
    test_ei = ac_model._compute_acq(test_x[:,np.newaxis])
    best_ei_score = np.max(test_ei)
    best_ei_x = test_x[np.argmax(test_ei)]

    # Define plotting function
        # Get lower and upper confidence bounds
        # lower, upper = rand_var.confidence_region()
        # Plot training data as black stars
    pre_mean = observed_pred_y
    pre_best_y = np.min(pre_mean)
    pre_best_x = test_x[np.argmin(pre_mean)]
    pre_up = observed_pred_y + observed_corv
    pre_low = observed_pred_y - observed_corv

    ax.plot(test_x, test_y, 'r-', linewidth=1, alpha=1)
    ax.plot(test_x, pre_mean[:,0], 'b-', linewidth=1, alpha=1)
    ax.plot(test_x, test_ei[:,0], 'g-', linewidth=1, alpha=1)

    ax.plot(train_x[:,0], train_y_temp[:,0], marker='*', color='black', linewidth=0,alpha=0.3)

    ax.plot(train_x[subset_id,0], train_y_temp[subset_id,0],marker='*', color='yellow', linewidth=0,alpha=1)


    # ax[0].plot(Ac_candi[:,0], Ac_candi_f[:,0], marker='*', color='orange', linewidth=0)
    ax.plot(Ac_candi[:, 0], 0, marker='*', color='orange', linewidth=0)
    ax.plot(best_ei_x, best_ei_score, marker='*', color='green', linewidth=0)
    # ax[0].plot(opt_x[:,0], opt_val, marker='*', color='red', linewidth=0)
    ax.plot(opt_x[:, 0], 0, marker='*', color='red', linewidth=0)
    ax.plot(pre_best_x, pre_best_y, marker='*', color='blue', linewidth=0)
    ax.fill_between(test_x, pre_up[:,0], pre_low[:,0], alpha=0.2, facecolor='blue')



    # ax.set_ylim([-3, 3])
    ax.legend(['True f(x)', 'Prediction', 'EI', 'Observed Data', 'Candidate'])
    ax.set_xlim([bounds[0][0], bounds[1][0]])
    num_sample = train_x.shape[0]
    ax.set_title(title + ' at Seed=' + str(Seed) + ' Sample(' + str(num_sample) + ')')


    plt.grid()
    if (show):
        plt.show()

    if not os.path.exists('{}/figs/oneD/{}/{}/{}'.format(Exper_floder, method, Seed, title)):
        os.makedirs('{}/figs/oneD/{}/{}/{}'.format(Exper_floder, method, Seed, title))

    plt.savefig('{}/figs/oneD/{}/{}/{}/{}.png'.format(Exper_floder, method, Seed, title, file_name), format='png')
    plt.close()


if __name__ == '__main__':
    Task_list = [
        'Sphere',
        'Ackley',
        'Griewank',
        'Levy',
        'StyblinskiTang',
        ]



