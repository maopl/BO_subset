import matplotlib.pyplot as plt
import numpy as np
import os

from Problem import Problem
import Plot
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def convergence_analysis(Exp_name, Dim, name):
    Exper_floder = './experiments/{}'.format(Exp_name)
    plot_convergence = 1
    plt.figure(figsize=(10, 5))
    for Method in Method_list:
        if Dim == 1:
            Seed_list = [0]
        elif Dim == 2:
            Seed_list = [1]
        elif Dim == 5:
            Seed_list = [1,2]
        elif Dim == 10:
            Seed_list = [1]
        elif Dim == 3:
            Seed_list = [1]
        elif Dim == 4:
            Seed_list = [0,1,3,4,5,6,7,8,9]
        elif Dim == 6:
            Seed_list = [0,1,3,4,5,6,7,8,9]
        elif Dim == 8:
            Seed_list = [0]
        else:
            raise ValueError
        INIT = Dim * 2

        best_y_list = []

        for Seed in Seed_list:
            target_fun = Problem.Select_test_fun(fun_name=name, input_dim=Dim, Seed=Seed, dtype=np.float64)
            opt = [target_fun.optimizers, target_fun.optimal_value]

            data_X = np.loadtxt('{}/data/{}/{}d/{}/{}/train_x.txt'.format(Exper_floder, Method, Dim, Seed, name))
            data_Y = np.loadtxt('{}/data/{}/{}d/{}/{}/train_y.txt'.format(Exper_floder, Method, Dim, Seed, name))

            if len(data_X.shape) == 1:
                data_X = data_X[:,np.newaxis]

            if not os.path.exists('{}/figs/convergence/{}/{}d/{}/{}/'.format(Exper_floder, Method, Dim, Seed, name)):
                os.makedirs('{}/figs/convergence/{}/{}d/{}/{}/'.format(Exper_floder, Method, Dim, Seed, name))

            def best(x, y):
                best_y = np.ones((1, len(y)))
                best_x = np.ones((len(x), len(x[0])))
                best_index = []
                for j in range(len(y)):
                    best_index.append(np.argmin(y[:j + 1]))
                    best_y[0, j] = y[best_index[-1]]
                    best_x[j] = x[best_index[-1]]
                return best_x, best_y, best_index

            b_x, b_y, b_index = best(data_X, data_Y)
            b_y = b_y[0]
            best_y_list.append(b_y)

            if plot_convergence == 1:
                Plot.plot_convergence_x(target_fun, data_X, data_Y, name, Method, Dim, Seed,
                                        INIT, b_x, b_index, b_y, Exper_floder, opt)
                Plot.plot_convergence_y(target_fun, data_X, data_Y, name, Method, Dim, Seed,
                                        INIT, b_x, b_index, b_y, Exper_floder, opt)

        best_y_list = np.asarray(best_y_list)
        if name in ['Rosenbrock', 'RotatedHyperEllipsoid']:
            best_y_list = np.log(best_y_list + 1)
            plt.ylabel('Best log(y+1)')
        else:
            plt.ylabel('Best y')

        iters = np.arange(len(best_y_list[0]))
        mean = np.median(best_y_list, axis=0)
        # plt.ylim(ymin= -70, ymax= 0)
        error_low = np.percentile(best_y_list, q=30, axis=0)
        error_high = np.percentile(best_y_list, q=70, axis=0)
        line = plt.plot(iters, mean, label=Method, linewidth=1.5)[0]
        plt.title('{}d_{}'.format(Dim, name))
        plt.axvline(x=INIT, alpha=0.4, color='r')
        plt.axvline(x=9*Dim, alpha=0.4, color='r')
        plt.fill_between(iters, error_low, error_high, alpha=0.2, facecolor=line.get_color())
        plt.xlabel('Iteration')

    plt.legend(loc="upper right")
    plt.draw()
    plt.savefig('{}/figs/convergence/{}d_{}.png'.format(Exper_floder, Dim, name))
    plt.clf()
    plt.close()


if __name__ == '__main__':

    # Task_list = [
    #     'Ackley_Wave',
    #     'Ackley_Wave1',
    #     'Ackley_Wave2',
    #     'Ackley_Wave3',
    #     'Ackley_Wave4',
    #     'Ackley_Wave5',
    #     'Ackley_Wave6',
    #     'Ackley_Wave7',
    #     'Ackley_Wave8',
    #     'Ackley_Wave9',
    #     ]

    # Task_list = [
    #     'Rastrigin',
    #     'Rastrigin_wave1',
    #     'Rastrigin_wave2',
    #     'Rastrigin_wave3',
    #     'Rastrigin_wave4',
    #     'Rastrigin_wave5',
    #     'Rastrigin_wave6',
    #     'Rastrigin_wave7',
    #
    #     ]
    Task_list = [
        # 'Ackley',
        'Schwefel',
        # 'Levy',
        # 'Rastrigin',
        # 'Griewank',
        # 'Rastrigin_stretch_0.7_shift_3.25',
        # 'Rastrigin_stretch_1.75_shift_-1.25',
        # 'Rastrigin',
        # 'Schwefel',
        # 'Schwefel_stretch_0.3_shift_223',
        # 'Schwefel_stretch_1.2_shift_-442',
        ]

    # Task_list = [
        # 'Movingpeak_0',
        # 'Movingpeak_1',
        # 'Movingpeak_2',
        # 'Movingpeak_3',
        # 'Movingpeak_4',
        # 'Movingpeak_5',
        # 'Movingpeak_6',
        # 'Movingpeak_7',
        # ]
    Dim_list = [1]
    Method_list = [
        'BO_random',
        # 'TBO_grid',
        # 'EBO_random',
        # 'TBO',
        # 'GYM_grid',
        # 'RSVD_random',
        'SPARSE_VDTC',
        # 'SPARSE_FITC',
        # 'SUBSET_decomposition',
        'SUBSET_clustering',
        # 'SUBSET_local',
    ]

    Exp_name = 'subset'
    for Dim in Dim_list:
        for name in Task_list:
            convergence_analysis(Exp_name, Dim, name)


