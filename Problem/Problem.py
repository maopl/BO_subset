

import math
import logging
import numpy as np
stand = 1
import matplotlib.pyplot as plt
import os
from operation.normalize import Normalize

class BASE():
    def __init__(
            self,
            input_dim=1,
            bounds=None,
            sd=None,
            RX=None,
            Seed=None,
            optimizers=None,
            dtype=np.float64
    ):
        self.xdim = input_dim
        self.dtype = dtype
        self.RX = np.array(RX, dtype=dtype)
        self.optimizers = self.normalize(np.array(optimizers, dtype=dtype))

        self.query_num = 0

        np.random.seed(Seed)
        if bounds is None:
            self.bounds = np.array([[-1.0] * self.xdim, [1.0] * self.xdim],  dtype=dtype) * stand
        else:
            self.bounds = bounds

        if sd is None:
            self.sd = 0
        else:
            self.sd = sd

    def transfer(self, X):
        return (X + 1) * (self.RX[:, 1] - self.RX[:, 0]) / 2 + (self.RX[:, 0])

    def normalize(self, X):
        return 2 * (X - (self.RX[:, 0]))/(self.RX[:, 1] - self.RX[:, 0]) - 1

    def noise(self, n):
        if self.sd == 0:
            noise = np.zeros(shape=(n, ), dtype=self.dtype)
        else:
            noise = np.random.normal(mean=0, std=self.sd, size=(n, ), dtype=self.dtype)

        return noise



class Sphere(BASE):
    '''
    Sphere function

    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''

    def __init__(
            self,
            fun_name,
            input_dim=1,
            bounds=None,
            sd=None,
            RX=None,
            Seed=0,
            shift=0,
            stretch=1,
            sinTrans=0,
            expTrans=0,
            cosTrans=0,
            inv=0,
            dtype=np.float64,
    ):
        self.name = fun_name
        self.optimal_value = 0.0
        self.shift = shift
        self.stretch = stretch
        self.sinTrans = sinTrans
        self.expTrans = expTrans
        self.cosTrans = cosTrans
        self.inv = inv
        RX = [(-5.12, 5.12) for _ in range(input_dim)] if RX is None else RX
        optimizers = [tuple(self.shift for _ in range(input_dim))]

        super(Sphere, self).__init__(
            input_dim=input_dim,
            bounds=bounds,
            sd=sd,
            RX=RX,
            optimizers=optimizers,
            Seed=Seed,
            dtype=dtype,
        )

    def f(self, X):
        X = self.transfer(X)

        if len(X.shape) == 1:
            X = X.reshape(shape=(1, self.xdim))

        n = X.shape[0]

        y = np.sum((self.stretch*X - self.shift)**2, axis=1)

        if self.sinTrans == 1:
            y = np.sin(y)
        if self.expTrans == 1:
            y = np.exp(y)
        if self.cosTrans == 1:
            y = np.cos(y)
        if self.inv == 1:
            y= -y

        return  y + self.noise(n)




class Rastrigin(BASE):
    '''
    Rastrigin function

    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''

    def __init__(
            self,
            fun_name,
            input_dim=1,
            bounds=None,
            sd=None,
            RX=None,
            Seed=0,
            shift=0,
            stretch=1,
            sinTrans=0,
            expTrans=0,
            cosTrans=0,
            inv=0,
            dtype=np.float64,
    ):
        self.name = fun_name
        self.optimal_value = 0.0
        self.shift = shift
        self.sinTrans = sinTrans
        self.expTrans = expTrans
        self.cosTrans = cosTrans
        self.inv = inv
        RX = [(-5.12, 5.12) for _ in range(input_dim)] if RX is None else RX

        if stretch == 1:
            self.w = 2
        else:
            self.w = stretch
        optimizers = [tuple(-self.shift for _ in range(input_dim))]

        super(Rastrigin, self).__init__(
            input_dim=input_dim,
            bounds=bounds,
            sd=sd,
            RX=RX,
            optimizers=optimizers,
            Seed=Seed,
            dtype=dtype,
        )

    def f(self, X):
        X = self.transfer(X)
        if len(X.shape) == 1:
            X = X.reshape(shape=(1, self.xdim))

        n = X.shape[0]
        pi = np.array([math.pi], dtype=self.dtype)
        y = 10.0 * self.xdim + np.sum((X + self.shift) ** 2 - 10.0 * np.cos(self.w * pi * (X+self.shift)), axis=1)


        if self.sinTrans == 1:
            y = np.sin(y)
        if self.expTrans == 1:
            y = np.exp(y)
        if self.cosTrans == 1:
            y = np.cos(y)
        if self.inv == 1:
            y= -y
        return y + self.noise(n)



class Ackley(BASE):
    '''
    Ackley function

    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''

    def __init__(
            self,
            fun_name,
            input_dim=1,
            bounds=None,
            sd=None,
            RX=None,
            Seed=0,
            shift=0,
            stretch=1,
            sinTrans=0,
            expTrans=0,
            cosTrans=0,
            inv=0,
            dtype=np.float64,
    ):
        self.name = fun_name
        self.optimal_value = 0.0
        self.shift = shift
        self.stretch = stretch
        self.sinTrans = sinTrans
        self.expTrans = expTrans
        self.cosTrans = cosTrans
        self.inv = inv
        RX = [(-32.768, 32.768) for _ in range(input_dim)] if RX is None else RX
        optimizers = [tuple(self.shift for _ in range(input_dim))]

        super(Ackley, self).__init__(
            input_dim=input_dim,
            bounds=bounds,
            sd=sd,
            RX=RX,
            optimizers=optimizers,
            Seed=Seed,
            dtype=dtype,
        )
        self.a = np.array([20], dtype=self.dtype)
        self.b = np.array([0.2], dtype=self.dtype)
        self.c = np.array([2 * math.pi], dtype=self.dtype)


    def f(self, X):
        X = self.transfer(X)
        if len(X.shape) == 1:
            X = X.reshape(shape=(1, self.xdim))

        n = X.shape[0]
        d = X.shape[1]
        a, b, c = self.a, self.b, self.c

        part1 = -a * np.exp(-b / math.sqrt(d) * np.linalg.norm((X - self.shift), axis=-1))
        part2 = -(np.exp(np.mean(np.cos(c * (self.stretch*X - self.shift)), axis=-1)))
        y = part1 + part2 + a + math.e

        if self.sinTrans == 1:
            y = np.sin(y)
        if self.expTrans == 1:
            y = np.exp(y)
        if self.cosTrans == 1:
            y = np.cos(y)
        if self.inv == 1:
            y= -y


        return y + self.noise(n)



class Schwefel(BASE):
    '''
    Schwefel function

    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''

    def __init__(
            self,
            fun_name,
            input_dim=1,
            bounds=None,
            sd=None,
            RX=None,
            Seed=0,
            shift=0,
            stretch=1,
            sinTrans=0,
            expTrans=0,
            cosTrans=0,
            inv=0,
            dtype=np.float64,
    ):
        self.name = fun_name
        self.optimal_value = 0.0
        self.shift = shift
        self.stretch = stretch
        self.sinTrans = sinTrans
        self.expTrans = expTrans
        self.cosTrans = cosTrans
        self.inv = inv
        RX = [[-500.0, 500.0] for _ in range(input_dim)] if RX is None else RX
        optimizers = [tuple((420.9687 / self.stretch + self.shift) for _ in range(input_dim))]

        super(Schwefel, self).__init__(
            input_dim=input_dim,
            bounds=bounds,
            sd=sd,
            RX=RX,
            optimizers=optimizers,
            Seed=Seed,
            dtype=dtype,
        )

    def f(self, X):
        self.query_num += 1
        X = self.transfer(X)
        if len(X.shape) == 1:
            X = X.reshape(shape=(1, self.xdim))

        n = X.shape[0]
        d = X.shape[1]

        y = 418.9829*d - np.sum(np.multiply(self.stretch*X-self.shift, np.sin(np.sqrt(abs(self.stretch*X-self.shift)))), axis=1)

        if self.sinTrans == 1:
            y = np.sin(y)
        if self.expTrans == 1:
            y = np.exp(y)
        if self.cosTrans == 1:
            y = np.cos(y)
        if self.inv == 1:
            y= -y

        return y + self.noise(n)



class Levy(BASE):
    '''
    Levy function

    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''

    def __init__(
            self,
            fun_name,
            input_dim=1,
            bounds=None,
            sd=None,
            RX=None,
            Seed=0,
            shift=0,
            stretch = 1,
            sinTrans=0,
            expTrans=0,
            cosTrans=0,
            inv=0,
            dtype=np.float64,
    ):
        self.name = fun_name
        self.optimal_value = 0.0
        self.shift = shift
        self.stretch = stretch
        self.sinTrans = sinTrans
        self.expTrans = expTrans
        self.cosTrans = cosTrans
        self.inv = inv
        RX = [(-10.0, 10.0) for _ in range(input_dim)] if RX is None else RX
        optimizers = [tuple(self.shift for _ in range(input_dim))]

        super(Levy, self).__init__(
            input_dim=input_dim,
            bounds=bounds,
            sd=sd,
            RX=RX,
            optimizers=optimizers,
            Seed=Seed,
            dtype=dtype,
        )

    def f(self, X):
        X = self.transfer(X)
        if len(X.shape) == 1:
            X = X.reshape(shape=(1, self.xdim))

        n = X.shape[0]
        w = 1.0 + (self.stretch*X - self.shift - 1.0) / 4.0
        pi = np.array([math.pi], dtype=self.dtype)
        part1 = np.sin(pi * w[..., 0]) ** 2
        part2 = np.sum(
            (w[..., :-1] - 1.0) ** 2
            * (1.0 + 10.0 * np.sin(math.pi * w[..., :-1] + 1.0) ** 2),
            axis=1,
        )
        part3 = (w[..., -1] - 1.0) ** 2 * (
            1.0 + np.sin(2.0 * math.pi * w[..., -1]) ** 2
        )
        y = part1 + part2 + part3

        self.query_num += 1

        if self.sinTrans == 1:
            y = np.sin(y)
        if self.expTrans == 1:
            y = np.exp(y)
        if self.cosTrans == 1:
            y = np.cos(y)
        if self.inv == 1:
            y= -y


        return y + self.noise(n)


class Griewank(BASE):
    '''
    Griewank function

    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''

    def __init__(
            self,
            fun_name,
            input_dim=1,
            bounds=None,
            sd=None,
            RX=None,
            Seed=0,
            shift=0,
            stretch=1,
            sinTrans=0,
            expTrans=0,
            cosTrans=0,
            inv=0,
            dtype=np.float64,
    ):
        self.name = fun_name
        self.optimal_value = 0.0
        self.shift = shift
        self.stretch = stretch
        self.sinTrans = sinTrans
        self.expTrans = expTrans
        self.cosTrans = cosTrans
        self.inv = inv
        RX = [[-600.0, 600.0] for _ in range(input_dim)] if RX is None else RX
        optimizers = [tuple(self.shift for _ in range(input_dim))]

        super(Griewank, self).__init__(
            input_dim=input_dim,
            bounds=bounds,
            sd=sd,
            RX=RX,
            optimizers=optimizers,
            Seed=Seed,
            dtype=dtype,
        )

    def f(self, X):
        X = self.transfer(X)
        if len(X.shape) == 1:
            X = X.reshape(shape=(1, self.xdim))

        n = X.shape[0]
        d = X.shape[1]

        div = np.arange(start=1, stop=d+1, dtype=self.dtype)
        part1 = np.sum((X-self.shift) ** 2 / 4000.0, axis=1)
        part2 = -np.prod(np.cos((self.stretch*X - self.shift) / np.sqrt(div)), axis=1)
        y = part1 + part2 + 1.0

        if self.sinTrans == 1:
            y = np.sin(y)
        if self.expTrans == 1:
            y = np.exp(y)
        if self.cosTrans == 1:
            y = np.cos(y)
        if self.inv == 1:
            y= -y

        return y + self.noise(n)






class Rosenbrock(BASE):
    '''
    Rosenbrock function

    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''

    def __init__(
            self,
            fun_name,
            input_dim=1,
            bounds=None,
            sd=None,
            RX=None,
            Seed=0,
            shift=0,
            stretch=1,
            sinTrans=0,
            expTrans=0,
            cosTrans=0,
            inv=0,
            dtype=np.float64,
    ):
        self.name = fun_name
        self.optimal_value = 0.0
        self.shift = shift
        self.stretch = stretch
        self.sinTrans = sinTrans
        self.expTrans = expTrans
        self.cosTrans = cosTrans
        self.inv = inv
        RX = [(-5.0, 10.0) for _ in range(input_dim)] if RX is None else RX
        optimizers = [tuple(1.0 for _ in range(input_dim))]

        super(Rosenbrock, self).__init__(
            input_dim=input_dim,
            bounds=bounds,
            sd=sd,
            RX=RX,
            optimizers=optimizers,
            Seed=Seed,
            dtype=dtype,
        )

    def f(self, X):
        X = self.transfer(X)
        if len(X.shape) == 1:
            X = X.reshape(shape=(1, self.xdim))

        n = X.shape[0]
        y = np.sum(
            100.0 * (X[..., 1:] - X[..., :-1] ** 2) ** 2 + (X[..., :-1] - 1) ** 2,
            axis=-1,
        )

        if self.sinTrans == 1:
            y = np.sin(y)
        if self.expTrans == 1:
            y = np.exp(y)
        if self.cosTrans == 1:
            y = np.cos(y)
        if self.inv == 1:
            y= -y

        return y + self.noise(n)





class Dropwave(BASE):
    def __init__(
            self,
            fun_name,
            input_dim=1,
            bounds=None,
            sd=None,
            RX=None,
            Seed=0,
            shift=0,
            stretch = 1,
            sinTrans=0,
            expTrans=0,
            inv=0,
            dtype=np.float64,
    ):
        self.name = fun_name
        self.optimal_value = 0.0
        self.shift = shift
        self.stretch = stretch
        self.sinTrans = sinTrans
        self.expTrans = expTrans
        self.inv = inv
        RX = [(-10.0, 10.0) for _ in range(input_dim)] if RX is None else RX
        optimizers = [tuple(self.shift for _ in range(input_dim))]

        super(Dropwave, self).__init__(
            input_dim=input_dim,
            bounds=bounds,
            sd=sd,
            RX=RX,
            optimizers=optimizers,
            Seed=Seed,
            dtype=dtype,
        )
        self.a = np.array([20], dtype=self.dtype)
        self.b = np.array([0.2], dtype=self.dtype)
        self.c = np.array([2 * math.pi], dtype=self.dtype)

    def f(self, X):
        X = self.transfer(X)
        if len(X.shape) == 1:
            X = X.reshape(shape=(1, self.xdim))

        n = X.shape[0]
        part1 = np.linalg.norm(X,axis=1)
        y = -(1 + np.cos(part1)) / (0.5*np.power(part1,2) + 2)


        return y


class Langermann(BASE):
    def __init__(
            self,
            fun_name,
            input_dim=1,
            bounds=None,
            sd=None,
            RX=None,
            Seed=0,
            shift=0,
            stretch = 1,
            sinTrans=0,
            expTrans=0,
            cosTrans=0,
            inv=0,
            dtype=np.float64,
    ):
        self.name = fun_name
        self.optimal_value = 0.0
        self.shift = shift
        self.stretch = stretch
        self.sinTrans = sinTrans
        self.expTrans = expTrans
        self.cosTrans = cosTrans
        self.inv = inv
        RX = [(0, 10.0) for _ in range(input_dim)] if RX is None else RX
        optimizers = [tuple(self.shift for _ in range(input_dim))]

        super(Langermann, self).__init__(
            input_dim=input_dim,
            bounds=bounds,
            sd=sd,
            RX=RX,
            optimizers=optimizers,
            Seed=Seed,
            dtype=dtype,
        )
        self.c = np.array([1,2,5])

        self.m = 3
        self.A = np.random.randint(1,10,(self.m,input_dim))

    def f(self, X):
        X = self.transfer(X)
        if len(X.shape) == 1:
            X = X.reshape(shape=(1, self.xdim))

        n = X.shape[0]
        d = X.shape[1]

        y =0
        for i in range(self.m):
            part1 = np.exp(-np.sum(np.power(X-self.A[i],2), axis=1)/np.pi)
            part2 = np.cos(np.sum(np.power(X-self.A[i],2),axis=1) * np.pi)
            y += part1*part2 *self.c[i]
        return y




def Select_test_fun(
        fun_name=None,
        input_dim=None,
        Seed=None,
        dtype=np.float64
):
    fun_only_2d = [
        'Beale',
        'Branin',
        'Bukin',
        'Cosine8',
        'EggHolder',
        'HolderTable',
        'Michalewicz',
        'SixHumpCamel',
        'ThreeHumpCamel'
    ]
    if fun_name in fun_only_2d and input_dim != 2:
        logging.error('ERROR: xdim of % only can be 2%s\n' % (fun_name))
        exit(-1)

    tmp_list = fun_name.split('_')
    fun = tmp_list[0]

    # if fun == 'Movingpeak':
    #     if fivePeak.name == fun_name:
    #         return fivePeak
    #     else:
    #         fivePeak.change()
    #         fivePeak.name = fun_name
    #         return fivePeak

    shift_var = 0
    stretch_var = 1
    sinTrans = 0
    expTrans = 0
    cosTrans = 0
    inv = 0
    for id, oper in enumerate(tmp_list[1:]):
        if oper == 'shift':
            shift_var = float(tmp_list[id+2])
        elif oper == 'stretch':
            stretch_var = float(tmp_list[id+2])
        elif oper == 'sin':
            sinTrans = 1
        elif oper == 'exp':
            expTrans = 1
        elif oper == 'inv':
            inv = 1
        elif oper == 'cos':
            cosTrans = 1
        else:
            continue

    return eval(fun)(fun_name=fun_name,input_dim=input_dim, shift = shift_var, stretch = stretch_var, sinTrans=sinTrans, cosTrans=cosTrans, expTrans=expTrans, inv=inv, Seed=Seed, dtype=dtype,)

# from Problem.MovingPeak import MovingPeak
# fivePeak = MovingPeak(name='Movingpeak_0',n_var=10)

def plot_true_contour(obj_fun_list, dim, dtype, Exper_floder=None):
    for i in obj_fun_list:
        obj_fun = Select_test_fun(fun_name=i, input_dim=dim, Seed=0, dtype=dtype)
        print(i)

        if not os.path.exists('{}/true_f/contour//'.format(Exper_floder, obj_fun.name)):
            os.makedirs('{}/true_f/contour/'.format(Exper_floder, obj_fun.name))
        name = obj_fun.name
        if '.' in obj_fun.name:
            name = name.replace('.','|')
        save_load = '{}/true_f/contour/{}'.format(Exper_floder, name)

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
        # plt.plot(optimizers[:, 0], optimizers[:, 1], marker='*', linewidth=0, color='white', markersize=10, label="GlobalOpt")
        plt.colorbar(a)
        plt.title(i)
        fig.legend(facecolor='gray')
        plt.draw()
        plt.savefig(save_load, dpi=300)
        plt.close()



def plot_true_oned(obj_fun_list, dim, dtype, Exper_floder=None):

    for i in obj_fun_list:
        f, ax = plt.subplots(1, 1, figsize=(16, 6))
        problem = Select_test_fun(fun_name=i, input_dim=dim, Seed=0, dtype=dtype)
        bounds = problem.bounds
        opt_x = problem.optimizers
        opt_val = problem.optimal_value
        test_x = np.arange(-1, 1.05, 0.005, dtype=dtype)
        test_y = problem.f(test_x[:, np.newaxis])
        # test_y = Normalize(test_y)
        ax.plot(test_x, test_y, 'r-', linewidth=1, alpha=1)
        ax.legend(['True f(x)'])
        ax.set_xlim([bounds[0][0], bounds[1][0]])
        ax.set_title(i)
        # plt.show()
        if not os.path.exists('{}/true_f/oneD/'.format(Exper_floder)):
            os.makedirs('{}/true_f/oneD/'.format(Exper_floder))
        name = problem.name
        if '.' in problem.name:
            name = name.replace('.','|')

        save_load = '{}/true_f/oneD/{}'.format(Exper_floder, name)

        plt.savefig(save_load+'')


if __name__ == '__main__':
    Dim = 2
    # obj_fun_list = [
    #     'Schwefel',
    #     'Schwefel_stretch_0.9_shift_10',
    #     'Schwefel_stretch_1.1_shift_-10',
    #     'Schwefel_stretch_0.98_shift_20',
    #     'Schwefel_stretch_1.2_shift_-20',
    #     'Schwefel_stretch_0.95_shift_30',
    #     'Schwefel_stretch_1.12_shift_-30',
    #     'Schwefel_stretch_1.05_shift_40',
    #     'Schwefel_stretch_0.96_shift_-40',
    # ]

    obj_fun_list = [
        'Schwefel',
        'Schwefel_stretch_0.2_shift_150',
        'Schwefel_stretch_10_shift_-250',
        'Schwefel_stretch_3.1_shift_422',
        'Schwefel_stretch_2.2_shift_-79',
        'Schwefel_stretch_-1.95_shift_339',
        'Schwefel_stretch_-5.1_shift_-312',
        'Schwefel_stretch_-6.25_shift_499',
        'Schwefel_stretch_0.96_shift_-17',
    ]

    for i in obj_fun_list:
        tmp_list = i.split('_')
        fun = tmp_list[0]
        for id, oper in enumerate(tmp_list[1:]):
            if oper == 'shift':
                shift_var = float(tmp_list[id + 2])
            elif oper == 'stretch':
                stretch_var = float(tmp_list[id + 2])
            elif oper == 'sin':
                sinTrans = 1
            elif oper == 'exp':
                expTrans = 1
            elif oper == 'inv':
                inv = 1
            elif oper == 'cos':
                cosTrans = 1
            else:
                continue

        obj_fun = Select_test_fun(fun_name=i, input_dim=Dim, Seed=0)
        optimizers = obj_fun.optimizers
        print(optimizers)
    # for i in obj_fun_list:
    #     fun = Select_test_fun(fun_name=i, input_dim=2, Seed=0,)
    #     print(tuple(fun.bounds[0]))
    #
    #     a = fun.f(np.array([[-0.5,0.1], [-0.3,-0.2]]))
    #     print(a)

    # plot_true_contour(obj_fun_list, Dim,  np.float64,'../experiments/plot_problem')
    plot_true_oned(obj_fun_list, 1,  np.float64,'../experiments/plot_problem')



    # fun = Select_gym_fun(fun_name='Sin_w_2_amplitude_10', input_dim=1, Seed=0, )
    # print(tuple(fun.bounds[0]))
    #
    # a = fun.f(np.array([[-0.5], [0.3]]))
    # print(a)
