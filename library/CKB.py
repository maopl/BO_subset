import numpy as np

def initial_ELLA(d =2,k=2, mu = 1, lam = 1, k_init = False):
    ELLA_dic = {}
    ELLA_dic['d'] = d
    ELLA_dic['k'] = k
    ELLA_dic['L'] = np.random.randn(d, k)
    ELLA_dic['A'] = np.zeros((d * k, d * k))
    ELLA_dic['b'] = np.zeros((d * k, 1))
    ELLA_dic['S'] = np.zeros((k, 0))
    ELLA_dic['T'] = 0
    ELLA_dic['mu'] = mu
    ELLA_dic['lam'] = lam
    ELLA_dic['k_init'] = k_init

    return ELLA_dic



def update_ELLA(old_ELLA_dic, new_ELLA_dic):
    old_ELLA_dic['L'] = new_ELLA_dic['L']
    old_ELLA_dic['A'] = new_ELLA_dic['A']
    old_ELLA_dic['b'] = new_ELLA_dic['b']
    old_ELLA_dic['S'] = new_ELLA_dic['S']
    old_ELLA_dic['T'] = new_ELLA_dic['T']

class KnowledgeBase():
    def __init__(self):
        self.name = []
        self.type = []
        self.model = []
        self.x = []
        self.y = []
        self.len = 0
        self.current_task_x = []
        self.local_optimal = []
        self.initial_x = []
        self.initial_y = []
        self.random_y = []

        self.ELLA_dic = initial_ELLA()

    def add(self, fun_name, type, x, y, optimal, initial_x = None, initial_y = None, model=None):
        if fun_name not in self.name:
            self.name.append(fun_name)
            self.type.append(type)
            self.model.append(model)
            self.x.append(x)
            self.y.append(y)
            self.len += 1
            self.local_optimal.append(optimal)
            if initial_x is not None:
                self.initial_x.append(initial_x)
            else:
                self.initial_x.append([])
            if initial_y is not None:
                self.initial_y.append(initial_y)
            else:
                self.initial_x.append([])
        else:
            ind = self.name.index(fun_name)
            self.x[ind] = x
            self.y[ind] = y
            self.local_optimal[ind] = optimal
            self.type[ind] = type
            self.model[ind] = model


    def search_task(self, fun_name):
        if fun_name not in self.name:
            return self.len
        else:
            return self.name.index(fun_name)

    def random_mean_std(self):
        Y = np.array(self.random_y)
        mean = np.mean(Y)
        std = np.std(Y)
        return mean, std