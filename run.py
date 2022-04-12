import logging
import os
import time
import argparse
import pickle
import numpy as np

#
from method import BO
from Subset import Sparse
from Subset import RSVD
from Subset import Subset
from Subset import Batch
from library import CKB
from Problem import Problem

#
from itertools import product


# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['MKL_NUM_THREADS'] = "1"
os.environ['NUMEXPR_NUM_THREADS'] = "1"
os.environ['OMP_NUM_THREADS'] = "1"



parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("-d", "--dim", type=int, default=1)  # 设置维度
parser.add_argument("-n", "--name", type=str, default='subset')  # 实验名称，保存在experiments中
parser.add_argument("-t", "--trial", type=int, nargs='+', default=[0])  # 设置随机种子，与迭代次数相关
parser.add_argument("-sm", "--save_mode", type=int, default=1)  # 控制是否保存模型
parser.add_argument("-lm", "--load_mode", type=int, default=0)  # 控制是否从头开始
parser.add_argument("-as", "--adjust_seq", type=int, default=0)  # 控制是否利用相似性调整任务顺序
parser.add_argument("-ac", "--acquisition_func", type=str, default='EI')  # 控制BO的acquisition function
args = parser.parse_args()



if __name__ == '__main__':


    #测试函数
    Task_list = [
        # 'Ackley',
        # 'Schwefel',
        'Levy',
        # 'Rastrigin',
        # 'Griewank',
        ]



    #方法：BO或者TBO或者ELLAGP
    Method_list = [
        'BO_random',
        'SPARSE_VDTC',
        'SPARSE_FITC',
        'SUBSET_clustering',
        'SUBSET_decomposition',
    ]


    Xdim = args.dim
    # Method = args.algs
    Trial = args.trial
    Exp_name = args.name
    Load_mode = args.load_mode
    Save_mode = args.save_mode
    Adjust_seq = args.adjust_seq
    Acfun = args.acquisition_func


    Plt = True
    Max_iter = 0
    Init = 0
    Grid_search_num = 0
    Exper_floder = './experiments/{}'.format(Exp_name)

    if not os.path.exists('{}/figs'.format(Exper_floder)):
        os.makedirs('{}/figs'.format(Exper_floder))
    if not os.path.exists('{}/data'.format(Exper_floder)):
        os.makedirs('{}/data'.format(Exper_floder))
    if not os.path.exists('{}/log'.format(Exper_floder)):
        os.makedirs('{}/log'.format(Exper_floder))
    if not os.path.exists('{}/model'.format(Exper_floder)):
        os.makedirs('{}/model'.format(Exper_floder))

    PID = os.getpid()
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M',
        handlers=[
            logging.FileHandler(Exper_floder + '/log/' + str(PID) + '.txt'),
            logging.StreamHandler()]
    )
    logging.getLogger('matplotlib').setLevel(logging.WARNING)


    for Method in Method_list:
        if Method.split('_')[0] == 'SUBSET':
            subset_method = Method.split('_')[-1]
        elif Method.split('_')[0] == 'SPARSE':
            sparse_method = Method.split('_')[-1]
        elif Method.split('_')[0] == 'BATCH':
            batch_size = int(Method.split('_')[-1])
        Init_method = 'random'

        Max_iter = 40 * Xdim
        Init = 20 * Xdim


        for Seed in Trial:
            KB = CKB.KnowledgeBase()
            Gym_KB = CKB.KnowledgeBase()

            if not os.path.exists('{}/model/{}d/{}/'.format(Exper_floder, Xdim, Method)):
                os.makedirs('{}/model/{}d/{}/'.format(Exper_floder, Xdim, Method))

            if Load_mode == 1:
                try:
                    with open('{}/model/{}d/{}/{}_KB.txt'.format(Exper_floder, Xdim, Method, Seed), 'rb') as f:
                        KB = pickle.load(f)
                    del Task_list[:KB.len]
                except:
                    print("Can not find file, please set 'lm=0'.")
                    print("KB Start from scratch!")
            time_trial_start = time.time()
            np.random.seed(Seed)

            for task_id, task in enumerate(Task_list):
                task_ind = KB.search_task(task)

                target_fun = Problem.Select_test_fun(fun_name=task, input_dim=Xdim, Seed=Seed, dtype=np.float32)
                logging.info('Runing(' + str(Seed) + '):' +
                             '\tMethod=' + Method +
                             '\tSeed=' + str(Seed) +
                             '\tTask=' + task +
                             '\tAcf=' + Acfun +
                             '\tN=' + str(Max_iter) +
                             '\tINIT=' + str(Init) +
                             '\txdim=' + str(Xdim))
                if Method.split('_')[0] == 'BO':
                    BO.BO(Dty=np.float64, Plt=False, Evol=Max_iter, Init=Init, GSN=Grid_search_num, Xdim=Xdim,
                          Task=target_fun, Acf=Acfun, Seed=Seed, Method=Method, KB=KB, Init_method=Init_method,
                          Save_mode=Save_mode, Exper_floder=Exper_floder)
                elif Method.split('_')[0] == 'SPARSE':
                    Sparse.SPARSE(Dty=np.float64, Plt=Plt, Evol=Max_iter, Init=Init, GSN=Grid_search_num, Xdim=Xdim,
                                  Task=target_fun, Acf=Acfun, Seed=Seed, Method=Method, KB=KB, Init_method=Init_method,
                                  sparse_method=sparse_method, Save_mode=Save_mode, Exper_floder=Exper_floder)
                elif Method.split('_')[0] == 'RSVD':
                    RSVD.RSVD(Dty=np.float64, Plt=Plt, Evol=Max_iter, Init=Init, GSN=Grid_search_num, Xdim=Xdim,
                              Task=target_fun, Acf=Acfun, Seed=Seed, Method=Method, KB=KB, Init_method=Init_method,
                              Save_mode=Save_mode, Exper_floder=Exper_floder)
                elif Method.split('_')[0] == 'SUBSET':
                    Subset.SUBSET(Dty=np.float64, Plt=Plt, Evol=Max_iter, Init=Init, Xdim=Xdim,
                                  Task=target_fun, Acf=Acfun, Seed=Seed, Method=Method, KB=KB, Init_method=Init_method,
                                  subset_method=subset_method, Save_mode=Save_mode, Exper_floder=Exper_floder)
                elif Method.split('_')[0] == 'BATCH':
                    Batch.BATCH(Dty=np.float64, Plt=Plt, Evol=Max_iter, Init=Init, Xdim=Xdim,
                                  Task=target_fun, Acf=Acfun, Seed=Seed, Method=Method, KB=KB, Init_method=Init_method,
                                  Save_mode=Save_mode, Exper_floder=Exper_floder, batch_size=batch_size)

