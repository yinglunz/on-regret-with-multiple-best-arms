import numpy as np
import time
import multiprocessing
import copy
from functools import partial
from datetime import date
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from regret_class import MOSS, Quantile, MOSSPLUS, Parallel
from regret_curve import generate_instance, calc_oracle_sample



def run_MOSS(n, alpha, oracle_or_not, instance_type, sigma, pull_max, update_interval):
    instance, m = generate_instance(n, alpha, pull_max)

    if oracle_or_not == 0:
        n_select = n
        alg_obj = MOSS(instance, n_select, instance_type, sigma, pull_max)
    elif oracle_or_not == 1:
        n_select = calc_oracle_sample(n, alpha, pull_max)
        alg_obj = MOSS(instance, n_select, instance_type, sigma, pull_max)

    while alg_obj.t <= pull_max:
        if alg_obj.t % update_interval == 0:
            print('run MOSS with alpha={} at time={}'.format(alpha, alg_obj.t))
        alg_obj.update()
    regret = alg_obj.regret
    return regret

def single_run_MOSS(n, alpha_list, oracle_or_not, instance_type, sigma, pull_max, update_interval):
    regret_list = []
    for alpha in alpha_list:
        regret = run_MOSS(n, alpha, oracle_or_not, instance_type, sigma, pull_max, update_interval)
        regret_list.append(regret)
    return regret_list

def run_Quantile(n, alpha, n_initial, instance_type, sigma, pull_max, update_interval):
    instance, m = generate_instance(n, alpha, pull_max)
    alg_obj = Quantile(instance, n_initial, instance_type, sigma, pull_max)
    while alg_obj.t <= pull_max:
        if alg_obj.t % update_interval == 0:
            print('run Quantile with alpha={} at time={}'.format(alpha, alg_obj.t))
        alg_obj.update()
    regret = alg_obj.regret
    return regret

def single_run_Quantile(n, alpha_list, n_initial, instance_type, sigma, pull_max, update_interval):
    regret_list = []
    for alpha in alpha_list:
        regret = run_Quantile(n, alpha, n_initial, instance_type, sigma, pull_max, update_interval)
        regret_list.append(regret)
    return regret_list

def run_MOSSPLUS(n, alpha, instance_type, sigma, pull_max, update_interval, beta):
    instance, m = generate_instance(n, alpha, pull_max)
    alg_obj = MOSSPLUS(instance, instance_type, sigma, pull_max, beta)
    while alg_obj.t <= pull_max:
        if alg_obj.t % update_interval == 0:
            print('run MOSS++ with alpha={} at time={}'.format(alpha, alg_obj.t))
        alg_obj.update_vanilla()
        # vanilla version of MOSS++
    regret = alg_obj.regret
    return regret

def single_run_MOSSPLUS(n, alpha_list, instance_type, sigma, pull_max, update_interval, beta):
    regret_list = []

    for alpha in alpha_list:
        regret = run_MOSSPLUS(n, alpha, instance_type, sigma, pull_max, update_interval, beta)
        regret_list.append(regret)
    return regret_list

def run_empMOSSPLUS(n, alpha, instance_type, sigma, pull_max, update_interval, beta):
    instance, m = generate_instance(n, alpha, pull_max)
    alg_obj = MOSSPLUS(instance, instance_type, sigma, pull_max, beta)
    while alg_obj.t <= pull_max:
        if alg_obj.t % update_interval == 0:
            print('run empMOSS++ with alpha={} at time={}'.format(alpha, alg_obj.t))
        alg_obj.update_emp()
    regret = alg_obj.regret
    return regret

def single_run_empMOSSPLUS(n, alpha_list, instance_type, sigma, pull_max, update_interval, beta):
    regret_list = []
    for alpha in alpha_list:
        regret = run_empMOSSPLUS(n, alpha, instance_type, sigma, pull_max, update_interval, beta)
        regret_list.append(regret)
    return regret_list


def run_Parallel(n, alpha, instance_type, sigma, pull_max, update_interval):
    instance, m = generate_instance(n, alpha, pull_max)
    alg_obj = Parallel(instance, instance_type, sigma, pull_max)
    while alg_obj.t <= pull_max:
        if alg_obj.t % update_interval == 0:
            print('run Parallel with alpha={} at time={}'.format(alpha, alg_obj.t))
        alg_obj.update()
    regret = alg_obj.regret
    return regret

def single_run_Parallel(n, alpha_list, instance_type, sigma, pull_max, update_interval):
    regret_list = []
    for alpha in alpha_list:
        regret = run_Parallel(n, alpha, instance_type, sigma, pull_max, update_interval)
        regret_list.append(regret)
    return regret_list


def single_sim(n, alpha_list, beta_list, n_initial, instance_type, sigma, pull_max, update_interval):

    np.random.seed()

    regret_list_multi = []

    oracle_or_not_list = [0, 1]
    for oracle_or_not in oracle_or_not_list:
        regret_list = single_run_MOSS(n, alpha_list, oracle_or_not, instance_type, sigma, pull_max, update_interval)
        regret_list_multi.append(regret_list)


    regret_list = single_run_Quantile(n, alpha_list, n_initial, instance_type, sigma, pull_max, update_interval)
    regret_list_multi.append(regret_list)


    regret_list = single_run_Parallel(n, alpha_list, instance_type, sigma, pull_max, update_interval)
    regret_list_multi.append(regret_list)


    for beta in beta_list:
        regret_list = single_run_MOSSPLUS(n, alpha_list, instance_type, sigma, pull_max, update_interval, beta)
        regret_list_multi.append(regret_list)

    for beta in beta_list:
        regret_list = single_run_empMOSSPLUS(n, alpha_list, instance_type, sigma, pull_max, update_interval, beta)
        regret_list_multi.append(regret_list)

    return regret_list_multi

def multi_sim(n_parallel, n_process, n, alpha_list, n_initial, instance_type, sigma, pull_max, update_interval, beta_list):
    time_start = time.time()

    single_sim_partial = partial(single_sim, n, alpha_list, beta_list, n_initial, instance_type, sigma, pull_max)

    pool = multiprocessing.Pool(processes = n_process)
    results = pool.map(single_sim_partial, list(map(int, update_interval * np.ones(n_parallel))))
    print(results)

    print('multi_sim got results!')

    # the order of the following sequences matters!!
    measures = ['regret']
    algs_MOSS = ['MOSS', 'MOSS Oracle']
    algs_MOSSPLUS = ['MOSS++_{} (ours)'.format(beta) for beta in beta_list]
    algs_empMOSSPLUS = ['empMOSS++_{} (ours)'.format(beta) for beta in beta_list]

    algs = algs_MOSS + [ 'Quantile', 'Parallel (ours)'] + algs_MOSSPLUS + algs_empMOSSPLUS

    if len(beta_list) == 1:
        algs = algs_MOSS + [ 'Quantile', 'Parallel (ours)', 'MOSS++ (ours)', 'empMOSS++ (ours)']

    dict_regret = dict(zip(algs, [[] for alg in algs]))

    # orders need to match the previous one!
    dict_results = dict(zip(measures, [dict_regret]))

    dict_results_ave = copy.deepcopy(dict_results)
    dict_results_std = copy.deepcopy(dict_results)

    for i in range(n_parallel):
        for j in range(len(measures)):
            for k in range(len(algs)):
                dict_results[measures[j]][algs[k]].append(results[i][k])
            # note here measures[0]
    print(dict_results)


    k = 4
    # we devide the std by k in the plot, k=4 means the shaded area represent 0.5 std
    for measure in measures:
        for alg in algs:
            dict_results_ave[measure][alg] = np.mean(dict_results[measure][alg], axis=0)
            dict_results_std[measure][alg] = np.std(dict_results[measure][alg], axis=0)/k

    print('---- final average results ----')
    print(dict_results_ave)

    time_end = time.time()
    print('total time spent', time_end - time_start)

    file = open('AlphaComparison{}n_total - {}alpha_list - {}beta_list - {}max - {}interval - {}n_parallel - {}instance_type -{}.txt'.\
              format(n, alpha_list, beta_list, pull_max, update_interval, n_parallel, instance_type, date.today()), 'w')
    file.write('{} - {}n_total  - {}alpha_list - {}beta_list - {}max - {}interval - {}n_parallel - {}instance_type\n'.\
              format(date.today(), n, alpha_list, beta_list, pull_max, update_interval, n_parallel, instance_type))

    file.write('total time spent = {}\n'.format(time_end - time_start))

    file.write('measures: {}\n'.format(measures))
    file.write('algs: {}\n'.format(algs))


    file.write('Following results are for all algs\n')
    for measure in measures:
        for alg in algs:
            file.write('measure:{}, alg:{}, ave:\n'.format(measure, alg))
            file.write('{}\n'.format(dict_results_ave[measure][alg]))

    for measure in measures:
        for alg in algs:
            file.write('measure:{}, alg:{}, std:\n'.format(measure, alg))
            file.write('{}\n'.format(dict_results_std[measure][alg]))


    fig = plt.figure(1)

    # we only create 6 different line style as following. requires len(algs)=6
    marker_list = [(0, (1, 1)), (0, (3, 1, 1, 1, 1, 1)), (0, (5, 1)), '-.', '--', '-']
    if len(algs) == 6:
        for i in range(len(algs)):
            ave = np.array(dict_results_ave[measures[0]][algs[i]])
            std = np.array(dict_results_std[measures[0]][algs[i]])
            plt.plot(alpha_list, ave, label=algs[i], linestyle=marker_list[i], linewidth=3)
            plt.fill_between(alpha_list, ave - std, ave + std, alpha=0.2)
    else:
        for i in range(len(algs)):
            ave = np.array(dict_results_ave[measures[0]][algs[i]])
            std = np.array(dict_results_std[measures[0]][algs[i]])
            plt.plot(alpha_list, ave, label=algs[i], linewidth=3)
            plt.fill_between(alpha_list, ave - std, ave + std, alpha=0.2)



    plt.xlabel(r'$\alpha$')
    plt.ylabel('Expected regret at time {}'.format(pull_max))
    plt.legend(loc=0)
    plt.grid(alpha=0.75)
    plt.savefig('varying_hardness.pdf')
    fig.set_size_inches(6, 4)
    plt.savefig('thumbnail_fig.png')
    plt.show()
    plt.close(fig)


if __name__ == '__main__':
    instance_type = 'bernoulli'
    n_parallel = 100
    n_process = 100
    n = 20000
    alpha_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    n_initial = 2
    sigma = 0.25
    pull_max = 50000
    update_interval = 50
    beta_list = [0.5]

    multi_sim(n_parallel, n_process, n, alpha_list, n_initial, instance_type, sigma, pull_max, update_interval,
              beta_list)


