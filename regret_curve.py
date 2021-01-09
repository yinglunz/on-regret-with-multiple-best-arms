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
from math import log, sqrt
import sys

from regret_class import MOSS, Quantile, MOSSPLUS, Parallel

# this function is used to generate the instance
def generate_instance(n, alpha, pull_max):
    n_sub_opt = 5
    m = int(np.ceil(n / (2 * pull_max ** alpha)))
    # according to our definition of alpha
    r = (n-m)%n_sub_opt
    k = int((n-m)/n_sub_opt)
    if r == 0:
        instance = 0.1 * np.ones(k)
    else:
        instance = 0.1 * np.ones(k + 1)

    for i in range(1, n_sub_opt):
        if i <= r - 1:
            instance = np.hstack((instance, (i+1) * 0.1 * np.ones(k+1)))
        else:
            instance = np.hstack((instance, (i+1) * 0.1 * np.ones(k)))

    instance = np.hstack((instance, 0.9*np.ones(m)))

    if len(instance) != n:
        raise Exception('Made a mistake in instance generating!!')

    return instance, m


def calc_oracle_sample(n, alpha, pull_max):

    return min(n, int(np.ceil((2*pull_max**alpha) * log(sqrt(pull_max)))))

def calc_oracle_sample_m(n, m, pull_max):

    return min(n, int(np.ceil((n/m) * log(sqrt(pull_max)))))


def single_run_MOSS(instance, n_select, instance_type, sigma, pull_max, update_interval):
    regret_list = []

    alg_obj = MOSS(instance, n_select, instance_type, sigma, pull_max)
    while alg_obj.t <= pull_max:

        if alg_obj.t % update_interval == 0:
            regret = alg_obj.regret
            regret_list.append(regret)
            print('run MOSS at time {} with regret {}'.format(alg_obj.t, regret))
        alg_obj.update()
    final_extra = 0

    return regret_list, final_extra


def single_run_Quantile(instance, n_initial, instance_type, sigma, pull_max, update_interval):
    regret_list = []

    alg_obj = Quantile(instance, n_initial, instance_type, sigma, pull_max)

    while alg_obj.t <= pull_max:

        if alg_obj.t % update_interval == 0:
            regret = alg_obj.regret
            regret_list.append(regret)
            print('run Quantile at time {} with regret {}'.format(alg_obj.t, regret))
        alg_obj.update()

    final_extra = 0

    return regret_list, final_extra



def single_run_MOSSPLUS(instance, instance_type, sigma, pull_max, update_interval, beta):
    # vanilla version of MOSS++
    regret_list = []

    alg_obj = MOSSPLUS(instance, instance_type, sigma, pull_max, beta)

    while alg_obj.t <= pull_max:

        if alg_obj.t % update_interval == 0:
            regret = alg_obj.regret
            regret_list.append(regret)
            print('run MOSS++ Vanilla at time {} with regret {}'.format(alg_obj.t, regret))
        alg_obj.update_vanilla()

    final_extra = alg_obj.instance_extra

    return regret_list, final_extra

def single_run_empMOSSPLUS(instance, instance_type, sigma, pull_max, update_interval, beta):
    regret_list = []

    alg_obj = MOSSPLUS(instance, instance_type, sigma, pull_max, beta)

    while alg_obj.t <= pull_max:

        if alg_obj.t % update_interval == 0:
            regret = alg_obj.regret
            regret_list.append(regret)
            print('run empMOSS++ at time {} with regret {}'.format(alg_obj.t, regret))
        alg_obj.update_emp()

    final_extra = alg_obj.instance_extra

    return regret_list, final_extra

def single_run_Parallel(instance, instance_type, sigma, pull_max, update_interval):
    regret_list = []

    # alg_obj = UCB_adaptive(instance, n_initial, instance_type, sigma, delta, pull_max)
    alg_obj = Parallel(instance, instance_type, sigma, pull_max)

    while alg_obj.t <= pull_max:
        if alg_obj.t % update_interval == 0:
            regret = alg_obj.regret
            regret_list.append(regret)
            print('run Parallel at time {} with regret {}'.format(alg_obj.t, regret))
        alg_obj.update()

    final_extra = 0
    SR_emp_regret = alg_obj.empirical_regret_SR
    SR_pulls = alg_obj.pulls_SR
    SR_opt_reward = alg_obj.SR_opt_reward


    return regret_list, final_extra, SR_emp_regret, SR_pulls, SR_opt_reward

def single_sim(instance, n_select_list, beta_list, n_initial, instance_type, sigma, pull_max, update_interval):

    np.random.seed()

    regret_list_multi = []
    final_extra_multi = []


    for n_select in n_select_list:
        regret_list, final_extra = single_run_MOSS(instance, n_select, instance_type, sigma, pull_max, update_interval)
        regret_list_multi.append(regret_list)
        final_extra_multi.append(final_extra)


    regret_list, final_extra = single_run_Quantile(instance, n_initial, instance_type, sigma, pull_max, update_interval)
    regret_list_multi.append(regret_list)
    final_extra_multi.append(final_extra)


    regret_list, final_extra, SR_emp_regret, SR_pulls, SR_opt_reward = single_run_Parallel(instance, instance_type, sigma, pull_max, update_interval)
    regret_list_multi.append(regret_list)
    final_extra_multi.append(final_extra)


    for beta in beta_list:
        regret_list, final_extra= single_run_MOSSPLUS(instance, instance_type, sigma, pull_max,
                                                         update_interval, beta)
        regret_list_multi.append(regret_list)
        final_extra_multi.append(final_extra)

    for beta in beta_list:
        regret_list, final_extra= single_run_empMOSSPLUS(instance, instance_type, sigma, pull_max,
                                                         update_interval, beta)
        regret_list_multi.append(regret_list)
        final_extra_multi.append(final_extra)



    return regret_list_multi, final_extra_multi, SR_emp_regret, SR_pulls, SR_opt_reward


def multi_sim(n_parallel, n_process, n, alpha, n_initial, instance_type, sigma, pull_max, update_interval, beta_list, real_data):
    time_start = time.time()

    if real_data == 0:
        instance, m = generate_instance(n, alpha, pull_max)
        print('mean of all rewards = {}'.format(np.mean(instance)))
        n_oracle_sample = calc_oracle_sample(n, alpha, pull_max)
        n_select_list = [n, n_oracle_sample]
    else:
        instance = np.load('652_contest.npy')
        m = 54
        n_oracle_sample = calc_oracle_sample_m(len(instance), m, pull_max)
        n_select_list = [len(instance), n_oracle_sample]

    for i in n_select_list:
        if i > len(instance):
            raise Exception('select too much')
            

    single_sim_partial = partial(single_sim, instance, n_select_list, beta_list, n_initial, instance_type, sigma, pull_max)

    pool = multiprocessing.Pool(processes = n_process)
    results = pool.map(single_sim_partial, list(map(int, update_interval * np.ones(n_parallel))))
    print(results)

    print('multi_sim got results!')

    # the order of the following sequences matters!!
    measures = ['regret', 'final_extra']

    algs_MOSS = ['MOSS', 'MOSS Oracle']
    algs_MOSSPLUS = ['MOSS++_{} (ours)'.format(beta) for beta in beta_list]
    algs_empMOSSPLUS = ['empMOSS++_{} (ours)'.format(beta) for beta in beta_list]
    # algs_Restarting_v = ['Restarting_v_{}'.format(beta) for beta in beta_list]

    algs = algs_MOSS + [ 'Quantile', 'Parallel (ours)'] + algs_MOSSPLUS + algs_empMOSSPLUS

    if len(beta_list) == 1:
        algs = algs_MOSS + [ 'Quantile', 'Parallel (ours)', 'MOSS++ (ours)', 'empMOSS++ (ours)']

    dict_regret = dict(zip(algs, [[] for alg in algs]))
    dict_extra = dict(zip(algs, [[] for alg in algs]))
    # orders need to match the previous one!
    dict_results = dict(zip(measures, [dict_regret, dict_extra]))

    dict_results_ave = copy.deepcopy(dict_results)
    dict_results_std = copy.deepcopy(dict_results)

    for i in range(n_parallel):
        for j in range(len(measures)):
            for k in range(len(algs)):
                dict_results[measures[j]][algs[k]].append(results[i][j][k])
    print(dict_results)

    Parallel_measure = ['SR_emp_regret', 'SR_pulls', 'SR_opt_reward']
    Parallel_dict = dict(zip(Parallel_measure, [[] for measure in Parallel_measure]))
    for i in range(n_parallel):
        for j in range(len(Parallel_measure)):
            Parallel_dict[Parallel_measure[j]].append(results[i][len(measures)+j])

    Parallel_dict_ave = copy.deepcopy(Parallel_dict)
    for measure in Parallel_measure:
        Parallel_dict_ave[measure] = np.mean(Parallel_dict[measure], axis=0)

    k = 4
    # we devide the std by k in the plot, the shaded area thus represent 0.5 std
    for measure in measures:
        for alg in algs:
            dict_results_ave[measure][alg] = np.mean(dict_results[measure][alg], axis=0)
            dict_results_std[measure][alg] = np.std(dict_results[measure][alg], axis=0)/k

    print('---- final average results ----')
    print(dict_results_ave)

    time_end = time.time()
    print('total time spent', time_end - time_start)

    if real_data == 0:
        file = open('{}n_total - {}m - {}alpha - {}max - {}interval - {}n_parallel - {}instance_type - {}.txt'. \
                    format(n, m, alpha, pull_max, update_interval, n_parallel, instance_type, date.today(), ), 'w')
    else:
        file = open('real_data - T = {} - {}n_total - {}m - {}alpha- {}interval - {}n_parallel - {}instance_type -{}.txt'. \
                    format(pull_max, n, m, alpha,  update_interval, n_parallel, instance_type, date.today()), 'w')
        file.write('m = {}; max(instance) = {}; mean(instance) = {}\n'.format(m, max(instance), np.mean(instance)))

    file.write('{} - {}n_total - {}m - {}alpha - {}max - {}interval - {}n_parallel - {}instance_type\n'.\
              format(date.today(), n, m, alpha, pull_max, update_interval, n_parallel, instance_type))

    file.write('mean of bandit instance = {}; max of bandit instance = {}\n'.format(np.mean(instance), max(instance)))

    file.write('total time spent = {}\n'.format(time_end - time_start))

    file.write('measures: {}\n'.format(measures))
    file.write('algs: {}\n'.format(algs))

    file.write('Parallel measure = {}\n'.format(Parallel_measure))
    file.write('Following results are for Alg Parallel\n')
    for measure in Parallel_measure:
        file.write('measure:{} ave:\n'.format(measure))
        file.write('{}\n'.format(Parallel_dict_ave[measure]))

    file.write('Following results are for all algs\n')
    for measure in measures:
        for alg in algs:
            file.write('measure:{}, alg:{}, ave:\n'.format(measure, alg))
            file.write('{}\n'.format(dict_results_ave[measure][alg]))

    for measure in measures:
        for alg in algs:
            file.write('measure:{}, alg:{}, std:\n'.format(measure, alg))
            file.write('{}\n'.format(dict_results_std[measure][alg]))

    x = list(range(0, pull_max+1, update_interval))
    fig = plt.figure(1)

    # we only create 6 different line style as following. require len(algs)=6
    marker_list = [(0, (1, 1)), (0, (3, 1, 1, 1, 1, 1)), (0, (5, 1)), '-.', '--', '-']

    if len(algs) == 6:
        for i in range(len(algs)):
            ave = np.array(dict_results_ave[measures[0]][algs[i]])
            std = np.array(dict_results_std[measures[0]][algs[i]])
            plt.plot(x, ave, label=algs[i], linestyle=marker_list[i], linewidth=3)
            plt.fill_between(x, ave - std, ave + std, alpha=0.2)
    else:
        for i in range(len(algs)):
            ave = np.array(dict_results_ave[measures[0]][algs[i]])
            std = np.array(dict_results_std[measures[0]][algs[i]])
            plt.plot(x, ave, label=algs[i], linewidth=3)
            plt.fill_between(x, ave - std, ave + std, alpha=0.2)



    plt.xlabel('Time')
    plt.ylabel('Expected regret')
    plt.legend(loc=0)
    plt.grid(alpha=0.75)
    plt.xlim(0, pull_max)

    if real_data == 0:
        plt.savefig('multi_best_n={}_alpha={}.pdf'.format(n, alpha))
        max_regret = max(dict_results_ave[measures[0]][algs[2]])
        # standard MOSS incurs large regret, so we truncate the figure to give a clear comparison of other methods
        plt.ylim(0, 1.2 * max_regret)
        plt.savefig('Truncate_multi_best_n={}_alpha={}.pdf'.format(n, alpha))
    else:
        plt.savefig('real_data_T={}_m={}.pdf'.format(pull_max, m))
        max_regret = max(dict_results_ave[measures[0]][algs[2]])
        # standard MOSS incurs large regret, so we truncate the figure to give a clear comparison of other methods
        plt.ylim(0, 1.2 * max_regret)
        plt.savefig('Truncate_real_data_T={}_m={}.pdf'.format(pull_max, m))

    plt.show()
    plt.close(fig)


def main(real_data):
    instance_type = 'bernoulli'
    n_parallel = 100
    n_process = 100
    n = 20000
    alpha_list = [0.25]
    n_initial = 2
    sigma = 0.25
    update_interval = 200
    beta_list = [0.5]


    if real_data == 0:
        print('******** synthetic data learning *********')
        pull_max = 50000
        for alpha in alpha_list:
            multi_sim(n_parallel, n_process, n, alpha, n_initial, instance_type, sigma, pull_max,
                      update_interval, beta_list, real_data)
    else:
        print('******** real-world data learning *********')
        for pull_max in [100000]:
            alpha = 0.3
            # this alpha doesn't do anything, just as a required input for the following
            multi_sim(n_parallel, n_process, n, alpha, n_initial, instance_type, sigma, pull_max,
                      update_interval, beta_list, real_data)


if __name__ == '__main__':
    main(int(sys.argv[1]))
