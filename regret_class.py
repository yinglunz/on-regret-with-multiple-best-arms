import numpy as np
import math
import copy
from math import log, sqrt


def get_reward(instance, arm, sigma, instance_type):

    if instance_type == 'bernoulli':
        if np.random.random() < instance[arm]:
            return 1
        else:
            return 0
    else:
        return np.random.normal(instance[arm], sigma)


class MOSS:
    def __init__(self, instance, n_select, instance_type, sigma, pull_max):
        self.n = len(instance)
        n = self.n
        self.instance = instance
        self.opt_reward = max(self.instance)
        self.Delta = max(instance) - instance
        self.n_select = n_select
        self.index_select = np.random.choice(self.n, min(self.n, self.n_select), replace=False)
        self.instance_type = instance_type
        self.sigma = sigma
        self.t = 0
        self.wins = np.zeros(n)
        self.pulls = np.zeros(n)
        self.rewards = np.zeros(n)
        self.ucbs = 10000 * np.ones(n)
        self.lcbs = np.zeros(n)
        self.regret = 0
        self.pull_max = pull_max
        self.reward = 0

    def compute_ci_MOSS(self, arm):
        # compute the confidence interval based on MOSS
        log_term = max(log(self.pull_max/(self.pulls[arm] * len(self.index_select))), 0)
        return math.sqrt(log_term / (self.pulls[arm]))


    def update(self):
        s_candidate = [(x, self.ucbs[x]) for x in self.index_select]
        candidate_value = max(s_candidate, key=lambda x: x[1])[1]
        s_candidate_index = [x for x, y in s_candidate if y == candidate_value]
        arm = np.random.choice(s_candidate_index)
        rwd = get_reward(self.instance, arm, self.sigma, self.instance_type)
        self.pulls[arm] += 1
        self.wins[arm] += rwd
        self.rewards[arm] = self.wins[arm] / self.pulls[arm]
        beta_tilde = self.compute_ci_MOSS(arm)
        self.ucbs[arm] = self.rewards[arm] + beta_tilde
        self.lcbs[arm] = self.rewards[arm] - beta_tilde
        self.regret += self.opt_reward - self.instance[arm]
        self.reward += self.instance[arm]
        self.t += 1


    def compute_regret(self):
        return np.inner(self.Delta[self.index_select], self.pulls[self.index_select])


class Quantile:
    # algorithm try to minimize "quantile regret".
    def __init__(self, instance, n_select_initial, instance_type, sigma, pull_max):
        self.n = len(instance)
        n = self.n
        self.instance = instance
        self.opt_reward = max(self.instance)
        self.Delta = max(instance) - instance
        self.n_select = n_select_initial
        # we need to have n_select_initial = 2 to satisfy the requirement in quantile regret paper
        self.index_select = np.random.choice(self.n, self.n_select, replace=False)
        self.instance_type = instance_type
        self.sigma = sigma
        self.alpha = 0.347
        self.t = 0
        self.l = 1
        self.wins = np.zeros(n)
        self.pulls = np.zeros(n)
        self.rewards = np.zeros(n)
        self.ucbs = 10000 * np.ones(n)
        self.lcbs = np.zeros(n)
        self.regret = 0
        self.reward = 0
        self.pull_max = pull_max

    def compute_ci_MOSS(self, arm):
        # compute the confidence interval based on MOSS
        pull_max_up_this_round = min((2 ** (self.l+1) - 2), self.pull_max)
        num_arms_selected = len(self.index_select)
        log_term = max(log(pull_max_up_this_round / (self.pulls[arm] * num_arms_selected)), 0)
        # with adjusted time horizon as suggested in the quantile regret paper
        return math.sqrt(log_term / (self.pulls[arm]))


    def update(self):

        if self.t == (2 ** (self.l+1) - 2):
            self.l = self.l + 1
            # re-calculate ucbs for each arm in the beginning of each iteration
            for arm in self.index_select:
                beta_tilde = self.compute_ci_MOSS(arm)
                self.ucbs[arm] = self.rewards[arm] + beta_tilde
                self.lcbs[arm] = self.rewards[arm] - beta_tilde

            index_remain = np.setdiff1d(np.arange(self.n), self.index_select)
            if len(index_remain) > 0:
                n_select_additional = int(np.ceil(2**(self.l * self.alpha)) - len(self.index_select))
                index_select_addtional = np.random.choice(index_remain, min(n_select_additional, len(index_remain)), replace=False)
                self.index_select = np.hstack((self.index_select, index_select_addtional))

        s_candidate = [(x, self.ucbs[x]) for x in self.index_select]
        candidate_value = max(s_candidate, key=lambda x: x[1])[1]
        s_candidate_index = [x for x, y in s_candidate if y == candidate_value]
        arm = np.random.choice(s_candidate_index)
        rwd = get_reward(self.instance, arm, self.sigma, self.instance_type)
        self.pulls[arm] += 1
        self.wins[arm] += rwd
        self.rewards[arm] = self.wins[arm] / self.pulls[arm]
        beta_tilde = self.compute_ci_MOSS(arm)
        self.ucbs[arm] = self.rewards[arm] + beta_tilde
        self.lcbs[arm] = self.rewards[arm] - beta_tilde
        self.regret += self.opt_reward - self.instance[arm]
        self.reward += self.instance[arm]
        self.t += 1

    def compute_regret(self):
        return np.inner(self.Delta[self.index_select], self.pulls[self.index_select])

class MOSSPLUS:
    # Algorithm MOSS++ and empMOSS++ in the paper
    def __init__(self, instance, instance_type, sigma, pull_max, beta):

        self.n = len(instance)
        n = self.n
        self.instance = instance
        self.opt_reward = max(instance)
        self.Delta = self.opt_reward - instance
        self.instance_type = instance_type
        self.sigma = sigma
        self.pull_max = pull_max
        # pull_max is the given time horizon T
        self.beta = beta
        self.t = 0
        # t is the current time
        self.l = 1
        # l represents the round, from 1 to np.ceil(log_2(T^\beta))
        self.wins = np.zeros(n)
        self.pulls = np.zeros(n)
        self.pulls_history = np.zeros(n)
        self.rewards = np.zeros(n)
        # rewards stand for empirical means here
        self.ucbs = 10000 * np.ones(n)
        # this is to make sure un-pulled arms will be pulled next if selection is based on ucb
        self.lcbs = np.zeros(n)
        self.regret = 0
        self.reward = 0

        ####### followings are extra "made-up"-arms/ empirical measure for MOSS++
        self.p = int(np.ceil(log((self.pull_max ** self.beta), 2)))
        # p represents the number of total iterations

        self.n_select = min(2 ** (self.p+1), self.n)
        # the number of arm selected in the first round, i.e., K_1 in the paper
        self.index_select = np.random.choice(self.n, self.n_select, replace=False)
        # one can also change to the setting of sampling with replacement 
        # and remove the upper bound n on n_select

        self.instance_extra = np.zeros(self.p)
        self.wins_extra = np.zeros(self.p)
        self.pulls_extra = np.zeros(self.p)
        self.pulls_extra_history = np.zeros(self.p)
        self.rewards_extra = np.zeros(self.p)
        self.ucbs_extra = 10000 * np.ones(self.p)
        self.lcbs_extra = np.zeros(self.p)

        self.Delta_extra = self.opt_reward - self.instance_extra


    def compute_ci_MOSS(self, arm):
        # compute the confidence interval based on MOSS
        pull_max_up_this_round = min(2 ** (self.p) * (2 ** (self.l+1) - 2), self.pull_max)
        num_arms = len(self.index_select) + self.l - 1
        log_term = max(log(pull_max_up_this_round / (self.pulls[arm] * num_arms)), 0)
        # with adjusted time horizon in the same way as suggested in the quantile regret paper
        return math.sqrt(log_term / (self.pulls[arm]))

    def compute_ci_MOSS_v(self, arm):
        # v stands for vanilla version
        # compute the confidence interval based on MOSS
        pull_max_this_round = min(2 ** (self.p) * (2 ** (self.l) ), self.pull_max)
        num_arms = len(self.index_select) + self.l - 1
        log_term = max(log(pull_max_this_round / (self.pulls[arm] * num_arms)), 0)
        return math.sqrt(log_term / (self.pulls[arm]))


    def compute_ci_MOSS_extra(self, arm):
        # compute the confidence interval based on MOSS
        pull_max_up_this_round = min(2 ** (self.p) * (2 ** (self.l+1) - 2), self.pull_max)
        num_arms = len(self.index_select) + self.l - 1
        log_term = max(log(pull_max_up_this_round / (self.pulls_extra[arm] * num_arms)), 0)
        # with adjusted time horizon in the same way as suggested in the quantile regret paper
        return math.sqrt(log_term / (self.pulls_extra[arm]))

    def compute_ci_MOSS_extra_v(self, arm):
        # v stands for vanilla version
        # compute the confidence interval based on MOSS
        pull_max_this_round = min(2 ** (self.p) * (2 ** (self.l) ), self.pull_max)
        num_arms = len(self.index_select) + self.l - 1
        log_term = max(log(pull_max_this_round / (self.pulls_extra[arm] * num_arms)), 0)
        return math.sqrt(log_term / (self.pulls_extra[arm]))


    def update_vanilla(self):
        # update for the vanilla version MOSS++

        if self.t == 2 ** (self.p) * (2 ** (self.l + 1) - 2):
            # time at which one should prepare for the next iteration
            pulls = copy.deepcopy(self.pulls)
            # pull distribution at the l-th iteration
            pulls_total = sum(pulls)
            # print('pulls_total = {}'.format(pulls_total))
            extra_pulls = self.pulls_extra
            extra_pulls_total = sum(extra_pulls)
            # print('pulls_extra_total = {}'.format(extra_pulls_total))
            pulls_combined_total = pulls_total + extra_pulls_total
            if pulls_combined_total != 2 ** (self.p + self.l):
                raise Exception('error in calculation of extra pulls!!')
            new_mean = (np.inner(self.instance, pulls) + np.inner(self.instance_extra, extra_pulls)) / (pulls_combined_total)
            self.instance_extra[self.l - 1] = new_mean
            self.Delta_extra = max(self.instance) - self.instance_extra
            self.wins = np.zeros(self.n)
            self.pulls = np.zeros(self.n)
            self.rewards = np.zeros(self.n)
            # rewards stand for empirical means here
            self.ucbs = 10000 * np.ones(self.n)
            self.lcbs = np.zeros(self.n)
            self.wins_extra = np.zeros(self.p)
            self.pulls_extra = np.zeros(self.p)
            self.pulls_extra_history = np.zeros(self.p)
            self.rewards_extra = np.zeros(self.p)
            self.ucbs_extra = 10000 * np.ones(self.p)
            self.lcbs_extra = np.zeros(self.p)

            self.l = self.l + 1
            n_select = max(int(len(self.index_select) / 2), 4)

            self.index_select = np.random.choice(self.n, min(n_select, self.n), replace=False)


        # following are the way to decide which arm to pull, considering both regular arms and empirical measures
        s_candidate = [(x, self.ucbs[x]) for x in self.index_select]
        candidate_value = max(s_candidate, key=lambda x: x[1])[1]

        if self.l > 1:
            s_candidate_extra = [(x, self.ucbs_extra[x]) for x in range(self.l - 1)]
            # l-1 previous empirical measure at the i-th iteration
            candidate_value_extra = max(s_candidate_extra, key=lambda x: x[1])[1]
        else:
            s_candidate_extra = [(0, 0)]
            candidate_value_extra = candidate_value - 1
            # it's enough to make sure it's smaller than candidate value of regular arm

        if candidate_value > candidate_value_extra:
            s_candidate_index = [x for x, y in s_candidate if y == candidate_value]
            arm = np.random.choice(s_candidate_index)
            rwd = get_reward(self.instance, arm, self.sigma, self.instance_type)
            self.pulls[arm] += 1
            self.wins[arm] += rwd
            self.rewards[arm] = self.wins[arm] / self.pulls[arm]
            beta_tilde = self.compute_ci_MOSS_v(arm)
            self.ucbs[arm] = self.rewards[arm] + beta_tilde
            self.lcbs[arm] = self.rewards[arm] - beta_tilde
            self.regret += self.opt_reward - self.instance[arm]
            self.reward += self.instance[arm]

        else:
            s_candidate_extra_index = [x for x, y in s_candidate_extra if y == candidate_value_extra]
            arm = np.random.choice(s_candidate_extra_index)
            rwd = get_reward(self.instance_extra, arm, self.sigma, self.instance_type)
            # it's enough to obtain reward in this way for Bernoulli feedback
            self.pulls_extra[arm] += 1
            self.wins_extra[arm] += rwd
            self.rewards_extra[arm] = self.wins_extra[arm] / self.pulls_extra[arm]
            beta_tilde = self.compute_ci_MOSS_extra_v(arm)
            self.ucbs_extra[arm] = self.rewards_extra[arm] + beta_tilde
            # print('extra arm ucb = {}'.format(self.ucbs_extra[arm]))
            self.lcbs_extra[arm] = self.rewards_extra[arm] - beta_tilde
            self.regret += self.opt_reward - self.instance_extra[arm]
            self.reward += self.instance_extra[arm]

        self.t += 1


    def update_emp(self):
        # update for the empirical version empMOSS++

        if self.t == 2**(self.p) * (2 ** (self.l+1) - 2):
            # time at which one should prepare for the next iteration
            pulls = self.pulls - self.pulls_history
            # pull distribution at the l-th iteration
            pulls_total = sum(pulls)
            extra_pulls = self.pulls_extra - self.pulls_extra_history
            extra_pulls_total = sum(extra_pulls)
            pulls_combined_total = pulls_total + extra_pulls_total
            if pulls_combined_total != 2**(self.p + self.l):
                raise Exception('error in calculation of extra pulls!!')
            new_mean = (np.inner(self.instance, pulls) + np.inner(self.instance_extra, extra_pulls)) / (pulls_combined_total)
            self.instance_extra[self.l - 1] = new_mean
            self.Delta_extra = max(self.instance) - self.instance_extra
            self.pulls_history = copy.deepcopy(self.pulls)
            self.pulls_extra_history = copy.deepcopy(self.pulls_extra)
            self.l = self.l + 1
            n_select = max(int(len(self.index_select)/2), 4)

            ranking_rewards = np.argsort(self.rewards)[::-1]
            self.index_select = ranking_rewards[:n_select]
            # self.index_select = np.random.choice(self.n, min(n_select, self.n), replace=False)

            # update confidence intervals at the beginning at each round
            for arm in self.index_select:
                if self.pulls[arm] > 0:
                    beta_tilde = self.compute_ci_MOSS(arm)
                    self.ucbs[arm] = self.rewards[arm] + beta_tilde
                    self.lcbs[arm] = self.rewards[arm] - beta_tilde

            for arm in range(self.l - 1):
                if self.pulls_extra[arm] > 0:
                    beta_tilde = self.compute_ci_MOSS_extra(arm)
                    self.ucbs_extra[arm] = self.rewards_extra[arm] + beta_tilde
                    self.lcbs_extra[arm] = self.rewards_extra[arm] - beta_tilde

        s_candidate = [(x, self.ucbs[x]) for x in self.index_select]
        candidate_value = max(s_candidate, key=lambda x: x[1])[1]

        if self.l > 1:
            s_candidate_extra = [(x, self.ucbs_extra[x]) for x in range(self.l - 1)]
            # l-1 previous empirical measure at the i-th iteration
            candidate_value_extra = max(s_candidate_extra, key=lambda x: x[1])[1]
        else:
            s_candidate_extra = [(0,0)]
            candidate_value_extra = candidate_value - 1
            # it's enough to make sure it's smaller than candidate value for regular arm

        if candidate_value > candidate_value_extra:
            s_candidate_index = [x for x, y in s_candidate if y == candidate_value]
            arm = np.random.choice(s_candidate_index)
            rwd = get_reward(self.instance, arm, self.sigma, self.instance_type)
            self.pulls[arm] += 1
            self.wins[arm] += rwd
            self.rewards[arm] = self.wins[arm] / self.pulls[arm]
            beta_tilde = self.compute_ci_MOSS(arm)
            self.ucbs[arm] = self.rewards[arm] + beta_tilde
            self.lcbs[arm] = self.rewards[arm] - beta_tilde
            self.regret += self.opt_reward - self.instance[arm]
            self.reward += self.instance[arm]

        else:
            s_candidate_extra_index = [x for x, y in s_candidate_extra if y == candidate_value_extra]
            arm = np.random.choice(s_candidate_extra_index)
            rwd = get_reward(self.instance_extra, arm, self.sigma, self.instance_type)
            # it's enough to obtain reward in this way for Bernoulli feedback
            self.pulls_extra[arm] += 1
            self.wins_extra[arm] += rwd
            self.rewards_extra[arm] = self.wins_extra[arm] / self.pulls_extra[arm]
            beta_tilde = self.compute_ci_MOSS_extra(arm)
            self.ucbs_extra[arm] = self.rewards_extra[arm] + beta_tilde
            self.lcbs_extra[arm] = self.rewards_extra[arm] - beta_tilde
            self.regret += self.opt_reward - self.instance_extra[arm]
            self.reward += self.instance_extra[arm]

        self.t += 1



    def compute_regret(self):
        return np.inner(self.Delta, self.pulls) + np.inner(self.Delta_extra, self.pulls_extra)


class SR_MOSS:
    # subroutine of Parallel using MOSS; no need to compute the regret of each subroutine actually
    def __init__(self, instance_input, alpha, instance_type, sigma, pull_max):
        n_input = len(instance_input)
        self.n_select = int(min(np.ceil((pull_max ** alpha) * log(sqrt(pull_max))), pull_max))
        self.instance = np.random.choice(instance_input, min( n_input, self.n_select), replace=False)
        # one can also change to the setting of sampling with replacement 
        # and remove the upper bound n on n_select
        self.n = len(self.instance)
        n = self.n
        self.Delta = max(self.instance) - self.instance
        self.instance_type = instance_type
        self.sigma = sigma
        self.t = 0
        self.wins = np.zeros(n)
        self.pulls = np.zeros(n)
        self.rewards = np.zeros(n)
        self.ucbs = 10000 * np.ones(n)
        self.lcbs = np.zeros(n)
        self.regret = 0
        self.pull_max = pull_max
        self.opt_reward = max(self.instance)
        self.reward = 0

    def compute_ci_MOSS(self, arm):
        # compute the confidence interval based on MOSS
        log_term = max(log(self.pull_max/(self.pulls[arm]*self.n)), 0)
        return math.sqrt(log_term / (self.pulls[arm]))


    def update(self):
        s_candidate = [(x, self.ucbs[x]) for x in range(self.n)]
        candidate_value = max(s_candidate, key=lambda x: x[1])[1]
        s_candidate_index = [x for x, y in s_candidate if y == candidate_value]
        arm = np.random.choice(s_candidate_index)
        rwd = get_reward(self.instance, arm, self.sigma, self.instance_type)
        self.reward += self.instance[arm]
        self.pulls[arm] += 1
        self.wins[arm] += rwd
        self.rewards[arm] = self.wins[arm] / self.pulls[arm]
        beta_tilde = self.compute_ci_MOSS(arm)
        self.ucbs[arm] = self.rewards[arm] + beta_tilde
        self.lcbs[arm] = self.rewards[arm] - beta_tilde
        self.t += 1
        # it's important to return empirical reward here
        return self.instance[arm], rwd

    def compute_regret(self):
        return np.inner(self.Delta, self.pulls)



class Parallel:
    # Algorithm Parallel in the paper
    def __init__(self, instance, instance_type, sigma, pull_max):
        self.p = int(np.ceil(log(pull_max)))
        # p represents the number of subroutines
        self.interval = int(np.ceil(sqrt(pull_max)))
        # interval represents the Delta in the algorithm
        self.opt_reward = max(instance)
        self.alpha_list = np.arange(1, self.p+1)/self.p
        self.SR = [SR_MOSS(instance, alpha, instance_type, sigma, pull_max) for alpha in self.alpha_list]
        self.index_SR = np.random.randint(0, self.p)
        self.pulls_SR = np.zeros(self.p)
        self.empirical_regret_SR = np.zeros(self.p)
        # empirical regret of each subroutine
        self.regret_SR = np.zeros(self.p)
        self.t = 0
        self.regret = 0
        self.reward = 0
        self.SR_opt_reward = [x.opt_reward for x in self.SR]

    def update(self):

        if self.t % self.interval == 0:
            # time at which one should prepare for the next iteration
            s_candidate = [(i, self.empirical_regret_SR[i]) for i in range(self.p)]
            lowest_regret = min(s_candidate, key=lambda x: x[1])[1]
            s_candidate_index = [x for x, y in s_candidate if y == lowest_regret]
            self.index_SR = np.random.choice(s_candidate_index)

        exp_reward, empirical_reward = self.SR[self.index_SR].update()
        self.regret += self.opt_reward - exp_reward
        self.reward += exp_reward
        self.regret_SR[self.index_SR] += self.opt_reward - exp_reward
        self.empirical_regret_SR[self.index_SR] += self.opt_reward - empirical_reward
        self.pulls_SR[self.index_SR] += 1
        self.t += 1

