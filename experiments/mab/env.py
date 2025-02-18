import copy
import datetime
import time
import random
import math
import itertools
import multiprocessing

import numpy as np
from scipy.optimize import linprog
from scipy.stats import truncnorm

from util import format_time, true_with_probability, bounded_normal


def generate_history_information_block(param):
    """Static method for generating history information via multiple processing.

    Arguments:
        param {[type]} -- [description]

    Returns:
        {dict} -- { "" }
    """

    import numpy as np
    import os
    from util.terminal_utils import bcolors
    from util import format_time, format_now

    block_index, env, size, log, log_prefix = param

    if log:
        print("{:s}block_index: {:2d}, size: {:d}, pid: {:d}, {:s}, {:s}BEGIN{:s}".format(log_prefix, block_index, size, os.getpid(), format_now(), bcolors.OKGREEN, bcolors.ENDC))
    
    # For logging
    block_begin_time = time.time()

    c_2darray = np.zeros((env.num_arm, size), dtype=float)
    d_2darray = np.zeros((env.num_arm, size), dtype=float)
    click_2darray = np.zeros((env.num_arm, size), dtype=int)

    for t in range(size):
        # Pull all the arms in each round
        selected_arm_indices = range(env.num_arm)

        # Get the observations by pulling the arms
        observation = env.pull_arms(selected_arm_indices)

        # Fill in the clicks and the observations (c and d)
        c_list = observation[0]
        d_list = observation[1]
        for index, arm_index in enumerate(selected_arm_indices):
            c = c_list[index]
            d = d_list[index]
            c_2darray[arm_index][t] = c
            d_2darray[arm_index][t] = d
            click_2darray[arm_index][t] = 1
    
    # For logging
    block_end_time = time.time()
    block_time_consumption = block_end_time - block_begin_time
    if log:
        print("{:s}block_index: {:2d}, size: {:d}, pid: {:d}, {:s}, {:s}END{:s}   {:s}".format(log_prefix, block_index, size, os.getpid(), format_now(), bcolors.OKBLUE, bcolors.ENDC, format_time(block_time_consumption)))

    return {\
        'c': c_2darray,\
        'd': d_2darray,\
        'click': click_2darray }


class Env:
    def __init__(self, name="Env"):
        self.name = name

        self.num_arm = 0 # The number of arms, which should be set in the __init__ method of subclasses.

    
    def pull_arms(self, arm_indices):
        return


class SimpleEnv(Env):
    def __init__(self, theta_list, name="SimpleEnv", **kwds):
        super().__init__(name=name, **kwds)

        self.theta_list = theta_list
        self.num_arm = len(theta_list)
    

    def pull_arms(self, arm_indices):
        return [ [ np.random.binomial(1, self.theta_list[arm_index]) for arm_index in arm_indices ] ]


class EmojiPredictionEnv(Env):
    def __init__(
        self,
        num_arm,
        num_selection,
        predictor_list,
        accuracy_list,
        arm_predictor_index_mapping_list,
        user_input_corpus,
        latency_mean_list,
        latency_std,
        latency_min,
        latency_max,
        beta=1e-3,
        e_array=None,
        b=None,
        mock_prediction=False,
        name="EmojiPredictionEnv",
        **kwds):
        """Constructor.

        Arguments:
            num_arm {int} -- The number of arms.
            predictor_list {list of EmojiPredictor} -- A list of EmojiPredictor objects.
            arm_predictor_index_mapping_list {list of int} -- List index: the arm index. List element: the predictor index.
            user_input_corpus {dict} -- { "text_list": text_list, "label_list": label_list }. This can be loaded by nlp.dataset_utils.load_dataset

        Keyword Arguments:
            beta {float} -- The factor for the latency term. (default: {1e-3})
            e_array {numpy 1darray} -- The array of the expected energy comsumption of all the arms. (default: None)
        """

        super().__init__(name=name, **kwds)

        self.num_arm = num_arm
        self.num_selection = num_selection
        self._predictor_list = predictor_list
        self._accuracy_list = accuracy_list
        self._arm_predictor_index_mapping_list = arm_predictor_index_mapping_list
        self._user_input_corpus = user_input_corpus
        self.beta = beta
        self.latency_mean_list = latency_mean_list
        self.latency_std = latency_std
        self.latency_min = latency_min
        self.latency_max = latency_max
        self.e_array = e_array
        self.b = b
        self.mock_prediction = mock_prediction

        self._user_input_corpus_size = len(self._user_input_corpus["text_list"])

        # self._latency_lower_bound = 2
        # # self._latency_fluctuation = 150
        # self._latency_fluctuation = 15
        # self._d_min = max(min(self.latency_mean_list) - self._latency_fluctuation, self._latency_lower_bound)
        # self._d_max = max(self.latency_mean_list) + self._latency_fluctuation

        # No normalization in the journal version
        # # For normalization of rewards (observations)
        # self._loc = self.beta * self._d_max
        # self._scale = 1 / (1 + self.beta * (self._d_max - self._d_min))

        # Calculate the minimum feasible budget value
        self._minimum_feasible_b = sum(sorted(self.e_array)[:self.num_selection])

        # Calculate the optimal reward mean
        self._calculate_optimal_r()
    

    def _calculate_optimal_r(self):
        # TODO: Update the optimal policy solving for regret calculation
        # # For optimal r and regret calculation
        # self._expected_r_array = []
        # for arm_index in range(self.num_arm):
        #     predictor_index = self._arm_predictor_index_mapping_list[arm_index]
        #     self._expected_r_array.append((self._accuracy_list[predictor_index] - self.beta * self._mean_of_latency_of_predictor(predictor_index) + self._loc) * self._scale)
        # self._expected_r_array = np.array(self._expected_r_array)
        # # print("self._arm_predictor_index_mapping_list:", self._arm_predictor_index_mapping_list)
        # # print("env._expected_array:", self._expected_r_array)
        # # Get all combinations (the list of all available super arms)
        # combinations = [ i for i in itertools.combinations(range(self.num_arm), self.num_selection) ]
        # # print(len(combinations))
        # # Calculate the weight vector c
        # c_array = np.array([ np.sum(self._expected_r_array[list(combination)]) for combination in combinations ])
        # # Calculate the array of energy consumption of all combinations
        # a_ub = np.array([ np.sum(self.e_array[[self._arm_predictor_index_mapping_list[arm_index] for arm_index in combination]]) for combination in combinations ]).reshape((1, len(combinations)))
        # # Linear programming
        # if self.b < self._minimum_feasible_b:
        #     print("Infeasible budget b = {:.2f}, replaced with the minimum feasible b = {:.2f}".format(self.b, self._minimum_feasible_b))
        #     bound = self._minimum_feasible_b
        # else:
        #     bound = self.b
        # result = linprog(-c_array, A_ub=a_ub, b_ub=np.array([bound]), A_eq=np.ones((1, len(combinations)), dtype=float), b_eq=np.array([1]), bounds=[(0.0, 1.0) for _ in range(len(combinations))])
        # # print(result.x)
        # # print(-result.fun)
        # # print("result.status:", result.status)
        # # print("result.message:", result.message)
        # # print(result)
        # self.optimal_r = -result.fun
        self.optimal_r = 0.0

    
    def pull_arms(self, arm_indices):
        # Sample one sample from user_input_corpus uniformly at random
        sample_index = random.randrange(self._user_input_corpus_size)
        text = self._user_input_corpus["text_list"][sample_index]
        true_label = self._user_input_corpus["label_list"][sample_index]

        # Observe the c list
        c_list = []
        click_list = []
        for arm_index in arm_indices:
            predictor_index = self._arm_predictor_index_mapping_list[arm_index]
            c, click = self._predicted_correct(predictor_index, text, true_label)
            c_list.append(c)
            click_list.append(click)

        # Observe the d list
        d_list = [ self._latency(self._arm_predictor_index_mapping_list[arm_index]) for arm_index in arm_indices ]

        # TODO: Fix regret calculation mechanism
        # # Calculate the regret for this round
        # compound_reward = np.sum(self._expected_r_array[arm_indices])
        # regret = self.optimal_r - compound_reward
        regret = 0.0

        # Calculate the energy consumption
        energy_consumption = 0.0
        for arm_index in arm_indices:
            predictor_index = self._arm_predictor_index_mapping_list[arm_index]
            energy_consumption += self.e_array[predictor_index]

        return c_list, d_list, regret, click_list, energy_consumption
    

    def _predicted_correct(self, predictor_index, text, true_label):
        """Returns C (user satisfaction, float) and click (indicator, int).
        """

        # If mock prediction
        if self.mock_prediction:
            if true_with_probability(self._accuracy_list[predictor_index]):
                click = 1
                c = 1.0
            else:
                click = 0
                c = 0.0
        # Else, truly use the predictor to predict
        else:
            # TODO: Conduct inference on GPU if possible for acceleration
            predictor = self._predictor_list[predictor_index]
            if true_label == predictor.predict(text, top_k=1)[0]:
                click = 1
                c = 1.0
            else:
                click = 0
                c = 0.0
        
        # No need for normalization for the journal version
        # # Normalization
        # c += self._loc
        # c *= self._scale
        
        return c, click
    

    def _mean_of_latency_of_predictor(self, predictor_index):
        return self.latency_mean_list[predictor_index]
    

    def _latency(self, predictor_index):
        # mean = self._mean_of_latency_of_predictor(predictor_index)
        # a = max(mean - self._latency_fluctuation, self._latency_lower_bound)
        # b = mean + self._latency_fluctuation
        # normalized_a, normalized_b = (a - mean) / self.latency_std, (b - mean) / self.latency_std

        # latency = truncnorm.rvs(normalized_a, normalized_b) * self.latency_std + mean

        # return latency

        return bounded_normal(
            self.latency_mean_list[predictor_index],
            self.latency_std,
            self.latency_min,
            self.latency_max
        )
    

    def generate_history_information(self, size, num_selection, log=True, num_process=1):
        # If multiple processing
        if num_process > 1:
            if log:
                print("Generating history information via multiple processing...")

            # Each process generates a block, thus the number of blocks equals to the number of processes
            num_block = num_process

            with multiprocessing.Pool(num_process) as pool:
                env_copy = copy.deepcopy(self)
                env_copy.agent = None
                env_list = [ copy.deepcopy(env_copy) for _ in range(num_block) ]
                normal_block_size = size // num_block
                last_block_size = normal_block_size + size % num_block
                size_list = [ normal_block_size if block_index < num_block - 1 else last_block_size for block_index in range(num_block) ]
                log_list = [ log for _ in range(num_block) ]
                log_prefix_list = [ "  " for _ in range(num_block) ]
                result_list = pool.map(generate_history_information_block, zip(range(num_block), env_list, size_list, log_list, log_prefix_list))
                
                # TODO: merge the results
                # for block_index, result in enumerate(result_list):
                c_2darray = np.concatenate([ result['c'] for result in result_list ], axis=1)
                d_2darray = np.concatenate([ result['d'] for result in result_list ], axis=1)
                click_2darray = np.concatenate([ result['click'] for result in result_list ], axis=1)

                return {\
                    'c': c_2darray,\
                    'd': d_2darray,\
                    'click': click_2darray }
        # Else, no multiple processing
        else:
            # For logging
            num_iteration = size
            iteration_time_consumption_array = np.zeros(num_iteration, dtype=float)
            log_interval = num_iteration // 100
            total_begin_datetime = datetime.datetime.now()
            num_digit = 0
            temp = num_iteration
            while temp > 1:
                temp /= 10
                num_digit += 1
            print("  EmojiPredictionEnv.generate_history_information() started at: {:s}".format(total_begin_datetime.strftime("%Y-%m-%d %H:%M:%S")))

            c_2darray = np.zeros((self.num_arm, size), dtype=float)
            d_2darray = np.zeros((self.num_arm, size), dtype=float)
            click_2darray = np.zeros((self.num_arm, size), dtype=int)
            
            for t in range(size):
                # For logging
                begin_time = time.time()

                # Pull all the arms in each round
                selected_arm_indices = range(self.num_arm)

                # Get the observations by pulling the arms
                observation = self.pull_arms(selected_arm_indices)

                # Fill in the clicks and the observations (c and d)
                c_list = observation[0]
                d_list = observation[1]
                for index, arm_index in enumerate(selected_arm_indices):
                    c = c_list[index]
                    d = d_list[index]
                    c_2darray[arm_index][t] = c
                    d_2darray[arm_index][t] = d
                    click_2darray[arm_index][t] = 1
                
                # For logging
                iteration_index = t
                end_time = time.time()
                iteration_time_consumption = end_time - begin_time
                iteration_time_consumption_array[iteration_index] = iteration_time_consumption
                average_time_consumption = np.mean(iteration_time_consumption_array[:iteration_index + 1])
                eta_datetime = total_begin_datetime + datetime.timedelta(seconds=average_time_consumption * num_iteration)
                if (iteration_index + 1) % log_interval == 0:
                    if iteration_index < num_iteration - 1:
                        eta_datetimestr = eta_datetime.strftime("%Y-%m-%d %H:%M:%S")
                        template = "  {:6.2f}% {:" + str(num_digit) + "d} iterations finished in {:>10s}. Average: {:>10s}/iter. ETA: {:s}."
                        print(template.format(
                            (iteration_index + 1) * 100 / num_iteration,
                            iteration_index + 1,
                            format_time(iteration_time_consumption),
                            format_time(average_time_consumption),
                            eta_datetimestr))
                    else:
                        completion_datetimestr = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        template = "  {:6.2f}% {:" + str(num_digit) + "d} iterations finished in {:>10s}. Average: {:>10s}/iter. FIN: {:s}."
                        print(template.format(
                            (iteration_index + 1) * 100 / num_iteration,
                            iteration_index + 1,
                            format_time(iteration_time_consumption),
                            format_time(average_time_consumption),
                            completion_datetimestr))
            
            return {\
                    'c': c_2darray,\
                    'd': d_2darray,\
                    'click': click_2darray }
