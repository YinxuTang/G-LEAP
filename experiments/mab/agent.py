import math
from random import randrange

import numpy as np

from util import true_with_probability, arg_top_k


class Agent(object):
    def __init__(self, env, num_selection, b, name="Agent"):
        self._env = env
        self.num_selection = num_selection
        self.name = name
        self.b = b
    

    def pull_arms(self):
        return
    

    def reset(self):
        self.q = 0.0


class SimpleUcbAgent(Agent):
    def __init__(self, env, num_selection, b, lamb=1.0, **kwds):
        super().__init__(env, num_selection, b, **kwds)

        self.lamb = lamb

        self.reset()
    

    def pull_arms(self):
        # Returns selected_arm_indices, observation
        
        # Increment the round t
        self._round += 1

        # Calculate the latest w
        for i in range(self._env.num_arm):
            if self._count_array[i]:
                self._w_array[i] = self._theta_array[i] + self.lamb * math.sqrt(math.log(self._round) / self._count_array[i])
            else:
                self._w_array[i] = 1.0

        # Select arms with top n w
        selected_arm_indices = arg_top_k(self._w_array, self.num_selection)

        # Observe from the environment
        observation = self._env.pull_arms(selected_arm_indices)

        # Update estimations of parameters and calculate the compound reward
        reward_list = observation[0]
        compound_reward = sum(reward_list)
        for index, arm_index in enumerate(selected_arm_indices):
            self._count_array[arm_index] += 1
            
            reward = reward_list[index]
            self._theta_array[arm_index] += (reward - self._theta_array[arm_index]) / self._count_array[arm_index]

        return selected_arm_indices, observation, compound_reward
    

    def reset():
        self._round = 0
        self._count_array = np.zeros(self._env.num_arm, dtype=int)
        self._theta_array = np.zeros(self._env.num_arm, dtype=float)
        self._w_array = np.zeros(self._env.num_arm, dtype=float)


class UniformlyRandomAgent(Agent):
    def __init__(self, env, num_selection, b, beta=1e-3, name="UniformlyRandomAgent", **kwds):
        super().__init__(env, num_selection, b, name=name, **kwds)

        self.beta = beta
    

    def pull_arms(self):
        # Select n from N unifromly randomly
        selected_arm_indices = np.random.choice(self._env.num_arm, self.num_selection, replace=False).tolist()

        # Observe from the environment
        observation = self._env.pull_arms(selected_arm_indices)

        # Parse the observation
        c_list = observation[0]
        d_list = observation[1]
        energy_consumption = observation[4]
        
        # Calculate the compound reward
        compound_reward = sum(c_list) - self.beta * ((max(d_list) - self._env.latency_min) / (self._env.latency_max - self._env.latency_min))
        
        # Update q
        self.q = max(self.q + energy_consumption - self.b, 0.0)
        
        payload = { "compound_reward": compound_reward, "q": self.q }

        return selected_arm_indices, observation, payload


class DataDrivenAgent(Agent):
    def __init__(self, env, num_selection, b, history_information=None, beta=1e-3, h=0, **kwds):
        super().__init__(env, num_selection, b, **kwds)

        self.history_information = history_information
        self.beta = beta
        self.h = h

        self.reset()
    

    def pull_arms(self):
        # Increment the round t
        self._round += 1

        # Calculate the latest r array and w array
        # In each iteration, i is the index of the current arm
        for i in range(self._env.num_arm):
            if self._count_array[i] > 0:
                self._r_array[i] = self._estimate_reward(i)
            else:
                # In our paper, both UCB and $\epsilon$-greedy initialize the estimated reward mean of each arm as the maximum value (1.0)
                self._r_array[i] = 1.0
            
            # Calculate w
            self._w_array[i] = self._calculate_w(i)

        # Select arms
        selected_arm_indices = self._select_arms()

        # Observe from the environment
        observation = self._env.pull_arms(selected_arm_indices)

        # Prepare payload
        payload = self._prepare_payload(selected_arm_indices, observation)

        return selected_arm_indices, observation, payload

    
    def _estimate_reward(self, i):
        raise NotImplementedError
    

    def _calculate_w(self, i):
        raise NotImplementedError
    

    def _select_arms(self):
        """Selects arms according to the latest w_array.
        """
        raise NotImplementedError

    
    def _prepare_payload(self, selected_arm_indices, observation):
        raise NotImplementedError
    

    def reset(self):
        super().reset()

        self._round = 0
        self._count_array = np.zeros(self._env.num_arm, dtype=int)
        self._c_array = np.zeros(self._env.num_arm, dtype=float)
        self._d_array = np.zeros(self._env.num_arm, dtype=float)

        # The estimate of the mean of X^1, X^2
        self._theta_1_hat_array = np.zeros(self._env.num_arm, dtype=float)
        self._theta_2_hat_array = np.zeros(self._env.num_arm, dtype=float)

        self._r_array = np.zeros(self._env.num_arm, dtype=float)
        self._w_array = np.zeros(self._env.num_arm, dtype=float)

        # Initialization by utilizing the offline history information (i.e., data-driven)
        if self.h > 0:
            c_2darray = self.history_information['c']
            d_2darray = self.history_information['d']
            click_2darray = self.history_information['click']

            sampled_history_indices = [ randrange(c_2darray.shape[1]) for _ in range(self.h) ]
            # print(sampled_history_indices)
            for arm_index in range(self._env.num_arm):
                self._count_array[arm_index] = np.sum(click_2darray[arm_index][sampled_history_indices])
                
                count = self._count_array[arm_index]
                # print('arm_index: {:d}, count: {:d}'.format(arm_index, count))
                # # Debugging...
                # print(count)
                if count > 0:
                    self._c_array[arm_index] = np.sum(c_2darray[arm_index][sampled_history_indices]) / count
                    self._d_array[arm_index] = np.sum(d_2darray[arm_index][sampled_history_indices]) / count
            
            # # Debugging
            # print(self._a_array - self.alpha * self._l_array)
            # print(sampled_history_indices)
            # print([self._count_array[arm_index] for arm_index in range(self._env.num_arm)])


class UcbAgent(DataDrivenAgent):
    """H-UCB agents. TODO: This is not equivalent to G-LEAP without control. This is a de-graded version of G-LEAP without control.
    """


    def __init__(self, env, num_selection, b, gamma=0.1, **kwds):
        self.gamma = gamma

        super().__init__(env, num_selection, b, **kwds)
    

    def _estimate_reward(self, i):
        return min(self._c_array[i] - self.beta * ((self._d_array[i] - self._env.latency_min) / (self._env.latency_max - self._env.latency_min)) + self.gamma * math.sqrt(math.log(self._round) / self._count_array[i]), 1.0)
    

    def _calculate_w(self, i):
        return self._r_array[i]
    

    def _select_arms(self):
        # return arg_top_k(self._w_array, self.num_selection)

        selected_arm_indices = arg_top_k(self._w_array, self.num_selection)

        # # TODO: Debugging...
        # print("H-UCB: selected_arm_indices:", selected_arm_indices)

        return selected_arm_indices
    

    def _prepare_payload(self, selected_arm_indices, observation):
        # Update estimations of parameters and calculate the compound reward
        c_list = observation[0]
        d_list = observation[1]
        energy_consumption = observation[4]
        for index, arm_index in enumerate(selected_arm_indices):
            self._count_array[arm_index] += 1

            c = c_list[index]
            d = d_list[index]

            self._c_array[arm_index] += (c - self._c_array[arm_index]) / self._count_array[arm_index]
            self._d_array[arm_index] += (d - self._d_array[arm_index]) / self._count_array[arm_index]
        compound_reward = sum(c_list) - self.beta * ((max(d_list) - self._env.latency_min) / (self._env.latency_max - self._env.latency_min))
        
        # Update q
        self.q = max(self.q + energy_consumption - self.b, 0.0)

        # Prepare payload
        payload = { "compound_reward": compound_reward, "q": self.q }

        return payload


class HEpsilonGreedyAgent(DataDrivenAgent):
    """H-$\epsilon$-Greedy agents.
    """


    def __init__(self, env, num_selection, b, epsilon=0.01, **kwds):
        self.epsilon = epsilon

        super().__init__(env, num_selection, b, **kwds)
    

    def _estimate_reward(self, i):
        return self._c_array[i] - self.beta * ((self._d_array[i] - self._env.latency_min) / (self._env.latency_max - self._env.latency_min))
    

    def _calculate_w(self, i):
        return self._r_array[i]
    

    def _select_arms(self):
        # Select arms uniformly randomly with probability $\epsilon$
        if true_with_probability(self.epsilon):
            return np.random.choice(self._env.num_arm, self.num_selection, replace=False).tolist()
        # Select arms with top n w with probability $1 - \epsilon$
        else:
            return arg_top_k(self._w_array, self.num_selection)
    

    def _prepare_payload(self, selected_arm_indices, observation):
        # Update estimations of parameters and calculate the compound reward
        c_list = observation[0]
        d_list = observation[1]
        energy_consumption = observation[4]
        for index, arm_index in enumerate(selected_arm_indices):
            self._count_array[arm_index] += 1

            c = c_list[index]
            d = d_list[index]

            self._c_array[arm_index] += (c - self._c_array[arm_index]) / self._count_array[arm_index]
            self._d_array[arm_index] += (d - self._d_array[arm_index]) / self._count_array[arm_index]
        compound_reward = sum(c_list) - self.beta * ((max(d_list) - self._env.latency_min) / (self._env.latency_max - self._env.latency_min))
        
        # Update q
        self.q = max(self.q + energy_consumption - self.b, 0.0)

        # Prepare payload
        payload = { "compound_reward": compound_reward, "q": self.q }

        return payload


class BaseGleapAgent(DataDrivenAgent):
    def __init__(self, env, num_selection, b, v=1.0, **kwds):
        super().__init__(env, num_selection, b, **kwds)

        self.v = v
        self.b = b
    

    def _estimate_reward(self, i):
        # This method should not be used for G-LEAP and its variants.
        return None
    

    def pull_arms(self):
        # Increment the round t
        self._round += 1

        # Calculate the latest HUCB estimates for theta_1 and theta_2, and the weights (self._w_array)
        for arm_index in range(self._env.num_arm):
            if self._count_array[arm_index] > 0:
                self._theta_1_hat_array[arm_index] = min(self._c_array[arm_index] + self.gamma * math.sqrt(math.log(self._round) / self._count_array[arm_index]), 1.0)
                self._theta_2_hat_array[arm_index] = min(((self._d_array[arm_index] - self._env.latency_min) / (self._env.latency_max - self._env.latency_min)) + self.gamma * math.sqrt(math.log(self._round) / self._count_array[arm_index]), 1.0)
            else:
                self._theta_1_hat_array[arm_index] = 1.0
                self._theta_2_hat_array[arm_index] = 1.0
            
            # Calculate the weight
            self._w_array[arm_index] = self._calculate_w(arm_index)

        # Select arms
        selected_arm_indices = self._select_arms()

        # Observe from the environment
        observation = self._env.pull_arms(selected_arm_indices)

        # # TODO: Debugging.
        # print("{:s}: selected_arm_indices:".format(self.name), selected_arm_indices, " energy consumption:", observation[4])

        # Prepare payload
        payload = self._prepare_payload(selected_arm_indices, observation)

        return selected_arm_indices, observation, payload
    

    def _calculate_w(self, i):
        predictor_index = self._env._arm_predictor_index_mapping_list[i]
        if self.v < math.inf:
            return self.v * self._theta_1_hat_array[i] - self.q * self._env.e_array[predictor_index]
        else:
            return self._theta_1_hat_array[i]
    

    def _select_arms(self):
        return self._select_best_arms_according_to_estimation()
    

    def _select_best_arms_according_to_estimation(self):
        optimal_solution = None
        minimum_value = math.inf
        for j in range(self._env.num_arm):
            solution = self._feasible_solution(j)

            # If there is a feasible solution
            if solution:
                weight_sum = sum([ self._w_array[i] for i in solution ])
                if self.v < math.inf:
                    current_value = self.v * self.beta * self._theta_2_hat_array[j] - weight_sum
                else:
                    current_value = self.beta * self._theta_2_hat_array[j] - sum([ self._theta_1_hat_array[i] for i in solution ])

                if current_value < minimum_value:
                    minimum_value = current_value
                    optimal_solution = solution
        
        return optimal_solution
    

    def _feasible_solution(self, j):
        # Construct the weight list without the j-th element
        arm_index_list_without_j_th_element = []
        for arm_index in range(self._env.num_arm):
            if arm_index != j:
                arm_index_list_without_j_th_element.append(arm_index)
        # print("j: {:d}".format(j))
        # print("arm_index_list_without_j_th_element:", arm_index_list_without_j_th_element)
        # print("arm_index_list_without_j_th_element:", arm_index_list_without_j_th_element) # 
        w_list_without_j_th_element = []
        for index, w in enumerate(self._w_array):
            if index != j:
                w_list_without_j_th_element.append(w)
        # print("w_list_without_j_th_element:", w_list_without_j_th_element)
        
        # Sort TODO: Is the sorting operation really necessary?
        # sorted_model_index_list = [ int(x) for x in np.argsort(w_list_without_j_th_element) ]
        sorted_arm_index_list = [ pair[0] for pair in sorted(zip(arm_index_list_without_j_th_element, w_list_without_j_th_element), key=lambda x: x[1], reverse=True) ]
        # print("sorted_arm_index_list:", sorted_arm_index_list)
        
        solution = [ j ]
        # for k in range(self._env.num_arm - 1): # k = 0, 1, ..., num_arm - 2
        for arm_index in sorted_arm_index_list:
            if self._theta_2_hat_array[arm_index] <= self._theta_2_hat_array[j]:
                solution.append(arm_index)
            
            if len(solution) == self.num_selection:
                # print("solution:", solution)
                return solution

        # print("None")
        return None
    

    def _prepare_payload(self, selected_arm_indices, observation):
        # Update estimations of parameters and calculate the compound reward
        c_list = observation[0]
        d_list = observation[1]
        energy_consumption = observation[4]
        for index, arm_index in enumerate(selected_arm_indices):
            self._count_array[arm_index] += 1

            c = c_list[index]
            d = d_list[index]

            self._c_array[arm_index] += (c - self._c_array[arm_index]) / self._count_array[arm_index]
            self._d_array[arm_index] += (d - self._d_array[arm_index]) / self._count_array[arm_index]
        compound_reward = sum(c_list) - self.beta * ((max(d_list) - self._env.latency_min) / (self._env.latency_max - self._env.latency_min))
        
        # Update q
        self.q = max(self.q + energy_consumption - self.b, 0.0)

        # Prepare payload
        payload = { "compound_reward": compound_reward, "q": self.q }

        return payload


class GleapEpsilonGreedyAgent(BaseGleapAgent):
    """G-LEAP-$\epsilon$-greedy with history and Lyapunov control mechanism. It can be viewed as a variant of G-LEAP, with the UCB part replaced by $\epsilon$-greedy.
    """


    def __init__(self, env, num_selection, b, epsilon=0.1, **kwds):
        self.epsilon = epsilon

        super().__init__(env, num_selection, b, **kwds)
    

    def pull_arms(self):
        # Increment the round t
        self._round += 1

        # Calculate the latest HUCB estimates for theta_1 and theta_2, and the weights (self._w_array)
        for arm_index in range(self._env.num_arm):
            if self._count_array[arm_index] > 0:
                self._theta_1_hat_array[arm_index] = self._c_array[arm_index]
                self._theta_2_hat_array[arm_index] = (self._d_array[arm_index] - self._env.latency_min) / (self._env.latency_max - self._env.latency_min)
            else:
                self._theta_1_hat_array[arm_index] = 1.0
                self._theta_2_hat_array[arm_index] = 1.0
            
            # Calculate the weight
            self._w_array[arm_index] = self._calculate_w(arm_index)

        # Select arms
        selected_arm_indices = self._select_arms()

        # Observe from the environment
        observation = self._env.pull_arms(selected_arm_indices)

        # # TODO: Debugging...
        # print("G-LEAP: selected_arm_indices:", selected_arm_indices, " energy consumption:", observation[4])

        # Prepare payload
        payload = self._prepare_payload(selected_arm_indices, observation)

        return selected_arm_indices, observation, payload
    

    def _select_arms(self):
        # Select arms uniformly randomly with probability epsilon
        if true_with_probability(self.epsilon):
            return np.random.choice(self._env.num_arm, self.num_selection, replace=False).tolist()
        # Select arms with top n w with probability 1 - epsilon
        else:
            return self._select_best_arms_according_to_estimation()


class GleapAgent(BaseGleapAgent):
    def __init__(self, env, num_selection, b, gamma=0.1, **kwds):
        self.gamma = gamma

        super().__init__(env, num_selection, b, **kwds)
