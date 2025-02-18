import copy
import pickle
import multiprocessing
import os
import time
from datetime import datetime, timedelta

import numpy as np

from .agent import GleapAgent, GleapEpsilonGreedyAgent, UcbAgent, UniformlyRandomAgent
from .util import format_time


def run_repetition(param):
    repetition_index, agent, num_round, log, log_prefix = param

    import numpy as np
    import os
    from util.terminal_utils import bcolors
    from util import format_time, format_now

    if log:
        print("{:s}repetition_index: {:2d}, pid: {:d}, {:s}, {:s}BEGIN{:s}".format(log_prefix, repetition_index, os.getpid(), format_now(), bcolors.OKGREEN, bcolors.ENDC))

    # For logging
    begin_time = time.time()

    # Reset the agent
    agent.reset()

    compound_reward_array = np.zeros(num_round, dtype=float)
    time_averaged_compound_reward_array = np.zeros(num_round, dtype=float)
    aggregated_compound_reward_array = np.zeros(num_round, dtype=float)
    regret_array = np.zeros(num_round, dtype=float)
    cumulative_regret_array = np.zeros(num_round, dtype=float)
    time_averaged_regret_array = np.zeros(num_round, dtype=float)
    user_satisfaction_array = np.zeros(num_round, dtype=float)
    time_averaged_user_satisfaction_array = np.zeros(num_round, dtype=float)
    user_click_array = np.zeros(num_round, dtype=int)
    time_averaged_total_prediction_accuracy_array = np.zeros(num_round, dtype=float)
    total_prediction_latency_array = np.zeros(num_round, dtype=float)
    time_averaged_total_prediction_latency_array = np.zeros(num_round, dtype=float)
    energy_consumption_array = np.zeros(num_round, dtype=float)
    time_averaged_energy_consumption_array = np.zeros(num_round, dtype=float)
    q_array = np.zeros(num_round, dtype=float)
    arm_selection_count_array = np.zeros(agent._env.num_arm, dtype=int)
    
    # Simulate the rounds
    for round_index in range(num_round):
        selected_arm_indices, observation, payload = agent.pull_arms()

        compound_reward = payload["compound_reward"]
        c_list = observation[0]
        d_list = observation[1]
        regret = observation[2]
        click_list = observation[3]
        energy_consumption = observation[4]
        
        # For compound reward
        compound_reward_array[round_index] = compound_reward
        time_averaged_compound_reward_array[round_index] = np.mean(compound_reward_array[:round_index + 1])
        if round_index:
            aggregated_compound_reward_array[round_index] = aggregated_compound_reward_array[round_index - 1] + compound_reward
        else:
            aggregated_compound_reward_array[round_index] = compound_reward
        
        # For regret
        regret_array[round_index] = regret
        if round_index:
            cumulative_regret_array[round_index] = cumulative_regret_array[round_index - 1] + regret
        else:
            cumulative_regret_array[round_index] = regret
        time_averaged_regret_array[round_index] = np.mean(regret_array[:round_index + 1])

        # For user satisfaction
        user_satisfaction_array[round_index] = np.sum(c_list)
        time_averaged_user_satisfaction_array[round_index] = np.mean(user_satisfaction_array[:round_index + 1])
        
        # For accuracy
        user_click_array[round_index] = 1 if np.sum(click_list) > 0 else 0
        time_averaged_total_prediction_accuracy_array[round_index] = np.mean(user_click_array[:round_index + 1])

        # For queue length
        q_array[round_index] = payload["q"]

        # For total prediction latency
        total_prediction_latency_array[round_index] = max(d_list)
        time_averaged_total_prediction_latency_array[round_index] = np.mean(total_prediction_latency_array[:round_index + 1])

        # For energy consumption
        energy_consumption_array[round_index] = energy_consumption
        time_averaged_energy_consumption_array[round_index] = np.mean(energy_consumption_array[:round_index + 1])

        # For arm selection count
        for selected_arm_index in selected_arm_indices:
            arm_selection_count_array[selected_arm_index] += 1
    
    # For logging
    end_time = time.time()
    repetition_time_consumption = end_time - begin_time
    # repetition_time_consumption_array[repetition_index] = repetition_time_consumption
    # if log:
    #     average_time_consumption = np.mean(repetition_time_consumption_array[:repetition_index + 1])
    #     eta_datetime = total_begin_datetime + timedelta(seconds=average_time_consumption * self.num_repetition)
    #     if repetition_index < self.num_repetition - 1:
    #         eta_datetimestr = eta_datetime.strftime("%Y-%m-%d %H:%M:%S")
    #         print("  Repetition #{:3d} finished in {:>10s}. Average: {:>10s}. ETA: {:s}.".format(repetition_index, format_time(repetition_time_consumption), format_time(average_time_consumption), eta_datetimestr))
    #     else:
    #         eta_datetimestr = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #         print("  Repetition #{:3d} finished in {:>10s}. Average: {:>10s}. FIN: {:s}.".format(repetition_index, format_time(repetition_time_consumption), format_time(average_time_consumption), eta_datetimestr))
    
    if log:
        print("{:s}repetition_index: {:2d}, pid: {:d}, {:s}, {:s}END  {:s} {:s}".format(log_prefix, repetition_index, os.getpid(), format_now(), bcolors.OKBLUE, bcolors.ENDC, format_time(repetition_time_consumption)))

    return {
        "compound_reward_array": compound_reward_array,
        "time_averaged_compound_reward_array": time_averaged_compound_reward_array,
        "aggregated_compound_reward_array": aggregated_compound_reward_array,
        "regret_array": regret_array,
        "cumulative_regret_array": cumulative_regret_array,
        "time_averaged_regret_array": time_averaged_regret_array,
        "user_satisfaction_array": user_satisfaction_array,
        "time_averaged_user_satisfaction_array": time_averaged_user_satisfaction_array,
        "user_click_array": user_click_array,
        "time_averaged_total_prediction_accuracy_array": time_averaged_total_prediction_accuracy_array,
        "total_prediction_latency_array": total_prediction_latency_array,
        "time_averaged_total_prediction_latency_array": time_averaged_total_prediction_latency_array,
        "energy_consumption_array": energy_consumption_array,
        "time_averaged_energy_consumption_array": time_averaged_energy_consumption_array,
        "q_array": q_array,
        "arm_selection_count_array": arm_selection_count_array
    }


class Simulation:
    def __init__(self, env, agent, num_round=1000, num_repetition=100):
        self.env = env
        self.agent = agent
        self.num_round = num_round
        self.num_repetition = num_repetition

        # For compound reward
        self.compound_reward_2darray = np.zeros((self.num_repetition, self.num_round), dtype=float)
        self.time_averaged_compound_reward_2darray = np.zeros((self.num_repetition, self.num_round), dtype=float)
        self.aggregated_compound_reward_2darray = np.zeros((self.num_repetition, self.num_round), dtype=float)
        
        # For regret
        self.regret_2darray = np.zeros((self.num_repetition, self.num_round), dtype=float)
        self.cumulative_regret_2darray = np.zeros((self.num_repetition, self.num_round), dtype=float)
        self.time_averaged_regret_2darray = np.zeros((self.num_repetition, self.num_round), dtype=float)
        
        # For user satisfaction
        self.user_satisfaction_2darray = np.zeros((self.num_repetition, self.num_round), dtype=float)
        self.time_averaged_user_satisfaction_2darray = np.zeros((self.num_repetition, self.num_round), dtype=float)

        # For prediction accuracy
        self.user_click_2darray = np.zeros((self.num_repetition, self.num_round), dtype=int)
        self.time_averaged_total_prediction_accuracy_2darray = np.zeros((self.num_repetition, self.num_round), dtype=float)
        
        # For prediction latency
        self.total_prediction_latency_2darray = np.zeros((self.num_repetition, self.num_round), dtype=float)
        self.time_averaged_total_prediction_latency_2darray = np.zeros((self.num_repetition, self.num_round), dtype=float)
        
        # For energy consumption
        self.energy_consumption_2darray = np.zeros((self.num_repetition, self.num_round), dtype=float)
        self.time_averaged_energy_consumption_2darray = np.zeros((self.num_repetition, self.num_round), dtype=float)
        
        # For queue backlog size
        self.q_2darray = np.zeros((self.num_repetition, self.num_round), dtype=float)

        # For arm selection count
        self.arm_selection_count_2darray = np.zeros((self.num_repetition, self.env.num_arm), dtype=int)


    def run(self, log=False, num_process=1):
        # For logging
        total_begin_datetime = datetime.now()
        if log:
            print("  Simulation.run() started at: {:s}".format(total_begin_datetime.strftime("%Y-%m-%d %H:%M:%S")))
        repetition_time_consumption_array = np.zeros(self.num_repetition, dtype=float)

        # If use multiple processing
        if num_process > 1:
            with multiprocessing.Pool(num_process) as pool:
                agent_list = [ copy.deepcopy(self.agent) for _ in range(self.num_repetition) ]
                num_round_list = [ self.num_round for _ in range(self.num_repetition) ]
                log_list = [ log for _ in range(self.num_repetition) ]
                log_prefix_list = [ "  " for _ in range(self.num_repetition) ]
                run_repetition_result_list = pool.map(run_repetition, zip(range(self.num_repetition), agent_list, num_round_list, log_list, log_prefix_list))
                # print("type(run_repetition_result_list):", type(run_repetition_result_list)) # <class 'list'>
                for repetition_index, result in enumerate(run_repetition_result_list):
                    self.compound_reward_2darray[repetition_index] = result["compound_reward_array"]
                    self.time_averaged_compound_reward_2darray[repetition_index] = result["time_averaged_compound_reward_array"]
                    self.aggregated_compound_reward_2darray[repetition_index] = result["aggregated_compound_reward_array"]
                    self.regret_2darray[repetition_index] = result["regret_array"]
                    self.cumulative_regret_2darray[repetition_index] = result["cumulative_regret_array"]
                    self.time_averaged_regret_2darray[repetition_index] = result["time_averaged_regret_array"]
                    self.user_satisfaction_2darray[repetition_index] = result["user_satisfaction_array"]
                    self.time_averaged_user_satisfaction_2darray[repetition_index] = result["time_averaged_user_satisfaction_array"]
                    self.user_click_2darray[repetition_index] = result["user_click_array"]
                    self.time_averaged_total_prediction_accuracy_2darray[repetition_index] = result["time_averaged_total_prediction_accuracy_array"]
                    self.total_prediction_latency_2darray[repetition_index] = result["total_prediction_latency_array"]
                    self.time_averaged_total_prediction_latency_2darray[repetition_index] = result["time_averaged_total_prediction_latency_array"]
                    self.energy_consumption_2darray[repetition_index] = result["energy_consumption_array"]
                    self.time_averaged_energy_consumption_2darray[repetition_index] = result["time_averaged_energy_consumption_array"]
                    self.q_2darray[repetition_index] = result["q_array"]
                    self.arm_selection_count_2darray[repetition_index] = result["arm_selection_count_array"]
        # Else, no multiple processing
        else:
            for repetition_index in range(self.num_repetition):
                # For logging
                begin_time = time.time()

                # Reset the agent
                self.agent.reset()
                
                # Simulate the rounds
                for round_index in range(self.num_round):
                    selected_arm_indices, observation, payload = self.agent.pull_arms()

                    # # Debugging...
                    # print(selected_arm_indices)
                    # print(self.agent._count_array)
                    # print([self.env._arm_predictor_index_mapping_list[arm_index] for arm_index in selected_arm_indices])
                    # print(self.agent._w_array)

                    compound_reward = payload["compound_reward"]
                    c_list = observation[0]
                    d_list = observation[1]
                    regret = observation[2]
                    click_list = observation[3]
                    energy_consumption = observation[4]
                    
                    # For compound reward
                    self.compound_reward_2darray[repetition_index][round_index] = compound_reward
                    self.time_averaged_compound_reward_2darray[repetition_index][round_index] = np.mean(self.compound_reward_2darray[repetition_index][:round_index + 1])
                    if round_index:
                        self.aggregated_compound_reward_2darray[repetition_index][round_index] = self.aggregated_compound_reward_2darray[repetition_index][round_index - 1] + compound_reward
                    else:
                        self.aggregated_compound_reward_2darray[repetition_index][round_index] = compound_reward
                    
                    # For regret
                    self.regret_2darray[repetition_index][round_index] = regret
                    if round_index:
                        self.cumulative_regret_2darray[repetition_index][round_index] = self.cumulative_regret_2darray[repetition_index][round_index - 1] + regret
                    else:
                        self.cumulative_regret_2darray[repetition_index][round_index] = regret
                    self.time_averaged_regret_2darray[repetition_index][round_index] = np.mean(self.regret_2darray[repetition_index][:round_index + 1])

                    # For user satisfaction
                    self.user_satisfaction_2darray[repetition_index][round_index] = np.sum(c_list)
                    self.time_averaged_user_satisfaction_2darray[repetition_index][round_index] = np.mean(self.user_satisfaction_2darray[repetition_index][:round_index + 1])
                    
                    # For accuracy
                    self.user_click_2darray[repetition_index][round_index] = 1 if np.sum(click_list) > 0 else 0
                    self.time_averaged_total_prediction_accuracy_2darray[repetition_index][round_index] = np.mean(self.user_click_2darray[repetition_index][:round_index + 1])

                    # For queue length
                    self.q_2darray[repetition_index][round_index] = payload["q"]

                    # For total prediction latency
                    self.total_prediction_latency_2darray[repetition_index][round_index] = max(d_list)
                    self.time_averaged_total_prediction_latency_2darray[repetition_index][round_index] = np.mean(self.total_prediction_latency_2darray[repetition_index][:round_index + 1])

                    # For energy consumption
                    self.energy_consumption_2darray[repetition_index][round_index] = energy_consumption
                    self.time_averaged_energy_consumption_2darray[repetition_index][round_index] = np.mean(self.energy_consumption_2darray[repetition_index][:round_index + 1])

                    # For arm selection count
                    for selected_arm_index in selected_arm_indices:
                        self.arm_selection_count_2darray[repetition_index][selected_arm_index] += 1
                
                # For logging
                end_time = time.time()
                repetition_time_consumption = end_time - begin_time
                repetition_time_consumption_array[repetition_index] = repetition_time_consumption
                if log:
                    average_time_consumption = np.mean(repetition_time_consumption_array[:repetition_index + 1])
                    eta_datetime = total_begin_datetime + timedelta(seconds=average_time_consumption * self.num_repetition)
                    if repetition_index < self.num_repetition - 1:
                        eta_datetimestr = eta_datetime.strftime("%Y-%m-%d %H:%M:%S")
                        print("  Repetition #{:3d} finished in {:>10s}. Average: {:>10s}. ETA: {:s}.".format(repetition_index, format_time(repetition_time_consumption), format_time(average_time_consumption), eta_datetimestr))
                    else:
                        eta_datetimestr = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        print("  Repetition #{:3d} finished in {:>10s}. Average: {:>10s}. FIN: {:s}.".format(repetition_index, format_time(repetition_time_consumption), format_time(average_time_consumption), eta_datetimestr))
    

    def prepare_dump(self):
        self.env._predictor_list = None
        self.env._user_input_corpus = None
        self.agent.history_information = None
