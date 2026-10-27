import numpy as np
import pandas as pd

class QLearningAgent:
    def __init__(self, env, discount_factor, q_table, alpha=0.1, max_steps_per_episode=20):
        self.env = env
        self.n_actions = env.getActionsCount()
        self.n_states = env.getStateShape()[0]
        self.alpha = alpha
        self.max_steps_per_episode = max_steps_per_episode
        self.discount_factor = discount_factor
        self.q_table = q_table

    def _epsilon_greedy(self, epsilon, q, state):
        if np.random.rand() > epsilon:
            # Exploitation: choose the action with the highest value.
            action = np.argmax(q.get_vals(state))
        else:
            # Exploration: choose a random action.
            action = np.random.randint(0, self.n_actions)
        return action

    def train_episode(self, epslion):
        G = 0
        s_counter = 0
        s = self.env.reset()

        agent_policy = lambda state: self._epsilon_greedy(epslion, self.q_table, state)
        opponent_policy = lambda state: self._epsilon_greedy(epslion, self.q_table, state)

        for _ in range(self.max_steps_per_episode):
            a = agent_policy(s)

            s_, d, r= self.env.two_steps(agent_policy, opponent_policy) 
            s_counter += 1
            
            G += r
            
            q_s = self.q_table.get_vals(s_)
            qsa = self.q_table.get_val(s, a)

            new_val_td =  self.alpha * (r + self.discount_factor * np.max(q_s) - qsa)
            self.q_table.set_val(s, a, qsa + new_val_td)
            s = s_
            
            if d:
                break
                
        return G, s_counter
            
    def act(self, state):
        return np.argmax(self.q_table.get_vals(state))