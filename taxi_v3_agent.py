import numpy as np
from collections import defaultdict
import random

class TaxiAgent:
    def __init__(self, env, gamma = 0.8, alpha = 1e-1,
                 start_epsilon = 1, end_epsilon = 1e-2, epsilon_decay = 0.999):
        
        self.env = env
        self.n_action = self.env.action_space.n
        self.gamma = gamma
        self.alpha = alpha
        
        #action values
        self.q = defaultdict(lambda: np.zeros(self.n_action)) #action value
        
        #epsilon greedy parameters
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.epsilon_decay = epsilon_decay

    #get epsilon
    def get_epsilon(self,n_episode):
        epsilon = max(self.start_epsilon * (self.epsilon_decay ** n_episode), self.end_epsilon)
        return(epsilon)
    
    #select action based on epsilon greedy
    def select_action(self,state,epsilon):
        #implicit policy; if we have action values for that state, choose the largest one, else random
        best_action = np.argmax(self.q[state]) if state in self.q else self.env.action_space.sample()
        if random.random() > epsilon:
            action = best_action
        else:
             action = self.env.action_space.sample()
        return(action)
    
    def on_policy_td_sarsa(self, state, action, reward, next_state, n_episode):
        """
        Implement On policy TD learning or SARSA
        YOUR CODE HERE
        """
        # computing the previous rewards and previous q values
        prev_q = self.q[state][action]
        reward = self.alpha * reward

        # update q vals w.r.t. previosu q valuess and reward,
        self.q[state][action] = (1 - self.alpha) * prev_q + reward

        # first, checking if next state exists for the given state and action
        if next_state:
            # select the action for the next state using ε-greedy policy
            action = self.select_action(next_state,
                                         self.get_epsilon(n_episode))

            # update the q value from the updated policy
            self.q[state][action] += self.alpha * self.gamma * self.q[
                next_state][action]

        # raise NotImplementedError
        
    def off_policy_td_q_learning(self, state, action, reward, next_state):
        """
        Implement Off policy TD learning ie SARSA-MAX/Q learning 
        YOUR CODE HERE
        """
        
        # computing the previous rewards and previous q values
        prev_q = self.q[state][action]
        reward = self.alpha * reward

        # update q values wrt previous q values and reward
        self.q[state][action] = (1 - self.alpha) * prev_q + reward + self.alpha * self.gamma * np.max(self.q[next_state])

        # raise NotImplementedError
        