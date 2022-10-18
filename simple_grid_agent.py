from unittest import result
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import random

class GridworldAgent:
    def __init__(self, env, policy, gamma = 0.9, 
                 start_epsilon = 0.9, end_epsilon = 0.1, epsilon_decay = 0.9):
        self.env = env
        self.n_action = len(self.env.action_space)
        self.policy = policy
        self.gamma = gamma
        self.v = dict.fromkeys(self.env.state_space,0)  # state value initiated as 0
        self.n_v = dict.fromkeys(self.env.state_space,0)  # number of actions performed: use it for MC state value prediction
        self.q = defaultdict(lambda: np.zeros(self.n_action))  # action value
        self.n_q = defaultdict(lambda: np.zeros(self.n_action))  # number of actions performed: use it for MC state-action value prediction
        
        # epsilon greedy parameters
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.epsilon_decay = epsilon_decay
    
    def get_epsilon(self,n_episode):
        epsilon = max(self.start_epsilon * (self.epsilon_decay**n_episode),self.end_epsilon)
        return(epsilon)
    
    def get_v(self,start_state,epsilon = 0.):
        episode = self.run_episode(start_state,epsilon)
        """
        Write the code to calculate and return the state value 
        given a deterministic policy. Should return a scalar. Study the components of episode to
        understand how to calculate return.
        YOUR CODE HERE
        """
        # returns sum of all discounted rewards for transitions of the episode.
        
        return np.sum(
            [result[2] * self.gamma**i for i, result in enumerate(episode)])
        
        # return (np.sum([res[2] * pow(self.gamma, i) for i, res in enumerate(episode)]))

        # raise NotImplementedError
    
    def get_q(self, start_state, first_action, epsilon=0.):
        episode = self.run_episode(start_state,epsilon,first_action)
        """
        Write the code to calculate and return the action value of a state 
        given a deterministic policy. Should return a scalar. Study the components of episode to
        understand how to calculate return.
        YOUR CODE HERE
        """
        # returns sum of all discounted rewards for transitions of the episode.

        # return (np.sum([res[2] * pow(self.gamma, i) for i, res in enumerate(episode)]))
        
        return np.sum(
            [result[2] * self.gamma**i for i, result in enumerate(episode)])
        # raise NotImplementedError
    
    def select_action(self,state,epsilon):
        best_action = self.policy[state]
        if random.random() > epsilon:
            action = best_action
        else:
             action = np.random.choice(np.arange(self.n_action))
        return(action)
    
    def print_policy(self):
        for i in range(self.env.sz[0]):
            print('\n----------')
            for j in range(self.env.sz[1]):
                p=self.policy[(i,j)]
                out = self.env.action_text[p]
                print(f'{out} |',end='')
    
    def print_v(self, decimal = 1):
        for i in range(self.env.sz[0]):
            print('\n---------------')
            for j in range(self.env.sz[1]):
                out=np.round(self.v[(i,j)],decimal)
                print(f'{out} |',end='')
    
    def run_episode(self, start, epsilon, first_action = None):
        result = []
        state = self.env.reset(start)
        #dictate first action to iterate q
        if first_action is not None:
            action = first_action
            next_state,reward,done,_ = self.env.step(action)
            result.append((state,action,reward,next_state,done))
            state = next_state
            if done: return(result)
        while True:
            action = self.select_action(state,epsilon)
            next_state,reward,done,_ = self.env.step(action)
            result.append((state,action,reward,next_state,done))
            state = next_state
            if done: break
        return(result)
    
    def update_policy_q(self):
        for state in self.env.state_space:
            self.policy[state] = np.argmax(self.q[state])
    
    def mc_predict_v(self,n_episode=10000,first_visit=True):
        for t in range(n_episode):
            traversed = {} #converting to hashmap for faster lookup

            e = self.get_epsilon(t)
            transitions = self.run_episode(self.env.start, e)
            states,actions,rewards,next_states,dones = zip(*transitions)

            #computing the discount to make the q-value update 
            disc = np.array([pow(self.gamma, i) for i in range(len(transitions) + 1)])

            for i in range(len(transitions)):
                if first_visit and (states[i] not in traversed):
                    """
                    Implement first-visit Monte Carlo for state values(see Sutton and Barto Section 5.1)
                    Comment each line of code with what part of the pseudocode you are implementing in that line
                    YOUR CODE HERE
                    """
                    # True if state is already traversed
                    traversed[states[i]] = True                    
                    #state counter
                    self.n_v[states[i]] += 1
                    # update the sum of rewards for the given state, V
                    self.v[states[i]] += np.sum(rewards[i:] *
                                                disc[:-(i + 1)])

                elif not first_visit:
                    """
                    Implement any-visit Monte Carlo for state values(see Sutton and Barto Section 5.1)
                    Comment each line of code with what part of the pseudocode you are implementing in that line
                    YOUR CODE HERE
                    """
                    #state counter
                    self.n_v[states[i]] += 1

                    # update the sum of rewards for the given state, V
                    self.v[states[i]] += np.sum(rewards[i:] * disc[:-(i+1)])

        for state in self.env.state_space:
            if state != self.env.goal:
                self.v[state] = self.v[state] / self.n_v[state]
            else:
                self.v[state] = 0
    
    def mc_predict_q(self,n_episode=10000,first_visit=True):
        for t in range(n_episode):
            traversed = {} # converting to hashmap for faster lookup
            e = self.get_epsilon(t)
            transitions = self.run_episode(self.env.start,e)
            states,actions,rewards,next_states,dones = zip(*transitions)

            # computing the discount to make the q-value update 
            disc = np.array(
                [self.gamma**i for i in range(len(transitions) + 1)])
            for i in range(len(transitions)):
                if first_visit and ((states[i],actions[i]) not in traversed):
                    """
                    Implement first-visit Monte Carlo for state-action values(see Sutton and Barto Section 5.2)
                    Comment each line of code with what part of the pseudocode you are implementing in that line
                    YOUR CODE HERE
                    """
                    # True if state is already traversed
                    traversed[states[i]] = True                    
                    # state counter
                    self.n_q[states[i]] += 1
                    # update the sum of rewards for the given state-action, Q
                    self.q[states[i]] += np.sum(rewards[i:] *
                                                disc[:-(i + 1)])

                elif not first_visit:
                    
                    """
                    Implement any-visit Monte Carlo for state-action values(see Sutton and Barto Section 5.2)
                    Comment each line of code with what part of the pseudocode you are implementing in that line
                    YOUR CODE HERE
                    """
                    #state counter
                    self.n_q[states[i]] += 1

                    # update the sum of rewards for the given state-action, Q
                    self.q[states[i]] += np.sum(rewards[i:] * disc[:-(i+1)])

        for state in self.env.state_space:
            for action in range(self.n_action):
                if state != self.env.goal:
                    self.q[state][action] = self.q[state][action] / self.n_q[state][action]
                else:
                    self.q[state][action] = 0
        
    def mc_control_q(self,n_episode=10000,first_visit=True):
        """
        Write the code to perform Monte Carlo Control for state-action values
        Hint: You just need to do prediction then update the policy
        YOUR CODE HERE
        """
        # using predict q and then updating the policy
        self.mc_predict_q(n_episode, first_visit)
        self.update_policy_q()

        # raise NotImplementedError
        
    def mc_control_glie(self,n_episode=10000,first_visit=True,lr=0.):
        """
        Bonus: Taking hints from the mc_predict_q and mc_control_q methods, write the code to
        perform GLIE Monte Carlo control. Comment each line of code with what part of the pseudocode you are implementing in that line
        YOUR CODE HERE
        """
        for t in range(n_episode):
            traversed = {}  # converting to hashmap for faster lookup
            e = self.get_epsilon(t)
            transitions = self.run_episode(self.env.start, e)
            states, actions, rewards, next_states, dones = zip(*transitions)

            # computing the discount to make the q-value update 
            disc = np.array([self.gamma**i for i in range(len(transitions) + 1)])

            for i in range(len(transitions)):
                # updating the condition based on the hashmap.
                if first_visit and (traversed.get((states[i], actions[i]))):

                    # if the state is already traversed
                    continue

                # True if state is already traversed
                traversed[(states[i], actions[i])] = True
                # state counter
                self.n_q[states[i]][actions[i]] += 1
                # calculating the discounted rewards i.e. G
                G = np.sum(rewards[i:] * disc[:-(i + 1)])

                # using the concept: 
                # if learning rate is zero then use the update rule, Q(S,A) <- (G-Q(S,A)) / N(S,A)
                if lr == 0:
                    self.q[states[i]][actions[i]] += (G - self.q[states[i]][
                        actions[i]]) / self.n_q[states[i]][actions[i]]

                # if learning rate is provided:
                # Q(S,A) <- (G-Q(S,A))*alpha
                else:
                    self.q[states[i]][actions[i]] += (G - self.q[states[i]][actions[i]]) * lr
        
        # raise NotImplementedError