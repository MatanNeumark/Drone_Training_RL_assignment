from __future__ import annotations
from QuadModel import QuadModel
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


# This code implements REINFORCE algorithm to train a drone to take off, hover, and ascend up to a target zone.
# Written by Matan Neumark as the assignment for the course AE4350 Bio-inspired Intelligence at the TU delft August 2024
# References: OpenAI Gym and PyTorch documentation

class Neural_Network_Policy(nn.Module):
    def __init__(self, S_space_dims: int, action_space_dims: int):

        super().__init__()
        hidden_layer1 = 32
        hidden_layer2 = 32

        # Shared Network
        self.shared_net = nn.Sequential(
            nn.Linear(S_space_dims, hidden_layer1),
            nn.Tanh(),
            nn.Linear(hidden_layer1, hidden_layer2),
            nn.Tanh(),)

        # Policy Mean specific Linear Layer
        self.policy_mean_net = nn.Sequential(
            nn.Linear(hidden_layer2, action_space_dims))
        # Policy Std Dev specific Linear Layer
        self.policy_stddev_net = nn.Sequential(
            nn.Linear(hidden_layer2, action_space_dims),
            nn.Softplus())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shared_features = self.shared_net(x.float())
        mu = self.policy_mean_net(shared_features)
        sigma = self.policy_stddev_net(shared_features)
        # mu is the action means, sigma is the standard deviation
        return mu, sigma


class REINFORCE:

    def __init__(self, S_space_dims: int, action_space_dims: int, learning_rate: float, gamma: float):

        # Hyperparameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        print(f'alpha={learning_rate}, gamma={gamma}')
        self.eps = 1e-6
        self.probs = []
        self.rewards = []
        self.net = Neural_Network_Policy(S_space_dims, action_space_dims)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)    # optimizer

    def sample_action(self, state: np.ndarray) -> float:
        state = torch.tensor(np.array([state]))
        mu, sigma = self.net(state)     # passes the state to the net and receives back mean and standard deviation
        distrib = Normal(mu[0] + self.eps, sigma[0] + self.eps)     # creates a normal distribution from mu and sigma
        action = distrib.sample()   # samples and action from the normal distribution (dimension 2)
        prob = distrib.log_prob(action)     # gives the log of the probability of the chosen action
        action = action.numpy()     # converts torch tensor to numpy array
        self.probs.append(prob)     # stores the log of probability to later use in the loss function
        return action

    def update(self):
        G = 0
        G_hist = []
        for R in self.rewards[::-1]:
            G = R + self.gamma * G      # Discounted return per time step in the episode
            G_hist.insert(0, G)  # vector of discounted return

        deltas = torch.tensor(G_hist)
        loss = 0
        for log_prob, delta in zip(self.probs, deltas):     # loss function. negative because optimizer find a minimum
            loss += log_prob.mean() * delta * (-1)

        # Update the policy network
        self.optimizer.zero_grad()  # resets the gradient
        loss.backward()             # calculate the gradient
        self.optimizer.step()       # take a step towards the minimum
        
        self.probs = []
        self.rewards = []



class QuadTraining:
    def __init__(self, lr=1e-4, gamma=0.99, episodes=1000):
        self.lr = lr                # learning rate alpha
        self.gamma = gamma          # discount factor gamma
        self.episodes = episodes    # number of episodes to run
    def training(self):
        mass = 1                    # drone's mass
        g = 9.81                    # gravity. dah
        drone_weight = mass * g     # weight
        motor_distance = 1          # diagonal motor distant
        max_steps = 300             # defines the maximum length on an episode
        env = QuadModel(mass, motor_distance, max_steps)   # creating the environment
        queue_length = 100
        score_queue = deque(maxlen=queue_length)
        step_queue = deque(maxlen=queue_length)
        S_space_dims = 6            # dimension of the space state: [x, x_dot, z, z_dot, alpha, alpha_dot]
        action_space_dims = 2       # dimension of action space: [thrust left, thrust right]
        agent = REINFORCE(S_space_dims, action_space_dims, learning_rate=self.lr, gamma=self.gamma)
        score_running_mean = []
        steps_running_mean = []
        steps_per_episode = []
        score_per_episode = []
        for episode in range(self.episodes+1):
            S = env.reset(mid_air=False)    # resetting the environment to the initial (stochastic) state
            score = 0
            n_steps = 0
            done = False
            while not done:
                action = agent.sample_action(S)     # sampling an action from the neural network with the state as input
                thrust = action * drone_weight      # used to denormalize the action value
                S, reward, terminated, truncated, cause = env.dynamics(thrust)  # getting a state and reward from env
                agent.rewards.append(reward)    # saving the reword to calculate the return
                score += reward
                n_steps += 1
                done = terminated or truncated
            if episode > self.episodes-1000:
                env.render(render=True)
            step_queue.append(n_steps)                              # queue of length 100
            steps_per_episode.append(step_queue[-1])                # vector with number of steps per episode
            score_queue.append(score)                               # queue of length 100
            score_per_episode.append(score_queue[-1])               # vector with score per episode
            steps_running_mean.append(np.mean(step_queue))          # running mean of number of steps per episode
            score_running_mean.append(np.mean(score_queue))         # running mean of score per episode
            agent.update()

            if episode % 1000 == 0:
                torch.save(agent.net.state_dict(), 'policy.pth')    # saving progress
                avg_reward = int(np.mean(score_queue))  # average reword per batch of episodes
                avg_step = int(np.mean(step_queue))     # average steps per batch of episodes
                print("Episode:", episode, "Average number of steps:", avg_step, "Average rewards:", avg_reward)

        env.close()     # closing the environment once all episodes are done

        return steps_per_episode, score_per_episode, steps_running_mean, score_running_mean


if __name__ == '__main__':
    lr = 1e-4
    gamma = 0.999
    episodes = 5000
    steps_per_episode, score_per_episode, steps_running_mean, score_running_mean = QuadTraining(lr=lr, gamma=gamma, episodes=episodes).training()

#########################################################################
#%% plots for trouble shooting and quick evaluation

#   #mean score per episode
    # plt.plot(score_per_episode)
    # plt.plot(score_running_mean)
    # plt.title('Rewards over Episodes')
    # plt.xlabel('Episodes')
    # plt.ylabel('Rewards')
    # plt.xlim(0, episodes)
    # #plt.ylim(0, 3000)
    # plt.grid(True, alpha=0.25, color='black', linestyle='-', linewidth=1)
    # plt.show()

#   #mean number of steps per episode
    # plt.plot(steps_per_episode)
    # plt.plot(steps_running_mean)
    # plt.hlines(y=300, xmin=0, xmax=episodes, color='black', linewidth=3)
    # plt.title('steps per Episode')
    # plt.xlim(0, episodes)
    # plt.ylim(0, 320)
    # plt.xlabel('Episodes', fontsize=14)
    # plt.ylabel('Steps', fontsize=14)
    # plt.grid(True, alpha=0.25, color='black', linestyle='-', linewidth=1)
    # plt.show()
