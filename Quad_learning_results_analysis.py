from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from REINFORCE_learning_Quad_V2 import QuadTraining
import pandas as pd
import math

# this script is used to run multiple batches of episodes and generate plots to decrease uncertainty

steps_per_episode = []
score_per_episode = []
steps_running_mean = []
score_running_mean =[]
mean_steps_over_runs = []
mean_score_over_runs = []
window = 100
#learning_rate = [1e-4, 5e-4]
learning_rate = [5e-5, 1e-4, 5e-4, 1e-3]
gamma = [0.999, 0.99, 0.95, 0.9]
n_runs = 4

episodes = 6000
# for i in range(len(learning_rate)):
#     for n in range(n_runs):
#         print(f'Run number: {n}')
#         results = QuadTraining(lr=learning_rate[i], gamma=gamma[1], episodes=episodes).training()
#
#         steps_per_episode.append(results[0])
#         score_per_episode.append(results[1])
#         steps_running_mean.append(results[2])
#         score_running_mean.append(results[3])
#         # print(steps_per_episode)
#         # print(score_per_episode)
#     mean_steps_over_runs.append(np.mean(steps_running_mean, axis=0))
#     mean_score_over_runs.append(np.mean(score_running_mean, axis=0))

#np.savetxt('mean_steps_over_runs_alpha.csv', mean_steps_over_runs, delimiter=',')
#np.savetxt('mean_score_over_runs_alpha.csv', mean_score_over_runs, delimiter=',')


#%%
mean_steps_over_runs_alpha = np.loadtxt('mean_steps_over_runs_alpha.csv', delimiter=',')
mean_score_over_runs_alpha = np.loadtxt('mean_score_over_runs_alpha.csv', delimiter=',')
mean_steps_over_runs_gamma = np.loadtxt('mean_steps_over_runs_gamma.csv', delimiter=',')
mean_score_over_runs_gamma = np.loadtxt('mean_score_over_runs_gamma.csv', delimiter=',')

# plt.plot(mean_score_over_runs, label='Mean return', linewidth=5, color='m')
# for i in range(n_runs):
#     plt.scatter(range(episodes), score_per_episode[i], s=14, color='#cce6ff', alpha=0.8, label='Spread')
# plt.legend(['', 'spread'], loc=(0, 0.8), fontsize=12)
# plt.xlabel('Episode', fontsize=14)
# plt.ylabel('Return', fontsize=14)
# plt.xlim(0, episodes)
# #plt.ylim(0, 3000)
# plt.grid(True, alpha=0.25, color='black', linestyle='-', linewidth=1)
# plt.show()

#%%
fig, ax = plt.subplots()

for i in range(len(gamma)):
    ax.plot(mean_score_over_runs_gamma[i], label=f'gamma={gamma[i]}', linewidth=4)
ax.legend(loc=(0, 0.68), fontsize=12)
ax.set_xlabel('Episode', fontsize=14)
ax.set_ylabel('Return', fontsize=14)
ax.set_xlim(0, episodes)
ax.set_ylim(0, 3000)
ax.grid(True, alpha=0.2, color='black', linestyle='-', linewidth=1)
ax.set_aspect(episodes/3000)

#plt.gca().set_aspect(2)

#plt.savefig("return_per_gamma_square.png", dpi=300)
plt.show()
#%%
fig, ax = plt.subplots()

for i in range(len(gamma)):
    ax.plot(mean_steps_over_runs_gamma[i], label=f'gamma={gamma[i]}', linewidth=4)
ax.legend(loc=(0.47, 0), fontsize=12)
ax.hlines(y=300, xmin=0, xmax=episodes, color='black', linewidth=5)
ax.set_xlim(0, episodes)
ax.set_ylim(0, 310)
ax.set_xlabel('Episode', fontsize=14)
ax.set_ylabel('Steps', fontsize=14)
ax.grid(True, alpha=0.2, color='black', linestyle='-', linewidth=1)
ax.set_aspect(episodes/310)
#ax.set_aspect(2)

#plt.savefig("steps_per_gamma_square.png", dpi=300)
plt.show()

#%%

fig, ax = plt.subplots()

for i in range(len(learning_rate)):
    ax.plot(mean_score_over_runs_alpha[i], linewidth=4)
ax.legend(['alpha=5e-5', 'alpha=1e-4', 'alpha=5e-4', 'alpha=1e-3'], loc=(0, 0.68), fontsize=12)
ax.set_xlabel('Episode', fontsize=14)
ax.set_ylabel('Return', fontsize=14)
ax.set_xlim(0, episodes)
ax.set_ylim(0, 3000)
ax.grid(True, alpha=0.2, color='black', linestyle='-', linewidth=1)
ax.set_aspect(2)
#plt.savefig("return_per_alpha_square.png", dpi=300)
plt.show()
#%%
fig, ax = plt.subplots()

for i in range(len(learning_rate)):
    ax.plot(mean_steps_over_runs_alpha[i], linewidth=4)
ax.legend(['alpha=5e-5', 'alpha=1e-4', 'alpha=5e-4', 'alpha=1e-3'], loc=(0.54, 0), fontsize=12)
ax.hlines(y=300, xmin=0, xmax=episodes, color='black', linewidth=5)
ax.set_xlim(0, episodes)
ax.set_ylim(0, 310)
ax.set_xlabel('Episode', fontsize=14)
ax.set_ylabel('Steps', fontsize=14)
ax.grid(True, alpha=0.2, color='black', linestyle='-', linewidth=1)
ax.set_aspect(episodes/310)
#plt.savefig("steps_per_alpha_square.png", dpi=300)
plt.show()


#%% Gamma plot
gamma = [1, 0.999, 0.99, 0.95, 0.9]
steps = np.arange(300)
fig, ax = plt.subplots()

for gamma in gamma:
    ax.plot(gamma**steps, linewidth=4)
ax.legend(['gamma = 1', 'gamma = 0.999', 'gamma = 0.99', 'gamma = 0.95', 'gamma = 0.9'], loc=(0.65, 0.15))
ax.set_xlim(0, len(steps))
ax.set_ylim(0, 1.05)
ax.set_xlabel('steps', fontsize=14)
ax.set_ylabel('Discount factor gamma', fontsize=14)
ax.grid(True, alpha=0.2, color='black', linestyle='-', linewidth=1)
#ax.hlines(y=0.5, xmin=0, xmax=len(steps), color='black', linewidth=3)
ax.set_aspect(120)
#plt.savefig("discount_factor.png", dpi=300)
plt.show()


