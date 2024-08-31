from QuadModel import QuadModel
import matplotlib.pyplot as plt
import math
import numpy as np

# This code is meant as a validation and testing tool of the custom environment.
# Can be used independent of RL training codes

mass = 1
g = 9.81
motor_distance = 2
env = QuadModel(mass, motor_distance, render_mode="human")
T_l = 4.95
T_r = 5
action = [T_l, T_r]

episodes = 1
s_0 = env.reset(mid_air=True)
x = [s_0[0]]
x_d = [s_0[1]]
y = [s_0[2]]
y_d = [s_0[3]]
v_total = [math.sqrt(x_d[0] ** 2 + y_d[0] ** 2)]
kinetic_energy = []
potential_energy = []
done = False
truncated = False
t = 0
score = 0
while not (done or truncated):
    t += 1
    s, reward, terminated, truncated, cause = env.dynamics(action)
    score += reward
    x.append(s[0])
    x_d.append(s[1])
    y.append(s[2])
    y_d.append(s[3])
    v_total.append(math.sqrt(x_d[t] ** 2 + y_d[t] ** 2))
    kinetic_energy.append(0.5 * mass * v_total[t] ** 2)
    potential_energy.append(g * mass * y[t])
    print(s, score, terminated, truncated, cause)
    if cause is not None:
        print(cause)
        print(s)
    if terminated:
        done = True
total_energy = np.array(kinetic_energy) + np.array(potential_energy)

env.close()

# plt.plot(x)
# plt.plot(y)
# plt.legend(['x', 'y'])
# plt.xlabel('Time')
# plt.ylabel('displacement [m]')
# plt.show()

# plt.plot(v_total)
# plt.legend(['v_total'])
# plt.xlabel('Time')
# plt.ylabel('Velocity [m/s]')
# plt.show()


# plt.plot(kinetic_energy)
# plt.plot(potential_energy)
# #plt.plot(total_energy)
# plt.legend(['kinetic energy', 'potential energy', 'total energy'])
# plt.xlabel('Time')
# plt.ylabel('energy [J]')
# plt.show()

plt.plot(x, y)
plt.scatter(x[0], y[0], c='green', s=60)
plt.scatter(x[-1], y[-1], c='red', s=60)
plt.title(f'initial state is: x={s_0[0]}, x_d={s_0[1]}, y={s_0[2]}, y_d={s_0[3]}, alpha={s_0[4]}, alpha_d={s_0[5]}, t={t}')
plt.xlim([-4.2, 4.2])
plt.ylim([0, 4.2])
plt.xlabel('x [m]')
plt.ylabel('height [m]')
plt.show()