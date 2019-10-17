#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import itertools
import gym
import numpy as np


def choose_action(state):
    """
    Chooses an action according to the learning previously performed
    """
    q_values = [q.get((state, a), 0.0) for a in actions]
    max_q = max(q_values)

    if random_state.uniform() < epsilon:
        return random_state.choice(actions)  # a random action is selected

    count = q_values.count(max_q)

    # In case there're several state-action max values
    # we select a random one among them
    if count > 1:
        best = [i for i in range(len(actions)) if q_values[i] == max_q]
        i = random_state.choice(best)
    else:
        i = q_values.index(max_q)

    return actions[i]


def learn(state, action, reward, next_state):
    """
    Performs a Q-learning update for a given state transition

    Q-learning update:
    Q(s, a) += alpha * (reward(s,a) + max(Q(s') - Q(s,a))
    """
    new_max_q = max([q.get((next_state, a), 0.0) for a in actions])
    old_value = q.get((state, action), 0.0)

    q[(state, action)] = old_value + alpha * (reward + gamma * new_max_q - old_value)


def run():
    """
    Runs the reinforcement learning agent with a given configuration.
    """
    # list that contains the amount of time-steps of the episode. It is used as a way to score the performance of
    # the agent.
    timesteps_of_episode = []
    # list that contains the amount of reward given to the agent in each episode
    reward_of_episode = []

    for i_episode in range(episodes_to_run):
        # an instance of an episode is run until it fails or until it reaches 200 time-steps

        # resets the environment, obtaining the first state observation
        state = env.reset()

        episode_reward = 0
        done = False
        t = 0

        while not done:

            # Pick an action based on the current state
            action = choose_action(state)
            # Execute the action and get feedback
            observation, reward, done, info = env.step(action)

            episode_reward += reward

            next_state = observation

            if not done:
                learn(state, action, reward, next_state)
                state = next_state
            else:
                learn(state, action, reward, next_state)
                timesteps_of_episode = np.append(timesteps_of_episode, [int(t + 1)])
                reward_of_episode = np.append(reward_of_episode, max(episode_reward, -100))

            t += 1

    return reward_of_episode.mean(), timesteps_of_episode, reward_of_episode


q = {}

# definimos sus híper-parámetros básicos

alpha = 1
gamma = 1
epsilon = 0.1
tau = 25

episodes_to_run = 500

env = gym.make("CliffWalking-v0")
actions = range(env.action_space.n)

# se declara una semilla aleatoria
random_state = np.random.RandomState(42)

# se realiza la ejecución del agente
avg_steps_per_episode, timesteps_ep, reward_ep = run()

episode_rewards = np.array(reward_ep)


# se suaviza la curva de convergencia
episode_number = np.linspace(1, len(episode_rewards) + 1, len(episode_rewards) + 1)
acumulated_rewards = np.cumsum(episode_rewards)

reward_per_episode = [acumulated_rewards[i] / episode_number[i] for i in range(len(acumulated_rewards))]

plt.plot(reward_per_episode)
plt.title('Recompensa acumulada por episodio')
plt.show()

# ---

# se muestra la curva de aprendizaje de los pasos por episodio
episode_steps = np.array(timesteps_ep)
plt.plot(np.array(range(0, len(episode_steps))), episode_steps)
plt.title('Pasos (timesteps) por episodio')
plt.show()

# se suaviza la curva de aprendizaje
episode_number = np.linspace(1, len(episode_steps) + 1, len(episode_steps) + 1)
acumulated_steps = np.cumsum(episode_steps)

steps_per_episode = [acumulated_steps[i] / episode_number[i] for i in range(len(acumulated_steps))]

plt.plot(steps_per_episode)
plt.title('Pasos (timesteps) acumulados por episodio')
plt.show()

# ---

n_rows = 4
n_columns = 12
n_actions = 4

# se procede con los cálculos previos a la graficación de la matriz de valor
q_value_matrix = np.empty((n_rows, n_columns))
for row in range(n_rows):
    for column in range(n_columns):

        state_values = []

        for action in range(n_actions):
            state_values.append(q.get((row * n_columns + column, action), -100))

        maximum_value = max(state_values)  # determinamos la acción que arroja máximo valor

        # el valor de la matriz para la mejor acción es el máximo valor por la probabilidad de que el mismo sea elegido
        # (que es 1-epsilon por la probabilidad de explotación más 1/4 * epsilon por probabilidad de que sea elegido al
        # azar cuando se opta por una acción exploratoria)
        q_value_matrix[row, column] = maximum_value

# el valor del estado objetivo se asigna en 1 (reward recibido al llegar) para que se coloree de forma apropiada
q_value_matrix[3, 11] = -1

# se grafica la matriz de valor
plt.imshow(q_value_matrix, cmap=plt.cm.RdYlGn)
plt.tight_layout()
plt.colorbar()

fmt = '.2f'
thresh = q_value_matrix.max() / 2.

for row, column in itertools.product(range(q_value_matrix.shape[0]), range(q_value_matrix.shape[1])):

    left_action = q.get((row * n_columns + column, 3), -1000)
    down_action = q.get((row * n_columns + column, 2), -1000)
    right_action = q.get((row * n_columns + column, 1), -1000)
    up_action = q.get((row * n_columns + column, 0), -1000)

    arrow_direction = '↓'
    best_action = down_action

    if best_action < right_action:
        arrow_direction = '→'
        best_action = right_action
    if best_action < left_action:
        arrow_direction = '←'
        best_action = left_action
    if best_action < up_action:
        arrow_direction = '↑'
        best_action = up_action
    if best_action == -1:
        arrow_direction = ''

    # notar que column, row están invertidos en orden en la línea de abajo porque representan a x,y del plot
    plt.text(column, row, arrow_direction, horizontalalignment="center")

plt.xticks([])
plt.yticks([])
plt.show()

print('\n Matriz de valor (en números): \n\n', q_value_matrix)

env.close()
