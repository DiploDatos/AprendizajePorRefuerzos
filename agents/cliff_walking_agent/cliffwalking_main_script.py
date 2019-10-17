#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from agents.cliff_walking_agent import CliffWalkingAgent as cW
import itertools

# definimos sus híper-parámetros básicos

alpha = 0.5
gamma = 1
epsilon = 0.1
tau = 25


# se declara una semilla aleatoria
random_state = np.random.RandomState(42)

# instanciamos nuestro agente
agent = cW.CliffWalkingAgent()

agent.set_hyper_parameters({"alpha": alpha, "gamma": gamma, "epsilon": epsilon})
agent.episodes_to_run = 500

agent.random_state = random_state


# inicializa el agente
agent.init_agent()

# reinicializa el conocimiento del agente
agent.restart_agent_learning()

# se realiza la ejecución del agente
avg_steps_per_episode = agent.run()

episode_rewards = np.array(agent.reward_of_episode)

# se suaviza la curva de convergencia
episode_number = np.linspace(1, len(episode_rewards) + 1, len(episode_rewards) + 1)
acumulated_rewards = np.cumsum(episode_rewards)

reward_per_episode = [acumulated_rewards[i] / episode_number[i] for i in range(len(acumulated_rewards))]

plt.plot(reward_per_episode)
plt.title('Recompensa acumulada por episodio')
plt.show()

# ---

# se muestra la curva de aprendizaje de los pasos por episodio
episode_steps = np.array(agent.timesteps_of_episode)
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
            state_values.append(agent.q.get((row * n_columns + column, action), -100))

        maximum_value = max(state_values)  # determinamos la acción que arroja máximo valor


        # el valor de la matriz para la mejor acción es el máximo valor por la probabilidad de que el mismo sea elegido
        # (que es 1-epsilon por la probabilidad de explotación más 1/4 * epsilon por probabilidad de que sea elegido al
        # azar cuando se opta por una acción exploratoria)
        q_value_matrix[row, column] = maximum_value

# el valor del estado objetivo se asigna en 1 (reward recibido al llegar) para que se coloree de forma apropiada
q_value_matrix[3, 11] = 1

# se grafica la matriz de valor
plt.imshow(q_value_matrix, cmap=plt.cm.RdYlGn)
plt.tight_layout()
plt.colorbar()

fmt = '.2f'
thresh = q_value_matrix.max() / 2.

for row, column in itertools.product(range(q_value_matrix.shape[0]), range(q_value_matrix.shape[1])):

    left_action = agent.q.get((row * n_columns + column, 3), -1000)
    down_action = agent.q.get((row * n_columns + column, 2), -1000)
    right_action = agent.q.get((row * n_columns + column, 1), -1000)
    up_action = agent.q.get((row * n_columns + column, 0), -1000)
    
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

agent.destroy_agent()
