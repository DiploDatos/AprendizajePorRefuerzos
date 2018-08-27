#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from agents.frozen_lake_agent import FrozenLakeAgent as fP


# se declara una semilla aleatoria
random_state = np.random.RandomState(20)

# el tiempo de corte del agente son 100 time-steps
cutoff_time = 100

# instanciamos nuestro agente
agent = fP.FrozenLakeAgent()

# definimos sus híper-parámetros básicos
# (también podrían establecerse los bins que hacen la división, modificando el método set_hyper_parameters)

alpha = 0.5
gamma = 0.9
epsilon = 0.1

agent.set_hyper_parameters({"alpha": alpha, "gamma": gamma, "epsilon": epsilon})

# declaramos como True la variable de mostrar video, para ver en tiempo real cómo aprende el agente. Borrar esta línea
# para acelerar la velocidad del aprendizaje
agent.display_video = True

# establece el tiempo de
agent.set_cutoff_time(cutoff_time)

# inicializa el agente
agent.init_agent()

# reinicializa el conocimiento del agente
agent.restart_agent_learning()

# se realiza la ejecución del agente
avg_steps_per_episode = agent.run()

# se muestra la curva de convergencia de las recompensas
episode_rewards = np.array(agent.reward_of_episode)
plt.scatter(np.array(range(0, len(episode_rewards))), episode_rewards, s=0.7)
plt.title('Recompensa por episodio')
plt.show()

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

value_matrix = np.zeros((4, 4))
for row in range(4):
    for column in range(4):

        state_values = []

        for action in range(4):
            state_values.append(agent._learning_algorithm.get_q(row * 4 + column, action))

        maximum_value = max(state_values)  # como usamos epsilon-greedy, determinamos la acción que arroja máximo valor
        state_values.remove(maximum_value)  # removemos el ítem asociado con la acción de máximo valor

        value_matrix[row, column] = maximum_value * (1 - epsilon)

        for non_maximum_value in state_values:
            value_matrix[row, column] += epsilon/4 * non_maximum_value

# el valor del estado objetivo se asigna en 1 (reward recibido al llegar) para que se coloree de forma apropiada
value_matrix[3, 3] = 1

plt.imshow(value_matrix, cmap=plt.cm.RdYlGn)
plt.tight_layout()
plt.colorbar()
plt.xticks([])
plt.yticks([])
plt.show()

print(value_matrix)

agent.destroy_agent()
