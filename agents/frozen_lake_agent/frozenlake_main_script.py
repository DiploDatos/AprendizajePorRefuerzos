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

agent.set_hyper_parameters({"alpha": 0.5, "gamma": 0.9, "epsilon": 0.01})

# declaramos como True la variable de mostrar video, para ver en tiempo real cómo aprende el agente. Borrar esta línea
# para acelerar la velocidad del aprendizaje
agent.display_video = True

# establece el tiempo de
agent.set_cutoff_time(cutoff_time)

# inicializa el agente
agent.init_agent()

# reinicializa el conocimiento del agente
agent.restart_agent_learning()

# run corre el agente devuelve la cantidad promedio de pasos por episodio
avg_steps_per_episode = agent.run()

# se muestra la curva de convergencia
episode_rewards = np.array(agent.reward_of_episode)
plt.scatter(np.array(range(0, len(episode_rewards))), episode_rewards)
plt.title('Recompensa')
plt.show()

# se suaviza la curva de aprendizaje
episode_number = np.linspace(1, len(episode_rewards) + 1, len(episode_rewards) + 1)
acumulated_rewards = np.cumsum(episode_rewards)

reward_per_episode = [acumulated_rewards[i] / episode_number[i] for i in range(len(acumulated_rewards))]

plt.plot(reward_per_episode)
plt.title('Recompensa por episodio')
plt.show()

# Se muestra el agente con su política óptima
print(agent._learning_algorithm.q)

agent.destroy_agent()
