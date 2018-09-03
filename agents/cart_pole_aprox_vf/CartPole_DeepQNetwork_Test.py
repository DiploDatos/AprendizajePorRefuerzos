import numpy as np
from agents.cart_pole_aprox_vf.CartPole_ApproxVF_DQN_Agent import CartPoleApproxVFDQNAgent
import matplotlib.pyplot as plt

# el tiempo de corte del agente son 200 time-steps (el cual es el máximo del entorno Cartpole; seguir iterando tras 200
# no cambiará el entorno)
cutoff_time = 200

# instanciamos nuestro agente
agent = CartPoleApproxVFDQNAgent()

# definimos sus híper-parámetros básicos
# (también podrían establecerse los bins que hacen la división, modificando el método set_hyper_parameters)

agent.set_hyper_parameters({
    "alpha": 0.01,
    "alpha_decay": 0.01,
    "gamma": 1.0,
    "epsilon_init": 1.0,
    "epsilon_decay": 0.995,
    "epsilon_min": 0.01,
    "batch_size": 64})

# declaramos como True la variable de mostrar video, para ver en tiempo real cómo aprende el agente. Borrar esta línea
# para acelerar la velocidad del aprendizaje
agent.display_video = False

# establece el tiempo de
agent.set_cutoff_time(cutoff_time)

# inicializa el agente
agent.init_agent()

# reinicializa el conocimiento del agente
agent.restart_agent_learning()

# run corre el agente devuelve el overall score, que es el promedio de recompensa de todos los episodios
overall_score = agent.run()
episode_reward = agent.reward_average
agent.destroy_agent()

plt.set_xlabel('Episode')
plt.set_ylabel('Avg. Reward')
plt.plot(episode_reward)
plt.show()