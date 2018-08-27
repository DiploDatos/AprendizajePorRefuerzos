#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from agents.cart_pole_aprox_vf.CartPole_ApproxVF_SGDRegressor_Agent import CartPoleApproxVFSGDRegressorAgent
# se declara una semilla aleatoria
random_state = np.random.RandomState(20)

# el tiempo de corte del agente son 200 time-steps (el cual es el máximo del entorno Cartpole; seguir iterando tras 200
# no cambiará el entorno)
cutoff_time = 200

# instanciamos nuestro agente
agent = CartPoleApproxVFSGDRegressorAgent()

# definimos sus híper-parámetros básicos
# (también podrían establecerse los bins que hacen la división, modificando el método set_hyper_parameters)

agent.set_hyper_parameters({"alpha": 0.5, "gamma": 0.9, "epsilon": 0.3})

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
agent.destroy_agent()

print("Overall score: {:0.2f}".format(overall_score))
