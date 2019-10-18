# Aprendizaje por Refuerzos

Repo oficial de la materia optativa Aprendizaje por Refuerzos de la Diplomatura en Ciencias de Datos, Aprendizaje 
Automático y sus Aplicaciones.

## Instalación

Para instalar el entorno virtual y poder ejecutar los agentes que vamos a ver, se requieren los paquetes:

- gym
- gym[atari] (opcional, en el caso que se quiera contar con los juegos de Atari de OpenAI)
- numpy
- scipy
- matplotlib
- torch

Pasos para instalarlos como un nuevo entorno virtual:

1. Descargar Anaconda para Python 3.7 desde [https://www.anaconda.com/download/](https://www.anaconda.com/download/).

2. Instalar el entorno virtual provisto, descargando el archivo python37_rl.yml y desde la consola ejecutar:

*conda env create -f python37_rl.yml*

## Cómo ejecutar los agentes RL

1. (Opcional) Descargar un entorno (ej: [Pycharm](https://www.jetbrains.com/pycharm/download/)) para poder realizar un 
debug paso a paso de los agentes. Si bien se puede trabajar desde jupyter lab, suele resultar mucho más sencillo 
hacer debug de los agentes desde un IDE.

2. Correr los agentes a partir de alguno de los scripts por ejemplo *frozenlake_main_script.py* o 
*cartpole_main_script.py*.

3. Para incorporar el nuevo kernel a Jupyter Notebook tiene que ejecutar los siguientes comandos:

*source activate tensorflow*

*python -m ipykernel install --name python37_rl*

*source deactivate*

4. Para incorporar el entorno tensorflow creado en Pycharm, seleccionarlo desde:

* File -> Settings -> Project -> Project interpreter -> Show all (seleccionado en lista de entornos) -> Add -> Conda environment -> Existing environment -> tensorflow


## Algunos links útiles para aprender más:

* [Excelente recurso para aprender Deep RL, de OpenAI](https://spinningup.openai.com/en/latest/spinningup/spinningup.html)

* [Completa entrada de blog de métodos de RL](https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html)

* [Tutorial para hacer un agente que juega al Atari con DQN](https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26)

* [Entrada de blog explicando el equilibrio entre sesgo y varianza en RL](https://medium.com/mlreview/making-sense-of-the-bias-variance-trade-off-in-deep-reinforcement-learning-79cf1e83d565)

* [Paper sobre vista general de Deep RL](https://arxiv.org/abs/1701.07274)

* [Libro principal de RL, por Sutton y Barto](https://drive.google.com/file/d/1opPSz5AZ_kVa1uWOdOiveNiBFiEOHjkG/view)

* [Paper sobre la exploración basada en curiosidad y recompensas intrínsecas](https://pathak22.github.io/noreward-rl/resources/icml17.pdf)

