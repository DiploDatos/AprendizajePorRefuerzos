# Aprendizaje por Refuerzos

Repo oficial de la materia optativa Aprendizaje por Refuerzos de la Diplomatura en Ciencias de Datos, Aprendizaje
Automático y sus Aplicaciones.

## Instalación y ejecución

### Instalación desde local

Pasos para instalar los paquetes requeridos con poetry:

Instalar [poetry](https://python-poetry.org/docs/#installation):

* Desde Linux/Mac/WSL:

        pip install poetry==1.1.13

* Desde Windows:

        (Invoke-WebRequest -Uri https://install.python-poetry.org/ -UseBasicParsing).Content | python -

Comprobar que se instaló correctamente:

        poetry --version

Instalamos las dependencias (parados desde la carpeta raíz del repo):

        poetry install  # instala las dependencias necesarias
        poetry install -E zoo  # (Opcional) instala las dependencias para usar rl-baselines-zoo
        poetry install -E dev_tools  # (Opcional) instala las dependencias para usar jupyter notebooks y otras las herramientas de desarrollo

Activamos el entorno virtual:

        poetry shell

Listo! Ya podemos ejecutar los notebooks.

### Ejecución

Los notebooks están preparados para ejecutarse tanto desde localhost, como desde Google Colab.
En general, las simulaciones de estos notebooks se pueden ejecutar sin problemas desde localhost, ya que no demandan demasiados recursos computacionales (excepto si se ejecutan entrenamientos completos en entornos muy complejos, como en los de Atari).
Algunas características sólo están disponibles en localhost, como las animaciones de los agentes en los entornos.

## Algunos links útiles para aprender más

* [Discord de RL](https://discord.gg/dBVVY8Sz7v) (muy activo y recomendado!) y su [wiki de recursos](https://github.com/andyljones/reinforcement-learning-discord-wiki/wiki)!

* [Discord de HuggingFace](http://hf.co/join/discord) (tiene canal dedicado a su curso de RL).

* [Comunidad de RL en reddit](https://old.reddit.com/r/reinforcementlearning).

* [Completa entrada de blog de métodos de RL](https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html).

* [Awesome RL envs](https://github.com/clvrai/awesome-rl-envs).

* [Awesome deep RL](https://github.com/kengz/awesome-deep-rl).

* [Libro principal de RL, por Sutton y Barto](http://incompleteideas.net/book/RLbook2020.pdf).
