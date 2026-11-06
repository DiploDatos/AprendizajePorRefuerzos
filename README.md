# Aprendizaje por Refuerzos

Repo oficial de la materia optativa Aprendizaje por Refuerzos de la Diplomatura en Ciencias de Datos, Aprendizaje
Automático y sus Aplicaciones.

## Instalación y ejecución

### Instalación local para las actividades de 2026

Con Python 3.12, desde la raíz del repositorio:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r 2026/huggingface/requirements.txt
```

El archivo de dependencias cubre las cuatro unidades de Hugging Face y el [notebook introductorio de referencia](2026/notebooks/notebook_1_intro_rl.ipynb). Los notebooks también incluyen instrucciones para ejecutarlos en Google Colab.

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
