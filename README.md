# Aprendizaje por Refuerzos

Repo oficial de la materia optativa Aprendizaje por Refuerzos de la Diplomatura en Ciencias de Datos, Aprendizaje 
Automático y sus Aplicaciones.

## Instalación y ejecución

### Instalación

Pasos para instalar los paquetes requeridos como un nuevo entorno de Conda:

#### Desde local

Descargar Anaconda desde [https://www.anaconda.com/download/](https://www.anaconda.com/download/), e instalarlo tras la descarga. Tras ello, ejecutamos los siguientes comandos desde consola:

1. Creamos el entorno local:

        conda create --name diplodatos_rl python=3.9
        conda activate diplodatos_rl
    
1. Añadimos el entorno a jupyter (fuente: https://stackoverflow.com/questions/53004311/how-to-add-conda-environment-to-jupyter-lab#53546634)
    
        conda install ipykernel
        ipython kernel install --user --name=diplodatos_rl
        conda deactivate
        conda activate diplodatos_rl

1. Instalamos librerías de aprendizaje por refuerzos profundo (excepto gym y ffmpeg, el resto son para los algoritmos de deep RL que veremos el segundo fin de semana)

        pip install gym pyglet stable-baselines3[extra,tests,docs]
        git clone --recursive https://github.com/DLR-RM/rl-baselines3-zoo
        cd rl-baselines3-zoo/
        conda install swig
        apt-get install cmake ffmpeg
        pip install -r requirements.txt


#### Desde Nabu

Para crear el entorno virtual desde Nabu, los primeros comandos son:

    wget https://repo.anaconda.com/archive/Anaconda3-2021.04-Linux-x86_64.sh
    chmod 755 Anaconda3-2020.11-Linux-x86_64.sh
    ./Anaconda3-2020.11-Linux-x86_64.sh
    conda create --name diplodatos_rl
    conda activate diplodatos_rl
    conda config --add channels conda-forge



### Ejecución

Los notebooks están preparados para ejecutarse tanto desde localhost, como desde Google Colab y Nabu.
En general, las simulaciones de estos notebooks se pueden ejecutar sin problemas desde localhost, ya que no demandan demasiados recursos computacionales (excepto si se ejecutan entrenamientos completos en entornos muy complejos, como en los de Atari).
Algunas características sólo están disponibles en localhost, como las animaciones de los agentes en los entornos.


## Algunos links útiles para aprender más:

* [Discord de RL](https://discord.gg/xhfNqQv) (muy activo y recomendado!) y su [wiki de recursos](https://github.com/andyljones/reinforcement-learning-discord-wiki/wiki)!

* [Comunidad de RL en reddit](https://old.reddit.com/r/reinforcementlearning).

* [Excelente recurso para aprender Deep RL, de OpenAI](https://spinningup.openai.com/en/latest/spinningup/spinningup.html)

* [Completa entrada de blog de métodos de RL](https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html)

* [Awesome RL envs](https://github.com/clvrai/awesome-rl-envs).

* [Awesome deep RL](https://github.com/kengz/awesome-deep-rl).

* [Environments zoo](https://github.com/tshrjn/env-zoo).

* [Libro principal de RL, por Sutton y Barto](https://drive.google.com/file/d/1opPSz5AZ_kVa1uWOdOiveNiBFiEOHjkG/view)



