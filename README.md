# RLDiplodatos

Repositorio oficial de la materia Aprendizaje por Refuerzos de la Diplomatura en Ciencias de Datos, Aprendizaje 
Automático y sus Aplicaciones.

# Cómo ejecutar los agentes RL

Para instalar el entorno virtual y poder ejecutar los laboratorios, se requiere:

1. Descargar Anaconda para Python 3.6 desde [https://www.anaconda.com/download/]().

2. Instalar el entorno virtual provisto, descargando el archivo tensorflow.yml y desde la consola ejecutar:

*conda env create -f tensorflow.yml*

3. (Opcional) Descargar un entorno (ej: [Pycharm](https://www.jetbrains.com/pycharm/download/)) para poder realizar un 
debug paso a paso de los agentes. Si bien se puede trabajar desde jupyter notebook, suele resultar mucho más sencillo 
debuguear los agentes desde un IDE.

4. Correr los agentes a partir de alguno de los scripts por ejemplo *frozenlake_main_script.py* o 
*cartpole_main_script.py*.

5. Para incorporar el nuevo kernel a Jupyter Notebook tiene que ejecutar los siguientes comandos:

*source activate tensorflow*

*python -m ipykernel install --name tensorflow*

*source deactivate*

6. Para incorporar el entorno tensorflow creado en Pycharm, seleccionarlo desde:

* File -> Settings -> Project -> Project interpreter -> Show all (seleccionado en lista de entornos) -> Add -> Conda environment -> Existing environment -> tensorflow


Algunos links útiles para aprender más:

• [Completa entrada de blog de métodos de RL hasta el momento](https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html)

• [Blog de muy buen curso de Deep RL](https://simoninithomas.github.io/Deep_reinforcement_learning_Course/)

• [Entrada de blog explicando el equilibrio entre sesgo y varianza en RL](https://medium.com/mlreview/making-sense-of-the-bias-variance-trade-off-in-deep-reinforcement-learning-79cf1e83d565)

• [Paper sobre vista general de Deep RL](https://arxiv.org/abs/1701.07274)

• [Paper sobre la exploración basada en curiosidad y recompensas intrínsecas](https://pathak22.github.io/noreward-rl/resources/icml17.pdf)

• [Libro de Sutton y Barto (libro principal de RL)](https://drive.google.com/file/d/1opPSz5AZ_kVa1uWOdOiveNiBFiEOHjkG/view)
