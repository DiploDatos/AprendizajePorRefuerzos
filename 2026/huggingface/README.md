# Actividades basadas en Hugging Face Deep RL

Estas cuatro actividades parten de los [notebooks oficiales](https://github.com/huggingface/deep-rl-class/tree/main/notebooks). Cada archivo conserva un enlace a su versión de origen. Se retiraron las salidas, las celdas incompletas que impiden ejecutar todo y las secciones de publicación o carga de modelos del Hub. Las copias de validación se ejecutaron en Google Colab desde Brave.

| Unidad | Tema | Validación en Colab |
| --- | --- | --- |
| [1](unidad_1.ipynb) | PPO y LunarLander-v3 | Prueba corta en Colab: instalación, 2048 pasos, guardado y evaluación ejecutados. El archivo del curso configura un millón de pasos, no ejecutados completos. |
| [2](unidad_2.ipynb) | Q-learning, FrozenLake y Taxi-v4 | [Copia de validación](https://colab.research.google.com/drive/1UKQmi0aBdxTb-kjVqE5iGQeblG2Pr1Ds): ambos agentes entrenados; Taxi obtuvo recompensa media 7,50 ± 2,76 en 100 episodios. |
| [3](unidad_3.ipynb) | DQN, Space Invaders y RL Zoo | Prueba corta en Colab: ALE cargó el entorno y RL Zoo entrenó 500 pasos, guardó el modelo y lo cargó para evaluar 50 pasos. El archivo del curso configura un millón de pasos. |
| [4](unidad_4.ipynb) | REINFORCE y CartPole-v1 | [Copia de validación](https://colab.research.google.com/drive/1Gn-X8RCJTNW4F8QcRAPfopjf9t0t8tOJ): 1000 episodios de entrenamiento y recompensa media 500 ± 0 en 10 episodios de evaluación. |

Copias completas listas para abrir en Colab: [unidad 1](https://colab.research.google.com/drive/176nZrSvdgZe84tedHzRTpMX80LBE3S5O), [unidad 2](https://colab.research.google.com/drive/1UKQmi0aBdxTb-kjVqE5iGQeblG2Pr1Ds), [unidad 3](https://colab.research.google.com/drive/1K9z4ESHwSK2Qlatv_ccnp3UhrZCG6sW9) y [unidad 4](https://colab.research.google.com/drive/1Gn-X8RCJTNW4F8QcRAPfopjf9t0t8tOJ).

Las pruebas cortas de las unidades 1 y 3 verifican compatibilidad y ejecución, no calidad de la política aprendida. Colab mantuvo las salidas privadas, por lo que las copias enlazadas pueden abrirse sin los resultados de esta sesión. Para repetirlas, usar **Entorno de ejecución → Ejecutar todas**.

Para ejecutar las cuatro unidades localmente, crear un entorno con Python 3.12 e instalar `python -m pip install -r 2026/huggingface/requirements.txt` desde la raíz del repositorio. El archivo reúne Box2D, Atari, Stable Baselines, RL Zoo, PyTorch y las bibliotecas de gráficos y video. En Colab cada notebook conserva su propia celda de instalación. En Python 3.14 o posterior, Box2D también puede requerir SWIG instalado en el sistema; por eso recomendamos Python 3.12.
