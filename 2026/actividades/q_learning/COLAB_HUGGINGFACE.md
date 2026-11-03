# Hands-on de Hugging Face (unidad 2) en Colab

Actividad formativa del primer encuentro: usar la [copia del curso](../huggingface/unidad_2.ipynb), basada en el [notebook oficial](https://github.com/huggingface/deep-rl-class/blob/main/notebooks/unit2/unit2.ipynb). En la [copia probada en Colab](https://colab.research.google.com/drive/1UKQmi0aBdxTb-kjVqE5iGQeblG2Pr1Ds) se entrenaron FrozenLake y Taxi-v4; Taxi obtuvo 7,50 ± 2,76 puntos de recompensa en 100 episodios. No hace falta publicar en Hugging Face.

## Cambios frente al original

- La instalación vieja de `requirements-unit2.txt` se reemplazó por paquetes actuales de Colab. Esa instalación fallaba con Python 3.13 por `pyyaml==6.0` y `pickle5`.
- Se retiró la celda que terminaba el proceso de Colab con `os.kill(os.getpid(), 9)`.
- Las celdas incompletas `TODO` se conservaron como texto de ejercicio; las soluciones ejecutables aparecen más adelante.
- Taxi usa `Taxi-v4`, disponible en la versión actual de Gymnasium.
- Se omitieron la publicación y la carga de modelos externos. La actividad termina con evaluación local.

Para ejecutarla, abrir la copia de Colab y usar **Entorno de ejecución → Ejecutar todas**. Sus salidas están configuradas como privadas y pueden no verse al abrirla; se regeneran al ejecutarla.
