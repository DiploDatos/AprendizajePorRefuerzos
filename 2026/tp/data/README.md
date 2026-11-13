# Datos del TP

Los archivos provienen del material de la edición anterior y se conservan sin modificar su contenido.

El TP vigente utiliza Netflix y Goodreads con 8 grupos:

- `train/netflix8_train.df` y `train/goodreads8_train.df`.
- `test_users/netflix8_test.json` y `test_users/goodreads8_test.json`.
- Las estadísticas correspondientes en `groups/`.

Los archivos adicionales de otras variantes se conservan para no perder material, pero no forman parte de las dos configuraciones de `config_dataset.py`.

Cargar los datos mediante [`config_dataset.py`](../config_dataset.py) o las funciones de [`protocolo/datos.py`](../protocolo/datos.py). Consultar el [protocolo](../protocolo/PROTOCOLO.md) para particiones, episodios y evaluación. Los archivos `groups/` son parámetros del generador sintético para análisis, no entradas para construir políticas.
