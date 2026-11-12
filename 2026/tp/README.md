# Trabajo Práctico 2026 — Decision Transformer e IQL para recomendaciones

En este trabajo se implementan y comparan dos métodos de aprendizaje por refuerzo offline, **Decision Transformer (DT)** e **Implicit Q-Learning (IQL)**, para recomendar ítems usando Netflix o Goodreads.

El objetivo es analizar cómo aprende cada método a partir de trayectorias registradas, cómo influye la calidad de esos datos y qué resultados obtiene frente a líneas base bajo un protocolo común.

## Material del TP

- [Consigna](CONSIGNA.md): actividades y entregables.
- [Guía de implementación](GUIA.md): conceptos, ejemplos y esqueletos de código de DT e IQL para completar.
- [Rúbrica de evaluación](RUBRICA.md): criterios y puntajes.
- [Protocolo de evaluación](protocolo/PROTOCOLO.md): reglas de las sesiones, métricas y líneas base.
- [Configuración de datasets](config_dataset.py): selección del dataset y rutas de los archivos.

## Preparación

Con Python 3.12, desde la raíz del repositorio:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r 2026/tp/requirements.txt
```

Los datos se encuentran en `tp_decision_transformer/data/`, en la raíz del repositorio. `config_dataset.py` resuelve sus rutas.

## Primeros pasos

1. Leer la consigna y las secciones de datos y formulación de la guía.
2. Verificar el protocolo desde `2026/tp`:

   ```bash
   cd 2026/tp
   python -m pytest protocolo/tests
   python -m protocolo.evaluar --dataset netflix --split val --output outputs/baselines_netflix_val
   ```

3. Implementar DT e IQL siguiendo los esqueletos de la guía y comparar con las líneas base.
4. Elegir hiperparámetros en validación y reservar test para la evaluación final.

La evaluación utiliza sesiones simuladas: cada rating se revela únicamente después de recomendar el ítem. Se comparan condiciones con grupo visible, grupo oculto e historial inicial de warm-up. Los experimentos incluyen condicionamiento por retorno y entrenamiento con logs de distinta calidad.

## Organización del código

Los esqueletos de **DT e IQL están en los bloques de código de `GUIA.md`**, en las secciones 4 y 5. Las partes marcadas con `TODO` deben completarse como parte del trabajo.

La carpeta `protocolo/` contiene el código ejecutable de carga y partición de datos, simulación de sesiones, líneas base, generación de logs, métricas, evaluación y pruebas. Los reportes de evaluación se guardan en la carpeta indicada con `--output`.
