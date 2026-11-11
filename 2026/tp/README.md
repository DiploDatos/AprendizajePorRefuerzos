# Único TP 2026 — Decision Transformer e IQL para recomendaciones

Continúa el trabajo que en 2025 se llamaba TP2. En 2026 se denomina **Trabajo Práctico** o **TP**, sin número. FrozenLake/Taxi y el notebook de RL para generación son actividades formativas de clase.

- [Consigna 2026 en revisión docente](CONSIGNA.md).
- [Guía para la clase y los grupos](GUIA.md): conceptos, uso del protocolo y esqueletos de DT e IQL.
- [Criterios de evaluación propuestos](RUBRICA.md).
- [Configuración y ubicación de los datos](config_dataset.py).
- Instalación: `python -m pip install -r 2026/tp/requirements.txt` (ver [la guía](GUIA.md#0-preparación)).
- [Protocolo común de evaluación](protocolo/PROTOCOLO.md).
- [Enunciado y guías históricos](../../tp_decision_transformer/01_README_TRABAJO_PRACTICO.md), como referencia de la edición anterior. Su evaluación y su implementación de referencia no son válidas para 2026.

Se conservan Netflix/Goodreads, preparación de trayectorias, modelo condicionado por retorno, comparación con baselines, experimentos de return conditioning y reporte. Se agregan IQL como segundo modelo obligatorio y un experimento con logs de un sistema anterior. No se exige entrenar un LLM ni desplegar en la Orin.

## Estado antes de distribuir

Este borrador unifica el TP y conserva su temática. La implementación de referencia de 2025 tiene problemas de evaluación y arquitectura identificados en la auditoría de la edición. El [protocolo común](protocolo/PROTOCOLO.md) resuelve la evaluación:

- sesiones simuladas en las que cada valoración se revela recién al recomendar;
- tres condiciones: grupo visible, cold-start estricto y warm-up;
- líneas base, incluida `popularity`, y tests;
- logs de un sistema anterior (`protocolo/logs.py`) para estudiar cómo influye la calidad de los datos.

No usar el siguiente ID de un listado ordenado como siguiente interacción real.

Los datos permanecen en `tp_decision_transformer/data/` para evitar duplicarlos. `config_dataset.py` devuelve rutas absolutas válidas desde cualquier directorio. Las referencias históricas conservan su configuración original; no son la versión técnica corregida de 2026.

Fechas, tamaño de grupo y modalidad de defensa siguen pendientes de acuerdo docente. No se trasladan fechas de 2025 ni se agrega una segunda entrega obligatoria.
