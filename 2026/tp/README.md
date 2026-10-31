# Único TP 2026 — Decision Transformer para recomendaciones

Continúa el trabajo que en 2025 se llamaba TP2. En 2026 se denomina **Trabajo Práctico** o **TP**, sin número. FrozenLake/Taxi y el notebook de RL para generación son actividades formativas de clase.

- [Consigna 2026 en revisión docente](CONSIGNA.md).
- [Criterios de evaluación propuestos](RUBRICA.md).
- [Configuración y ubicación de los datos](config_dataset.py).
- [Entorno reproducible y extra de Hugging Face](../../ENTORNO.md).
- [Enunciado y guías históricos](../../archive/2025/tp_decision_transformer/01_README_TRABAJO_PRACTICO.md), como referencia de la edición anterior.

Se conservan Netflix/Goodreads, preparación de trayectorias, modelo condicionado por retorno, comparación con baselines, experimentos de return conditioning y reporte. No se exige entrenar un LLM ni desplegar en la Orin.

## Estado antes de distribuir

Este borrador unifica el TP y conserva su temática. La implementación de referencia de 2025 tiene problemas de evaluación y arquitectura identificados en la [auditoría](../00_AUDITORIA.md). Antes de publicarlo hay que cerrar y ejecutar un protocolo común que distinga ratings de secuencias, usuario nuevo de cold-start y DT original de una variante condicionada. No usar el siguiente ID de un listado ordenado como siguiente interacción real.

Los datos permanecen en `archive/2025/tp_decision_transformer/data/` para evitar duplicarlos. `config_dataset.py` devuelve rutas absolutas válidas desde cualquier directorio. Las referencias históricas conservan su configuración original; no son la versión técnica corregida de 2026.

Fechas, tamaño de grupo y modalidad de defensa siguen pendientes de acuerdo docente. No se trasladan fechas de 2025 ni se agrega una segunda entrega obligatoria.
