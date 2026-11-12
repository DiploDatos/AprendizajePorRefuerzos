# Rúbrica de evaluación — TP 2026

El trabajo práctico se evalúa sobre un total de **100 puntos**, según los siguientes criterios.

| Componente | Puntos | Evidencia |
|---|---:|---|
| Datos y formulación | 15 | Semántica del dataset, MDP explícito, trayectorias/RTG coherentes, separación de particiones |
| Decision Transformer | 15 | Condicionamiento, máscaras y targets explicados; ejecución reproducible |
| IQL | 15 | V, Q y π bien definidas; expectil, objetivo TD y AWR explicados; comparación γ = 0 frente a γ > 0; π frente al máximo de Q |
| Baselines y evaluación | 20 | Comparación bajo protocolo común, métricas válidas, test reservado, varias semillas |
| Return conditioning | 15 | Intervención controlada, resultados por grupo, incertidumbre y límites |
| Calidad de los logs | 10 | Ambos modelos con logs originales y `mixto`; interpretación del cambio en modelos y líneas base |
| Reporte y comprensión | 10 | Comparación razonada DT/IQL, explicaciones sustentadas, bitácora de decisiones y contribuciones |

No se exige superar un puntaje universal ni obtener mejoras sobre todas las líneas base. Los resultados negativos bien verificados también son válidos: se evalúan el diseño experimental, la interpretación de la evidencia y el diagnóstico y corrección de errores.

Una pérdida de entrenamiento descendente no demuestra calidad de recomendación. En IQL, las pérdidas de Q y π pueden subir mientras la política mejora; las conclusiones deben apoyarse en la evaluación con el protocolo común.

El uso de IA es opcional y se aplican los mismos criterios a todos los grupos. Si se utiliza, se debe indicar su uso y qué se verificó. Se evalúan la comprensión y la justificación de las decisiones, no la cantidad de prompts.
