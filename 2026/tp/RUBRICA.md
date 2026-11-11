# Rúbrica propuesta — único TP 2026

Para revisión docente antes de publicar. Conserva los componentes del trabajo anterior, agrega IQL y la calidad de los logs, y añade trazabilidad de las decisiones. Se usa una sola calificación de TP; las actividades de clase no generan otra entrega.

| Componente | Puntos | Evidencia |
|---|---:|---|
| Datos y formulación | 15 | Semántica del dataset, MDP explícito, trayectorias/RTG coherentes, separación de particiones |
| Decision Transformer | 15 | Condicionamiento, máscaras y targets explicados; ejecución reproducible |
| IQL | 15 | V, Q y π bien definidas; expectil, objetivo TD y AWR explicados; comparación γ = 0 frente a γ > 0; π frente al máximo de Q |
| Baselines y evaluación | 20 | Comparación bajo protocolo común, métricas válidas, test reservado, varias semillas |
| Return conditioning | 15 | Intervención controlada, resultados por grupo, incertidumbre y límites |
| Calidad de los logs | 10 | Ambos modelos con logs originales y `mixto`; interpretación del cambio en modelos y líneas base |
| Reporte y comprensión | 10 | Comparación razonada DT/IQL, explicaciones sustentadas, bitácora de decisiones y contribuciones |

No exigir superar un score universal. Dar crédito a resultados negativos bien verificados. No confundir una curva de loss descendente con calidad de recomendación: en IQL, las pérdidas de Q y π pueden subir mientras la política mejora. Valorar el diagnóstico y corrección de errores.

IA opcional y mismos criterios para todos. No usar detectores de texto generado ni puntuar la cantidad de prompts. La modalidad de explicación individual y las reglas institucionales de aprobación se definen y comunican antes de la entrega.
