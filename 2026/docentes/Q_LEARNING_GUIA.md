# Guía docente — actividad formativa de Q-learning

Esta actividad reemplaza la propuesta anterior de TP1. No tiene puntos, entrega ni defensa obligatoria. El único TP es Decision Transformer. Usar el modo corto en clase y la evidencia de cinco semillas para discutir lo que una corrida pequeña no permite concluir.

## Organización

Duración sugerida: 45 minutos dentro del segundo encuentro. Proveer notebook, `experimentos.py` y dependencias de `actividades/q_learning/`. El script `referencia_q_learning.py` y el notebook ejecutado de esta carpeta son apoyo docente. La hoja de trabajo es opcional.

Observar si pueden explicar el target, distinguir entrenamiento/evaluación y reconocer un proxy inadecuado. El feedback es formativo; no aplicar la antigua rúbrica de 100 puntos ni pedir informe de 2–4 páginas. Si no se termina, cerrar con los resultados de referencia.

## Contraejemplos

**max frente a argmax:** Q siguiente `[8, 1, 0, 0]`, r=0, γ=0.9, α=1, Q actual=0. Target correcto 7.2; usar el índice argmax da 0.

**TimeLimit:** r=1, max Q siguiente=10, γ=0.9, no terminal pero truncado. Target 10 si el corte es externo al MDP; en terminación real, target 1.

**Bonus por vivir:** finalizar da 1 y termina; permanecer da 0.05 indefinidamente, con γ=0.99 valor 5. Cambiamos el objetivo. No prometer que toda corrida corta muestre el efecto; apoyar la discusión con una trayectoria y el experimento completo guardado.

## Preguntas de cierre

- ¿Qué cambia si reemplazo max por argmax?
- ¿Por qué evaluar no debe modificar Q?
- ¿Mayor recompensa de entrenamiento implica mayor éxito?
- ¿Qué resultado refutaría tu predicción?

IA opcional para todos; no usar detectores ni premiar modelos pagos. Pedir la misma explicación conceptual a quienes usen IA y a quienes no. El leaderboard de Hugging Face es optativo y no otorga nota.
