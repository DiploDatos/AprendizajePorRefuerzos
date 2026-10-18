# Actividad formativa — Q-learning: resolver, intervenir y auditar

DiploDatos 2026. Actividad de clase para el primer fin de semana, sin calificación, entrega ni defensa obligatoria. El único TP del curso es [Decision Transformer](../../tp/README.md). La práctica ampliada es optativa.

## Objetivo

Completar Q-learning tabular y discutir qué aprendió el agente. Punto de partida: [hands-on de Hugging Face](https://huggingface.co/learn/deep-rl-course/unit2/hands-on). Entornos: FrozenLake-v1 y Taxi-v3, con Gymnasium 1.2.1. No hace falta cuenta, certificación ni publicación en el Hub.

## En clase: 45 minutos

1. **Predicción, 5 min.** Antes de ejecutar, anticipar qué cambia al volver resbaladizo FrozenLake y qué podría provocar un bonus por permanecer vivo. Anotar una métrica que permita comprobar cada predicción.
2. **Actualización, 10 min.** Completar `q_update` y resolver los controles numéricos de `max` frente a `argmax` y de terminación frente a truncación.
3. **Experimento, 15 min.** Ejecutar el modo `demo` del notebook. Comparar recompensa original, éxito, duración y truncación. Una corrida corta no demuestra convergencia.
4. **Discusión, 10 min.** Contrastar lago resbaladizo con/sin bonus. Explicar una trayectoria y un posible error de evaluación. Si el modo corto no muestra el efecto, usar los resultados docentes completos.
5. **Cierre, 5 min.** Explicar una actualización y una limitación de las conclusiones. Compartir una decisión verificada, con o sin IA.

## Actualización que implementamos

\[
Q(s,a)\leftarrow Q(s,a)+\alpha[r+\gamma(1-\mathbb{1}_{terminated})\max_{a'}Q(s',a')-Q(s,a)].
\]

TimeLimit es un corte externo en esta actividad: conservamos bootstrap cuando solo se trunca. Si el horizonte es parte del MDP, hay que representar el tiempo restante. Evaluación: política fija y ninguna actualización de Q.

## Ampliación optativa

Cambiar `MODE` a `completo`: cinco semillas, 100.000 pasos por semilla y 100 episodios de evaluación por política/configuración. Los escenarios incluyen lago determinista, lago resbaladizo, Taxi y lago con bonus `+0.05` por transición no terminal. Mantener presupuesto, semillas, α, γ y ε al comparar el bonus. Evaluar con recompensa original y bonus desactivado.

Usar validación para decidir hiperparámetros y test una sola vez al final. Informar la dispersión entre entrenamientos y conservar resultados negativos. La [hoja de trabajo](INFORME_TEMPLATE.md) es opcional; no hay informe para entregar. Los checkpoints y registros pueden quedar como notas personales.

## IA opcional

Se puede usar IA para código o discusión y se puede resolver sin ella. En ambos casos comprobar propuestas con un ejemplo numérico o experimento. No se evalúa acceso a herramientas ni cantidad de prompts. La disciplina de formular predicciones y revisar evidencia se retomará en el único TP.
