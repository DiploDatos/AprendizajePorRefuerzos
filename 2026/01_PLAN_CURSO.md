# Plan de curso 2026

Revisión 31/10/2026. Propuesta de cuatro encuentros de cuatro horas, basada en Organización 2025. Juan ubicó el desarrollo de RL para LLMs en el encuentro 3 (segundo fin de semana). Hay un único TP: Decision Transformer, que se presenta recién en el encuentro 4. No anunciarlo ni mostrar la consigna en los encuentros 1–3. Fechas, duración detallada y reparto docente pendientes de acordar; no reutilizar fechas de 2025.

| Encuentro | Minutos | Contenido y actividad | Resultado |
|---|---:|---|---|
| 1 | 0–40 | Juan: motivación, dos slides RL en LLMs e interfaz agente–entorno | Reconocer dónde aplicar RL |
| 1 | 40–100 | Jorge: MDP, retorno, valor y Bellman | Resolver una transición a mano |
| 1 | 100–115 | Recreo | |
| 1 | 115–165 | Juan: políticas y Lab 0 | Explicar política y objetivo |
| 1 | 165–205 | Exploración, bandidos, panorama de métodos | Comparar explorar/explotar |
| 1 | 205–240 | Preparación del laboratorio, ejercicios y consultas | Llegar al encuentro 2 con el entorno y las nociones básicas |
| 2 | 0–40 | Juan: bandidos, ε-greedy, softmax | Vincular distribución y decisiones |
| 2 | 40–110 | Jorge: TD, SARSA y Q-learning | Comprender on/off-policy y bootstrap |
| 2 | 110–125 | Recreo | |
| 2 | 125–170 | Actividad formativa FrozenLake/Taxi: predicción, implementación y discusión | Una actualización y una limitación explicadas |
| 2 | 170–210 | Puesta en común con resultados docentes: recompensa y solución defectuosa | Un contraejemplo reproducible, sin entrega |
| 2 | 210–240 | Ética en RL: cuatro casos, actividad de preferencias y cierre (28 min + 2 de margen) | Justificar un control y reconocer límites de la evidencia |
| 3 | Distribución a ensayar | Deep RL: redes, DQN, gradientes de política, actor-crítico y PPO | Conectar valor, política y aproximación |
| 3 | Integrado al recorrido | RL en modelos de lenguaje: RLHF/RLVR, RLOO/GRPO y demo acotada | Entender la actualización sin desplazar los otros temas |
| 3 | Mantener en el hilo | Offline RL, modelado de secuencias, Decision Transformer y aplicaciones | Distinguir aprendizaje on-policy y desde trayectorias registradas |
| 4 | 0–45 | Profundización de DT y evaluación desde datos registrados | Retomar condicionamiento por retorno y preparar el TP |
| 4 | 45–100 | Aplicaciones seleccionadas; robótica y transferencia | Analizar supuestos y límites |
| 4 | 100–115 | Recreo | |
| 4 | 115–180 | Único TP: Decision Transformer / clínica de evaluación | Objetivo y métricas coherentes |
| 4 | 180–240 | Plan de experimentos del TP, consultas y cierre | Próximos pasos de cada grupo |

## Qué priorizar

La sección 2026 del Programa abarca más contenidos que los que caben con profundidad en 16 horas. Núcleo evaluable: MDP, Q-learning/SARSA, evaluación experimental, distinción valor/política, gradientes de política, diferencias REINFORCE/PPO/RLOO/GRPO, origen de señal RLHF/RLVR y lectura crítica de resultados. DPO, SAC, variantes avanzadas y agentes con herramientas quedan como comparación conceptual y lecturas; Decision Transformer es el único TP, según la decisión de Juan. Los otros temas se trabajan en actividades formativas y preguntas de clase; no agregan informes, defensas ni entregas obligatorias.

## Ubicación de la demo confirmada por Juan

**Deck Clase 2 / bloque deep RL:** la guía de Wolfe orienta la ampliación sobre LLMs y la progresión hacia GRPO, no reemplaza el recorrido completo. Se conservan redes, DQN, gradientes de política, actor-crítico/PPO, Offline RL, Decision Transformer y aplicaciones. La explicación de DT forma parte de la clase; sólo la presentación del TP corresponde al encuentro 4.

La [guía de presentación](clase2/GUIA_PRESENTACION.md) refleja el flujo corregido. Hay 115 páginas principales y 39 de profundización, más la slide del TP omitida. El PDF contiene las 154 páginas sin el TP. Esto es un banco de material con continuidad, no una promesa de cubrir 115 slides y un laboratorio de 45 minutos en cuatro horas. Ajustar selección, ritmo y distribución entre encuentros 3–4 con Jorge mediante ensayo; no relegar bloques completos para hacer lugar a GRPO.

El laboratorio [llm_policy_gradients.ipynb](clase2/llm_policy_gradients.ipynb) puede usar SFT/RM y resultados preparados. Mini RLHF es una alternativa de profundidad, no otra práctica obligatoria. Juan informó que recuperó acceso a Orin; su diagnóstico y entrenamiento siguen pendientes.

## Continuidad entre actividades

Actividad Q-learning: “¿El agente aprendió lo que digo que aprendió?”. Clase 2: “¿Cómo aprendemos valores, políticas y acciones, con experiencia nueva o registrada?”. En su bloque LLM: “¿Cómo construimos GRPO y qué mide su recompensa?”. Único TP: “¿Mi evaluación mide decisiones útiles fuera de los datos observados?”. Los tres problemas comparten semillas, baselines, controles y reconocimiento de incertidumbre.

## Introducción renovada

El primer encuentro usa [dos slides introductorias](clase1/SLIDES_INTRO_LLM.md), de 6–8 minutos, en lugar del bloque histórico de overhype/HARKing. Presenta preferencias → RLHF y comparación de respuestas → GRPO. La pregunta por estado, acción y recompensa abre los fundamentos; el encuentro 3 retoma el mismo ejemplo para desarrollar el algoritmo.

## Ética renovada y publicación del TP

El cierre del encuentro 2 usa las slides 80–86 de Clase 1: Navier–Stokes (crédito, confianza y prioridades de investigación), reward hacking, el incidente OpenAI–Hugging Face de 2026 y pluralidad de preferencias. La [guía docente](clase1/ETICA_RL.md) distingue fuentes, declaraciones e incertidumbre y propone una votación A/B de seis minutos. El bloque vuelve sobre el laboratorio de Q-learning y prepara la discusión de recompensas del encuentro 3.

El índice y los archivos del TP se conservan para preparación docente. Seleccionar el material estudiantil de cada encuentro: la presentación y publicación de la consigna del único TP corresponden al encuentro 4. Esta decisión no modifica permisos de Drive ni elimina los archivos de trabajo.
