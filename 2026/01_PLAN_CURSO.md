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
| 3 | 0–20 | Puente Q-table/DQN → política neuronal | Distinguir valor y distribución de acciones |
| 3 | 20–60 | Juan: generación como RL, retorno, ventaja y KL | Mapear prompt/prefijo/token/EOS |
| 3 | 60–95 | REINFORCE, baseline y PPO | Explicar gradiente y clipping con números |
| 3 | 95–110 | Recreo | |
| 3 | 110–160 | Preferencias/verificador, RLOO y GRPO | Separar fuente de reward y algoritmo |
| 3 | 160–205 | Notebook REINFORCE/RLOO/GRPO con generador pequeño | Ajuste desde mismo SFT; cambiar reward |
| 3 | 205–240 | Evaluación ciega, DPO/variantes y puente a agentes | Reconocer proxy, presupuestos y máscaras |
| 4 | 0–45 | Offline RL y DT | Entender condicionamiento por retorno |
| 4 | 45–100 | Aplicaciones seleccionadas; robótica y transferencia | Analizar supuestos y límites |
| 4 | 100–115 | Recreo | |
| 4 | 115–180 | Único TP: Decision Transformer / clínica de evaluación | Objetivo y métricas coherentes |
| 4 | 180–240 | Plan de experimentos del TP, consultas y cierre | Próximos pasos de cada grupo |

## Qué priorizar

La sección 2026 del Programa abarca más contenidos que los que caben con profundidad en 16 horas. Núcleo evaluable: MDP, Q-learning/SARSA, evaluación experimental, distinción valor/política, gradientes de política, diferencias REINFORCE/PPO/RLOO/GRPO, origen de señal RLHF/RLVR y lectura crítica de resultados. DPO, SAC, variantes avanzadas y agentes con herramientas quedan como comparación conceptual y lecturas; Decision Transformer es el único TP, según la decisión de Juan. Los otros temas se trabajan en actividades formativas y preguntas de clase; no agregan informes, defensas ni entregas obligatorias.

## Ubicación de la demo confirmada por Juan

**Deck Clase 2 / bloque deep RL:** Juan pidió que el hilo conductor siga la [guía de RL para LLMs de Wolfe](https://cameronrwolfe.substack.com/p/llm-rl). El desarrollo se ubica en el tercer encuentro; se propone usar el bloque completo para mejorar la progresión hacia GRPO: gradiente de política, baseline/ventaja, PPO y comparación por grupos. Se reservan 45 minutos de notebook; preferencias y SFT/RM pueden prepararse antes. [Guion y alternativa de 90 minutos](clase2/GUIA_CLASE.md).

Las [slides de Clase 2](clase2/GUIA_PRESENTACION.md) ya tienen 32 páginas principales para el encuentro 3; el material del encuentro 4 está omitido al final. El laboratorio principal es [llm_policy_gradients.ipynb](clase2/llm_policy_gradients.ipynb). La demo de mini RLHF de 65 minutos queda como alternativa de profundidad en preferencias, no una segunda práctica obligatoria. Juan informó que recuperó acceso a la Orin el 31/10. Su diagnóstico y entrenamiento siguen pendientes; esta revisión trabaja sobre las slides.

## Continuidad entre actividades

Actividad Q-learning: “¿El agente aprendió lo que digo que aprendió?”. Encuentro 3: “¿Cómo construimos y justificamos una actualización GRPO, y qué mide su recompensa?”. Único TP: “¿Mi evaluación mide decisiones útiles fuera de los datos observados?”. Los tres problemas comparten semillas, baselines, controles y reconocimiento de incertidumbre.

## Introducción renovada

El primer encuentro usa [dos slides introductorias](clase1/SLIDES_INTRO_LLM.md), de 6–8 minutos, en lugar del bloque histórico de overhype/HARKing. Presenta preferencias → RLHF y comparación de respuestas → GRPO. La pregunta por estado, acción y recompensa abre los fundamentos; el encuentro 3 retoma el mismo ejemplo para desarrollar el algoritmo.

## Ética renovada y publicación del TP

El cierre del encuentro 2 usa las slides 80–86 de Clase 1: Navier–Stokes (crédito, confianza y prioridades de investigación), reward hacking, el incidente OpenAI–Hugging Face de 2026 y pluralidad de preferencias. La [guía docente](clase1/ETICA_RL.md) distingue fuentes, declaraciones e incertidumbre y propone una votación A/B de seis minutos. El bloque vuelve sobre el laboratorio de Q-learning y prepara la discusión de recompensas del encuentro 3.

El índice y los archivos del TP se conservan para preparación docente. Seleccionar el material estudiantil de cada encuentro: la presentación y publicación de la consigna del único TP corresponden al encuentro 4. Esta decisión no modifica permisos de Drive ni elimina los archivos de trabajo.
