# Clase 2 — deep RL, modelos de lenguaje y Decision Transformer

Revisión del 31/10/2026 según la orientación indicada por Juan. “Clase 2” es el bloque de diapositivas de deep RL. Juan confirmó su desarrollo en el encuentro 3, segundo fin de semana. La clase mantiene el flujo general; el desarrollo de LLMs se integra a él. La distribución detallada entre los encuentros 3–4 debe ensayarse; sólo el TP se presenta recién en el 4.

La [guía de Cameron R. Wolfe](https://cameronrwolfe.substack.com/p/llm-rl) sirve como hilo de lectura: fundamentos de RL para generación, gradientes de política, algoritmos y problemas de implementación. Juan aclaró que el propósito es mejorar el camino hacia GRPO. Los temas se presentan como respuestas a problemas sucesivos y convergen en ese algoritmo. Las ecuaciones y decisiones de implementación se contrastan con los artículos originales; las variantes recientes quedan como lecturas optativas.

## Criterio de progresión

Seguir el [camino hacia GRPO](CAMINO_A_GRPO.md): política parametrizada → REINFORCE → baseline/ventaja; desde allí, construir tanto el mecanismo de actualización PPO como la comparación entre respuestas del mismo prompt. GRPO reúne esas piezas. Mantener los fundamentos del bloque deep RL y usar generación de texto como ejemplo continuo.

## Resultado que buscamos en el subbloque de LLM

Al terminar, cada estudiante puede representar generación autoregresiva como RL, calcular una ventaja, explicar por qué necesitamos política actual/vieja/referencia y reconstruir GRPO y justificar por qué aparecen ventajas de grupo, ratios y clipping. También puede distinguir fuente de recompensa, actualización y evaluación: GRPO no define por sí mismo si un experimento es RLHF o RLVR.

Prerrequisitos: MDP, retorno, políticas estocásticas, derivación elemental y log-probabilidad. No exigir experiencia con Transformers. Abrir con una transición de FrozenLake de la actividad Q-learning y trasladar estado/acción/recompensa a una respuesta de texto.

## Recorrido de la clase completa

| Bloque | Continuidad |
|---|---|
| Tablas → redes → DQN | Por qué aproximar valores y cómo usa experiencia DQN |
| Gradientes de política | Aprender directamente una distribución y estimar su gradiente |
| Actor-crítico y PPO | Baseline, función de valor y límites de la actualización |
| RL en modelos de lenguaje | Reutilizar esos fundamentos para generación, RLHF/RLVR y RLOO/GRPO |
| Offline RL y Decision Transformer | Cambiar de experiencia nueva a trayectorias registradas; predecir acciones condicionadas por return |
| Aplicaciones y conclusiones | Retomar los conceptos en juegos, recomendación, control y robótica |

GRPO es una ampliación dentro del cuarto bloque. Los fundamentos originales ya presentan REINFORCE y PPO: los repasos nuevos redundantes quedan como consulta, no como un reinicio de la clase. Se conservan las aplicaciones en el recorrido. No imponer un guion de cuatro horas centrado exclusivamente en GRPO.

La [guía de presentación](GUIA_PRESENTACION.md) contiene los números actuales. Ajustar tiempos y reparto con Jorge sin omitir bloques completos. La demo puede proyectar etapas/resultados preparados y profundizarse en una actividad posterior. El TP se presenta en el encuentro 4, pero su explicación teórica de DT no queda postergada por esa regla.

## Material para proyectar

- [Slides aplicadas](SLIDES_LLM_RL.md): 22 nuevas, de las cuales 14 se integran al recorrido general y ocho quedan como profundización; [guía de presentación y números actuales](GUIA_PRESENTACION.md). El PDF completo conserva teoría y apéndice, excluyendo sólo el TP.
- [Notebook principal](llm_policy_gradients.ipynb): REINFORCE, RLOO y GRPO ejecutables; ejemplo numérico de PPO, sin entrenamiento PPO. Implementación propia en [llm_policy_gradients.py](llm_policy_gradients.py).
- [Notebook de preferencias](mini_rlhf.ipynb) y [guía detallada de 65 minutos](MINI_RLHF_GUIA.md): alternativa si se decide profundizar exclusivamente en anotación, RM y REINFORCE. No sumar ambos laboratorios completos dentro del mismo encuentro.
- [Preparación Orin](JETSON.md) y [extensión con LLM preentrenado](ORIN_LLM_PLAN.md). Acceso a la placa recuperado según Juan el 31/10; validación remota del entrenamiento todavía pendiente. No es un prerrequisito para revisar/usar el material CPU.

## Ideas que deben quedar claras

Para prompt $x$, $s_t=(x,y_{<t})$, $a_t=y_t$ y $\pi_\theta(a_t\mid s_t)$ modela el siguiente token. Con recompensa de resultado y $\gamma=1$, el resultado final informa todas las acciones de la respuesta. Un score por paso cambia la asignación de crédito. Contexto, PAD y observaciones de herramientas no deben contribuir como acciones generadas.

El objetivo docente es $J(\theta)=\mathbb E[r(x,y)]-\beta D_{KL}(\pi_\theta\Vert\pi_{ref})$. La referencia conserva el comportamiento inicial; beta controla esa presión. No garantiza corrección, diversidad ni alineación. En nuestra ruta REINFORCE/RLOO, la penalización muestreada entra al retorno; en GRPO entra como término diferenciable de loss.

REINFORCE utiliza una estimación Monte Carlo del gradiente con retorno menos baseline. El baseline debe ser independiente de la acción para mantener la justificación del estimador; usar media que incluye la propia muestra introduce dependencia. Nuestra REINFORCE usa un promedio de batches anteriores. RLOO compara muestras del mismo prompt excluyendo la propia recompensa del promedio; ver [Ahmadian et al.](https://arxiv.org/abs/2402.14740).

PPO añade una forma de limitar actualizaciones mediante ratios y clipping; en la variante actor-critic se estiman ventajas con una función de valor. $\pi_{old}$ generó los rollouts y queda fija mientras los reutilizamos. $\pi_{ref}$ ancla a un checkpoint y puede permanecer fija entre muchos rollouts. El critic predice retorno esperado; el RM puntúa resultados. El clipping es direccional y no impone una cota estricta de KL; ver [PPO](https://arxiv.org/abs/1707.06347).

GRPO estima ventajas mediante grupos de respuestas a la misma pregunta, prescindiendo del critic. La [formulación original de DeepSeekMath](https://arxiv.org/abs/2402.03300) combina ventajas normalizadas, ratios por token, clipping y regularización KL en la loss. Eso es lo que ejecutamos, con recompensa final y un generador diminuto. Requiere más muestras por prompt; eliminar critic no vuelve gratuito el entrenamiento.

## Dos ejes para clasificar un experimento

| Fuente de señal | Cómo se obtiene | Algoritmos posibles |
|---|---|---|
| Demostraciones | Texto objetivo | SFT, antes del loop RL |
| Preferencias humanas | Personas → comparaciones → reward model | REINFORCE, RLOO, PPO o GRPO para RLHF |
| Preferencias simuladas | Etiquetas generadas por reglas para el ensayo | Mismo loop, identificado como simulación |
| Verificador | Criterio ejecutable de formato, respuesta correcta o test | REINFORCE, RLOO, PPO o GRPO para RLVR |
| Pares preferido/rechazado para objetivo directo | Dataset y referencia | DPO: sin entrenar explícitamente RM + loop RL; [paper](https://arxiv.org/abs/2305.18290) |

Un verificador de formato permite evaluar exactamente **ese contrato**, no utilidad general. El notebook usa prefijos, longitud, ausencia de repeticiones y EOS: “claro vete” puede aprobar. Pedir ese contraejemplo antes de mirar la curva. En la ruta aprendida, las etiquetas simuladas sirven para probar software; recoger preferencias reales es necesario para describir la corrida como RLHF humano.

## Qué mirar en la demo

El modelo es una GRU de 13.923 parámetros entrenada primero con 18 frases; RM de 1.089 parámetros con bolsa de palabras. No tiene conocimiento general ni capacidades comparables a un LLM preentrenado. El notebook enseña mecanismos de generación y ajuste; la transferencia a un transformer pequeño requiere otro ensayo.

Comparar checkpoint SFT, REINFORCE, RLOO y GRPO con 180 muestras por política y tres prompts conocidos. Registrar proxy, KL de secuencia muestreada, diversidad, longitud, EOS y clipping. Medir también el contrato cuando se optimiza un RM. Cada corrida usa 1.440 respuestas de train, pero GRPO hace 180 actualizaciones contra 60 de los otros métodos: no declarar ganador sin igualar presupuesto, ajustar hiperparámetros en validación y repetir semillas.

Cuando un grupo tiene rewards idénticos, la ventaja de tarea GRPO es cero; la KL todavía puede actuar. k1 puede dar una media negativa en pocas muestras; k3 es no negativo por acción, pero una muestra antigua no transforma el surrogate en una medida exacta de KL actual. Mantener esta diferencia visible.

## Actividad y evaluación compatibles con IA

Dar a todos la afirmación: “GRPO es RLHF sin reward model”. En diez minutos deben corregirla, clasificar dos corridas del notebook y presentar un contraejemplo ejecutado. Pueden consultar IA o fuentes; todos entregan evidencia y explican individualmente estado, acción, señal y update. No puntuar el uso de herramientas.

Pregunta de salida: “Un grupo tiene rewards [1,1,1,1] y la política difiere de referencia. ¿Hay gradiente de tarea? ¿Puede haber actualización?”. Respuesta esperada: ventaja de tarea cero; regularización KL puede actualizar. No aceptar “no aprende nada” sin especificar qué término se considera.

## Profundización posterior

| Tema | Pregunta de lectura | Fuente primaria |
|---|---|---|
| DAPO | ¿Cómo afectan clipping, grupos sin señal y agregación de tokens? | [DAPO](https://arxiv.org/abs/2503.14476) |
| Dr. GRPO | ¿Qué cambia al quitar escalado por std y usar denominador fijo? | [Understanding R1-Zero-Like Training](https://arxiv.org/abs/2503.20783) |
| GSPO | ¿Qué significa usar un ratio por secuencia? | [GSPO](https://arxiv.org/abs/2507.18071) |
| Implementación vigente | ¿Qué loss y beta usa realmente la biblioteca elegida? | [Documentación TRL](https://huggingface.co/docs/trl/main/grpo_trainer), consultada 12/10/2026; no asumir que sus defaults replican GRPO original |
| Agentes | ¿Quién produjo cada token: política o entorno? ¿Dónde aparece el crédito de largo plazo? | Extensión conceptual sobre el MDP del curso; práctica con herramientas fuera de este laboratorio |

TIS/CISPO, entrenamiento asíncrono, aprendizaje continuo y escalado quedan como mapa de investigación. No derivarlos ni convertir afirmaciones empíricas de trabajos particulares en garantías generales.
