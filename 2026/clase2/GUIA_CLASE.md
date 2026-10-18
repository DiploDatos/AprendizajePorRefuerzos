# Clase 2 — de gradientes de política a GRPO

Revisión del 12/10/2026 según la orientación indicada por Juan. “Clase 2” es el bloque de diapositivas de deep RL. Juan confirmó su desarrollo en el encuentro 3, segundo fin de semana. La distribución detallada de las cuatro horas sigue como propuesta.

La [guía de Cameron R. Wolfe](https://cameronrwolfe.substack.com/p/llm-rl) sirve como hilo de lectura: fundamentos de RL para generación, gradientes de política, algoritmos y problemas de implementación. Juan aclaró que el propósito es mejorar el camino hacia GRPO. Los temas se presentan como respuestas a problemas sucesivos y convergen en ese algoritmo. Las ecuaciones y decisiones de implementación se contrastan con los artículos originales; las variantes recientes quedan como lecturas optativas.

## Criterio de progresión

Seguir el [camino hacia GRPO](CAMINO_A_GRPO.md): política parametrizada → REINFORCE → baseline/ventaja; desde allí, construir tanto el mecanismo de actualización PPO como la comparación entre respuestas del mismo prompt. GRPO reúne esas piezas. Mantener los fundamentos del bloque deep RL y usar generación de texto como ejemplo continuo.

## Resultado que buscamos

Al terminar, cada estudiante puede representar generación autoregresiva como RL, calcular una ventaja, explicar por qué necesitamos política actual/vieja/referencia y reconstruir GRPO y justificar por qué aparecen ventajas de grupo, ratios y clipping. También puede distinguir fuente de recompensa, actualización y evaluación: GRPO no define por sí mismo si un experimento es RLHF o RLVR.

Prerrequisitos: MDP, retorno, políticas estocásticas, derivación elemental y log-probabilidad. No exigir experiencia con Transformers. Abrir con una transición de FrozenLake de la actividad Q-learning y trasladar estado/acción/recompensa a una respuesta de texto.

## Guion principal: 240 minutos

| Minutos | Contenido | Acción o pregunta |
|---|---|---|
| 0–20 | De Q-table a redes: DQN y política parametrizada | ¿Predecir valor y producir una distribución son lo mismo? DQN queda como puente, no como otro laboratorio |
| 20–40 | Generación como MDP y como bandido contextual | Identificar prompt, prefijo, token, EOS y respuesta; ¿qué cambia con herramientas? |
| 40–60 | Objetivo y REINFORCE | Derivar gradiente de logprob por retorno y predecir su signo |
| 60–75 | Baseline, ventaja y actor-critic | ¿Cómo bajar varianza? Separar critic de RM y justificar el baseline |
| 75–95 | PPO y reutilización de rollouts | Calcular cuatro casos de clipping; señalar política vieja versus referencia |
| 95–110 | Recreo | |
| 110–130 | Comparación de respuestas al mismo prompt | Construir un baseline sin critic; preparar grupos y aclarar origen del reward |
| 130–140 | RLOO como puente | Calcular baseline excluyendo la propia respuesta y contrastar con media del grupo |
| 140–160 | Ensamblar GRPO | Combinar ventaja normalizada y surrogate clipped; caso de rewards iguales |
| 160–205 | Notebook de gradientes de política | Partir del SFT/RM preparados; seguir ventajas, old/ref, máscaras y actualización GRPO |
| 205–220 | Evaluación antes/después | Juicio ciego y diversidad; reconocer presupuestos distintos y límites del proxy |
| 220–230 | DPO y variantes de GRPO | Comparar objetivos; un problema concreto de longitud, normalización o clipping |
| 230–240 | Cierre individual y puente a agentes | Explicar una máscara y clasificar un experimento; observaciones externas no son acciones de la política |

Si no hay cuatro horas disponibles, usar una versión de 90 minutos: 15 generación/objetivo, 15 REINFORCE/baseline, 10 PPO, 15 RLOO/GRPO, 25 notebook con etiquetas/checkpoints preparados y 10 evaluación. Reducir DQN, DPO y variantes a lectura. La selección temporal debe acordarse con Jorge para conservar continuidad con el único TP de Decision Transformer.

Las preferencias pueden etiquetarse en una actividad previa y el SFT/RM prepararse antes de proyectar. Así el laboratorio dedica tiempo al camino de actualización y a GRPO; la anotación humana completa sigue disponible en la alternativa mini RLHF.

## Material para proyectar

- [Storyboard listo para trasladar a slides](SLIDES_LLM_RL.md): contenido y notas docentes de 22 diapositivas; correspondencia con el deck histórico en [plan de edición](../docentes/SLIDES_2026.md).
- [Notebook principal](llm_policy_gradients.ipynb): REINFORCE, RLOO y GRPO ejecutables; ejemplo numérico de PPO, sin entrenamiento PPO. Implementación propia en [llm_policy_gradients.py](llm_policy_gradients.py).
- [Notebook de preferencias](mini_rlhf.ipynb) y [guía detallada de 65 minutos](MINI_RLHF_GUIA.md): alternativa si se decide profundizar exclusivamente en anotación, RM y REINFORCE. No sumar ambos laboratorios completos dentro del mismo encuentro.
- [Preparación Orin](JETSON.md) y [extensión con LLM preentrenado](ORIN_LLM_PLAN.md). Equipo desconectado: validación remota aplazada a pedido de Juan. No es un prerrequisito para revisar/usar el material CPU.

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
