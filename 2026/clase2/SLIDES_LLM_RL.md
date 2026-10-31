# Bloque LLM integrado en Clase 2

Revisión de continuidad: 14 slides nuevas se integran al flujo general y ocho quedan como profundización. Las explicaciones originales de DQN, políticas, actor-crítico, PPO y DT se conservan. Los tiempos del guion anterior centrado en GRPO quedan sin efecto. [Mapa actual](GUIA_PRESENTACION.md).

## 01 — ¿Cómo cambia una respuesta mediante RL? (slide 117, profundización)

Prompt: «¿Cómo empiezo con RL?»

A: «Aprende todo primero.»

B: «Probemos una acción y observemos el resultado.»

¿Qué acciones tomó el modelo? ¿Quién decide la recompensa?

Notas: Consulta optativa: repaso o ejercicio adicional; no reemplaza los fundamentos originales de la clase. Apertura, 3 min. Recuperar FrozenLake: acción, recompensa y cambio de política. No hay una preferencia universal preasignada a A/B: pedir criterios antes de votar. La pregunta guía es cómo cambia la probabilidad de generar cada respuesta.

## 02 — De una tabla a una política neuronal (slide 118, profundización)

Una red puede aproximar valores o producir probabilidades.

VALOR

Estado → red → Q(s, a)

Elegir usando los valores

POLÍTICA

Estado → red → π(a | s)

Muestrear una acción

Notas: Consulta optativa: repaso o ejercicio adicional; no reemplaza los fundamentos originales de la clase. Puente, 3 min. DQN aproxima Q; la política usa esos valores para elegir. En la otra rama la red parametriza directamente la distribución. Ambas requieren experiencia y evaluación. Continuamos por gradientes de política.

## 03 — Una respuesta es una trayectoria (slide 53, principal)

Estado: prompt + tokens ya generados

Acción: siguiente token

Transición: anexar ese token al prefijo

Final: EOS o presupuesto de generación

«¿Cómo empiezo?»  →  «hola»  →  «hola puedo»

¿Qué observa la política antes de elegir el próximo token?

Notas: Integrada en la clase general; los tiempos de la versión centrada en GRPO quedan sin efecto. MDP, 7 min. x es el prompt, s_t=(x,y_<t), a_t=y_t. Transición determinista al concatenar; selección estocástica. EOS es una acción válida. Con herramientas hay observaciones externas: no son acciones del modelo.

## 04 — Dos formas de mirar la misma respuesta (slide 54, principal)

MDP de tokens: una acción en cada paso.

Bandido contextual: la respuesta completa es una acción.

La probabilidad de la respuesta se factoriza por token.

log π(y | x) = Σₜ log π(yₜ | x, y₍<t₎)

Notas: Integrada en la clase general; los tiempos de la versión centrada en GRPO quedan sin efecto. 7 min. pi(y|x)=producto_t pi(y_t|x,y_<t); log pi(y|x)=suma_t log pi(y_t|x,y_<t). El bandido ayuda con REINFORCE de secuencia; el MDP con critic, máscaras y asignación de crédito. No extender esta simplificación a cualquier agente con herramientas.

## 05 — ¿Cuándo llega la recompensa? (slide 119, profundización)

Resultado: un score al terminar la respuesta.

Proceso: señales en pasos intermedios.

Con reward final y γ = 1, todos los tokens reciben la señal del resultado.

Una respuesta falla al final: ¿qué parte debería cambiar?

Notas: Consulta optativa: repaso o ejercicio adicional; no reemplaza los fundamentos originales de la clase. 5 min. El retorno final informa el gradiente de cada token, pero no prueba qué token fue responsable del resultado. Distinguir recompensa por paso de reward model aprendido. Preguntar dónde ubicar una señal cuando falla una herramienta.

## 06 — Reward model, critic y ventaja (slide 56, principal)

Son cantidades diferentes: puntuar no es predecir el retorno.

| Cantidad | Qué hace |
| --- | --- |
| Reward model | Asigna un score a la respuesta |
| Critic V(s) | Estima retorno esperado desde s |
| Ventaja A | Compara resultado y baseline |

Notas: Integrada en la clase general; los tiempos de la versión centrada en GRPO quedan sin efecto. 5 min. RM r_phi(x,y) puntúa una respuesta. Critic V_psi(s) estima retorno esperado desde un estado y puede servir de baseline. Ventaja compara con baseline, aprendido o calculado. En nuestra demo RM es un MLP pequeño, no otro LLM. No entrenamos critic.

## 07 — Optimizar sin alejarse demasiado (slide 57, principal)

La política aprende a obtener recompensa.

La referencia conserva el checkpoint de partida.

β controla cuánto penalizamos alejarnos de ella.

J(θ) = E[r(x, y)] − β KL(πθ || πref)

Notas: Integrada en la clase general; los tiempos de la versión centrada en GRPO quedan sin efecto. 4 min. Expectativa sobre prompts y respuestas de política actual. KL de distribuciones de respuestas condicionadas al prompt; en código usamos estimaciones por token/secuencia. Referencia congelada. Penalizar KL no garantiza corrección, seguridad o diversidad. En REINFORCE/RLOO la KL muestreada entra al retorno; GRPO usa un término diferenciable en la loss.

## 08 — REINFORCE: cambiar probabilidades (slide 120, profundización)

Ventaja positiva → aumentar la probabilidad.

Ventaja negativa → disminuir la probabilidad.

Muestrear respuestas propias forma parte del entrenamiento.

∇J ≈ (1/N) Σᵢ (Rᵢ − bᵢ) ∇ log πθ(yᵢ | xᵢ)

REINFORCE / RLOO · Ahmadian et al.

Notas: Consulta optativa: repaso o ejercicio adicional; no reemplaza los fundamentos originales de la clase. 6 min. Estimador del término de retorno; la regularización se trata según la implementación. En PyTorch minimizamos la loss negativa. R_i es el retorno usado en esa ruta, b_i baseline. Explicar derivada de logprob sin derivar a través del muestreo discreto. Fuente: Ahmadian et al.; no generalizar superioridad empírica.

## 09 — Baseline: ¿mejor que qué? (slide 121, profundización)

Un baseline reduce la varianza del gradiente.

Puede usar batches previos o un critic del estado.

La media que incluye mi propia respuesta depende de esa muestra.

¿Qué baseline usarías para la tercera respuesta de [0, 0, 1, 1]?

Notas: Consulta optativa: repaso o ejercicio adicional; no reemplaza los fundamentos originales de la clase. 15 min con actor-critic. Un baseline independiente de la acción no cambia el gradiente esperado de tarea bajo los supuestos usuales. detach evita gradientes, no dependencia estadística. Para [0,0,1,1], baseline leave-one-out de la tercera respuesta=1/3; ventaja=2/3. Desarrollar esta cuenta antes de GRPO.

## 10 — PPO: reutilizar un rollout (slide 122, profundización)

Generar respuestas con πold y guardar logprobabilidades.

Mantener πold y las ventajas fijas durante los updates.

Actualizar πθ varias veces y luego generar otro rollout.

ρₜ = πθ(aₜ | sₜ) / πold(aₜ | sₜ)

PPO · Schulman et al.

Notas: Consulta optativa: repaso o ejercicio adicional; no reemplaza los fundamentos originales de la clase. 10 min. rho por token y estado del rollout. PPO actor-critic suele usar critic/GAE. Aquí solo se calcula el surrogate y su gradiente: no se entrena PPO. Distinguir reutilización limitada de un rollout de replay arbitrario off-policy.

## 11 — Clipping: predecir cuatro casos (slide 58, principal)

ε = 0,2. ¿Cuándo queda activo el tramo plano?

| Ratio ρ | Ventaja A | ¿Tramo plano? |
| --- | --- | --- |
| 1,5 | +1 | Predecir |
| 0,5 | +1 | Predecir |
| 1,5 | −1 | Predecir |
| 0,5 | −1 | Predecir |

PPO · objetivo clipped

Notas: Integrada en la clase general; los tiempos de la versión centrada en GRPO quedan sin efecto. 5 min, calcular antes de revelar. Se maximiza min(rho*A, clip(rho,.8,1.2)*A). Respuestas por fila: 1.2, .5, -1.5, -.8. Tramo plano sí/no/no/sí. Fuera del intervalo no siempre desaparece el gradiente. Clipping no es cota estricta de KL. Ejecutar celda correspondiente.

## 12 — Tres políticas, tres funciones (slide 59, principal)

¿Qué denominador usa el ratio? ¿Qué política aparece en KL?

| Política | Función | Cuándo cambia |
| --- | --- | --- |
| Actual πθ | Se optimiza | En cada update |
| Vieja πold | Generó el rollout | Al renovar rollout |
| Referencia πref | Ancla al punto inicial | Fija en esta corrida |

Notas: Integrada en la clase general; los tiempos de la versión centrada en GRPO quedan sin efecto. 5 min. Al principio pueden coincidir; después de actualizar divergen. Vieja se renueva por rollout y queda fija durante sus updates. Referencia queda fija durante toda esta corrida. Ubicar el recreo según el desarrollo de la clase general.

## 13 — La recompensa y el algoritmo son dos ejes (slide 55, principal)

REINFORCE, RLOO, PPO y GRPO pueden usar distintas señales.

| Señal | Cómo obtenemos el score |
| --- | --- |
| Preferencias humanas | Personas → pares → reward model |
| Verificador | Criterio ejecutable → score |
| Etiquetas simuladas | Reglas → pares → RM de ensayo |

¿Qué falta en la frase «GRPO es RLHF sin reward model»?

Notas: Integrada en la clase general; los tiempos de la versión centrada en GRPO quedan sin efecto. 10 min. RLHF: personas generan comparaciones, se aprende RM y luego RL. RLVR: verificador ejecutable. Simulado: reglas etiquetan pares para probar pipeline; no describirlo como preferencias humanas. GRPO no define origen de reward. DPO se verá luego como objetivo directo sobre pares.

## 14 — Etiquetar preferencias exige un criterio (slide 123, profundización)

Prompt: «¿Qué significa explorar en RL?»

A: «Probar acciones para obtener información.»

B: «Elegir a veces algo distinto de lo que hoy parece mejor.»

Opciones: A, B, empate u omitir.

¿Para quién resulta más útil cada respuesta?

Notas: Consulta optativa: repaso o ejercicio adicional; no reemplaza los fundamentos originales de la clase. 10 min. Ambas respuestas pueden ser razonables. Registrar criterio y desacuerdos; no asignar una preferencia universal. Preparar 24 pares sin nombres, separar pares para RM del juicio ciego de nuevas generaciones. Los seis pares reservados en demo comparten contextos/frases y no son test semántico fuerte.

## 15 — RLOO: comparar sin incluirme (slide 60, principal)

Generar G respuestas al mismo prompt, con G ≥ 2.

Comparar cada recompensa con la media de las otras.

[0, 0, 1, 1] → [−2/3, −2/3, +2/3, +2/3]

Aᵢ = Rᵢ − promedio de las otras recompensas

RLOO · Ahmadian et al.

Notas: Integrada en la clase general; los tiempos de la versión centrada en GRPO quedan sin efecto. 10 min. Baseline leave-one-out excluye muestra propia. No mezclar preguntas distintas. En laboratorio se hace un update on-policy por batch, sin critic. Para la ruta regularizada usar el retorno correspondiente. Distinguir este estimador de normalizar por std en GRPO.

## 16 — GRPO: una ventaja relativa al grupo (slide 61, principal)

Comparar respuestas al mismo prompt.

Centrar y normalizar recompensas, sin critic aprendido.

[0, 0, 1, 1] → [−1, −1, +1, +1]

Aᵢ = (rᵢ − media(r)) / (std(r) + ε)

GRPO · DeepSeekMath

Notas: Integrada en la clase general; los tiempos de la versión centrada en GRPO quedan sin efecto. 10 min. Usamos std poblacional en demo, epsilon numérico. La ventaja de resultado es la misma para todos los tokens de esa respuesta. Eliminar critic requiere generar grupos; no vuelve gratuito el entrenamiento. GRPO combina estas ventajas con el mecanismo de actualización siguiente.

## 17 — GRPO: juntar las piezas (slide 62, principal)

Ventaja de grupo + ratio por token + clipping + KL.

Promediar tokens de cada respuesta y luego el batch.

Rewards iguales → ventaja de tarea cero; KL aún puede actuar.

max  mediaᵢ mediaₜ [min(ρᵢₜ Aᵢ, clip(ρᵢₜ) Aᵢ) − β k3ᵢₜ]

GRPO · objetivo simplificado de DeepSeekMath

Notas: Integrada en la clase general; los tiempos de la versión centrada en GRPO quedan sin efecto. 10 min. Objetivo original simplificado para recompensa final; no DAPO ni Dr. GRPO. Loss implementada: media_i media_t[-min(rho_it*A_i,clip(rho_it,1-eps,1+eps)*A_i)+beta*k3_it]. Tokens válidos incluyen EOS y excluyen prompt/PAD. k3=exp(logpref-logp)-(logpref-logp)-1; muestreo viejo no hace exacta la KL actual. Ratio con old; k3 con ref. GRPO loss diferenciable, old/ref congeladas.

## 18 — Los detalles del código también enseñan (slide 124, profundización)

Desplazar entradas y targets un token.

Excluir prompt y PAD; incluir EOS.

Guardar logprobs viejas y congelar la referencia.

Muestrear y puntuar con la misma distribución.

¿Los tokens que devuelve una herramienta son acciones de la política?

Notas: Consulta optativa: repaso o ejercicio adicional; no reemplaza los fundamentos originales de la clase. Profundización opcional del laboratorio. Mostrar máscaras y teacher forcing. k1 puede dar negativo en muestra finita; k3 es no negativo por acción. No agregar top-k o temperatura sin adaptar evaluación de probabilidades. Ver asserts del notebook; gradientes solo por política actual.

## 19 — Laboratorio: misma base, distintos updates (slide 63, principal)

GRU de 13.923 parámetros; vocabulario cerrado.

Ejercicio validado en CPU. Ensayo en Orin pendiente.

Preferencias simuladas por defecto; humanas como alternativa.

SFT + RM  →  Tres copias  →  REINFORCE / RLOO / GRPO

Abrir notebook de gradientes de política

Notas: Integrada en la clase general; los tiempos de la versión centrada en GRPO quedan sin efecto. Laboratorio con duración a ensayar dentro de la clase general; usar etapas preparadas cuando haga falta. SFT y RM preparados. Abrir notebook, no entrenar ambos laboratorios completos. El generador es GRU, no transformer preentrenado. REINFORCE/RLOO/GRPO usan 1440 respuestas de train; GRPO 180 updates, otros 60. No ranking de superioridad. Jetson recuperó acceso el 31/10; falta validar rendimiento. Notebook alternativo mini_rlhf profundiza RM/etiquetado.

## 20 — Más recompensa: ¿mejores respuestas? (slide 64, principal)

Contrato: prefijo, longitud, EOS y no repetición.

«claro vete» puede cumplirlo sin ser buena asistencia.

Comparar reward, KL, diversidad y juicio humano ciego.

¿Qué contraejemplo buscarías aunque la curva de reward suba?

Notas: Integrada en la clase general; los tiempos de la versión centrada en GRPO quedan sin efecto. 15 min. Evaluar antes/después con nuevos textos, criterio explícito, ocultar identidad de política y revelar al final. Comparar contrato y RM aun si optimizamos solo uno. No llamar RLVR de razonamiento a verificación de formato. Para comparar algoritmos igualar presupuestos, ajustar en validación y repetir semillas.

## 21 — DPO y variantes: qué cambia (slide 65, principal)

DPO optimiza directamente pares y referencia.

No entrena explícitamente un RM seguido de RL on-policy.

En variantes de GRPO cambian normalización, ratios o agregación.

DPO · Rafailov et al.

Notas: Integrada en la clase general; los tiempos de la versión centrada en GRPO quedan sin efecto. 10 min. DPO fuente Rafailov et al. Para variantes elegir una pregunta, sin otro laboratorio. DAPO https://arxiv.org/abs/2503.14476; Dr. GRPO https://arxiv.org/abs/2503.20783; GSPO https://arxiv.org/abs/2507.18071. Lecturas optativas ya documentadas. No presentar una variante como universalmente superior ni usar defaults sin inspección.

## 22 — De políticas a secuencias de decisiones (slide 66, principal)

En RL on-policy, la política genera experiencia nueva.

En offline RL, aprendemos de trayectorias registradas.

Un Transformer también puede aprender a predecir acciones.

El return deseado puede formar parte de su contexto.

Notas: Puente hacia el bloque de Offline RL y Decision Transformer, que sigue en el recorrido principal. La política de un LLM genera tokens y se actualiza a partir de nuevas respuestas. En DT entrenamos una política condicionada al return-to-go mediante predicción supervisada de acciones sobre trayectorias registradas. Compartir arquitectura secuencial no implica compartir objetivo de entrenamiento. No confundir DT con aplicar GRPO a un Transformer. Preguntar cómo representa la secuencia estados, acciones y returns. La explicación de DT pertenece a esta clase; sólo el lanzamiento del TP queda para el encuentro 4.
