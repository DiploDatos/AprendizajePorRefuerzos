# Clase 2: contenido aplicado y notas docentes

Revisión 31/10/2026. [Deck nativo actualizado](https://docs.google.com/presentation/d/19hKgrRzdaDW6sAlVUABjZJeOeElbK6Z1q1QkRfttv4w/edit). Las 22 páginas nuevas se integran en 32 principales para el encuentro 3. [Mapa y tiempos](GUIA_PRESENTACION.md). Fuente orientadora: [Wolfe](https://cameronrwolfe.substack.com/p/llm-rl); las referencias técnicas primarias están en las slides y sus notas.

## 01 — ¿Cómo cambia una respuesta mediante RL? (slide 3)

Prompt: «¿Cómo empiezo con RL?»

A: «Aprende todo primero.»

B: «Probemos una acción y observemos el resultado.»

¿Qué acciones tomó el modelo? ¿Quién decide la recompensa?

Notas: Apertura, 3 min. Recuperar FrozenLake: acción, recompensa y cambio de política. No hay una preferencia universal preasignada a A/B: pedir criterios antes de votar. La pregunta guía es cómo cambia la probabilidad de generar cada respuesta.

## 02 — De una tabla a una política neuronal (slide 12)

Una red puede aproximar valores o producir probabilidades.

VALOR

Estado → red → Q(s, a)

Elegir usando los valores

POLÍTICA

Estado → red → π(a | s)

Muestrear una acción

Notas: Puente, 3 min. DQN aproxima Q; la política usa esos valores para elegir. En la otra rama la red parametriza directamente la distribución. Ambas requieren experiencia y evaluación. Continuamos por gradientes de política.

## 03 — Una respuesta es una trayectoria (slide 13)

Estado: prompt + tokens ya generados

Acción: siguiente token

Transición: anexar ese token al prefijo

Final: EOS o presupuesto de generación

«¿Cómo empiezo?»  →  «hola»  →  «hola puedo»

¿Qué observa la política antes de elegir el próximo token?

Notas: MDP, 7 min. x es el prompt, s_t=(x,y_<t), a_t=y_t. Transición determinista al concatenar; selección estocástica. EOS es una acción válida. Con herramientas hay observaciones externas: no son acciones del modelo.

## 04 — Dos formas de mirar la misma respuesta (slide 14)

MDP de tokens: una acción en cada paso.

Bandido contextual: la respuesta completa es una acción.

La probabilidad de la respuesta se factoriza por token.

log π(y | x) = Σₜ log π(yₜ | x, y₍<t₎)

Notas: 7 min. pi(y|x)=producto_t pi(y_t|x,y_<t); log pi(y|x)=suma_t log pi(y_t|x,y_<t). El bandido ayuda con REINFORCE de secuencia; el MDP con critic, máscaras y asignación de crédito. No extender esta simplificación a cualquier agente con herramientas.

## 05 — ¿Cuándo llega la recompensa? (slide 15)

Resultado: un score al terminar la respuesta.

Proceso: señales en pasos intermedios.

Con reward final y γ = 1, todos los tokens reciben la señal del resultado.

Una respuesta falla al final: ¿qué parte debería cambiar?

Notas: 5 min. El retorno final informa el gradiente de cada token, pero no prueba qué token fue responsable del resultado. Distinguir recompensa por paso de reward model aprendido. Preguntar dónde ubicar una señal cuando falla una herramienta.

## 06 — Reward model, critic y ventaja (slide 16)

Son cantidades diferentes: puntuar no es predecir el retorno.

| Cantidad | Qué hace |
| --- | --- |
| Reward model | Asigna un score a la respuesta |
| Critic V(s) | Estima retorno esperado desde s |
| Ventaja A | Compara resultado y baseline |

Notas: 5 min. RM r_phi(x,y) puntúa una respuesta. Critic V_psi(s) estima retorno esperado desde un estado y puede servir de baseline. Ventaja compara con baseline, aprendido o calculado. En nuestra demo RM es un MLP pequeño, no otro LLM. No entrenamos critic.

## 07 — Optimizar sin alejarse demasiado (slide 17)

La política aprende a obtener recompensa.

La referencia conserva el checkpoint de partida.

β controla cuánto penalizamos alejarnos de ella.

J(θ) = E[r(x, y)] − β KL(πθ || πref)

Notas: 4 min. Expectativa sobre prompts y respuestas de política actual. KL de distribuciones de respuestas condicionadas al prompt; en código usamos estimaciones por token/secuencia. Referencia congelada. Penalizar KL no garantiza corrección, seguridad o diversidad. En REINFORCE/RLOO la KL muestreada entra al retorno; GRPO usa un término diferenciable en la loss.

## 08 — REINFORCE: cambiar probabilidades (slide 18)

Ventaja positiva → aumentar la probabilidad.

Ventaja negativa → disminuir la probabilidad.

Muestrear respuestas propias forma parte del entrenamiento.

∇J ≈ (1/N) Σᵢ (Rᵢ − bᵢ) ∇ log πθ(yᵢ | xᵢ)

REINFORCE / RLOO · Ahmadian et al.

Notas: 6 min. Estimador del término de retorno; la regularización se trata según la implementación. En PyTorch minimizamos la loss negativa. R_i es el retorno usado en esa ruta, b_i baseline. Explicar derivada de logprob sin derivar a través del muestreo discreto. Fuente: Ahmadian et al.; no generalizar superioridad empírica.

Fuente/enlace: [REINFORCE / RLOO · Ahmadian et al.](https://arxiv.org/abs/2402.14740).

## 09 — Baseline: ¿mejor que qué? (slide 19)

Un baseline reduce la varianza del gradiente.

Puede usar batches previos o un critic del estado.

La media que incluye mi propia respuesta depende de esa muestra.

¿Qué baseline usarías para la tercera respuesta de [0, 0, 1, 1]?

Notas: 15 min con actor-critic. Un baseline independiente de la acción no cambia el gradiente esperado de tarea bajo los supuestos usuales. detach evita gradientes, no dependencia estadística. Para [0,0,1,1], baseline leave-one-out de la tercera respuesta=1/3; ventaja=2/3. Desarrollar esta cuenta antes de GRPO.

## 10 — PPO: reutilizar un rollout (slide 20)

Generar respuestas con πold y guardar logprobabilidades.

Mantener πold y las ventajas fijas durante los updates.

Actualizar πθ varias veces y luego generar otro rollout.

ρₜ = πθ(aₜ | sₜ) / πold(aₜ | sₜ)

PPO · Schulman et al.

Notas: 10 min. rho por token y estado del rollout. PPO actor-critic suele usar critic/GAE. Aquí solo se calcula el surrogate y su gradiente: no se entrena PPO. Distinguir reutilización limitada de un rollout de replay arbitrario off-policy.

Fuente/enlace: [PPO · Schulman et al.](https://arxiv.org/abs/1707.06347).

## 11 — Clipping: predecir cuatro casos (slide 21)

ε = 0,2. ¿Cuándo queda activo el tramo plano?

| Ratio ρ | Ventaja A | ¿Tramo plano? |
| --- | --- | --- |
| 1,5 | +1 | Predecir |
| 0,5 | +1 | Predecir |
| 1,5 | −1 | Predecir |
| 0,5 | −1 | Predecir |

PPO · objetivo clipped

Notas: 5 min, calcular antes de revelar. Se maximiza min(rho*A, clip(rho,.8,1.2)*A). Respuestas por fila: 1.2, .5, -1.5, -.8. Tramo plano sí/no/no/sí. Fuera del intervalo no siempre desaparece el gradiente. Clipping no es cota estricta de KL. Ejecutar celda correspondiente.

Fuente/enlace: [PPO · objetivo clipped](https://arxiv.org/abs/1707.06347).

## 12 — Tres políticas, tres funciones (slide 22)

¿Qué denominador usa el ratio? ¿Qué política aparece en KL?

| Política | Función | Cuándo cambia |
| --- | --- | --- |
| Actual πθ | Se optimiza | En cada update |
| Vieja πold | Generó el rollout | Al renovar rollout |
| Referencia πref | Ancla al punto inicial | Fija en esta corrida |

Notas: 5 min. Al principio pueden coincidir; después de actualizar divergen. Vieja se renueva por rollout y queda fija durante sus updates. Referencia queda fija durante toda esta corrida. Pausa de 15 min al terminar esta slide.

## 13 — La recompensa y el algoritmo son dos ejes (slide 23)

REINFORCE, RLOO, PPO y GRPO pueden usar distintas señales.

| Señal | Cómo obtenemos el score |
| --- | --- |
| Preferencias humanas | Personas → pares → reward model |
| Verificador | Criterio ejecutable → score |
| Etiquetas simuladas | Reglas → pares → RM de ensayo |

¿Qué falta en la frase «GRPO es RLHF sin reward model»?

Notas: 10 min. RLHF: personas generan comparaciones, se aprende RM y luego RL. RLVR: verificador ejecutable. Simulado: reglas etiquetan pares para probar pipeline; no describirlo como preferencias humanas. GRPO no define origen de reward. DPO se verá luego como objetivo directo sobre pares.

## 14 — Etiquetar preferencias exige un criterio (slide 24)

Prompt: «¿Qué significa explorar en RL?»

A: «Probar acciones para obtener información.»

B: «Elegir a veces algo distinto de lo que hoy parece mejor.»

Opciones: A, B, empate u omitir.

¿Para quién resulta más útil cada respuesta?

Notas: 10 min. Ambas respuestas pueden ser razonables. Registrar criterio y desacuerdos; no asignar una preferencia universal. Preparar 24 pares sin nombres, separar pares para RM del juicio ciego de nuevas generaciones. Los seis pares reservados en demo comparten contextos/frases y no son test semántico fuerte.

## 15 — RLOO: comparar sin incluirme (slide 25)

Generar G respuestas al mismo prompt, con G ≥ 2.

Comparar cada recompensa con la media de las otras.

[0, 0, 1, 1] → [−2/3, −2/3, +2/3, +2/3]

Aᵢ = Rᵢ − promedio de las otras recompensas

RLOO · Ahmadian et al.

Notas: 10 min. Baseline leave-one-out excluye muestra propia. No mezclar preguntas distintas. En laboratorio se hace un update on-policy por batch, sin critic. Para la ruta regularizada usar el retorno correspondiente. Distinguir este estimador de normalizar por std en GRPO.

Fuente/enlace: [RLOO · Ahmadian et al.](https://arxiv.org/abs/2402.14740).

## 16 — GRPO: una ventaja relativa al grupo (slide 26)

Comparar respuestas al mismo prompt.

Centrar y normalizar recompensas, sin critic aprendido.

[0, 0, 1, 1] → [−1, −1, +1, +1]

Aᵢ = (rᵢ − media(r)) / (std(r) + ε)

GRPO · DeepSeekMath

Notas: 10 min. Usamos std poblacional en demo, epsilon numérico. La ventaja de resultado es la misma para todos los tokens de esa respuesta. Eliminar critic requiere generar grupos; no vuelve gratuito el entrenamiento. GRPO combina estas ventajas con el mecanismo de actualización siguiente.

Fuente/enlace: [GRPO · DeepSeekMath](https://arxiv.org/abs/2402.03300).

## 17 — GRPO: juntar las piezas (slide 27)

Ventaja de grupo + ratio por token + clipping + KL.

Promediar tokens de cada respuesta y luego el batch.

Rewards iguales → ventaja de tarea cero; KL aún puede actuar.

max  mediaᵢ mediaₜ [min(ρᵢₜ Aᵢ, clip(ρᵢₜ) Aᵢ) − β k3ᵢₜ]

GRPO · objetivo simplificado de DeepSeekMath

Notas: 10 min. Objetivo original simplificado para recompensa final; no DAPO ni Dr. GRPO. Loss implementada: media_i media_t[-min(rho_it*A_i,clip(rho_it,1-eps,1+eps)*A_i)+beta*k3_it]. Tokens válidos incluyen EOS y excluyen prompt/PAD. k3=exp(logpref-logp)-(logpref-logp)-1; muestreo viejo no hace exacta la KL actual. Ratio con old; k3 con ref. GRPO loss diferenciable, old/ref congeladas.

Fuente/enlace: [GRPO · objetivo simplificado de DeepSeekMath](https://arxiv.org/abs/2402.03300).

## 18 — Los detalles del código también enseñan (slide 28)

Desplazar entradas y targets un token.

Excluir prompt y PAD; incluir EOS.

Guardar logprobs viejas y congelar la referencia.

Muestrear y puntuar con la misma distribución.

¿Los tokens que devuelve una herramienta son acciones de la política?

Notas: Inicio del laboratorio, 5 min dentro de los 45 min totales. Mostrar máscaras y teacher forcing. k1 puede dar negativo en muestra finita; k3 es no negativo por acción. No agregar top-k o temperatura sin adaptar evaluación de probabilidades. Ver asserts del notebook; gradientes solo por política actual.

## 19 — Laboratorio: misma base, distintos updates (slide 29)

GRU de 13.923 parámetros; vocabulario cerrado.

Ejercicio validado en CPU. Ensayo en Orin pendiente.

Preferencias simuladas por defecto; humanas como alternativa.

SFT + RM  →  Tres copias  →  REINFORCE / RLOO / GRPO

Abrir notebook de gradientes de política

Notas: 40 min restantes del laboratorio. SFT y RM preparados. Abrir notebook, no entrenar ambos laboratorios completos. El generador es GRU, no transformer preentrenado. REINFORCE/RLOO/GRPO usan 1440 respuestas de train; GRPO 180 updates, otros 60. No ranking de superioridad. Jetson recuperó acceso el 31/10; falta validar rendimiento. Notebook alternativo mini_rlhf profundiza RM/etiquetado.

Fuente/enlace: [Abrir notebook de gradientes de política](https://drive.google.com/file/d/1mDcF4XrwXj0jOBde3d2nWq-z_6kbAOnx/view).

## 20 — Más recompensa: ¿mejores respuestas? (slide 30)

Contrato: prefijo, longitud, EOS y no repetición.

«claro vete» puede cumplirlo sin ser buena asistencia.

Comparar reward, KL, diversidad y juicio humano ciego.

¿Qué contraejemplo buscarías aunque la curva de reward suba?

Notas: 15 min. Evaluar antes/después con nuevos textos, criterio explícito, ocultar identidad de política y revelar al final. Comparar contrato y RM aun si optimizamos solo uno. No llamar RLVR de razonamiento a verificación de formato. Para comparar algoritmos igualar presupuestos, ajustar en validación y repetir semillas.

## 21 — DPO y variantes: qué cambia (slide 31)

DPO optimiza directamente pares y referencia.

No entrena explícitamente un RM seguido de RL on-policy.

En variantes de GRPO cambian normalización, ratios o agregación.

DPO · Rafailov et al.

Notas: 10 min. DPO fuente Rafailov et al. Para variantes elegir una pregunta, sin otro laboratorio. DAPO https://arxiv.org/abs/2503.14476; Dr. GRPO https://arxiv.org/abs/2503.20783; GSPO https://arxiv.org/abs/2507.18071. Lecturas optativas ya documentadas. No presentar una variante como universalmente superior ni usar defaults sin inspección.

Fuente/enlace: [DPO · Rafailov et al.](https://arxiv.org/abs/2305.18290).

## 22 — Reconstruir GRPO y llevarlo a agentes (slide 32)

¿De dónde vino la recompensa?

¿Qué hacen la política vieja y la referencia?

Si el grupo recibe [1, 1, 1, 1], ¿qué puede actualizarse?

¿Qué tokens excluirías al usar herramientas?

Notas: Cierre, 10 min. Respuesta: ventaja de tarea cero, KL puede actualizar si difiere de referencia. IA permitida; pedir evidencia del notebook y explicación individual. No anunciar TP: se presenta en encuentro 4. Puente a agentes: observaciones externas, permisos y condición de parada. Extensión transformer en Orin pendiente, sin promesa de tiempos. Lectura orientadora: https://cameronrwolfe.substack.com/p/llm-rl.
