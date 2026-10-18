# Storyboard — Clase 2: camino hacia GRPO

Contenido original listo para trasladar a una copia 2026 del deck. 22 diapositivas, con momentos de notebook y discusión. Es una propuesta en Markdown: las slides de Drive todavía no fueron editadas. [Lectura de orientación indicada por Juan](https://cameronrwolfe.substack.com/p/llm-rl); fundamentos contrastados con fuentes primarias enlazadas abajo.

## Criterio de secuencia

Seguir el [camino hacia GRPO](CAMINO_A_GRPO.md). Cada slide debe resolver una necesidad de la siguiente: actualizar probabilidades, bajar varianza, construir ventajas, reutilizar rollouts y comparar respuestas sin critic. PPO aporta el mecanismo de actualización; los grupos aportan las ventajas. GRPO es la llegada del bloque, y las variantes quedan después.

## 01 — ¿Cómo cambia una respuesta mediante RL?

En pantalla: un prompt, dos respuestas y tres preguntas: “¿Qué acciones tomó?”, “¿Quién decidió la recompensa?”, “¿Qué probabilidad cambia?”.

Notas: volver al lago de la actividad Q-learning. La recompensa es una elección del diseñador; una curva creciente no responde por sí sola si el resultado es deseable. No abrir con una lista de modelos comerciales.

## 02 — De valores tabulares a una política neuronal

En pantalla: Q-table → aproximación de Q (DQN); otra rama: red → distribución de acciones. Mostrar que ambas necesitan experiencia y evaluación.

Notas: dedicar a DQN solo el puente necesario. El resto del bloque seguirá la rama de gradientes de política. Preguntar cuál produce directamente una distribución.

## 03 — Una respuesta es una trayectoria

En pantalla: prompt $x$, prefijo $y_{<t}$, acción $y_t$, estado $(x,y_{<t})$, transición “anexar token”, finalización EOS/presupuesto.

Notas: pedir que identifiquen la transición “hola” → “hola puedo”. La transición de generación es determinista; la selección de token es estocástica. Con herramientas aparecen observaciones y transiciones externas.

## 04 — Dos abstracciones útiles

En pantalla: MDP de tokens frente a bandido contextual con respuesta completa como acción. La probabilidad de una respuesta es producto de probabilidades condicionales; su log-probabilidad es una suma.

Notas: usar el bandido para explicar REINFORCE de secuencia y el MDP para máscaras, critic y crédito por token. No afirmar que todos los agentes son bandidos.

## 05 — Señal de resultado y señal por paso

En pantalla: recompensa al terminar frente a recompensa de proceso. Para resultado y gamma=1, todos los tokens reciben información del resultado final.

Notas: un token aislado no “merece” necesariamente el score de la respuesta. Atribución de crédito es parte del problema. Preguntar dónde poner reward si se usa una herramienta que falla.

## 06 — Tres cantidades diferentes

En pantalla: reward model $r_\phi(x,y)$ puntúa; critic $V_\psi(s)$ predice retorno esperado; ventaja $A$ compara resultado con una expectativa/baseline.

Notas: el RM no es el critic. Un MLP puede ser RM en una tarea pequeña; no requiere otro LLM del mismo tamaño. [Preferencias humanas](https://arxiv.org/abs/1706.03741).

## 07 — Objetivo y referencia

En pantalla: $J=\mathbb E[r]-\beta D_{KL}(\pi_\theta\Vert\pi_{ref})$. Referencia = checkpoint inicial; beta = presión por conservarlo.

Notas: beta no es una garantía de alineación. La demo mantiene referencia congelada. Mostrar un caso donde más recompensa produce menos diversidad.

## 08 — REINFORCE: un gradiente legible

En pantalla: $\nabla J\approx\frac1N\sum_i(R_i-b_i)\nabla\log\pi_\theta(y_i|x_i)$. Aumentar probabilidad de respuestas con ventaja positiva; disminuir con negativa.

Notas: PyTorch minimiza, por eso loss negativa. Muestrear respuestas propias es parte del entrenamiento, no solo una evaluación. [REINFORCE para preferencias](https://arxiv.org/abs/2402.14740).

## 09 — Baseline: bajar varianza sin mirar la propia acción

En pantalla: comparar baseline de batches previos, critic del estado y leave-one-out. Señalar que media del grupo incluyendo la propia respuesta depende de esa muestra.

Notas: pedir baseline de [0,0,1,1] para la tercera respuesta. Distinguir independencia de acción de detener gradientes: `detach` por sí solo no resuelve la dependencia estadística.

## 10 — PPO: conservar datos mientras cambia la política

En pantalla: rollout con $\pi_{old}$ → guardar logprobs/ventajas → varias actualizaciones de $\pi_\theta$ → nuevo rollout. Ratio $\rho=\pi_\theta/\pi_{old}$.

Notas: critic y GAE sirven para estimar ventajas en el PPO actor-critic; aquí no implementamos ese entrenamiento. [PPO](https://arxiv.org/abs/1707.06347).

## 11 — Clipping: cuatro casos, no una consigna

En pantalla: tabla ratio [1.5,0.5,1.5,0.5], ventaja [+1,+1,−1,−1], epsilon .2. Activación [sí,no,no,sí].

Notas: ejecutar la celda de gradientes. Objetivo mínimo entre versión libre y clipped. No todo ratio fuera del intervalo queda sin gradiente. Tampoco hay una cota estricta de KL.

## 12 — Tres políticas con trabajos distintos

En pantalla: tabla actual = aprende; vieja = generó el rollout; referencia = checkpoint que ancla. Vieja cambia al siguiente rollout; referencia se conserva durante esta corrida.

Notas: en el primer rollout coinciden; después de actualizar divergen. Preguntar qué denominador corresponde al clipping y cuál a KL.

## 13 — El origen de la recompensa es otro eje

En pantalla: personas → pares → RM (RLHF); verificador → score (RLVR); etiquetas simuladas → ensayo identificado. Debajo: REINFORCE / RLOO / PPO / GRPO pueden consumir distintas señales.

Notas: “GRPO es RLHF sin reward model” no clasifica bien el experimento. Para DPO reservar slide 21: su objetivo se optimiza directamente sobre pares. [InstructGPT](https://arxiv.org/abs/2203.02155).

## 14 — Etiquetar no es revelar una verdad universal

En pantalla: un par A/B sin preferencia preasignada; opciones A, B, empate, omitir. Criterio: útil, pertinente, respetuoso; registrar desacuerdos.

Notas: usar 24 pares sin nombres de estudiantes. Separar pares de entrenamiento del juicio de textos generados. Los seis pares de validación comparten frases/contextos: no son un test semántico fuerte.

## 15 — RLOO: comparar sin incluirme

En pantalla: $A_i=R_i-\frac{\sum_{j\ne i}R_j}{G-1}$, respuestas del mismo prompt. Ejemplo [0,0,1,1] → [−2/3,−2/3,+2/3,+2/3].

Notas: no mezcla recompensas de preguntas distintas. Hace falta G≥2. En el laboratorio una actualización on-policy; no entrenamos critic. [RLOO](https://arxiv.org/abs/2402.14740).

## 16 — GRPO: una ventaja relativa al grupo

En pantalla: $A_i=(r_i-\bar r)/(\mathrm{std}_{pop}(r)+\epsilon)$; grupo [0,0,1,1] → [−1,−1,+1,+1]. Sin critic aprendido.

Notas: eliminar critic intercambia memoria/complejidad por más muestras. No garantiza menor costo total. [DeepSeekMath](https://arxiv.org/abs/2402.03300).

## 17 — GRPO: qué objetivo ejecutamos

En pantalla: ventaja relativa + surrogate clipped por token + beta·k3; promedio por respuesta y luego por batch. Grupo de rewards idénticos → ventaja de tarea cero.

Notas: formulation original simplificada con outcome reward, no DAPO/Dr. GRPO. KL puede seguir actuando cuando desaparece el término de tarea. Un default de biblioteca puede usar otra agregación; inspeccionarlo.

## 18 — Los detalles del código cambian el experimento

En pantalla: desplazar tokens causalmente; excluir prompt/PAD; incluir EOS; mantener viejas logprobs fijas; misma distribución para muestrear y puntuar.

Notas: notebook demuestra coincidencia entre logprob de muestreo y teacher forcing. k1 y k3 son estimadores, no KL exacta medida sobre todas las trayectorias. No añadir temperatura/top-k al muestreo sin revisar cómo se evalúa la probabilidad usada en la loss.

## 19 — Laboratorio: misma base, distintos updates

En pantalla: SFT GRU (13.923 parámetros) → preferencias/RM → tres copias → REINFORCE/RLOO/GRPO. Tres contextos, vocabulario cerrado, CPU u Orin después de validar.

Notas: abrir [notebook](llm_policy_gradients.ipynb). Cada método usa 1.440 respuestas de train; GRPO hace 180 updates y los otros 60. No presentar barras como ranking de superioridad. No es un LLM preentrenado.

## 20 — Cambiar reward; mirar algo más que reward

En pantalla: contrato de prefijo/longitud/EOS frente a recompensa aprendida. Medir ambos, KL, diversidad y preferencias ciegas.

Notas: “claro vete” satisface contrato sin ser buena asistencia. El verificador enseña RLVR de formato, no razonamiento matemático. Preparar otra evaluación humana y revelar clave después; proxy creciente no la reemplaza.

## 21 — DPO y los detalles que investigamos después

En pantalla: DPO = objetivo directo sobre pares y referencia. GRPO original / DAPO / Dr. GRPO / GSPO = decisiones diferentes de agrupamiento, ratio y agregación.

Notas: elegir solo una diferencia para discutir, por ejemplo media por respuesta versus por tokens. No sumar otro laboratorio ni prometer que una variante siempre gana. [DPO](https://arxiv.org/abs/2305.18290), [DAPO](https://arxiv.org/abs/2503.14476), [Dr. GRPO](https://arxiv.org/abs/2503.20783), [GSPO](https://arxiv.org/abs/2507.18071).

## 22 — Reconstruir GRPO y transferir lo aprendido

En pantalla: preguntas de salida: “¿Qué señal y optimizador usaste?”, “¿Qué hacen old y ref?”, “¿Qué pasa si el grupo recibe [1,1,1,1]?”, “¿Qué tokens excluirías en un agente con herramientas?”.

Notas: aceptar consulta de IA, exigir evidencia y explicación individual. Conectar TP1 (proxy) y TP2 (evaluación fuera de datos). Extensión en Orin con transformer pequeño pendiente de implementación y medición; la placa está desconectada y el ensayo queda para después.
