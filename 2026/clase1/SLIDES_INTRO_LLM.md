# Clase 1 — dos slides para anticipar RL en LLMs

Revisión 17/10/2026. Contenido aplicado en la copia 2026 de Clase 1: slides 14 y 15, con títulos abreviados, notas y fuentes. Ver [guion de ensayo](GUIA_PRESENTACION.md). Duración total: 6–8 minutos dentro de la motivación del primer encuentro. El desarrollo técnico ocurre en el **encuentro 3, segundo fin de semana**.

## Sustitución en el deck histórico

Retirar del recorrido principal las slides 14–20: overhype, crisis de reproducibilidad alrededor de 2016, HARKing, papers de reproducibilidad, meme y anécdota de Atari. Reemplazar también las slides 21–22 de actualidad/RLHF por las dos propuestas siguientes para evitar repetición. Conservar la slide 23 como transición hacia los desafíos y los fundamentos de RL. Los IDs originales están en el [plan de edición](../docentes/SLIDES_2026.md).

AlphaGo puede conservarse como un hito breve en la motivación previa; no abrir otra explicación histórica de 2016. La evaluación experimental se trabaja con ejemplos concretos en el laboratorio, sin añadir otra introducción sobre HARKing.

## Slide A — RL también puede mejorar cómo responde un modelo

**Texto visible**

> Un modelo puede aprender de ejemplos… y también de cómo evaluamos sus respuestas.

**2017 · Preferencias:** comparar comportamientos permite aprender una recompensa.

**2022 · InstructGPT:** ejemplos de buenas respuestas → preferencias humanas → ajuste con RL.

**La conexión con RL:** el modelo genera una respuesta, recibe una señal y cambia la probabilidad de volver a producirla.

**Visual sugerido**

Tres bloques horizontales: `Prompt → Dos respuestas → Preferencia humana`; debajo, una flecha de feedback hacia `Modelo`. Una sola respuesta se resalta como preferida. No presentar aún critic, ratios o función de pérdida.

**Notas docentes, 3–4 minutos**

Usar un prompt cotidiano: «Explicame qué es una política con un ejemplo». Comparar una respuesta correcta pero ininteligible con una explicación clara. Pedir una preferencia y el criterio que la justifica. La señal de preferencia puede convertirse en una recompensa aprendida; luego RL modifica la política de generación.

Los ejemplos supervisados y el feedback cumplen papeles diferentes. La comparación humana no implica que una persona puntúe cada respuesta durante cada paso de entrenamiento. Tampoco significa que todos los modelos de lenguaje se entrenen con RL.

Fuentes: [Christiano et al., preferencias humanas, 2017](https://arxiv.org/abs/1706.03741), antecedente general, no un trabajo de LLMs; [Ouyang et al., InstructGPT, 2022](https://arxiv.org/abs/2203.02155).

## Slide B — De preferencias a respuestas verificables: el camino hacia GRPO

**Texto visible**

**2024 · DeepSeekMath:** GRPO compara varias respuestas al mismo problema para construir una señal relativa.

**2025 · DeepSeek-R1:** RL cobra protagonismo en tareas de razonamiento con recompensas verificables.

> Mismo problema → varias respuestas → evaluación → actualizar el modelo.

**En el encuentro 3:** construiremos el camino REINFORCE → ventaja y PPO → comparación por grupos → GRPO.

**Visual sugerido**

Una tarjeta de problema y cuatro tarjetas de respuesta, dos correctas y dos incorrectas. Un evaluador asigna `[0, 1, 0, 1]` y una flecha señala «aumentar la probabilidad de respuestas mejor evaluadas». No usar el texto «GRPO = respuestas correctas».

**Notas docentes, 3–4 minutos**

Ejemplo: «¿Cuánto es 17 × 23?». Mostrar 391 y 381; aquí podemos verificar el resultado automáticamente. Contrastar con la explicación pedagógica de la slide anterior, donde la calidad requiere otro criterio.

Separar dos preguntas: **¿de dónde sale la recompensa?** —preferencia humana o verificador— y **¿cómo actualizamos el modelo?** —algoritmo—. GRPO responde a la segunda. El ejemplo binario es una simplificación didáctica, no una descripción exhaustiva de los experimentos de DeepSeekMath.

GRPO toma ideas de PPO y utiliza comparaciones dentro de un grupo de respuestas sin entrenar un critic de valor separado. Se nombra, pero su ecuación se reserva para el encuentro 3. RLOO será un puente conceptual, no un supuesto hito histórico anterior a GRPO.

DeepSeek-R1-Zero y DeepSeek-R1 tienen recetas distintas: no generalizar «sin SFT» a toda la familia. Un resultado verificado no demuestra que la explicación intermedia sea fiel ni que el modelo sea correcto en cualquier tarea.

Fuentes: [DeepSeekMath / GRPO, 2024](https://arxiv.org/abs/2402.03300), [DeepSeek-R1, publicación original de 2025](https://arxiv.org/abs/2501.12948v1).

## Salida hacia los fundamentos

Pregunta oral breve: «Si una respuesta es una secuencia de decisiones, ¿cuáles serían el estado, la acción y la recompensa?». Recoger intuiciones; no resolver aún toda la correspondencia token/MDP. Transición: «Para entender esa actualización vamos a empezar por políticas, retorno y aprendizaje en entornos pequeños».

Al comenzar el encuentro 3, recuperar el mismo ejemplo y pasar al [camino hacia GRPO](../clase2/CAMINO_A_GRPO.md). La introducción anticipa el destino; la práctica de Q-learning construye el vocabulario y la disciplina experimental.
