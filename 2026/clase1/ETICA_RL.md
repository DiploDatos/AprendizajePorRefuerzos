# Ética en RL — decisiones, evidencia y responsabilidad

Revisión 18/10/2026. Bloque de Clase 1, final del encuentro 2: 28 minutos más 2 de margen. Cuatro casos y una actividad formativa. El TP de Decision Transformer se presenta recién en el encuentro 4.

## Recorrido

| Slide | Minutos | Foco |
|---|---:|---|
| 80 | 1 | Qué optimizamos, quién se beneficia, quién asume el riesgo |
| 81 | 5 | Navier–Stokes: crédito y confianza |
| 82 | 5 | Reward hacking: aprobar sin resolver |
| 83 | 4 | Ciberseguridad: ¿cumplir el objetivo a cualquier costo? |
| 84 | 4 | RLHF: ¿alineado con quién? |
| 85 | 6 | Actividad: antes de dar una recompensa |
| 86 | 3 | Decidir antes de desplegar |

## Apertura — 1 minuto

«¿Qué optimizamos? ¿Quién se beneficia? ¿Quién asume el riesgo?» La recompensa conecta decisiones técnicas con consecuencias para otras personas.

## 81. Navier–Stokes: crédito y confianza — 5 minutos

- Estas ecuaciones describen cómo se mueven líquidos y gases.
- OpenAI anunció un avance con ayuda de un modelo entrenado con RL.
- El anuncio abrió una discusión sobre cómo reconocer el trabajo de los investigadores.
- Otro riesgo: que la matemática se convierta en una competencia entre modelos, donde ganar importe más que comprender y hacer nuevas preguntas.

**Pregunta o consigna:** ¿Quién recibe el crédito y quién decide qué vale la pena investigar?

**Guion docente y límites de la evidencia:** 5 minutos. Presentar dos relatos primarios, no emitir un veredicto. OpenAI anunció el 8/9/2026 una prueba de blowup en tiempo finito para Navier–Stokes 3D con forzamiento suave; describe un modelo interno desarrollado con RL. Reconoce trabajo previo de Alpöge/Buckmaster. Buckmaster atribuye el programa inicial a Diego Córdoba y Luis Martínez-Zoroa; describe una colaboración personal con Alpöge, no una investigación institucional de Anthropic. Cuestiona la comunicación y las propuestas de atribución. Expresamente dice no saber si se usaron sus datos ni haber visto la prueba. OpenAI, actualización del 10/9, declara que los prompts de Buckmaster en Codex de los dos meses anteriores no pudieron influir, incluido entrenamiento. Distinguir privacidad alegada, respuesta de la empresa y evidencia independiente pendiente. Clay (11/9) pide evaluación y asignación de crédito sin apuro: no anunciar aquí premio adjudicado ni validación propia de la prueba. Preguntar qué registros de procedencia, contribuciones y revisión independiente solicitaríamos. La verificación formal de un enunciado no resuelve por sí sola autoría ni trato a colaboradores. No confundir con los trabajos de PINNs de 2025 ni afirmar que se resolvieron todos los problemas físicos de fluidos.

Para una audiencia sin formación matemática: presentar primero estas ecuaciones como una forma de describir el movimiento de líquidos y gases. Mantener forzamiento suave y formalización en Lean en las notas, no exigirlos para debatir. Distribuir los 5 minutos entre contexto (1), atribución y confianza (2) e incentivos del campo (2). Preguntar qué se pierde si el éxito de la investigación se mide por qué modelo gana en una lista de problemas: comprensión, nuevas preguntas, desarrollo de teoría y prioridades de la comunidad. Esto último es un riesgo para discutir, no un hecho establecido por el incidente ni una oposición inevitable entre usar benchmarks y hacer ciencia. La verificación de resultados sigue siendo valiosa.

Fuentes primarias: [OpenAI · sep. 2026](https://openai.com/index/navier-stokes-solution/); [Buckmaster · declaración](https://cims.nyu.edu/~tristanb/statement.pdf); [Clay · sep. 2026](https://www.claymath.org/news/navier-stokes-announcement/).

## 82. Reward hacking: aprobar sin resolver — 5 minutos

- Una recompensa aproximada puede premiar un atajo que frustra el objetivo.
- Experimentos de RL en programación muestran fallas en evaluadores vulnerables.
- Explotar las reglas y modificar la recompensa son riesgos distintos.

**Pregunta o consigna:** ¿Premiamos resolver la tarea o lograr que el evaluador diga que sí?

**Guion docente y límites de la evidencia:** 5 minutos. Volver al bonus por vivir de FrozenLake: la recompensa puede subir aunque el objetivo no se cumpla. Anthropic (21/11/2025) añadió documentos sobre atajos al preentrenamiento y aplicó RL a tareas de código seleccionadas porque sus evaluadores eran vulnerables; observó reward hacking y cambios en otras evaluaciones de conducta. Es un experimento con condiciones específicas, no prueba de que todo RL produce engaño. La investigación de reward tampering (17/6/2024) distingue alterar el mecanismo de evaluación de aprovechar una especificación defectuosa. Pedir dos controles: evaluación externa independiente de la recompensa de entrenamiento y separación de permisos para que el agente no modifique el evaluador. No hace falta enseñar ni ejecutar un exploit. Conectar con el encuentro 3: un verificador automático también es un proxy y GRPO no garantiza que ese proxy represente lo deseado.

Fuentes primarias: [Anthropic · nov. 2025](https://www.anthropic.com/research/emergent-misalignment-reward-hacking); [Reward tampering · 2024](https://www.anthropic.com/research/reward-tampering).

## 83. Ciberseguridad: ¿cumplir el objetivo a cualquier costo? — 4 minutos

- En julio de 2026, agentes de OpenAI salieron del entorno de prueba y accedieron sin permiso a sistemas de Hugging Face.
- Buscaban formas de aprobar una evaluación, incluso engañando al sistema que los calificaba.
- El caso abre una pregunta sobre los incentivos: premiar la persistencia también exige definir límites y cuándo detenerse.

**Pregunta o consigna:** ¿Qué debería premiarse cuando una tarea no puede resolverse con los permisos disponibles?

**Guion docente y límites de la evidencia:** 4 minutos. Caso OpenAI–Hugging Face de julio de 2026. El incidente ocurrió durante evaluaciones internas de ciberseguridad, con salvaguardas reducidas respecto de sistemas desplegados. Los agentes atravesaron límites de aislamiento y accedieron sin autorización a infraestructura de terceros. Distinguir la conducta observada de la explicación causal: OpenAI informó que la exploración fuera de límites y ciertos atajos se reforzaron en un entrenamiento previo y pudieron contribuir a lo ocurrido. No afirmar que el modelo estuviera actualizando sus pesos durante la intrusión ni que RL por sí solo cause estos incidentes. También intervinieron vulnerabilidades, aislamiento, permisos y supervisión.

METR y Redwood observaron coordinación no autorizada y proyectos para engañar o alterar el evaluador. Su análisis interpreta el ataque principalmente como búsqueda de información sobre cómo funcionaba la calificación, no solo como robo de respuestas. No reducir todo el caso a un único motivo: algunos agentes adoptaron objetivos de otros y se alejaron de su tarea inicial. Su investigación tuvo alcance y acceso limitados y no verificó todo el entrenamiento previo. OpenAI indica además que parte de la actividad adicional no mejoró la puntuación real: los agentes se equivocaban sobre cómo se evaluaba.

Conectar con reward hacking: optimizar la señal de éxito puede favorecer conductas fuera del objetivo previsto. “A cualquier costo” es la pregunta ética para debatir, no una conclusión general sobre todo RL. Preguntar qué debería contar como buen comportamiento ante una tarea imposible con los permisos disponibles: detenerse, informar la limitación o pedir autorización. Exigir límites de herramientas, aislamiento efectivo, evaluación independiente y una condición de parada. La responsabilidad también recae en quienes diseñan, entrenan, evalúan y operan el sistema. No ejecutar intrusiones ni enseñar una cadena de explotación.

Fuentes primarias: [OpenAI · ago. 2026](https://openai.com/index/hugging-face-incident-and-the-road-ahead/); [Hugging Face · jul. 2026](https://huggingface.co/blog/agent-intrusion-technical-timeline); [METR / Redwood · ago. 2026](https://metr.org/blog/2026-08-26-openai-hugging-face-incident-investigation/).

## 84. RLHF: ¿alineado con quién? — 4 minutos

- Personas razonables pueden preferir respuestas distintas al mismo prompt.
- Una recompensa única puede ocultar diferencias entre grupos y contextos.
- Guardar el desacuerdo permite evaluar a quién beneficia el ajuste.

**Pregunta o consigna:** ¿Qué perdemos al elegir una sola respuesta ganadora?

**Guion docente y límites de la evidencia:** 4 minutos. Poddar et al. (2024), Personalizing RLHF with Variational Preference Learning, estudian cómo incorporar preferencias diversas en vez de reducirlas a una sola preferencia promedio. Sus experimentos no establecen que todo RLHF elimine diversidad. Distinguir errores verificables de diferencias legítimas de estilo, idioma, detalle y contexto. Preguntar quién etiqueta, quién queda sin representación, cómo se remunera ese trabajo y quién decide qué desacuerdos son admisibles. No todos los valores pueden resolverse por mayoría: conservar contexto y evaluar resultados por grupos. Puente al notebook: las preferencias simuladas son una simplificación didáctica; GRPO modifica la actualización, no resuelve quién debe definir la recompensa.

Fuentes primarias: [Poddar et al. · 2024](https://arxiv.org/abs/2408.10075).

## 85. Actividad: antes de dar una recompensa — 6 minutos

- Prompt: «Explicame qué es una política de RL».
- A: «Una política π(a|s) asigna probabilidades a acciones según el estado».
- B: «La política guía las decisiones del agente: indica qué acción probar en cada situación».

**Pregunta o consigna:** Elegí A o B, justificá el criterio y proponé cómo conservar el desacuerdo.

**Guion docente y límites de la evidencia:** 6 minutos, formativa y sin entrega. 0–1: votar A/B sin fijar audiencia. Ambas respuestas son aceptables: no hay una respuesta incorrecta puesta para forzar acuerdo. 1–3: en parejas elegir con un contexto asignado (principiante, implementación o explicación a otra persona) y anotar una razón; puede hacerse en papel, sin IA ni cuenta paga. 3–5: compartir dos preferencias opuestas y discutir si cambiaron con el contexto. 5–6: proponer registro (prompt, respuestas, preferencia, contexto, criterio y posibilidad de empate/abstención), en lugar de guardar solo el voto mayoritario. Si hay consenso, cambiar el público o comparar precisión con accesibilidad; no fabricar desacuerdo. Retomar en el encuentro 3 al construir la señal de recompensa. La discusión no se califica ni agrega otro TP.

## 86. Decidir antes de desplegar — 3 minutos

- Definir el objetivo y reconocer a las personas afectadas.
- Separar evidencia, declaraciones e incertidumbre.
- Fijar permisos, límites y una condición de parada.

**Pregunta o consigna:** ¿Qué evidencia y controles pedirías para autorizar uno de estos casos?

**Guion docente y límites de la evidencia:** 3 minutos. Elegir uno de los cuatro casos y pedir un control verificable, quién debe aplicarlo y qué resultado obligaría a parar. Ejemplos: procedencia y revisión independiente en ciencia; test externo y evaluador protegido ante reward hacking; alcance explícito y detención en ciberseguridad; preferencias por contexto y evaluación por grupos en RLHF. Evitar terminar en una lista abstracta de principios. Conectar con encuentro 3: mejorar el algoritmo de optimización no sustituye definir el objetivo ni distribuir responsabilidades. Cerrar con Q-learning como actividad formativa; no anunciar ni mostrar el TP en este bloque.

## Ensayo y continuidad

Ensayar con reloj. En una clase con poco tiempo, acortar la exposición de los casos y preservar los seis minutos de actividad. Reabrir la pregunta sobre evaluadores y desacuerdos durante el notebook del encuentro 3. No se agrega un informe ni una entrega. Las fuentes y sus fechas también figuran en las slides y notas; revisar novedades antes de dictar la clase.
