# Aula virtual 2026 — preparación docente

Estado al 31/10/2026: montaje inicial publicado en el [aula 2026](https://famaf.aulavirtual.unc.edu.ar/course/view.php?id=779). Acceso docente confirmado y recorrido revisado con la simulación de rol Student de Moodle. Faltan calendario confirmado, revisión del programa heredado y cierre de la evaluación.

## Cambios aplicados en Moodle

- Agregada bienvenida 2026 conservando docentes, carga horaria y programa existente.
- Actualizada «Material de Clases» con apartados para los cuatro encuentros, instrucciones del entorno, PDF 2026, warm-up, Q-learning y demo de gradientes de política para lenguaje. Son apartados dentro de la pestaña existente.
- Retirados de esa portada los enlaces al material 2025 y al práctico histórico. Decision Transformer se conserva como contenido teórico de Clase 2.
- Ocultada la pestaña Evaluación completa: contenía dos entregas y fechas de noviembre de 2025. Las actividades se conservan para revisión docente; todavía no se adaptaron fechas, grupos, restricciones o calificaciones. No habilitar la pestaña completa sin resolver esos puntos y retirar el TP1 histórico de la oferta 2026.
- Comprobado con rol Student: se ve el material nuevo; Evaluación aparece atenuada y no permite abrirse. Esta simulación no sustituye una prueba con una cuenta estudiantil real.
- Los PDF enlazan a GitHub: la consulta de permisos de Drive mostró sólo usuarios docentes, por lo que no se usaron esos enlaces restringidos ni se modificaron permisos.
- No se crearon entregas ni notas para Q-learning o la demo. No se anunciaron el TP ni fechas sin confirmar.

Quedan por revisar Bibliografía, los dos foros «Avisos» heredados y la pestaña vacía «Topic 4». No se alteraron matrículas ni calificaciones. El curso se dejó en el rol docente habitual y sin modo de edición.

## Estructura propuesta

| Sección | Contenido | Qué comprobar |
|---|---|---|
| Información general | Bienvenida, programa, calendario, docentes y canal de consultas | Fechas y horarios 2026 confirmados; no copiar los de 2025 |
| Preparación | Instrucciones del [entorno](../../ENTORNO.md), notebook introductorio y prueba de instalación | Enlaces accesibles y archivos actuales; instrucciones para las plataformas admitidas |
| Encuentro 1 | Fundamentos, material de Clase 1 y warm-up | Orden de lectura y actividad de clase claros |
| Encuentro 2 | Material de Clase 1, actividad Q-learning y ética | Notebook estudiantil; actividad formativa sin nota ni entrega obligatoria |
| Encuentro 3 | Material de deep RL, RL en LLMs, Offline RL/DT y demo | PDF completo sin el anuncio del TP; mantener el hilo general de la clase |
| Encuentro 4 | Profundización, consigna y rúbrica del único TP | Publicar el TP recién en este encuentro, una vez cerrado su protocolo |

Las presentaciones abarcan más de un encuentro. No equiparar «Clase 2» con el segundo encuentro. La teoría de Decision Transformer pertenece al recorrido general; lo reservado al encuentro 4 es la presentación del TP.

## Material para seleccionar

- [Índice 2026](../README.md) y [plan](../01_PLAN_CURSO.md): base de organización docente.
- [Notebook introductorio](../actividades/lab_0_warmup_rl.ipynb).
- [PDF Clase 1](../clase1/Clase_1_2026.pdf).
- [Consigna formativa de Q-learning](../actividades/q_learning/CONSIGNA.md) y [notebook estudiantil](../actividades/q_learning/lab_1_q_learning_auditoria.ipynb).
- [PDF Clase 2 sin anuncio del TP](../clase2/Clase_2_2026.pdf) y [notebook de gradientes de política](../clase2/llm_policy_gradients.ipynb), acompañado por sus módulos Python.
- [TP](../tp/CONSIGNA.md) y [rúbrica](../tp/RUBRICA.md): preparación docente, todavía sujetos a revisión técnica y publicación en el encuentro 4.

No distribuir como portada el índice docente o el ZIP completo: incluyen referencias, soluciones y borradores. Seleccionar recursos por encuentro. No cargar `docentes/` como material estudiantil. El notebook de Q-learning conserva un TODO para estudiantes; la referencia ejecutada es docente.

## Texto inicial para el aula

Bienvenidas y bienvenidos a Aprendizaje por Refuerzos 2026.

En el curso vamos a trabajar con los fundamentos de RL, métodos tabulares y deep RL, aplicaciones y aprendizaje a partir de trayectorias registradas. También exploraremos cómo se usan estas ideas para ajustar modelos de lenguaje.

Antes de comenzar, revisen las instrucciones del entorno y el material introductorio. Los recursos y actividades estarán organizados por encuentro. El uso de IA es opcional: nos interesa poder explicar las decisiones y comprobar los resultados.

El calendario y el canal de consultas deben completarse con la información confirmada antes de publicar esta bienvenida.

## Montaje y revisión

- [x] Confirmar URL del aula, permisos de edición y edición 2026.
- [x] Revisar materiales y evaluación existentes de 2025 antes de modificar.
- [ ] Cargar bienvenida, calendario confirmado e instrucciones del entorno.
- [x] Organizar por encuentro y enlazar los materiales seleccionados.
- [x] Mantener el TP oculto/sin publicar hasta el encuentro 4; no anunciarlo en encuentros 1–3.
- [ ] Cerrar protocolo, consigna, rúbrica, fecha, grupos y modalidad antes de habilitar su entrega.
- [ ] Probar accesos, descargas y rutas de los notebooks con vista estudiante.
- [ ] Confirmar que las actividades formativas no generan entregas ni calificaciones obligatorias.
- [ ] Verificar que no queden fechas antiguas ni enlaces a borradores como recursos definitivos.
- [x] Completar el borrador de Slack para estudiantes con el enlace real; omitir calendario aún no confirmado.

Criterio de cierre: el aula está navegable desde una cuenta/vista estudiantil, sus recursos actuales se abren y la disponibilidad de actividades coincide con las decisiones del curso.
