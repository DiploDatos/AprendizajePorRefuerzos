# Cambios propuestos para las diapositivas

Revisión 31/10/2026. Clase 1 editada (87 slides); Clase 2 actualizada (32 principales y 123 omitidas). Las [copias editables 2026 en Drive](../DRIVE_2026.md) ya están creadas. IDs verificados por outline nativo el 12/10/2026. Editar las copias 2026; actualizar el año dentro de las slides y los enlaces. No suponer que deck “Clase 2” es encuentro 2.

## Clase 1 — cambios aplicados

Números actuales, deck 2026 de 87 slides. Se conservan las notas y fuentes; el [guion de ensayo](../clase1/GUIA_PRESENTACION.md) detalla el recorrido.

| Slide / ID | Estado y propósito |
|---|---|
| 1 / `p` | Portada 2026. Afiliaciones de slide 2 pendientes de confirmación docente |
| 3 / `gb2e30c18f0_0_263` | Agenda y notas de prácticas formativas; no se presenta el TP |
| 14–15 / `gb2e30c18f0_0_199`, `gb2e30c18f0_0_205` | Dos slides de preferencias/RLHF y comparación de respuestas/GRPO; 6–8 min, anticipo del encuentro 3 |
| 20 / `gb49cbc1f9c_0_119` | Distinción online/offline; experiencia generada o registrada |
| 80 / `gb4eea7d43a_0_10` | Apertura de ética: objetivo, beneficiarios y riesgo |
| 81 / `rl26_ethics_navier` | Crédito y confianza en Navier–Stokes; riesgo de orientar la ciencia según rankings de modelos |
| 82 / `rl26_ethics_hacking` | Reward hacking y modificación del evaluador |
| 83 / `rl26_ethics_cyber` | Incidente OpenAI–Hugging Face 2026: incentivos, permisos y cuándo detenerse |
| 84 / `rl26_ethics_plural` | Diversidad de preferencias: ¿alineado con quién? |
| 85 / `rl26_ethics_activity` | Actividad A/B de 6 min con dos respuestas correctas |
| 86 / `rl26_ethics_closing` | Elegir controles y una condición de parada |
| 87 / `gb4eea7d43a_0_5` | Notebook Q-learning, IA opcional, sin entrega obligatoria; sin anuncio del TP |

El bloque 80–86 reemplaza las seis slides anteriores de ética y conserva el diseño nativo. [Guía, preguntas, tiempos y fuentes](../clase1/ETICA_RL.md). La notación general de Bellman y la ampliación max/argmax siguen como revisión docente pendiente; esta edición no las da por corregidas.

El TP se presenta recién en el encuentro 4. No se agrega su anuncio al primer fin de semana ni al encuentro 3. Los archivos se mantienen en el índice docente para preparación y se publican a estudiantes cuando corresponda.

## Clase 2 — cambios aplicados el 31/10/2026

Se editaron las copias 2026 conservando el tema Tropic, placeholders y medios útiles. Las diez slides originales de RLHF/DPO se reemplazaron por 22 páginas nuevas. El origen 2025 sigue en el archivo. [Guía y mapa actual](../clase2/GUIA_PRESENTACION.md), [contenido y notas](../clase2/SLIDES_LLM_RL.md), [validación](../validacion/clase2_revision.json).

| Páginas actuales | Función |
|---|---|
| 1–14 | Apertura, puente de Q-learning a redes, DQN, dos aplicaciones y generación como trayectoria |
| 15–19 | Recompensa, RM/critic/ventaja, objetivo/KL, REINFORCE y baseline |
| 20–22 | PPO: rollout, clipping, política actual/vieja/referencia |
| 23–27 | Fuente de recompensa, preferencias, RLOO y GRPO |
| 28–30 | Código, notebook enlazado y evaluación ciega |
| 31–32 | DPO/variantes como comparación breve y reconstrucción de GRPO |
| 33–144 | Consulta histórica de algoritmos y aplicaciones; omitida |
| 145–155 | Offline RL/Decision Transformer y único TP, encuentro 4; omitido |

Los IDs nuevos son `rl26_c2_01` a `rl26_c2_22`; el registro incluye su número actual. Las 32 principales están contiguas al inicio. En presentación normal se omite el resto. El PDF preparado contiene sólo esas 32; compartirlo con estudiantes, no el deck completo, antes del encuentro 4.

El notebook enseña REINFORCE/RLOO/GRPO sobre una GRU de 13.923 parámetros, con preferencias simuladas o etiquetas humanas explícitas; PPO se ilustra sin entrenarlo. Acceso a Orin recuperado según Juan, pero extensión con transformer y rendimiento todavía pendientes. Se corrigió el texto de DT que prometía alcanzar returns solicitados; no implica que el bloque completo del encuentro 4 ni su evaluador hayan sido revisados.

La consulta histórica conserva afirmaciones y fórmulas que pueden requerir revisión antes de volver a usarse en vivo. No se revalidaron todos sus enlaces ni se repararon todos los problemas previos de maquetación. El ensayo oral y el ajuste final del tiempo siguen pendientes.
