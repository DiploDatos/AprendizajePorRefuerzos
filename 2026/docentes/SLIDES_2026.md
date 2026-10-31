# Cambios propuestos para las diapositivas

Revisión 31/10/2026. Clase 1 editada (87 slides); Clase 2 corregida para conservar el flujo general (115 principales, 39 de profundización y TP omitido). Las [copias editables 2026 en Drive](../DRIVE_2026.md) ya están creadas. IDs verificados por outline nativo el 12/10/2026. Editar las copias 2026; actualizar el año dentro de las slides y los enlaces. No suponer que deck “Clase 2” es encuentro 2.

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

## Clase 2 — continuidad recuperada

La edición anterior reducía el recorrido principal a GRPO y omitía fundamentos, aplicaciones y DT. Juan indicó conservarlos. La corrección restaura el orden de la clase y usa 14 slides nuevas dentro del bloque LLM; los ocho repasos restantes van a profundización. Se conservan todas las páginas de DT que estaban presentes antes de la reorganización.

| Slides actuales | Bloque |
|---|---|
| 1–19 | Problemas tabulares y redes neuronales |
| 20–34 | Valor y DQN |
| 35–50 | Gradientes de política, actor-crítico, DDPG y PPO |
| 51–66 | RL en modelos de lenguaje, RLHF/RLVR, RLOO/GRPO y demo |
| 67–75 | Offline RL y Decision Transformer, dentro del recorrido visible |
| 76–115 | Aplicaciones y conclusiones |
| 116–154 | Profundización: ocho repasos LLM y aplicaciones detalladas; omitidas |
| 155 | Único TP: presentar en encuentro 4; omitido |

El PDF contiene 154 páginas, incluyendo profundización y DT, sin el anuncio del TP. Las slides 1–115 están visibles en presentación normal. Conservar la selección docente y el ritmo del material completo; no prometer que toda esa extensión más un laboratorio quepa en cuatro horas sin ensayo.

[Guía de presentación](../clase2/GUIA_PRESENTACION.md), [contenido y notas del bloque nuevo](../clase2/SLIDES_LLM_RL.md) y [registro de corrección](../validacion/clase2_revision.json). La demo sigue siendo una GRU validada en CPU, no un transformer preentrenado. La revisión integral del evaluador del TP continúa pendiente.
