# Clase 2 — guía del recorrido general

Revisión de continuidad, 31/10/2026. [Deck editable](https://docs.google.com/presentation/d/19hKgrRzdaDW6sAlVUABjZJeOeElbK6Z1q1QkRfttv4w/edit) y [PDF completo sin el TP](Clase_2_2026.pdf).

Se conserva el flow de la clase: aproximación neuronal → DQN → gradientes de política → actor-crítico/PPO → RL en modelos de lenguaje → Offline RL/Decision Transformer → aplicaciones y conclusiones. GRPO ocupa un subbloque dentro de ese hilo. La edición anterior de 32 slides principales queda reemplazada.

## Mapa actual

| Slides | Bloque y función |
|---|---|
| 1–19 | Límites de tablas, estados complejos, aproximación y repaso de redes |
| 20–34 | DQN: target, replay, arquitectura y variantes |
| 35–50 | Política parametrizada, REINFORCE, actor-crítico, DDPG y PPO |
| 51–66 | RL en LLMs: generación como trayectoria, preferencias/reward, KL, RLOO/GRPO y demo |
| 67–75 | Offline RL y Decision Transformer: aprender acciones de trayectorias registradas |
| 76–115 | Aplicaciones de juegos, recomendación, control/robótica; conclusiones |
| 116–154 | Consulta y profundización: ejercicios LLM y detalle de aplicaciones |
| 155 | TP único, reservado al encuentro 4 |

Las primeras 115 páginas están visibles en presentación normal; las 40 últimas están omitidas. El PDF incluye todas salvo la última: 154 páginas, con teoría de DT y apéndices. Omitir no restringe acceso: distribuir el PDF sin TP antes del encuentro 4. No se cambiaron permisos de Drive.

## Cómo integrar GRPO sin reiniciar la clase

Los gradientes de política y PPO ya se explicaron en el bloque anterior. Retomarlos para generación de tokens; no volver a derivar todo desde cero. Dentro del bloque LLM se integran 14 páginas nuevas. Los ocho repasos adicionales quedan en profundización para consultas o práctica posterior.

El cierre del bloque LLM ahora abre la pregunta por aprender de datos registrados. En Decision Transformer se conserva contexto de estados/acciones/return-to-go y predicción de acciones; no confundirlo con hacer GRPO sobre un Transformer. Pedir un return alto no garantiza alcanzarlo. Sólo el lanzamiento del TP está reservado al encuentro 4: esa regla no excluye la teoría de DT.

## Laboratorio y tiempos

Usar [llm_policy_gradients.ipynb](llm_policy_gradients.ipynb), enlazado en la slide 63. Preparar SFT/RM y resultados para integrar una demo acotada. La práctica completa puede profundizarse aparte. No asignar toda la clase a 45 minutos de laboratorio y al desarrollo de GRPO sin revisar qué lugar conservan los demás bloques.

La demo sigue siendo una GRU de 13.923 parámetros, validada en CPU. Las etiquetas iniciales son simuladas; las humanas se incorporan explícitamente. El ensayo en Orin y la extensión con transformer preentrenado siguen pendientes. No hay mediciones nuevas en esta edición.

La cantidad de páginas es un banco de material, no una duración garantizada. Ensayar ritmo y acordar con Jorge el reparto entre encuentros 3–4, manteniendo el hilo completo. Se retira el cronograma anterior de cuatro horas centrado en GRPO.

## Ejercicios y respuestas docentes

- Slide 58: clipping para ratios [1,5;0,5;1,5;0,5] y ventajas [+1;+1;−1;−1]. Objetivo [1,2;0,5;−1,5;−0,8]; tramo plano en primera y cuarta.
- Slide 59: ratio con política vieja; KL con referencia.
- Slide 60: RLOO de [0,0,1,1] da [−2/3,−2/3,+2/3,+2/3]. Slide 61: GRPO da aproximadamente [−1,−1,+1,+1] con std poblacional.
- Slide 62: rewards iguales anulan la ventaja de tarea; KL puede seguir actualizando.
- Slide 64: buscar contraejemplos al proxy; no interpretar más reward como mejor asistencia ni como ranking válido de algoritmos.

Las páginas históricas conservan medios y enlaces, con limitaciones previas de contenido y maquetación que pueden requerir revisión. Esta corrección recupera continuidad; no da por corregido el evaluador del TP ni valida todos los casos como actuales.

[Notas de las slides nuevas](SLIDES_LLM_RL.md). [Registro de revisión](../validacion/clase2_revision.json).
