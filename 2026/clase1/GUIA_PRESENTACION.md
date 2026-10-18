# Clase 1 — recorrido y guion de ensayo

Revisión 18/10/2026. [Presentación editable](https://docs.google.com/presentation/d/1cBZzjKvEYI1Cok3yE6IBRSureGgnzkK5SrfsgF7YsL8/edit), 87 slides; [copia PDF](Clase_1_2026.pdf). Este deck abarca fundamentos del primer fin de semana. El desarrollo de GRPO se reserva para el encuentro 3.

## Dos slides introductorias: 6–8 minutos

Los tiempos siguientes son una propuesta de ensayo. Se verificó el recorrido lógico y el espacio en las slides; el ensayo oral con reloj queda a cargo de Juan.

| Tiempo acumulado | Slide | Acción docente |
|---|---|---|
| 0:00–1:00 | 14, RL en modelos de lenguaje | Explicar que un modelo puede aprender de ejemplos y de evaluaciones de sus respuestas. Presentar preferencias e InstructGPT como hitos |
| 1:00–2:30 | 14 | Preguntar qué explicación de una política prefieren: una correcta pero ininteligible o una clara. Recoger criterios |
| 2:30–3:30 | 14 | Recorrer prompt → respuestas → preferencia → ajuste. Aclarar que una persona no puntúa cada rollout y que SFT y RL cumplen funciones distintas |
| 3:30–4:30 | 15, El camino hacia GRPO | Usar 17 × 23 = 391. Comparar las cuatro respuestas y el verificador 0/1 |
| 4:30–6:00 | 15 | Separar fuente de recompensa y algoritmo. GRPO compara respuestas del mismo problema; no enseñar aún la loss |
| 6:00–7:00 | 16, transición | Preguntar cuáles serían estado, acción y recompensa. Recoger intuiciones y anunciar que volveremos al ejemplo en el encuentro 3 |
| 7:00–8:00 | Margen opcional | Una aclaración breve o participación adicional; luego pasar a nociones básicas |

## Guion oral sugerido

**Slide 14.** «Hasta acá vimos agentes que aprenden a tomar decisiones. Un modelo de lenguaje también toma decisiones: elige cómo continuar una respuesta. Además de mostrarle ejemplos, podemos comparar lo que produce. Si dos respuestas explican el mismo concepto, ¿cuál elegirían y por qué? Esa evaluación puede servir para aprender una recompensa y ajustar el comportamiento del modelo. InstructGPT es un ejemplo de combinación de demostraciones supervisadas, preferencias humanas y RL. No significa que una persona mire cada paso de entrenamiento ni que todos los modelos usen la misma receta».

**Slide 15.** «En algunas tareas podemos comprobar un resultado automáticamente. Acá sabemos que 17 por 23 da 391. Generamos varias respuestas al mismo problema y las evaluamos. Una comparación dentro del grupo puede dar una señal relativa para aprender: esa es una de las piezas de GRPO. La pregunta por quién evalúa y la pregunta por cómo actualizamos son diferentes. Una preferencia humana y un verificador son fuentes de recompensa; GRPO es un algoritmo. En el encuentro 3 vamos a construirlo desde gradientes de política, ventajas, PPO y comparación por grupos».

**Slide 16.** «Para entender por qué funciona una actualización necesitamos precisar qué observa el agente, qué decide y qué recompensa recibe. Vamos a empezar con problemas pequeños. Después recuperaremos este mismo ejemplo de generación».

## Control del recorrido

- La introducción anticipa el objetivo sin exigir conocer Transformers, PPO ni un critic.
- El ejemplo verificado no se presenta como la receta completa de DeepSeekMath ni como prueba de fidelidad del razonamiento.
- Los fundamentos continúan con agente, entorno, política, retorno y MDP; el vocabulario se retoma en el encuentro 3.
- La agenda, el cierre y sus notas no presentan el TP. Decision Transformer se introduce recién en el encuentro 4.
- El warm-up y Q-learning son prácticas de clase. La última slide enlaza el notebook, declara IA opcional y no pide informe, defensa ni entrega obligatoria.
- La práctica de Q-learning tiene su propia propuesta de 45 minutos; los 6–8 minutos corresponden solo al anticipo de RL en LLMs.

## Ética: 28 minutos al cierre del encuentro 2

Slides 80–86: apertura (1 min), Navier–Stokes (5), reward hacking (5), ciberseguridad (4), pluralismo en RLHF (4), actividad A/B (6) y cierre (3). Reservar dos minutos adicionales de margen. [Contenido, fuentes y guion completo](ETICA_RL.md).

La actividad contrasta dos explicaciones correctas de una política; cada pareja justifica su preferencia según el público y propone registrar el desacuerdo. No agrega entrega ni requiere IA. En los casos reales se distinguen resultados de experimentos, relatos de empresas y asuntos no comprobados.

## Verificación realizada

La revisión introductoria del 17/10 se conserva en el [registro anterior](../validacion/clase1_revision.json). La revisión del bloque de ética y del cierre se registra por separado en [clase1_etica_revision.json](../validacion/clase1_etica_revision.json). Fuentes y notas docentes están dentro de la presentación; el PDF corresponde a la versión actual.

Siguiente revisión docente: ensayar en voz alta, confirmar afiliaciones de la slide 2 si cambiaron y ajustar tiempos sin sumar derivaciones a la introducción. La edición de Clase 2 y el cierre del protocolo de evaluación del único TP son trabajos posteriores.
