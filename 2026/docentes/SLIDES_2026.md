# Cambios propuestos para las diapositivas

Revisión 18/10/2026. Clase 1 editada y revisada (87 slides); plan de Clase 2 pendiente. Las [copias editables 2026 en Drive](../DRIVE_2026.md) ya están creadas. IDs verificados por outline nativo el 12/10/2026. Editar las copias 2026; actualizar el año dentro de las slides y los enlaces. No suponer que deck “Clase 2” es encuentro 2.

## Clase 1 — cambios aplicados

Números actuales, deck 2026 de 87 slides. Se conservan las notas y fuentes; el [guion de ensayo](../clase1/GUIA_PRESENTACION.md) detalla el recorrido.

| Slide / ID | Estado y propósito |
|---|---|
| 1 / `p` | Portada 2026. Afiliaciones de slide 2 pendientes de confirmación docente |
| 3 / `gb2e30c18f0_0_263` | Agenda y notas de prácticas formativas; no se presenta el TP |
| 14–15 / `gb2e30c18f0_0_199`, `gb2e30c18f0_0_205` | Dos slides de preferencias/RLHF y comparación de respuestas/GRPO; 6–8 min, anticipo del encuentro 3 |
| 20 / `gb49cbc1f9c_0_119` | Distinción online/offline; experiencia generada o registrada |
| 80 / `gb4eea7d43a_0_10` | Apertura de ética: objetivo, beneficiarios y riesgo |
| 81 / `rl26_ethics_navier` | Crédito, confianza y verificación en Navier–Stokes |
| 82 / `rl26_ethics_hacking` | Reward hacking y modificación del evaluador |
| 83 / `rl26_ethics_cyber` | Uso dual, autonomía y límites del permiso |
| 84 / `rl26_ethics_plural` | Diversidad de preferencias: ¿alineado con quién? |
| 85 / `rl26_ethics_activity` | Actividad A/B de 6 min con dos respuestas correctas |
| 86 / `rl26_ethics_closing` | Elegir controles y una condición de parada |
| 87 / `gb4eea7d43a_0_5` | Notebook Q-learning, IA opcional, sin entrega obligatoria; sin anuncio del TP |

El bloque 80–86 reemplaza las seis slides anteriores de ética y conserva el diseño nativo. [Guía, preguntas, tiempos y fuentes](../clase1/ETICA_RL.md). La notación general de Bellman y la ampliación max/argmax siguen como revisión docente pendiente; esta edición no las da por corregidas.

El TP se presenta recién en el encuentro 4. No se agrega su anuncio al primer fin de semana ni al encuentro 3. Los archivos se mantienen en el índice docente para preparación y se publican a estudiantes cuando corresponda.

## Clase 2

| Slide / ID | Cambio | Motivo |
|---|---|---|
| 50 / `ge90cb544c9_0_20` | Mantener PPO; aclarar diferencias con REINFORCE docente | No atribuir PPO al notebook |
| 53 / `g36fb9856750_1_83` | Presentar SFT → preferencias → RM → RL + KL | Define las etapas que se ejecutan |
| 54 / `g1e842fd07c2_2_0` | Sustituir “RM: otro LLM” por “modelo que asigna score; puede compartir arquitectura o ser mucho más pequeño” | La demo usa MLP, válido para enseñar preferencias |
| 55 / `g36fb9856750_1_76` | Insertar enlace a `llm_policy_gradients.ipynb`; ejecutar comparación REINFORCE/RLOO/GRPO | Convertir diagrama en experiencia observable |
| 56 / `g3a0d1898e12_0_10` | Separar RLHF y RLVR según origen de recompensa | No todo refuerzo de razonamiento usa feedback humano |
| 57–62 / `g385e773c5af_0_0` a `g36fb9856750_1_91` | DPO como comparación breve; objetivo directo, referencia y pares | Evitar otro bloque de implementación largo |
| 70 / `g39ef4f1d585_0_6` | Cambiar “genera acciones que logran ese return” por “condiciona acciones en el return deseado; no garantiza alcanzarlo” | Distinguir deseo y resultado |
| 71 / `g39ef4f1d585_0_12` | Añadir límites de extrapolar fuera de returns del dataset | Evitar prometer rendimiento fuera de distribución |
| 112 / `gf9badb265c_0_401` | Renombrar como único TP: Decision Transformer, con evaluación revisada | El ranking de IDs actual no mide satisfacción |

### Nuevo hilo del bloque deep RL

Juan pidió mejorar el camino hacia GRPO, usando la guía de RL para LLMs como referencia. Usar [storyboard de 22 slides](../clase2/SLIDES_LLM_RL.md) y [guion](../clase2/GUIA_CLASE.md): generación como MDP, objetivo y KL, REINFORCE/baseline, PPO, fuentes de recompensa, RLOO/GRPO, implementación y evaluación. Reemplazar la secuencia breve de RLHF por ese desarrollo, reduciendo aplicaciones para dar lugar a la práctica.

1. Antes de PPO (slide 50), introducir MDP de tokens, ventaja, baseline y REINFORCE.
2. En PPO, usar cuatro casos numéricos y distinguir actual/vieja/referencia. Notebook muestra loss; no entrena PPO.
3. En slides 53–56, conservar preferencias/RM y separar fuente RLHF/RLVR de algoritmo.
4. Añadir RLOO/GRPO antes del laboratorio; insertar enlace a `llm_policy_gradients.ipynb` en lugar de poner la demo REINFORCE como única práctica.
5. Después del notebook, evaluar a ciegas y discutir diversidad, grupos sin señal y presupuestos.
6. DPO y variantes quedan como comparación final/lecturas; DT mantiene su bloque aparte en el encuentro 4, donde recién se presenta el único TP.

Modelo ejecutado: GRU de 13.923 parámetros; no LLM preentrenado. La extensión Orin queda pendiente para cuando Juan conecte el equipo. Material nuevo ejecutado localmente, deck Drive sin editar.

## Criterio de reducción

La segunda presentación tiene 143 slides, con aplicaciones detalladas en apéndice. Para habilitar la práctica, mantener una aplicación de juegos, una de recomendación/control y el caso de preferencias. El resto queda como lectura. Preservar los enlaces a fuentes, revisando los que se usen en vivo durante el ensayo final.
