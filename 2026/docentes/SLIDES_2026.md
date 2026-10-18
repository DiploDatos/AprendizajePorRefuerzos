# Cambios propuestos para las diapositivas

Revisión 17/10/2026. Clase 1 editada y revisada (87 slides); plan de Clase 2 pendiente. Las [copias editables 2026 en Drive](../DRIVE_2026.md) ya están creadas. IDs verificados por outline nativo el 12/10/2026. Editar las copias 2026; actualizar el año dentro de las slides y los enlaces. No suponer que deck “Clase 2” es encuentro 2.

## Clase 1

Cambios de introducción, portada, agenda, transición, noción online/offline, preferencias y actividad final aplicados. Las nuevas slides son 14 y 15. Se actualizaron los enlaces de laboratorio y las notas de Q-learning/recompensa. La revisión amplia de la notación de Bellman y las afiliaciones no se da por cerrada. Ver [ensayo y registro](../clase1/GUIA_PRESENTACION.md). Los números siguientes se refieren a la versión 2025.

| Slide / ID | Cambio | Motivo |
|---|---|---|
| 1 / `p` | Año 2026; verificar afiliaciones docentes de slide 2 con Juan/Jorge | Evitar copiar datos profesionales desactualizados |
| 3 / `gb2e30c18f0_0_263` | Presentar recorrido, actividades formativas y único TP de Decision Transformer | Distinguir práctica de clase y entrega |
| 14–22 / IDs detallados abajo | Reemplazar bloque de overhype/HARKing y actualidad por dos slides introductorias de RL en LLMs | Anticipar el encuentro 3 con preferencias, RLHF y GRPO |
| 27 / `gb49cbc1f9c_0_119` | Cambiar “no hay dataset” por “en online RL el agente genera experiencia; en offline RL usa experiencia registrada” | Evitar contradicción con offline RL |
| 35 / `gb49cbc1f9c_0_17` | Revisar notación de Bellman y retorno también en speaker notes | Fetch incluye una nota TODO sobre notación |
| 70 / `gb49cbc1f9c_0_346` | Añadir ejemplo max frente a argmax y máscara terminal | Alineación con implementación nueva |
| 88 / `gb49cbc1f9c_0_149` | Añadir FrozenLake: recompensa por vivir frente a llegar | Caso de ética y proxy ejecutable |
| 93 / `g39cfa96d3e5_0_0` | Explicitar sesgos/desacuerdos en feedback humano | Preparar actividad RLHF |
| 94 / `gb4eea7d43a_0_5` | Reemplazar por actividad formativa Q-learning y aclarar que el único TP es DT | Sin entrega obligatoria de tabular |

### Dos slides introductorias en lugar del bloque histórico

Usar [contenido y notas listos para revisión](../clase1/SLIDES_INTRO_LLM.md), 6–8 minutos en total:

1. **RL también puede mejorar cómo responde un modelo:** preferencias como señal, InstructGPT y ejemplo de comparación de explicaciones.
2. **De preferencias a respuestas verificables: el camino hacia GRPO:** DeepSeekMath, DeepSeek-R1 y comparación de respuestas al mismo problema; anticipo del encuentro 3.

Retirar del recorrido principal estas slides originales y sustituir el conjunto por las dos nuevas. Los números son los del deck 2025; usar IDs al editar:

| Número | ID original | Contenido que se sustituye |
|---|---|---|
| 14 | `gb2f99adadb_0_34` | Overhype |
| 15 | `gb2e30c18f0_0_199` | Crisis alrededor de 2016 |
| 16 | `gbc696bf0ec_0_1` | Ablaciones y HARKing |
| 17 | `gbc2dfa48f5_0_0` | Definición de HARKing |
| 18 | `gbc2dfa48f5_0_9` | Papers de reproducibilidad |
| 19 | `gb2e30c18f0_0_254` | Meme de Irpan |
| 20 | `gb2e30c18f0_0_241` | Anécdota de Atari |
| 21 | `gb2e30c18f0_0_205` | Actualidad general |
| 22 | `g25f39d90f12_0_0` | RLHF |

Conservar la slide 23 como transición hacia desafíos y fundamentos. La evaluación experimental se enseña con el laboratorio: comparar entrenamiento/evaluación, inspeccionar un contraejemplo y distinguir recompensa de éxito. La versión de cinco semillas es ampliación optativa; no agregar una presentación de otro TP ni pedir informes de Q-learning.

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
6. DPO y variantes quedan como comparación final/lecturas; DT mantiene su bloque aparte para el único TP.

Modelo ejecutado: GRU de 13.923 parámetros; no LLM preentrenado. La extensión Orin queda pendiente para cuando Juan conecte el equipo. Material nuevo ejecutado localmente, deck Drive sin editar.

## Criterio de reducción

La segunda presentación tiene 143 slides, con aplicaciones detalladas en apéndice. Para habilitar la práctica, mantener una aplicación de juegos, una de recomendación/control y el caso de preferencias. El resto queda como lectura. Preservar los enlaces a fuentes, revisando los que se usen en vivo durante el ensayo final.
