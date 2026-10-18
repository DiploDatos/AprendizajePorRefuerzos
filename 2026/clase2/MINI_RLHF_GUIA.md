# Alternativa — mini RLHF centrado en preferencias

Alternativa de 65 minutos a la práctica principal de [gradientes de política](GUIA_CLASE.md). Duración, para el bloque de deep RL del deck Clase 2, ubicación confirmada por Juan. Versión de contingencia de 20 minutos al final. El programa ya menciona RLHF, DPO y RLVR; este ejercicio muestra el pipeline explícito de preferencias y recompensa.

## Propósito

Que los estudiantes puedan identificar qué parte aprende de demostraciones, qué parte aprende preferencias y qué parte se optimiza por RL. El resultado es un modelo de lenguaje pequeño, no un asistente de propósito general. Se ajusta una GRU con vocabulario cerrado, entrenada primero en un corpus de respuestas docentes. No se descargan pesos.

El hardware confirmado es **Orin Nano de 8 GB**. La GRU queda como demo mínima y fallback validado localmente; la latencia en la placa aún debe medirse. Se preparó además un [plan de extensión con LLM preentrenado](ORIN_LLM_PLAN.md). Usar PyTorch compatible con el stack detectado según [NVIDIA](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html).

## Secuencia de 65 minutos

| Minutos | Acción | Pregunta |
|---|---|---|
| 0–8 | Comparar respuestas del modelo tras SFT | ¿Imitar un corpus equivale a responder como queremos? |
| 8–20 | Etiquetar 24 pares A/B en grupos; acordar criterio | ¿Hay una preferencia universal? ¿Cuándo usar empate? |
| 20–30 | Entrenar reward model; ver loss y pares reservados | ¿Aprendió la preferencia o un atajo lexical? |
| 30–43 | Congelar referencia y reward model; actualizar política | ¿Qué hace la KL y por qué no es Q-learning? |
| 43–55 | Evaluar salidas nuevas a ciegas; medir proxy y repetición | ¿Más recompensa significa mejor texto? |
| 55–65 | Comparar beta y conectar con DPO/RLVR | ¿De dónde sale la señal en cada método? |

## El modelo y sus límites

- Generador: embedding de 24, GRU de 48 y head de vocabulario; **13.923 parámetros** en la versión validada. Tres tokens de contexto y máximo diez acciones por respuesta. Usa EOS; no calcula loss sobre PAD ni logprob después de EOS.
- SFT: 18 respuestas docentes, incluidas respuestas útiles, bruscas, largas y repetitivas. Esto es un corpus artificial para la actividad, no un preentrenamiento representativo.
- Reward model: **1.089 parámetros**, features de bolsa de palabras, longitud, repetición y contexto. Pierde orden y semántica; ideal para discutir un proxy imperfecto.
- Preferencias: 18 pares de entrenamiento y seis de validación. Separación por par, con contextos y respuestas compartidos. No equivale a holdout semántico.
- RL: REINFORCE on-policy, baseline móvil independiente del batch actual y penalización de log-ratio respecto de referencia congelada. Sin critic ni PPO clipping. El uso de feedback humano determina si la corrida es RLHF humano; el ensayo sintético se etiqueta como simulado.
- Evaluación: 90 muestras por política y 12 pares antes/después cegados. Una sola semilla no permite afirmar robustez estadística ni calidad general. Pedir razones humanas y revisar diversidad, relevancia y fallos, además del score.

El uso de una recompensa aprendida de comparaciones sigue el enfoque de [Christiano et al.](https://arxiv.org/abs/1706.03741). El pipeline de [InstructGPT](https://arxiv.org/abs/2203.02155) sirve para contrastar esta escala docente con el ajuste de grandes modelos. No reproducimos su arquitectura ni su algoritmo PPO.

## Ejecución con etiquetas humanas

1. Generar plantilla: `python mini_rlhf.py --export-pairs preferencias_humanas.json`.
2. Etiquetar con `python mini_rlhf.py --label-pairs preferencias_humanas.json`, o editar el JSON poniendo `source: human` y `choice: a/b/tie/skip` en cada par. No modificar textos ni particiones.
3. En notebook, reemplazar `PREFERENCE_FILE = None` por el path del archivo. Ejecutar todas las celdas desde un kernel nuevo.
4. Revisar `feedback_source: HUMAN` en métricas. La evaluación humana de salidas nuevas sigue pendiente: el código conserva `human_quality_verified: false` porque el origen humano del train no prueba calidad del resultado.
5. Otro grupo etiqueta `evaluacion_ciega.json` sin consultar `clave_evaluacion_docente.json`. Recién al final contar preferencias por política, empates y skips. No reutilizar ese test para ajustar beta.

No se guardan nombres de estudiantes ni datos privados. La actividad solo requiere preferencias sobre frases docentes. Si hay desacuerdo, conservarlo y discutirlo; no convertir una opinión mayoritaria en verdad objetiva.

## Checklist de ensayo docente

- Ejecutar diagnóstico y script en la Jetson identificada; anotar dispositivo real, versiones, tiempo, memoria y posibles advertencias.
- Tener checkpoint SFT/RL y métricas de una corrida previa, más CPU local disponible.
- Probar con anotaciones humanas reales; no prometer que todas las preferencias producirán las mismas respuestas.
- Mostrar que la referencia no cambia y que la política sí cambia (asserts incluidos).
- Revisar una salida que contradiga el reward model, por ejemplo una negación o palabras repetidas fuera del corpus.
- Si la instalación Jetson falla, usar CPU y mostrar el límite encontrado; no perder la clase resolviendo dependencias.

## Cápsula de 20 minutos para el segundo encuentro

3 min: mostrar dos salidas; 5 min: comparar cuatro pares; 5 min: visualizar el entrenamiento y explicar SFT/recompensa/política; 5 min: revisar antes/después; 2 min: volver a la pregunta de la actividad Q-learning. Usar checkpoint preparado y derivar REINFORCE más adelante.

## Comparación conceptual

| Método | Señal | Qué mostrar |
|---|---|---|
| SFT | Respuesta demostrada | Imita la distribución de corpus |
| RLHF de esta demo | Preferencia humana → recompensa aprendida | Muestrea salidas y optimiza política con KL |
| DPO | Pares preferido/rechazado | Objetivo directo sobre log-probabilidades, sin loop RM + RL explícito; [paper](https://arxiv.org/abs/2305.18290) |
| RLVR | Verificador de tarea | Recompensa por respuesta verificable; no presupone feedback humano |

Una extensión con un transformer preentrenado pequeño requiere un ensayo propio de compatibilidad, memoria y entrenamiento. No se afirma que un LLM de cientos de millones de parámetros sea ajustable aquí por usar LoRA o cuantización.
