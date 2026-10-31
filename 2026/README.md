# Aprendizaje por Refuerzos — edición 2026

Revisión: 31/10/2026. Material en preparación, rama `2026`, con [copias de trabajo en Drive](DRIVE_2026.md); aún sin publicación en Campus. Clase 1 ya fue editada y revisada en Drive; Clase 2 recupera el recorrido general: 115 slides principales de deep RL, RL en modelos de lenguaje, Offline RL/Decision Transformer y aplicaciones; 39 de profundización y una del TP omitidas. El PDF conserva todas las páginas salvo el TP. La Orin Nano de 8 GB volvió a estar accesible según Juan el 31/10; todavía no se ensayó el entrenamiento en la placa.

## Recorrido

1. [Clase 1 actualizada en Drive](https://docs.google.com/presentation/d/1cBZzjKvEYI1Cok3yE6IBRSureGgnzkK5SrfsgF7YsL8/edit), [PDF](clase1/Clase_1_2026.pdf), [guion de ensayo](clase1/GUIA_PRESENTACION.md) y [nuevo bloque de ética](clase1/ETICA_RL.md). [Plan de cuatro encuentros](01_PLAN_CURSO.md) y [dos slides introductorias de RL en LLMs](clase1/SLIDES_INTRO_LLM.md).
2. [Warm-up](actividades/lab_0_warmup_rl.ipynb), [actividad Q-learning](actividades/q_learning/CONSIGNA.md), [notebook inicial](actividades/q_learning/lab_1_q_learning_auditoria.ipynb) y [hoja de trabajo opcional](actividades/q_learning/INFORME_TEMPLATE.md). Son actividades formativas, sin nota ni entrega obligatoria.
3. Encuentro 3, bloque deep RL: [slides actualizadas](https://docs.google.com/presentation/d/19hKgrRzdaDW6sAlVUABjZJeOeElbK6Z1q1QkRfttv4w/edit), [PDF completo sin el TP](clase2/Clase_2_2026.pdf), [guía de presentación](clase2/GUIA_PRESENTACION.md), [camino hacia GRPO](clase2/CAMINO_A_GRPO.md), [guion](clase2/GUIA_CLASE.md), [storyboard](clase2/SLIDES_LLM_RL.md) y [notebook REINFORCE/RLOO/GRPO](clase2/llm_policy_gradients.ipynb).
4. Encuentro 4, preparación docente: [único TP, Decision Transformer para recomendaciones](tp/README.md), continuidad del anterior TP2. [Consigna 2026 en revisión](tp/CONSIGNA.md).
5. Docentes: [guía formativa de Q-learning](docentes/Q_LEARNING_GUIA.md), [plan de edición de slides](docentes/SLIDES_2026.md), [auditoría](00_AUDITORIA.md), [validación](validacion/RESULTADOS.md) y [próximos pasos](02_NEXT_STEPS.md).
6. [Archivo 2025](../archive/2025/EDICION.md) y [registro de esta reorganización](CAMBIOS_2026.md).

## Decisiones vigentes

- La introducción reemplaza el bloque de crisis de reproducibilidad de 2016/HARKing por dos slides: de preferencias a RLHF y de comparación de respuestas a GRPO. Anticipan el encuentro 3 sin derivar algoritmos.
- Hay **un único TP obligatorio**, sobre Decision Transformer. Se presenta recién en el encuentro 4; hasta entonces distribuir únicamente los materiales seleccionados para cada clase. Se abandona la numeración TP1/TP2 en el material vigente. Q-learning y RL para generación quedan como actividades de clase.
- Q-learning parte del [hands-on de Hugging Face](https://huggingface.co/learn/deep-rl-course/unit2/hands-on); conserva contraejemplos, semillas y análisis de recompensa. Su versión extensa es optativa.
- El notebook de deep RL ajusta un modelo de lenguaje GRU diminuto de 13.923 parámetros. Compara REINFORCE, RLOO y GRPO; ilustra PPO sin entrenarlo. No es un LLM preentrenado.
- Las preferencias iniciales son **simuladas**. Para realizar RLHF con feedback humano hay que etiquetar pares y seleccionar ese archivo. El verificador de formato constituye otra fuente de recompensa.
- Usar IA es opcional. El TP evalúa decisiones, experimentos y comprensión; no exige cuentas pagas ni publicación de modelos.

## Ejecutar las actividades

La instalación recomendada es el [entorno uv del repositorio](../ENTORNO.md): `uv sync --locked` y `uv run --locked jupyter lab`, desde la raíz. Cubre ambas actividades y las dependencias de datos/PyTorch para DT. El extra `tp-hf` añade Transformers/Accelerate. Las instrucciones siguientes describen también la opción tabular ligera y las dependencias mínimas para entornos propios.

Q-learning: Python 3.11–3.13 en un entorno separado, instalar [requirements.txt](actividades/q_learning/requirements.txt) y abrir el notebook desde la raíz o desde su carpeta. El modo `demo` sirve para la clase; `completo` amplía el experimento a cinco semillas sin crear una entrega adicional.

Deep RL: PyTorch funcional y Matplotlib. Desde la raíz, en el entorno local existente:

```bash
.venv/bin/python 2026/clase2/llm_policy_gradients.py --output /tmp/rl-policy-demo
```

El notebook no descarga pesos ni datos. [Mini RLHF](clase2/mini_rlhf.ipynb) es una alternativa para profundizar preferencias, no una segunda práctica obligatoria. Para Orin consultar [preparación](clase2/JETSON.md) y [extensión propuesta con LLM](clase2/ORIN_LLM_PLAN.md); no instalar allí las dependencias generales del repositorio.

Los resultados de ejecución en `outputs/` se ignoran por Git. `docentes/` contiene soluciones; distribuir a estudiantes solo los archivos de actividad seleccionados.
