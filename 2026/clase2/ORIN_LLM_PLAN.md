# Extensión: ajustar un LLM pequeño en la Orin Nano de 8 GB

Propuesta preparada tras confirmar el hardware. **Todavía no implementada ni medida en la placa:** Juan confirmó que el equipo está desconectado y aplazó el ensayo. Mantener el notebook de gradientes de política con GRU como ejercicio del pipeline y fallback; no describir esta extensión como entrenamiento ya logrado.

## Candidato inicial

[HuggingFaceTB/SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct): transformer de 135 millones de parámetros, licencia Apache 2.0 y orientado al inglés. La model card indica que su versión instruct ya fue ajustada mediante SFT y DPO. Para la actividad se parte de ese checkpoint y se hace un ajuste adicional con preferencias propias; no se atribuye su entrenamiento original al curso.

La elección es una hipótesis de viabilidad por tamaño, no un benchmark de Orin. Usar consignas/respuestas cortas en inglés y explicaciones docentes en español para no confundir limitación multilingüe con efecto de RL.

## Pipeline propuesto

1. Diagnosticar Python, PyTorch/CUDA, RAM disponible y servicios activos. No detener Ollama/WebUI automáticamente; decidir compatibilidad de recursos con la carga real.
2. Crear entorno aislado. Fijar versiones comprobadas en aarch64 y revisión exacta del modelo/tokenizador, con licencia y checksum registrados.
3. Medir generación base con contexto de hasta 128 tokens y salida de hasta 24 tokens, batch 1 y atención estándar.
4. Añadir adaptadores LoRA (rank 4 u 8); mantener backbone congelado. Preferir FP32 inicialmente para estabilidad; ensayar precisión reducida solo si el stack y las mediciones lo justifican.
5. Recopilar 24–40 pares humanos sobre una preferencia acotada (por ejemplo, respuesta breve, útil y sin repetición). Separar preferencias de recompensa y evaluación final; conservar empates y desacuerdos.
6. Entrenar un reward model pequeño con features/representaciones congeladas. Comprobar que no premie solo brevedad, palabras específicas o respuestas vacías.
7. Muestrear respuestas on-policy y ajustar LoRA con REINFORCE + KL respecto del checkpoint de partida. Conservar referencia congelada. Añadir RLOO y luego GRPO con grupos pequeños solo después de medir generación y memoria. Mantener la secuencia conceptual del notebook `llm_policy_gradients.ipynb`; PPO queda como comparación sin critic ejecutable.
8. Guardar adaptadores, métricas, ejemplos, pico de memoria y tiempo; comparar antes/después a ciegas. Medir tanto reward proxy como preferencia humana, diversidad y pertinencia.

## Memoria y ensayo

135 millones de parámetros requieren aproximadamente 540 MB en FP32 o 270 MB en FP16 **solo para pesos**. Una referencia duplicada suma pesos; runtime, activaciones y optimizador agregan consumo. LoRA reduce los parámetros entrenables, pero no elimina el backbone. La RAM de la Jetson es compartida entre CPU/GPU: no tratar sus 8 GB como memoria enteramente libre.

Empezar por un forward/backward mínimo y 5–10 actualizaciones. Solo aumentar el presupuesto después de medir. No usar bitsandbytes/QLoRA como dependencia inicial sin verificar su soporte real para el stack aarch64 instalado.

## Material a producir tras el ensayo

- `orin_llm_rlhf.ipynb`, separado del ejercicio GRU, con etapas y resultados visibles.
- Entorno fijado, revisión del modelo y comandos reproducibles.
- JSON de preferencias humanas y evaluación ciega, sin nombres ni datos personales.
- Adaptadores y métricas fuera de Git; paths documentados en la placa.
- Tabla de tiempo/memoria y decisión docente de incluirlo en el bloque deep RL.

Si se elige DPO por simplicidad, presentarlo explícitamente como otra actividad: no muestra el entrenamiento de reward model seguido por un loop on-policy de RL que buscamos enseñar aquí.
