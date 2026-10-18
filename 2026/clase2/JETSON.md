# Jetson Orin Nano 8 GB — preparación y conexión

Actualizado el 12/10/2026. **Equipo confirmado por Juan: Orin Nano de 8 GB.** La demo pertenece al **bloque de diapositivas de deep RL (Clase 2)**, no al segundo encuentro.

## Información localizada

La documentación local de `home_management` identifica el alias SSH **`jetson`**, y la configuración SSH de esta computadora coincide con ese inventario. El archivo `devices/jetson-orin-nano/README.md` registra JetPack 6, L4T R36.5.2 y CUDA 12.6; son datos documentados, pendientes de contrastar en vivo. No hace falta volver a pedir usuario ni dirección mientras ese inventario siga vigente.

Se intentó SSH con la clave ya configurada. La conexión agotó el tiempo de espera. También agotaron el tiempo las conexiones a los servicios de Ollama y WebUI documentados. La ruta de esta computadora apunta directamente a la misma subred por `en0`; hay una entrada ARP para el equipo. Eso no confirma que esté encendido ni que conserve su dirección. **No se logró abrir una sesión remota, entrenar ni medir rendimiento en la placa.** Juan confirmó que el equipo está desconectado y pidió dejar la prueba para después; no seguir intentando hasta que lo conecte.

## Elección del experimento

El notebook principal `llm_policy_gradients.ipynb` añade RLOO/GRPO sobre la misma GRU; `mini_rlhf.ipynb` sigue siendo el ejercicio mínimo para enseñar todo el pipeline y un fallback local ya validado. Con una **Orin Nano de 8 GB** también conviene ensayar una extensión sobre un transformer preentrenado pequeño; ver [plan de LLM para Orin](ORIN_LLM_PLAN.md). No se supone que la ejecución de un modelo cuantizado en Ollama permita ajustar sus pesos: inferencia y entrenamiento requieren entornos y formatos diferentes.

## Diagnóstico antes de instalar

Usar la [guía NVIDIA de PyTorch para Jetson](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html) y la matriz correspondiente a la versión detectada. Las wheels deben corresponder a JetPack, Python y aarch64. Mantener un entorno aislado por experimento, según el runbook de `home_management`.

`diagnostico_jetson.py` lee modelo, L4T, memoria y versiones, y prueba un backward CUDA mínimo si está disponible. No cambia configuración. El ejercicio GRU usa solo PyTorch; evita Transformers, TRL, bitsandbytes, NumPy y Gymnasium. Su compatibilidad en la placa todavía no se comprobó.

## Ejecución una vez restablecido SSH

Ejemplos desde la raíz de este repo. Usar el alias existente; no copiar claves privadas ni direcciones domésticas al material público.

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 jetson 'python3 -V; cat /etc/nv_tegra_release; cat /proc/device-tree/model; free -h'
ssh jetson 'mkdir -p ~/workspace/rl2026-demo'
scp 2026/clase2/mini_rlhf.py 2026/clase2/llm_policy_gradients.py 2026/clase2/diagnostico_jetson.py jetson:~/workspace/rl2026-demo/
ssh jetson 'cd ~/workspace/rl2026-demo && python3 diagnostico_jetson.py'
ssh jetson 'cd ~/workspace/rl2026-demo && python3 llm_policy_gradients.py --device cpu --output outputs/simulado'
```

Si PyTorch se encuentra en un venv, usar ese intérprete después de verificarlo, o crear uno dedicado compatible. No reemplazar una instalación funcional para satisfacer el `pyproject.toml` general del curso. Si CUDA pasa el diagnóstico, comparar explícitamente `--device cuda`; para esta GRU CPU puede ser más rápida.

El runbook doméstico indica `MAXN_SUPER` para inferencia/builds y `15W` en reposo. Registrar el modo al medir, y seguir ese protocolo si se necesita cambiarlo. La prueba de conectividad no cambió modos de potencia, servicios ni sistema operativo.

## Feedback humano

Etiquetar en la laptop, transferir el archivo y entrenar en la placa:

```bash
python3 2026/clase2/mini_rlhf.py --export-pairs /tmp/preferencias_humanas.json
python3 2026/clase2/mini_rlhf.py --label-pairs /tmp/preferencias_humanas.json
scp /tmp/preferencias_humanas.json jetson:~/workspace/rl2026-demo/
ssh jetson 'cd ~/workspace/rl2026-demo && python3 llm_policy_gradients.py --preferences preferencias_humanas.json --device cuda --output outputs/humano'
scp -r jetson:~/workspace/rl2026-demo/outputs/humano /tmp/rl2026-jetson-humano
```

Usar `cpu` si CUDA no fue validada. No se necesita Jupyter en la placa para la ruta CLI: se guarda el mismo checkpoint, métricas y evaluación ciega. El notebook puede proyectarse desde la laptop con resultados recuperados; un kernel remoto es opcional.

## Criterios de cierre

- Import y backward funcionan en el device elegido.
- SFT + reward model + RL terminan, con checkpoint y métricas recuperados.
- Política cambia y referencia permanece intacta (asserts incluidos).
- Registrar modelo, RAM, L4T, Python, PyTorch, device, modo de potencia y tiempo real.
- Realizar una corrida con etiquetas humanas y revisar salidas nuevas a ciegas.
- Ensayar la proyección de clase y mantener fallback CPU local.

Los pesos FP32 de la GRU ocupan unos 55 KB; el checkpoint local medido ocupa unos 123 KB. **Esto no estima la RAM total ni la latencia de la Orin:** deben medirse junto con runtime, optimizador, referencia y activaciones.
