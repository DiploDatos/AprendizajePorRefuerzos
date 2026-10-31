# Entorno del curso 2026

La instalación vigente usa **uv 0.9.2**, Python **3.12** por defecto y `uv.lock` versionado. Las dependencias directas están fijadas en `pyproject.toml`; el lock fija también las transitivas. No usar el Poetry histórico para instalar la edición 2026.

## Instalación desde un clon nuevo

Instalar [uv 0.9.2](https://docs.astral.sh/uv/getting-started/installation/#installing-a-specific-version) y ejecutar:

```bash
git clone --branch 2026 https://github.com/DiploDatos/AprendizajePorRefuerzos.git
cd AprendizajePorRefuerzos
uv python install 3.12
uv sync --locked
uv run --locked jupyter lab
```

Jupyter se inicia desde la raíz del repo. Abrir los notebooks dentro de `2026/`. No hace falta activar el entorno ni instalar paquetes desde celdas. Cerrar Jupyter con Ctrl+C.

El entorno cubre Q-learning, las demos REINFORCE/RLOO/GRPO, SB3 con DQN/PPO, gráficos y datos/PyTorch para implementar DT. Usa CPU en Linux/Windows; macOS Apple Silicon instala el wheel de PyPI. No hace falta GPU para estas actividades. [Configuración de PyTorch con uv](https://docs.astral.sh/uv/guides/integration/pytorch/).

Plataformas del lock: Linux x86_64, Windows x86_64 y macOS Apple Silicon. Python admitido: 3.11–3.13; se recomienda 3.12. Python 3.14, macOS Intel y Jetson requieren otra configuración. No se presume que un wheel CPU de ARM sirva para CUDA en Jetson.

## Decision Transformer con Hugging Face

La implementación en PyTorch no requiere Transformers. Para la alternativa con Hugging Face:

```bash
uv sync --locked --extra tp-hf
uv run --locked --extra tp-hf jupyter lab
```

Mantener `--extra tp-hf` también en `uv run`: uv sincroniza el entorno con las opciones de cada comando. La verificación instancia modelos pequeños desde configuración; no descarga pesos ni requiere cuentas de Hugging Face.

Los datos del TP están en `archive/2025/tp_decision_transformer/data/`; `2026/tp/config_dataset.py` resuelve las rutas. La prueba lee ambos datasets. **El entorno funcional no corrige ni valida el protocolo de evaluación histórico del TP**, que sigue en revisión docente.

## Comprobaciones reproducibles

Desde la raíz:

```bash
uv lock --check
uv run --locked python 2026/validacion/verificar_entorno.py --report outputs/environment.json
uv run --locked python 2026/validacion/verificar_notebooks.py --output outputs/notebooks
```

Para comprobar el extra:

```bash
uv run --locked --extra tp-hf python 2026/validacion/verificar_entorno.py --hf --report outputs/environment-hf.json
```

Se prueba reset/step/render RGB de tres entornos, entrenamiento corto de PPO, lectura de datos y los invariantes matemáticos del material. `--hf` añade causalidad y backward de DT. Los notebooks docentes se ejecutan con kernels nuevos del mismo intérprete, en una copia temporal, sin cambiar originales ni resultados locales.

El notebook estudiantil de Q-learning tiene un TODO intencional. Se comprueba su sintaxis; se usa la referencia docente para ejecución completa. El warm-up es introductorio y actualmente no tiene celdas de código.

El workflow `.github/workflows/curso.yml` instala desde cero y ejecuta las comprobaciones en Linux (Python 3.11/3.12/3.13), macOS ARM y Windows (3.12). Consultar el resultado real de Actions; una matriz configurada no equivale a una ejecución aprobada.

## Conservar otro entorno local

Para mantener un entorno anterior se puede elegir otro directorio. En macOS/Linux:

```bash
UV_PROJECT_ENVIRONMENT=.venv-2026 uv sync --locked --extra tp-hf
UV_PROJECT_ENVIRONMENT=.venv-2026 uv run --locked --extra tp-hf jupyter lab
```

En PowerShell, establecer `$env:UV_PROJECT_ENVIRONMENT = '.venv-2026'` antes de los comandos `uv`. En la computadora de Juan se preparó `.venv-2026` manteniendo `.venv` intacto. No se versionan estos directorios: se reconstruyen desde el lock.

No mezclar instalaciones con `pip install` ni usar `--upgrade` durante la clase. Para cambiar dependencias, editar `pyproject.toml`, regenerar `uv.lock` y repetir las comprobaciones. `uv sync --locked` falla si manifiesto y lock no coinciden, en lugar de resolver nuevas versiones silenciosamente. [Lock y sincronización](https://docs.astral.sh/uv/concepts/projects/sync/).

## Archivo y Jetson

El material 2025 conserva su `pyproject.toml` y `poetry.lock` originales en `archive/2025/`. No se promete ejecutar todos los agentes históricos con el entorno 2026. RL Zoo, Atari, MuJoCo, Box2D y simuladores externos quedan fuera de la instalación básica: no son necesarios para las actividades vigentes.

La Orin Nano usa un entorno propio compatible con JetPack/PyTorch de NVIDIA. Seguir [JETSON.md](2026/clase2/JETSON.md) antes de instalar allí. Esta configuración no demuestra compatibilidad ni tiempos de entrenamiento en la placa.
