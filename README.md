# Aprendizaje por Refuerzos — DiploDatos 2026

Material de la materia optativa de la Diplomatura en Ciencias de Datos, Aprendizaje Automático y sus Aplicaciones. La edición en preparación está en la rama `2026`.

## Ejecutar el material

Con [uv 0.9.2](https://docs.astral.sh/uv/getting-started/installation/#installing-a-specific-version), desde un clon de la rama `2026`:

```bash
uv python install 3.12
uv sync --locked
uv run --locked jupyter lab
```

[Instalación, plataformas, extra de Hugging Face y pruebas](ENTORNO.md). Python 3.12 es la ruta recomendada; las dependencias completas están fijadas en `uv.lock`. La configuración de Jetson se maneja por separado.

## Material

- [Material y recorrido 2026](2026/README.md).
- [Clase 1 actualizada en Drive](https://docs.google.com/presentation/d/1cBZzjKvEYI1Cok3yE6IBRSureGgnzkK5SrfsgF7YsL8/edit), [PDF](2026/clase1/Clase_1_2026.pdf) y [guion de ensayo](2026/clase1/GUIA_PRESENTACION.md).
- [Actividad de Q-learning](2026/actividades/q_learning/CONSIGNA.md): práctica formativa de clase, sin entrega obligatoria.
- [Actividades de las unidades 1–4 de Hugging Face](2026/actividades/huggingface/README.md), con pruebas en Colab y alcance de cada validación.
- [Encuentro 3: camino hacia GRPO](2026/clase2/CAMINO_A_GRPO.md) y [notebook](2026/clase2/llm_policy_gradients.ipynb).
- [Único trabajo práctico: Decision Transformer para recomendaciones](2026/tp/README.md), continuación del anterior TP2.
- [Archivo de la edición 2025](archive/2025/EDICION.md) y [agentes históricos](archive/agents/).

Las copias editables 2026 de ambas clases ya están actualizadas. Clase 2 recupera el recorrido general: 115 slides principales de deep RL, RL en modelos de lenguaje, Offline RL/Decision Transformer y aplicaciones; 39 de profundización y una del TP omitidas. El PDF conserva todas las páginas salvo el TP. Los decks de 2025 y sus copias PDF archivadas conservan su contenido histórico. Las instrucciones de ejecución y la evidencia están en el índice 2026. Juan recuperó acceso a la Orin Nano; todavía falta validar el entrenamiento en la placa.
