# Reorganización de la edición 2026

Fecha: 17/10/2026. Decisiones indicadas por Juan.

| Antes | Ahora |
|---|---|
| Introducción centrada en crisis de reproducibilidad de 2016, HARKing y anécdotas históricas | Dos slides sobre preferencias/RLHF y evolución hacia GRPO, como anticipo del encuentro 3 |
| Nuevo TP1 propuesto de FrozenLake/Taxi | Actividad formativa de Q-learning en `actividades/q_learning/`, sin nota ni entrega obligatoria |
| TP2 de Decision Transformer | Único TP del curso, con entrada vigente en `tp/` y revisión técnica explícita |
| Material 2025 en la raíz del repositorio | PDFs, labs, imágenes y TP anterior en `archive/2025/` |
| Checkout de la rama `2025` | Rama local `2026`, creada desde `22824ca` |

## Conservación

El traslado del material antiguo se registra con hashes en [MANIFIESTO.json](../archive/2025/MANIFIESTO.json). Se conservó el notebook anterior con los cambios locales de Juan, sin editar sus celdas. La configuración local `pyproject.toml`, `uv.lock`, los entornos y los archivos de su editor no se modificaron. Las copias históricas de configuración proceden del commit de origen.

Los datos de recomendaciones se mantienen en una sola ubicación, dentro del archivo 2025. La entrada 2026 los resuelve por ruta explícita, sin duplicar los datasets.

## Estado de las presentaciones

Se actualizaron los guiones locales y la correspondencia con IDs de las slides originales. Se crearon [copias editables 2026 en Drive](DRIVE_2026.md); Clase 1 ya fue editada; Clase 2 conserva el contenido anterior. Los originales y los PDFs históricos siguen sin editar. Las siguientes ediciones se harán sobre esas copias; los IDs y el detalle están en [SLIDES_2026.md](docentes/SLIDES_2026.md).

## Alcance de validación

La reorganización no vuelve a entrenar agentes ni valida la implementación histórica de Decision Transformer. Se comprueban rutas, integridad del archivo y las invariantes de los laboratorios ejecutados. El cierre técnico de la evaluación del único TP sigue como próximo paso antes de publicarlo.
