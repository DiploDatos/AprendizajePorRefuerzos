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

Se actualizaron los guiones locales y la correspondencia con IDs de las slides originales. Se crearon [copias editables 2026 en Drive](DRIVE_2026.md); Clase 1 ya fue editada; Clase 2 se actualizó el 31/10 con el recorrido hacia GRPO. Los originales y los PDFs históricos siguen sin editar. Las siguientes ediciones se harán sobre esas copias; los IDs y el detalle están en [SLIDES_2026.md](docentes/SLIDES_2026.md).

## Alcance de validación

La reorganización no vuelve a entrenar agentes ni valida la implementación histórica de Decision Transformer. Se comprueban rutas, integridad del archivo y las invariantes de los laboratorios ejecutados. El cierre técnico de la evaluación del único TP sigue como próximo paso antes de publicarlo.

## Ética y momento de presentación del TP — 18/10/2026

Clase 1 conserva 87 slides y sustituye las seis páginas antiguas de ética por Navier–Stokes, reward hacking, ciberseguridad, pluralidad en RLHF, actividad A/B y cierre sobre controles. Incluye notas y fuentes primarias con límites de evidencia explícitos. La sección dura 28 minutos al final del encuentro 2, más dos de margen. [Guía](clase1/ETICA_RL.md).

Se eliminan el anuncio del TP del cierre y las referencias de sus notas en Clase 1. Juan fijó su presentación recién en el encuentro 4; el plan ya no la ubica en el encuentro 1. Los archivos del TP siguen disponibles para preparación docente. El protocolo de evaluación sigue pendiente. Clase 2 se actualizó el 31/10, según el registro siguiente. No se modifican originales de 2025 ni configuraciones locales del usuario.

### Ajuste de los dos casos — 18/10/2026

Navier–Stokes se presenta sin exigir vocabulario matemático especializado e incorpora el riesgo de orientar el campo según la competencia entre modelos. La pregunta de cierre vincula crédito y prioridades de investigación. Se mantienen los detalles técnicos y los límites de evidencia en las notas.

Ciberseguridad reemplaza el caso Claude Code de 2025 por el incidente OpenAI–Hugging Face de julio de 2026. Incluye fuentes de OpenAI, Hugging Face y METR/Redwood. Las notas distinguen la intrusión durante evaluación de los comportamientos reforzados en entrenamiento previo y evitan atribuir el incidente exclusivamente a RL. La discusión incluye detenerse, informar o pedir autorización como comportamientos valiosos. Se conserva el bloque de 28 minutos y la presentación del TP en el encuentro 4.

## Clase 2 — 31/10/2026

Se aplica el camino hacia GRPO en el deck nativo 2026: 22 slides nuevas, tablas editables, ejemplos numéricos, notas y fuentes. El recorrido de encuentro 3 contiene 32 slides; 123 se omiten en presentación y se reservan a consulta/encuentro 4. Se reemplazan las diez slides antiguas de RLHF/DPO; el origen 2025 permanece archivado. Se corrigen las promesas de return deseado en DT y la descripción de replay de PPO. El TP queda sólo en el bloque final del encuentro 4. El PDF estudiantil incluye únicamente las 32 principales.

La demo sigue siendo la GRU validada en CPU; no se atribuye ajuste de un LLM preentrenado ni rendimiento medido a la Orin. Juan informó que recuperó acceso a la placa; su ensayo es un siguiente paso. La duración de cuatro horas requiere ensayo oral.

## Corrección de continuidad de Clase 2

Juan señaló que el recorte de 32 slides rompía el flujo y relegaba DT y otros contenidos valiosos. Se recupera el orden general de la clase: redes, DQN, políticas, actor-crítico/PPO, LLMs, Offline RL/DT, aplicaciones y conclusiones. GRPO queda integrado al bloque LLM; ocho repasos nuevos pasan a consulta. Quedan 115 slides visibles, 39 de profundización y sólo el lanzamiento del TP reservado al encuentro 4. El PDF completo de 154 páginas excluye únicamente ese lanzamiento. El guion anterior de cuatro horas centrado en GRPO queda sin efecto.

## Entorno y publicación de la rama 2026

Se completa la migración local a uv: un único pyproject.toml, Python 3.12 recomendado (3.11–3.13 admitido), versiones directas fijadas y uv.lock versionado. El Poetry raíz duplicado se retira; el archivo 2025 conserva sus originales. Las actividades vigentes se prueban desde un entorno nuevo. El extra tp-hf añade Transformers/Accelerate; el stack de Jetson se mantiene separado. Se añaden instrucciones y pruebas de Actions en Linux/macOS/Windows. Esto no corrige el protocolo del TP.

La rama 2026 se publicó con el material acumulado y el entorno. [PR #69 en borrador](https://github.com/DiploDatos/AprendizajePorRefuerzos/pull/69), sin integrar en master. Las cinco plataformas de CI pasaron; se corrigió UTF-8 explícito a partir de un fallo real en Windows.
