# Próximos pasos

Revisión 31/10/2026: rama local `2026`, material 2025 archivado y un único TP. Las actividades de Q-learning y RL para generación son formativas.

## Antes de publicar

| Prioridad | Acción | Responsable propuesto | Criterio de cierre |
|---|---|---|---|
| P0 | Cerrar y probar protocolo de evaluación y arquitectura del único TP de Decision Transformer | Juan y Jorge | Datos y secuencias distinguidos; baseline común, máscaras y test válidos |
| P0 | Revisar consigna/rúbrica del TP y fijar fecha, grupos y explicación individual | Juan y Jorge | Un solo enunciado, presentación y publicación recién en el encuentro 4 |
| P0 | Ensayar la clase general y el lugar de la demo de LLMs | Docentes | Continuidad entre fundamentos, LLMs, DT y aplicaciones; tiempos ensayados |
| Hecho | Aplicar el guion de Clase 2 en su copia editable 2026 | Asistencia | Flujo general restaurado; DT visible; GRPO integrado como subbloque; PDF sin TP |
| P1 | Ensayar actividad Q-learning de 45 min en otra máquina | Docentes | Modo corto funcional, sin pedir informe ni corrida completa |
| P1 | Etiquetar preferencias humanas y evaluar salidas nuevas | Docentes | Feedback humano separado del modo simulado |
| P1 | Publicar materiales seleccionados en Campus | Docentes | Índice por encuentro, TP recién en el 4 y actividades diferenciadas; sin soluciones docentes |
| Implementado | Entorno uv, lock, instrucciones y CI | Asistencia | Instalación limpia local, notebooks y pruebas de compatibilidad; ver ENTORNO.md y Actions |
| P1 | Diagnóstico y ensayo con LLM preentrenado en Orin Nano 8 GB | Juan + asistencia | Acceso recuperado según Juan; faltan compatibilidad, memoria y entrenamiento medidos |

## Orden de trabajo sugerido

1. Resolver primero el protocolo del único TP; no trasladar como referencia válida el evaluador histórico del siguiente ID.
2. Ensayar las dos slides introductorias (6–8 min) y el nuevo bloque de ética (28 min), con puente hacia el encuentro 3.
3. Ensayar las slides de Clase 2 ya actualizadas y validar la demo en Orin antes de reemplazar resultados CPU.
4. Probar el paquete estudiantil, fijar calendario y publicar.

## Estado operativo

La rama `2026` está creada localmente y la organización está aplicada en los archivos de trabajo, con la revisión de Clase 1 aplicada. Consultar el historial de Git para el registro local; el estado remoto está en [GitHub, rama 2026](https://github.com/DiploDatos/AprendizajePorRefuerzos/tree/2026). La rama `2025` mantiene su referencia anterior. El archivo registra hashes y preserva el notebook anterior con los cambios locales de Juan. Los entornos, `pyproject.toml`, `uv.lock` y archivos del editor se conservaron.

Se creó la carpeta 2026 en Drive con copias del material, incluidas las presentaciones editables. Clase 1 incorpora los cuatro casos de ética y la actividad de preferencias; el TP queda fuera de su agenda y cierre. Está actualizada y revisada visualmente; Clase 2 conserva su recorrido general, incluyendo DT, con GRPO integrado. Sólo el lanzamiento del TP está reservado al encuentro 4. Los originales no se modificaron. No se publicó en Campus, no se creó un PR y no se enviaron mensajes a colegas. Los notebooks de RL para generación ya se ejecutaron completos en CPU; ver [validación](validacion/RESULTADOS.md). Juan recuperó acceso a la Orin; falta la validación del entrenamiento.
