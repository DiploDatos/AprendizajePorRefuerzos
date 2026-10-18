# Próximos pasos

Revisión 17/10/2026: rama local `2026`, material 2025 archivado y un único TP. Las actividades de Q-learning y RL para generación son formativas.

## Antes de publicar

| Prioridad | Acción | Responsable propuesto | Criterio de cierre |
|---|---|---|---|
| P0 | Cerrar y probar protocolo de evaluación y arquitectura del único TP de Decision Transformer | Juan y Jorge | Datos y secuencias distinguidos; baseline común, máscaras y test válidos |
| P0 | Revisar consigna/rúbrica del TP y fijar fecha, grupos y explicación individual | Juan y Jorge | Un solo enunciado y condiciones comunicadas |
| P0 | Ensayar encuentro 3: camino hacia GRPO y notebook de 45 min | Docentes | Progresión y duración compatibles con los fundamentos |
| P1 | Aplicar el guion de Clase 2 en su copia editable 2026; Clase 1 ya está revisada | Juan | Dos slides introductorias nuevas; bloque de deep RL revisado; URLs comprobadas |
| P1 | Ensayar actividad Q-learning de 45 min en otra máquina | Docentes | Modo corto funcional, sin pedir informe ni corrida completa |
| P1 | Etiquetar preferencias humanas y evaluar salidas nuevas | Docentes | Feedback humano separado del modo simulado |
| P1 | Publicar materiales seleccionados en Campus | Docentes | Índice, un único TP y actividades diferenciadas; sin soluciones docentes |
| P2 | Unificar entorno y metadatos de dependencias | Mantenedor | Un camino de instalación validado |
| Aplazado | Diagnóstico y ensayo con LLM preentrenado en Orin Nano 8 GB | Juan + asistencia | Retomar cuando Juan conecte el equipo; memoria y entrenamiento medidos |

## Orden de trabajo sugerido

1. Resolver primero el protocolo del único TP; no trasladar como referencia válida el evaluador histórico del siguiente ID.
2. Revisar las dos slides de introducción y ensayar el puente hacia el encuentro 3.
3. Trasladar los guiones a las copias 2026 de Slides ya creadas y enlazar los notebooks.
4. Probar el paquete estudiantil, fijar calendario y publicar.

## Estado operativo

La rama `2026` está creada localmente y la organización está aplicada en los archivos de trabajo, con la revisión de Clase 1 aplicada. Consultar el historial de Git para el registro local; no se publicó la rama remota. La rama `2025` mantiene su referencia anterior. El archivo registra hashes y preserva el notebook anterior con los cambios locales de Juan. Los entornos, `pyproject.toml`, `uv.lock` y archivos del editor se conservaron.

Se creó la carpeta 2026 en Drive con copias del material, incluidas las presentaciones editables. Clase 1 ya está actualizada y revisada visualmente; Clase 2 conserva el contenido anterior. Los originales no se modificaron. No se publicó en Campus, no se creó un PR y no se enviaron mensajes a colegas. Los notebooks de RL para generación ya se ejecutaron completos en CPU; ver [validación](validacion/RESULTADOS.md). Orin sigue aplazada a pedido de Juan.
