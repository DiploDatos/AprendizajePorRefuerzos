# Auditoría curricular y técnica

Fecha: 12/10/2026. Checkout revisado: rama `2025`, commit `22824ca`. El remoto local corresponde a [DiploDatos/AprendizajePorRefuerzos](https://github.com/DiploDatos/AprendizajePorRefuerzos). No se verificó si el remoto tiene commits posteriores: el diagnóstico de código corresponde a este checkout.

## Actualización curricular del 17/10/2026

La auditoría describe el material encontrado en el checkout de 2025. Juan indicó después retirar el bloque introductorio de HARKing/crisis de 2016, anticipar RL en LLMs con dos slides y mantener un único TP: Decision Transformer. Q-learning pasa a actividad formativa. Los materiales de origen se trasladaron a `archive/2025/` y el desarrollo sigue en la rama local `2026`. Ver [registro de cambios](CAMBIOS_2026.md). Los nombres TP1/TP2 en los hallazgos siguientes identifican los trabajos históricos.

## Alcance y trazabilidad

Se inventariaron los archivos versionados, se leyeron los notebooks y documentos curriculares, los módulos históricos, la configuración y las estructuras de JSON de test. Se extrajo texto de ambos PDFs locales: 94 y 143 páginas. Coinciden en cantidad de páginas y temas con los decks nativos; no se auditó visualmente cada imagen, fórmula rasterizada ni video enlazado. No se ejecutaron todos los agentes antiguos ni se entrenó el TP2.

En Drive se leyó el contenido completo disponible de las dos presentaciones y de Organización, además del Programa descubierto en la carpeta. El inventario de diapositivas, IDs, revisiones y los 12 archivos de la carpeta está en [inventario_fuentes.json](validacion/inventario_fuentes.json). Los IDs permiten localizar cada recomendación aunque cambie el orden.

| Fuente | Estado y hallazgo | Acción |
|---|---|---|
| [Clase 1](https://docs.google.com/presentation/d/1s6e6-wCMOJ1nit9EHYREiFJe38MGcOVisFDdUvCWHkk/edit) | 94 slides; introducción, MDP, bandidos, TD, modelos, ética | Reutilizar fundamentos y reproducibilidad; renovar actividad 1 |
| [Clase 2](https://docs.google.com/presentation/d/1505Z4vPX_KDFcOiACzhdm8k5bVfGqN_RFfAbKvbrS8s/edit) | 143 slides; deep RL, RLHF, DPO, offline RL, DT y aplicaciones | Reorganizar bloque con RL para generación, REINFORCE/PPO/RLOO/GRPO; reducir aplicaciones |
| [Organización 2025](https://docs.google.com/document/d/1e0Vx6fdyRi0wxkxlAyO3kwwb2pRhZGproqHZbkZGvwI/edit) | Cuatro encuentros de 4 h; Juan y Jorge; los dos decks abarcan más de un encuentro | No confundir deck “Clase 2” con segundo encuentro |
| [Programa](https://docs.google.com/document/d/1NmWlabRgdbZUyWpmEL6iClS22tUarIcF5oWDTXEolYw/edit) | Contiene secciones **2025 y 2026**; modificado 11/06/2026. Ya agrega DPO, RLVR y agentes | Mantener coherencia con esa propuesta; confirmar prioridades con Jorge |
| `lab_0_warmup_rl.ipynb` | Ejercicios manuales de política y valor | Mantener para diagnóstico inicial |
| `lab_1_intro_rl.ipynb` | CliffWalking, SARSA/Q-learning, hiperparámetros, opcionales Softmax/Dyna-Q | Archivado en `archive/2025/`; actividad formativa en `2026/actividades/q_learning` |
| `lab_2_decision_transformer.md` y `tp_decision_transformer/` | Guías extensas, datos Netflix/Goodreads y alternativas PyTorch/HF | Archivado; continúa como único TP 2026, con revisión técnica pendiente |
| `archive/agents/` | APIs antiguas Gym, cuatro retornos de `step`, render antiguo; también notebook SB3 | Referencia histórica; no distribuir como ejecución soportada |
| `README.md`, `pyproject.toml`, locks | README menciona lab 2 SB3; metadatos 2024; Poetry y PEP 621 divergentes | Añadir entrada 2026; luego unificar dependencias con un entorno validado |
| [Q-Lab0](https://docs.google.com/spreadsheets/d/1ujRM9TjZEZVUeq4akfNCjK_fXGbWDaw68HxEy1WPS88/edit) | Tabla de Q y transiciones manuales; α=0.1, γ=0.9 | Mantener como puente entre Bellman y código |
| `images/` | Seis imágenes para ejercicios y redes | Reutilizables; no renovadas |
| `reinforcement_learning/`, `.venv/` | Entornos locales, no código docente | No incluir en paquete ni en escaneo de contenido curricular |

## Hallazgos prioritarios

1. **Q-learning:** celda Markdown 6 del notebook actual escribe `gamma * argmax Q`. La actualización debe usar `gamma * max Q`: `argmax` devuelve la acción, no su valor. En la versión nueva se explicita esta diferencia.
2. **Terminación:** el loop viejo no entrega `terminated` a la función de aprendizaje, por lo que no puede expresar directamente la máscara terminal. En un problema de duración indefinida cortado por TimeLimit se conserva bootstrap al truncar y se elimina al terminar. Si el horizonte finito forma parte del MDP, hay que representar el tiempo restante. [Documentación Gymnasium](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/).
3. **Métricas:** el loop guarda `max(episode_reward, -100)`. Esto aplana episodios peores y dificulta diagnosticar aprendizaje. Guardar retorno original, pasos, éxito y truncación por separado.
4. **Reproducibilidad:** la semilla de la selección de acciones no alcanza para fijar las transiciones del entorno; faltan semillas de reset y un protocolo separado de evaluación.
5. **Demo MountainCar:** celda 15 consulta `done` sin actualizarlo a partir de sus propias banderas. Renderizados antiguos no configuran `render_mode`.
6. **Versiones:** el tutorial HF conserva `Taxi-v3`; la documentación actual ya describe `Taxi-v4`. La nueva actividad fija Gymnasium **1.2.1** y `Taxi-v3`, probados localmente, para evitar mezclar benchmarks. Actualizar ambos juntos si se migra a v4. [Historial oficial de Taxi](https://gymnasium.farama.org/environments/toy_text/taxi/).
7. **TP2 / objetivo:** en `03_REFERENCIA_COMPLETA.md`, evaluación recorre `items=[0,1,...]` del test como si fuese historia y toma el siguiente ID como target. El JSON contiene ratings exhaustivos, no una secuencia de consumo observada. Medir siguiente ID no mide satisfacción. Propuesta: ranking de relevancia (`rating >= 4`) o simulación de sesión declarada, con ratings ocultos a la política hasta recomendar.
8. **TP2 / arquitectura:** la referencia PyTorch calcula `action_emb` pero no lo usa; suma estado, RTG, tiempo y grupo. Es una variante condicionada, no la tokenización intercalada del DT original. Revisar además alineación de targets y máscara de padding. [Paper DT](https://arxiv.org/abs/2106.01345).
9. **TP2 / cold-start:** el texto menciona “sin historial” pero evalúa tras 20 ítems. Distinguir cold-start estricto (sin historial), warm-up de sesión y usuario no visto en entrenamiento.
10. **Jetson:** Juan confirmó **Orin Nano de 8 GB**. `home_management` registra JetPack 6 / L4T R36.5.2 / CUDA 12.6 y el alias SSH `jetson`; esas versiones todavía no se verificaron en vivo porque las conexiones agotaron el tiempo de espera. Juan confirmó posteriormente que el equipo está desconectado y aplazó la prueba. Las wheels de PyTorch deben corresponder a JetPack, Python y aarch64; no instalar sin revisión las dependencias generales del repo. [NVIDIA](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html).

Los cambios locales previos se preservaron: `lab_1_intro_rl.ipynb` se trasladó íntegro a `archive/2025/`; `pyproject.toml` y el `uv.lock` no versionado siguen en la raíz. Estos problemas se documentan sin reescribir ese trabajo.

## Resto de la carpeta 2025

Se inventariaron planillas de correcciones y grupos, Q-Lab0, una planilla histórica de entregas/correcciones, un notebook de entrega estudiantil, el notebook Colab de lab 1 y dos shortcuts. Q-Lab0 se leyó y permite conservar el ejercicio manual. La planilla histórica, aunque su título dice “Lab 2023”, contiene entregas y feedback, no un enunciado nuevo: aparecen oportunidades recurrentes de interpretar exploración, muestrear softmax y justificar hiperparámetros. La bitácora y explicación del único TP atienden esos puntos; las actividades de clase dan feedback sin agregar entregas. No se copiaron nombres, notas ni entregas estudiantiles al repositorio; la planilla interna de correcciones 2025 y la de grupos solo se inventariaron. El fetch del Colab no entregó texto legible; la revisión detallada del notebook se apoya en el archivo local. Los shortcuts no devolvieron `targetId` en la metadata disponible: sus destinos siguen sin verificar. Esto limita el escaneo de anexos, no la revisión de los materiales principales.

## Lectura pedagógica

El material histórico dedica varias slides a la crisis de reproducibilidad y HARKing. La revisión 2026 reemplaza esa introducción por la evolución de RL en LLMs. El pensamiento experimental se conserva mediante predicciones, controles y análisis de recompensa en los laboratorios y en el único TP. La habilidad evaluada pasa de completar celdas a **formular una predicción, diseñar una comparación válida, identificar errores y defender evidencia**. Conviene mantener una resolución manual pequeña para verificar comprensión individual sin penalizar a quienes no usen IA.

## Orientación posterior confirmada por Juan

Clase 2 debe seguir el enfoque de [RL para LLMs de Wolfe](https://cameronrwolfe.substack.com/p/llm-rl). Se prepararon guion, storyboard de 22 slides y un notebook ejecutado que compara REINFORCE/RLOO/GRPO, ilustra el objetivo clipped de PPO y separa reward aprendido de verificador. La demo original de preferencias queda como alternativa; Drive permanece sin editar.
