# Trabajo Práctico — Decision Transformer e IQL para recomendaciones

DiploDatos 2026. **Trabajo práctico obligatorio del curso.** Las fechas, la organización de los grupos y las condiciones de aprobación se publican en Campus.

## Objetivo y alcance

Implementar y comparar dos enfoques de aprendizaje por refuerzo offline para recomendar, usando Netflix o Goodreads:

- **Decision Transformer (DT):** un recomendador condicionado por retorno. Un retorno objetivo es una condición de entrada: no garantiza obtener ese resultado.
- **Implicit Q-Learning (IQL):** un método basado en funciones de valor, que extrae una política sin evaluar acciones fuera de los datos.

Ambos se comparan con líneas base bajo el mismo protocolo. Además se estudia qué cambia al pedirle al DT distintos retornos y qué cambia en los dos modelos cuando los logs vienen de una política mejor que el azar.

La [guía](GUIA.md) explica cada parte y da esqueletos de código. El trabajo comprende exploración y preparación de datos, implementación y entrenamiento, comparación con líneas base, evaluación, experimentos de condicionamiento por retorno y reporte. La actividad Q-learning y el notebook GRPO no forman parte de las entregas de este TP.

## 1. Datos y construcción del problema

Elegir un dataset e inspeccionar ratings, identificadores, grupos y particiones. Documentar qué representa cada campo, cómo se construye una trayectoria y cómo se calcula el retorno acumulado futuro. Separar los usuarios de entrenamiento, validación y test sin filtrar ratings reservados hacia la política.

Los JSON de test contienen valoraciones por ítem; el orden de sus IDs no demuestra orden temporal de consumo. Si se usa un simulador de sesiones, declarar sus reglas, qué observa el agente, cuándo recibe cada rating y cómo se eligen las trayectorias. Si se adopta ranking estático, presentarlo como evaluación de ranking, sin atribuirle un retorno de interacción medido en usuarios reales.

Formular el problema como MDP: estado, acción, recompensa, horizonte y factor de descuento. Explicar qué información del estado necesita cada modelo.

## 2. Modelos y entrenamiento

Los dos modelos usan los mismos episodios de entrenamiento y la misma partición de validación.

### 2.1 Decision Transformer

Implementar Decision Transformer o una variante condicionada por retorno claramente identificada. Documentar tokens/representaciones de estado, acción y retorno; máscara causal, padding y alineación de targets. El target no puede aparecer en el contexto que lo predice.

### 2.2 IQL

Implementar IQL con tres componentes:

- V(s), ajustada por regresión de expectil sobre Q(s, a) de las acciones registradas;
- Q(s, a), ajustada por diferencias temporales con r + γ V(s');
- una política π(a | s), extraída por regresión ponderada por ventaja (AWR).

Documentar la representación del estado, la recompensa, el tratamiento del último paso del episodio y el uso de una red objetivo. Elegir el expectil τ, la temperatura β y γ en validación, con el protocolo y no con las pérdidas.

Comparar al menos γ = 0 con un γ > 0 e interpretar la diferencia: ¿qué parte del valor viene de la secuencia? Evaluar también, como comparación, la política que elige el máximo de Q entre los ítems disponibles, y explicar en qué se diferencia de π.

### 2.3 Para ambos modelos

Guardar configuración, semillas, curvas sin recortes, checkpoint e instrucciones de reproducción. Si se usa una biblioteca, explicar qué partes resuelve y verificar el contrato de entradas/salidas.

## 3. Baselines y evaluación

Comparar DT e IQL con al menos dos líneas base y justificarlas. Usar los mismos usuarios, candidatos, historial disponible y presupuesto. Separar selección de hiperparámetros de evaluación final.

Evaluar con el [protocolo común](protocolo/PROTOCOLO.md): sesiones de 20 recomendaciones sin repetición, donde cada valoración se revela recién al mostrar el ítem. Métrica principal: retorno normalizado entre política aleatoria y oráculo. Como secundarias, hit rate y NDCG@10 del primer paso. Incluir como mínimo `item_mean` y una línea base personalizada. No evaluar «acertar el siguiente ID». No comparar métricas de protocolos diferentes como si fueran equivalentes.

Reportar las tres condiciones del protocolo: grupo visible, grupo oculto (cold-start estricto) y warm-up de 5 ítems. Esta última no es cold-start.

Entrenar cada modelo final con al menos 3 semillas. Reportar la media y la dispersión entre semillas, separada del intervalo de confianza por usuario que da el protocolo: la primera mide la variabilidad del entrenamiento y la segunda la de la evaluación. Una diferencia con la línea base menor que la dispersión entre semillas no es una mejora.

## 4. Return conditioning

Con el DT, comparar varios retornos solicitados dentro del soporte de entrenamiento. Mantener las condiciones de evaluación y mostrar resultados por grupo además del agregado. Explicar diferencias entre retorno solicitado y obtenido, incertidumbre y límites de extrapolación. Un resultado sin mejora es válido si el experimento permite interpretarlo.

## 5. Calidad de los logs

Los logs originales vienen de una política uniforme al azar. `protocolo/logs.py` genera, con los mismos ratings, los logs de un sistema anterior que elegía qué mostrar (presets `mixto` y `experto`; ver el protocolo). Entrenar DT e IQL con los logs originales y con los `mixto`, ajustando las líneas base con los mismos logs que los modelos. Incluir `popularity` y el sistema anterior como referencias.

Explicar qué aprende cada modelo en cada caso, si supera a la política que generó sus datos y por qué cambia el rendimiento de `popularity` e `item_mean` entre un dataset y otro.

## 6. Entrega y explicación

- Notebooks ejecutados o scripts equivalentes para exploración, entrenamiento de ambos modelos, evaluación, return conditioning y calidad de los logs.
- Código, versiones, semillas, configuración y checkpoints; no duplicar los datasets proporcionados.
- Tabla de resultados de DT, IQL y líneas base, y curvas relevantes.
- Reporte de 4–6 páginas o Markdown equivalente: problema, protocolo, decisiones, evidencia y límites. Incluir una comparación razonada entre DT e IQL.
- Bitácora breve de contribuciones y decisiones verificadas. IA opcional: indicar si se utilizó y qué se comprobó; no entregar conversaciones completas.

La explicación individual se centra en una decisión de modelado y un resultado de evaluación; su modalidad se comunicará antes de la entrega. Tener acceso a modelos pagos no otorga puntos. Consultar la [rúbrica de evaluación](RUBRICA.md).
