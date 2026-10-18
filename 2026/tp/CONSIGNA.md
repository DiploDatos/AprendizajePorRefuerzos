# Trabajo Práctico — Decision Transformer para recomendaciones

DiploDatos 2026. **Único TP obligatorio del curso**, continuación del anterior TP2. Borrador para revisión docente; fecha, grupos y condiciones finales de aprobación se publicarán en Campus.

## Objetivo y alcance

Implementar y analizar un recomendador condicionado por retorno usando Netflix o Goodreads. Comparar su comportamiento con una línea base bajo el mismo protocolo y estudiar qué cambia al pedir distintos retornos. Un retorno objetivo es una condición de entrada: no garantiza obtener ese resultado.

Se mantiene la estructura del trabajo anterior: exploración/preparación, implementación/entrenamiento, baselines/evaluación, return conditioning y reporte. La actividad Q-learning y el notebook GRPO no forman parte de las entregas de este TP.

## 1. Datos y construcción del problema

Elegir un dataset e inspeccionar ratings, identificadores, grupos y particiones. Documentar qué representa cada campo, cómo se construye una trayectoria y cómo se calcula el retorno acumulado futuro. Separar los usuarios de entrenamiento, validación y test sin filtrar ratings reservados hacia la política.

Los JSON de test contienen valoraciones por ítem; el orden de sus IDs no demuestra orden temporal de consumo. Si se usa un simulador de sesiones, declarar sus reglas, qué observa el agente, cuándo recibe cada rating y cómo se eligen las trayectorias. Si se adopta ranking estático, presentarlo como evaluación de ranking, sin atribuirle un retorno de interacción medido en usuarios reales.

## 2. Modelo y entrenamiento

Implementar Decision Transformer o una variante condicionada por retorno claramente identificada. Documentar tokens/representaciones de estado, acción y retorno; máscara causal, padding y alineación de targets. El target no puede aparecer en el contexto que lo predice.

Guardar configuración, semillas, curvas sin recortes, checkpoint e instrucciones de reproducción. Si se usa una biblioteca, explicar qué partes resuelve y verificar el contrato de entradas/salidas.

## 3. Baseline y evaluación

Comparar con al menos una línea base sencilla y justificarla. Usar los mismos usuarios, candidatos, historial disponible y presupuesto. Separar selección de hiperparámetros de evaluación final.

El protocolo común de evaluación debe cerrarse y probarse por el equipo docente antes de publicar esta consigna: ranking de relevancia o simulación de sesión declarada. Las métricas deben corresponder a la tarea elegida. No evaluar «acertar el siguiente ID». No comparar métricas de protocolos diferentes como si fueran equivalentes.

Distinguir usuario no visto en entrenamiento, historial inicial vacío y sesión con warm-up. No describir una evaluación con 20 interacciones iniciales como cold-start estricto.

## 4. Return conditioning

Comparar varios retornos solicitados dentro del soporte de entrenamiento. Mantener las condiciones de evaluación y mostrar resultados por grupo además del agregado. Explicar diferencias entre retorno solicitado y obtenido, incertidumbre y límites de extrapolación. Un resultado sin mejora es válido si el experimento permite interpretarlo.

## 5. Entrega y explicación

- Notebooks ejecutados o scripts equivalentes para exploración, entrenamiento, evaluación y return conditioning.
- Código, versiones, semillas, configuración y checkpoint; no duplicar los datasets proporcionados.
- Tabla de resultados frente al baseline y curvas relevantes.
- Reporte de 3–5 páginas o Markdown equivalente: problema, protocolo, decisiones, evidencia y límites.
- Bitácora breve de contribuciones y decisiones verificadas. IA opcional: indicar si se utilizó y qué se comprobó; no entregar conversaciones completas.

La explicación individual se centra en una decisión de modelado y un resultado de evaluación; su modalidad se comunicará antes de la entrega. Tener acceso a modelos pagos no otorga puntos. Consultar la [rúbrica propuesta](RUBRICA.md).
