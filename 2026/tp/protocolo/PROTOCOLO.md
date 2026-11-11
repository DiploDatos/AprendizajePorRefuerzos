# Protocolo común de evaluación del TP

Todos los grupos evalúan con este protocolo. Esto permite comparar modelos y líneas base bajo las mismas reglas. El código está en este directorio. Se ejecuta desde `2026/tp`:

```bash
python -m protocolo.evaluar --dataset netflix --output outputs/baselines
python -m pytest protocolo/tests
```

## Qué contienen los datos

- **Train** (`train/*_train.df`): 16.000 usuarios, 2.000 por grupo. Cada uno tiene entre 25 y 200 ítems valorados, sin repeticiones, en el orden registrado.
- **Test** (`test_users/*_test.json`): 1.600 usuarios nuevos, 200 por grupo. Para cada uno se conoce la valoración de **todos** los ítems del catálogo. El orden de `items` (0, 1, 2, …) es el del catálogo, no un orden de consumo.
- **Grupos** (`groups/*.csv`): media, desvío y conteo por grupo e ítem, de 8 × n_items. Son parámetros del generador sintético. Sirven para analizar, no para construir políticas.

Dos propiedades de train condicionan el diseño:

1. **La política de registro es aleatoria.** El rating medio no varía con la posición en la trayectoria, y la frecuencia de cada ítem no se correlaciona con su conteo por grupo. Los datos sirven para aprender qué ítems valora bien cada tipo de usuario, no para imitar un recomendador previo.
2. **Hay escalas distintas en Goodreads.** En train los ratings son enteros de 1 a 5. En test son continuos, con dos decimales. El umbral de relevancia (rating ≥ 4) se aplica sobre el valor continuo. Reportar esta diferencia si se compara entre datasets.

## Tarea: sesión de recomendación

Una sesión dura `H = 20` pasos. En cada paso la política recomienda un ítem que el usuario no vio y recibe su valoración como recompensa. La valoración permanece oculta hasta que el ítem se muestra. No hay repeticiones.

La política implementa dos métodos (ver `simulador.Policy`):

```python
reset(group: int | None, horizon: int) -> None
scores(items_seen, ratings_seen) -> np.ndarray  # un puntaje por ítem del catálogo
```

El protocolo enmascara lo ya mostrado y elige el ítem disponible de mayor puntaje, con desempate al azar y semilla fija. La política nunca accede a las valoraciones ocultas. `items_seen` y `ratings_seen` son copias de lo observado en la sesión.

### Condiciones

| Condición | Qué cambia | Qué mide |
|---|---|---|
| `warmup = 0`, grupo visible | Historial vacío; se conoce el segmento | Personalización por segmento |
| `warmup = 0`, grupo oculto | Historial vacío y sin segmento: **cold-start estricto** | Adaptación sólo con feedback |
| `warmup = 5` | Los 5 primeros ítems los elige la política de registro, al azar, y sus ratings se revelan | Usuario no visto con historial inicial (no es cold-start) |

En todos los casos el usuario de test no aparece en train. Las métricas cuentan sólo los pasos que decide la política: 20 − warmup.

### Particiones

- `split_by_user`: 90 % de los usuarios de train para ajustar y 10 % para validación, estratificado por grupo y con semilla 0.
- La selección de hiperparámetros, incluido el retorno pedido en DT, se hace en **validación**. Allí la sesión se simula sobre el catálogo restringido a los ítems que cada usuario valoró (`--split val`).
- Test se usa una sola vez, para la tabla final.

## Métricas

Por sesión, sobre los pasos controlados:

- `return`: suma de valoraciones obtenidas.
- `normalized`: (return − aleatorio) / (oráculo − aleatorio). El aleatorio es el retorno esperado de la política de registro. El oráculo elige los mejores ítems del usuario. 0 equivale al azar y 1 al óptimo. Es la métrica principal, porque es comparable entre usuarios y condiciones.
- `hit_rate`: fracción de recomendaciones con rating ≥ 4.
- `ndcg@10`, `precision@10`: ranking de los puntajes del **primer** paso controlado, con relevancia binaria rating ≥ 4. Es una evaluación de ranking estático, no de sesión.

`metricas.summarize` promedia primero las semillas de cada usuario. Después informa la media y el IC 95 % con n = usuarios. Reportar el agregado **y** cada grupo (`resumen_por_grupo.csv`).

## Líneas base

| Nombre | Descripción |
|---|---|
| `random` | Política de registro. Referencia del 0 normalizado. |
| `popularity` | Ítems ordenados por cantidad de apariciones en los logs de train. Con logs uniformes rinde como `random`. |
| `item_mean` | Rating medio de cada ítem en train, contraído hacia la media global. No personaliza. |
| `group_mean` | Rating medio por grupo cuando el grupo es visible; si no, igual a `item_mean`. |
| `group_posterior` | Infiere el grupo con las valoraciones observadas (modelo gaussiano por grupo e ítem) y recomienda el mayor rating esperado. No explora. |

Resultados en test, 3 semillas, horizonte 20, media de `normalized`:

| | Netflix, grupo visible | Netflix, grupo oculto | Goodreads, grupo visible | Goodreads, grupo oculto |
|---|---:|---:|---:|---:|
| `random` | 0,00 | 0,00 | 0,00 | 0,00 |
| `popularity` | −0,06 | −0,06 | 0,04 | 0,04 |
| `item_mean` | 0,69 | 0,69 | 0,42 | 0,42 |
| `group_mean` | 0,74 | 0,69 | 0,49 | 0,42 |
| `group_posterior` | 0,74 | 0,71 | 0,49 | 0,46 |

`item_mean` ya es una línea base fuerte. Un modelo que no la supera no demuestra personalización, aunque su loss baje.

## Logs de un sistema anterior

`logs.py` genera un dataset de entrenamiento derivado. No cambia la evaluación, que sigue siendo en test con este protocolo. En vez de la política uniforme original, los ítems registrados los elige un "sistema anterior": conoce el grupo, muestrea con softmax(media del ítem en el grupo / τ) y explora con probabilidad ε. Corre en el simulador sobre cada usuario de train, con el catálogo restringido a los ítems que valoró. No se inventa ninguna valoración. Las medias que usa con cada usuario excluyen sus propias valoraciones.

| Preset | ε por sesión | τ | Retorno medio de un episodio de train |
|---|---|---:|---:|
| `random` (original) | 1 | — | 64,3 |
| `mixto` | 0; 0,25; 0,5; 1 | 0,25 | 74,6 |
| `experto` | 0,1 | 0,1 | 82,5 |

Uso: `logs.logged_episodes("mixto", ...)` devuelve los episodios (y los guarda en `outputs/logs/`). `python -m protocolo.evaluar --logs mixto` evalúa las líneas base ajustadas con esos logs, junto con el sistema anterior con ε = 0.

Las baselines se ajustan con los mismos logs que el modelo. Con logs sesgados, `popularity` hereda lo que sabía el sistema anterior. `item_mean` empeora, porque cada ítem se mostró sobre todo a los grupos que lo valoran bien (sesgo de selección). `group_mean` lo evita al condicionar por grupo.

Netflix, test, warm-up 0, `normalized` (visible / oculto):

| | `random` | `mixto` | `experto` |
|---|---:|---:|---:|
| `popularity` | −0,06 | 0,69 | 0,69 |
| `item_mean` | 0,69 | 0,67 | 0,44 |
| `group_mean` | 0,74 / 0,69 | 0,74 / 0,67 | 0,74 / 0,44 |
| `group_posterior` | 0,74 / 0,71 | 0,74 / 0,71 | 0,74 / 0,71 |
| Sistema anterior con ε = 0 | — | 0,59 / 0,54 | 0,69 / 0,66 |

## Reglas para Decision Transformer

- Un episodio de train es un tramo de 20 pasos consecutivos de un usuario. Usar `datos.make_episodes`. R̂_t es la suma de ratings desde t hasta el final del **episodio**, no de toda la trayectoria. Así la escala coincide con la de la sesión evaluada.
- El retorno pedido es una condición de entrada, no una garantía. Elegirlo en validación. Reportar el barrido de retornos pedidos frente a obtenidos, con el soporte de train marcado.
- Con warm-up, R̂ se actualiza restando también las valoraciones del warm-up.
- Una variante que no intercala tokens (R̂, s, a), por ejemplo sumando estado, R̂ y grupo en un solo token, es válida si se identifica como tal.

## Reglas para IQL

- Usar los mismos episodios de 20 pasos que el DT. La recompensa del paso t es la valoración revelada en t; puede centrarse o escalarse si se declara.
- El último paso del episodio es terminal: su objetivo TD es sólo r, sin V(s').
- El estado sólo puede incluir lo observable en la sesión: grupo si es visible, ítems mostrados y valoraciones reveladas, y el número de paso.
- La política descarta los ítems ya mostrados, igual que el protocolo. Al extraer π o elegir el máximo de Q, sólo se comparan ítems disponibles.
- τ, β y γ se eligen con el protocolo en validación. Las pérdidas de Q y π no miden la calidad de la política.
