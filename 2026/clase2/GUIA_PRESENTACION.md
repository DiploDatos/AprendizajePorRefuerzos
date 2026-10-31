# Clase 2 — guía para presentar el encuentro 3

Revisión 31/10/2026. [Slides editables](https://docs.google.com/presentation/d/19hKgrRzdaDW6sAlVUABjZJeOeElbK6Z1q1QkRfttv4w/edit) y [PDF del recorrido principal](Clase_2_2026.pdf). La copia tiene 155 slides: **32 principales**, 112 de consulta y 11 reservadas al encuentro 4. Las 123 últimas están omitidas en la presentación. Omitir no restringe acceso: antes del encuentro 4 compartir el PDF de 32 páginas y los notebooks seleccionados, no el deck docente completo.

## Recorrido de 240 minutos

| Minutos | Slides | Trabajo |
|---|---|---|
| 0–40 | 1–14 | Apertura; puente de Q-learning a redes y DQN; Atari/Spotify como ejemplos; generación como MDP/bandido |
| 40–60 | 15–18 | Señal final/proceso, RM/critic/ventaja, objetivo con KL y REINFORCE |
| 60–75 | 19 | Baseline y cuenta manual de leave-one-out |
| 75–95 | 20–22 | PPO, cuatro casos de clipping, actual/vieja/referencia |
| 95–110 | — | Recreo |
| 110–130 | 23–24 | Fuente de recompensa y preferencias con empate/desacuerdo |
| 130–140 | 25 | RLOO: calcular ventajas |
| 140–160 | 26–27 | GRPO: normalización de grupo, surrogate y KL |
| 160–205 | 28–29 | Notebook: máscaras, logprobs, old/ref, actualización y comparación |
| 205–220 | 30 | Evaluación a ciegas y contraejemplos al proxy |
| 220–230 | 31 | DPO y una diferencia de variantes; lecturas optativas |
| 230–240 | 32 | Reconstrucción individual y puente a agentes |

Es una distribución propuesta, no una duración ensayada. El repaso de redes se reduce a lo necesario para gradientes; el detalle anterior queda en consulta. Se conservan un ejemplo de juegos y uno de recomendación. Ajustar el reparto con Jorge antes de fijar el cronograma.

## Preparación del laboratorio

Abrir [llm_policy_gradients.ipynb](llm_policy_gradients.ipynb) y preparar SFT/RM para dedicar los 45 minutos a los updates. El enlace de slide 29 apunta a la copia del notebook en Drive. Conservar resultados CPU preparados. No agregar la práctica completa de mini RLHF de 65 minutos a este mismo encuentro.

La demo es una GRU de 13.923 parámetros, no un transformer preentrenado. Preferencias simuladas por defecto; si se etiqueta un archivo humano, identificarlo y separar el juicio ciego final. La recuperación de acceso a Orin no equivale a haber probado la demo allí: falta el ensayo y la extensión con un transformer pequeño.

## Respuestas docentes que conviene tener a mano

- Slide 19: baseline leave-one-out para la tercera muestra de [0,0,1,1] = 1/3; ventaja = 2/3. `detach` no garantiza independencia estadística.
- Slide 21: objetivo de PPO para las cuatro filas = [1,2; 0,5; −1,5; −0,8]. El tramo plano está activo en primera y cuarta. El clipping no impone una cota estricta de KL. El notebook ilustra la loss; no entrena PPO.
- Slides 22/27: ratio con política vieja; regularización con referencia. No intercambiarlas.
- Slides 25/26: RLOO da [−2/3,−2/3,+2/3,+2/3]; GRPO con std poblacional da aproximadamente [−1,−1,+1,+1]. El epsilon introduce una diferencia numérica mínima.
- Slide 32: rewards iguales anulan la ventaja de tarea. La KL puede seguir actualizando. El texto de una herramienta es observación; no se trata como acción generada.

## Evaluación y límites

Medir recompensa proxy, diversidad, longitud, EOS, KL y juicio ciego sobre salidas nuevas. «claro vete» puede satisfacer el contrato de formato sin ser útil. La demo no compara algoritmos con el mismo número de updates: no presentar un ranking de superioridad. Pedir predicción, evidencia y explicación individual; usar IA es opcional.

Los apéndices 33–144 preservan material histórico y no se revalidaron por completo. El bloque 145–155 corresponde al encuentro 4; sólo se corrigieron promesas sobre returns deseados y se renombró el TP único. Su revisión integral y el protocolo de evaluación del TP siguen pendientes. No anunciar ese TP en el encuentro 3.

[Contenido y notas de las 22 nuevas slides](SLIDES_LLM_RL.md). [Registro de validación](../validacion/clase2_revision.json).
