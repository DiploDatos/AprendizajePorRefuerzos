# 📚 TRABAJO PRÁCTICO: DECISION TRANSFORMER PARA RECOMENDACIONES

---

## 🎬📚 ELECCIÓN DE DATASET

Este trabajo práctico permite a cada grupo elegir entre **dos dominios diferentes**:

| Dataset | Items | Dominio | Características |
|---------|-------|---------|-----------------|
| **🎬 Netflix** | 752 películas | Películas/series | Recomendado, dominio familiar |
| **📚 Goodreads** | 472 libros | Libros | Alternativa, menos items (training más rápido) |

✅ **Ambos tienen la misma estructura** - El código es idéntico  
✅ **Solo cambian:** Paths de archivos y número de items  
✅ **Cada grupo elige según su interés** - No hay diferencia en dificultad

---

## 🗂️ Estructura de Documentos

Este trabajo práctico cuenta con **2 documentos principales**:

---

### 1️⃣ **03_REFERENCIA_COMPLETA.md** (Documento Principal)
📄 **[Leer aquí](./03_REFERENCIA_COMPLETA.md)**

**Contenido:**
- ✅ Descripción completa del problema
- ✅ Explicación detallada de los datasets disponibles (Netflix o Goodreads)
- ✅ Código de REFERENCIA completo (Decision Transformer, training, evaluation)
- ✅ Especificaciones técnicas de cada parte
- ✅ Criterios de evaluación

**Cuándo usarlo:**
- Para entender el problema a fondo
- Para ver ejemplos de código funcionando
- Para consultar detalles técnicos

---

### 2️⃣ **02_GUIA_IMPLEMENTACION_GRUPOS.md** (Guía Práctica) ⭐
📄 **[Leer aquí](./02_GUIA_IMPLEMENTACION_GRUPOS.md)**

**Contenido:**
- ✅ QUÉ implementar vs QUÉ es código de referencia
- ✅ Checklist paso a paso de tareas
- ✅ Esqueletos de código con TODOs
- ✅ Criterios de éxito para cada parte
- ✅ Consejos y troubleshooting

**Cuándo usarlo:**
- **¡EMPEZAR POR AQUÍ!** 👈
- Cuando el grupo no sepa por dónde empezar
- Para verificar que no falta nada
- Durante la implementación (checklist)

---

## 🔧 PASO 0: CONFIGURACIÓN INICIAL

Antes de empezar, el grupo debe elegir el dataset:

1. **Abre** `config_dataset.py`
2. **Modifica** la línea:
   ```python
   DATASET = 'netflix'    # Cambiar a 'goodreads' si prefieren libros
   ```
3. **Verifica** ejecutando: `python config_dataset.py`

✅ Todo el código del grupo usará este dataset automáticamente

---

## 📦 Estructura de Entrega Esperada

```
apellido_nombre_tp_dt/
│
├── data/                          # (proporcionado - no entregar)
│   ├── train/netflix8_train.df
│   └── test_users/netflix8_test.json
│
├── notebooks/
│   ├── 01_exploracion_dataset.ipynb       ✅ Parte 1
│   ├── 02_training.ipynb                  ✅ Parte 2
│   ├── 03_evaluation.ipynb                ✅ Parte 3
│   └── 04_return_conditioning.ipynb       ✅ Parte 4
│
├── src/
│   ├── data/
│   │   ├── load_data.py                   ✅ Implementar
│   │   ├── preprocessing.py               ✅ Implementar
│   │   └── dataset.py                     ✅ Implementar
│   │
│   ├── models/
│   │   ├── decision_transformer.py        ✅ Copiar/adaptar
│   │   └── baselines.py                   ✅ Implementar
│   │
│   ├── training/
│   │   └── trainer.py                     ✅ Implementar
│   │
│   └── evaluation/
│       ├── metrics.py                     ✅ Implementar
│       └── evaluate.py                    ✅ Implementar
│
├── results/
│   ├── figures/                           ✅ Gráficos generados
│   ├── logs/                              ✅ Logs de training
│   └── checkpoints/                       ✅ Modelo entrenado
│
├── REPORTE.pdf                            ✅ Parte 5 (3-5 páginas)
├── README.md                              ✅ Instrucciones de uso
└── requirements.txt                       ✅ Dependencias
```

---

## 📊 Criterios de Evaluación

La evaluación del trabajo práctico se basa en los siguientes componentes:

### **Parte 1: Exploración y Preparación del Dataset**

**Se evalúa:**
- Análisis exploratorio completo (distribuciones, estadísticas clave)
- Visualizaciones claras e informativas (mínimo 3 gráficos relevantes)
- Correcta implementación del preprocesamiento (returns-to-go, formato DT)
- Validación de que los datos procesados son correctos

**Entregables:**
- Notebook `01_exploracion_dataset.ipynb` ejecutado
- Script `data_preprocessing.py` funcional
- Dataset procesado guardado

---

### **Parte 2: Implementación del Modelo**

**Se evalúa:**
- Arquitectura del Decision Transformer funcional
- Training loop implementado correctamente
- Dataset y DataLoader de PyTorch funcionando
- El loss disminuye durante el entrenamiento
- Modelo entrenado guardado (checkpoint)

**Entregables:**
- `src/models/decision_transformer.py`
- `src/data/dataset.py`
- `src/training/trainer.py`
- Notebook `02_training.ipynb` con logs y gráficos de loss
- Modelo entrenado (`.pth` o `.pt`)

---

### **Parte 3: Baselines y Evaluación**

**Se evalúa:**
- Implementación de al menos 1 baseline (ej: Popularity, Behavior Cloning)
- Métricas correctamente implementadas (Hit Rate@K, NDCG@K, MRR)
- Evaluación completa en test set (cold-start users)
- Tabla comparativa de resultados (DT vs Baselines)

**Entregables:**
- `src/models/baselines.py`
- `src/evaluation/metrics.py`
- `src/evaluation/evaluate.py`
- Notebook `03_evaluation.ipynb` con resultados

---

### **Parte 4: Experimentos con Return Conditioning**

**Se evalúa:**
- Análisis del efecto de diferentes valores de R̂ objetivo
- Gráficos mostrando Return vs Performance
- Análisis de performance por grupo de usuarios
- Interpretación de resultados (¿el conditioning funciona?)

**Entregables:**
- Notebook `04_return_conditioning.ipynb` ejecutado
- Gráficos claros del efecto del conditioning
- Análisis cold-start por grupo

---

### **Parte 5: Reporte Final**

**Se evalúa:**
- Claridad en la presentación del problema y solución
- Correcta explicación de la metodología
- Análisis crítico de resultados
- Conclusiones bien fundamentadas
- Calidad de visualizaciones y presentación

**Entregables:**
- `REPORTE.pdf` (3-5 páginas)
- Incluye: Introducción, Metodología, Resultados, Conclusiones

---

### **Trabajo Adicional (Opcional)**

**Puede incluir:**
- Implementación propia del transformer (en vez de copiar código de referencia)
- Comparación con baselines adicionales (ej: Matrix Factorization, LSTM)
- Análisis de attention weights y embeddings (t-SNE)
- Experimentos creativos (ej: multi-objective conditioning)
- Código particularmente bien documentado y organizado

---

### **Aspectos Generales Evaluados en Todo el TP:**

✅ **Código:** Funcional, legible, bien organizado  
✅ **Reproducibilidad:** Instrucciones claras, semilla fijada, requirements.txt  
✅ **Documentación:** Comentarios útiles, README con instrucciones de uso  
✅ **Presentación:** Notebooks ejecutables, gráficos profesionales

---

## 💡 Recursos Adicionales

### **Papers:**
- Decision Transformer: [https://arxiv.org/abs/2106.01345](https://arxiv.org/abs/2106.01345)
- RLT4Rec: [https://arxiv.org/abs/2412.07403](https://arxiv.org/abs/2412.07403)
- Offline RL Tutorial: [https://arxiv.org/abs/2005.01643](https://arxiv.org/abs/2005.01643)

### **Código de Referencia:**
- Decision Transformer oficial: [https://github.com/kzl/decision-transformer](https://github.com/kzl/decision-transformer)
- RLT4Rec repo: [https://github.com/dilina-r/RLT4Rec](https://github.com/dilina-r/RLT4Rec)
- MinGPT (transformers simples): [https://github.com/karpathy/minGPT](https://github.com/karpathy/minGPT)

### **Tutoriales PyTorch:**
- Transformers: [https://pytorch.org/tutorials/beginner/transformer_tutorial.html](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)
- Custom Datasets: [https://pytorch.org/tutorials/beginner/data_loading_tutorial.html](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)

---

## ❓ FAQ

**Q: ¿Cómo se conforma un grupo?**
A: Máximo 3 personas.

**Q: ¿Dónde conseguimos el dataset?**
A: Está en la carpeta `data/` del repositorio. Pueden elegir entre Netflix (películas) o Goodreads (libros).

**Q: ¿Necesitamos GPU?**
A: Recomendado pero no obligatorio. Google Colab gratuito tiene GPU suficiente.

**Q: ¿Cuánto tarda el training?**
A: En GPU: 1-2 horas. En CPU: 4-8 horas.

**Q: Nuestro modelo no mejora, ¿qué hacemos?**
A: 
1. Verificar preprocesamiento (returns-to-go correctos?)
2. Verificar shapes de tensores
3. Probar learning rate más bajo (1e-5)
4. Verificar causal mask del transformer

**Q: ¿Qué specs de hardware mínimas?**
A: 8GB RAM, 10GB espacio disco. GPU opcional.

