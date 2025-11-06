# 🎯 GUÍA DE IMPLEMENTACIÓN PARA GRUPOS
## Decision Transformer para Recomendaciones

Esta guía clarifica **QUÉ DEBE IMPLEMENTAR CADA GRUPO** vs **QUÉ ES CÓDIGO DE REFERENCIA**

---

## 🎬📚 IMPORTANTE: ELEGIR DATASET

Antes de empezar, elige uno de estos dos datasets:

| Dataset | Items | Dominio | Dificultad |
|---------|-------|---------|------------|
| **Netflix** 🎬 | 752 películas | Películas/series | Media |
| **Goodreads** 📚 | 472 libros | Libros | Ligeramente menor |

**Ambos tienen la misma estructura.** Solo cambian:
- Número de items (`num_items` en el modelo)
- Paths de archivos

**💡 Sugerencia:** Usa Netflix si no tienes preferencia. Usa Goodreads si prefieres dominio de libros o training más rápido (menos items).

### **🔧 Archivo de Configuración Proporcionado**

Para facilitar el cambio entre datasets, se incluye `config_dataset.py`:

```python
# En config_dataset.py, solo modifica esta línea:
DATASET = 'netflix'    # o 'goodreads'

# Luego, en tu código:
from config_dataset import DATASET, NUM_ITEMS, get_paths

paths = get_paths()
df_train = pd.read_pickle(paths['train'])

model = DecisionTransformer(
    num_items=NUM_ITEMS,  # Se ajusta automáticamente
    ...
)
```

**✅ Ventajas:**
- Un solo lugar para cambiar el dataset
- Todos los paths se actualizan automáticamente
- Menos errores de configuración

---

## 📋 RESUMEN EJECUTIVO

| Componente | ¿Qué hacer? | Dificultad | Tiempo est. |
|------------|-------------|------------|-------------|
| **Parte 1: Exploración** | Implementar análisis y preprocesamiento | ⭐⭐ Fácil | 4-6 horas |
| **Parte 2: Modelo** | Copiar/adaptar código de referencia | ⭐⭐⭐ Media | 8-12 horas |
| **Parte 3: Baselines** | Implementar 2-3 métodos simples | ⭐⭐ Fácil-Media | 6-8 horas |
| **Parte 4: Experimentos** | Ejecutar y analizar experimentos | ⭐⭐ Media | 4-6 horas |
| **Parte 5: Reporte** | Escribir documento final | ⭐⭐ Media | 4-6 horas |

**Tiempo total estimado:** 26-38 horas (distribuir en 3-4 semanas)

---

## 🚦 PARTE 1: EXPLORACIÓN Y PREPARACIÓN

### ✅ QUÉ IMPLEMENTAR

#### **1.1. Script de carga de datos:**

**Archivo:** `src/data/load_data.py`

```python
# 🎯 IMPLEMENTAR: Funciones básicas de carga
# Usar el código de ejemplo del TP como guía

import pandas as pd
import json

# ============================================
# CONFIGURACIÓN: Elegir dataset
# ============================================
DATASET = 'netflix'    # O 'goodreads'
NUM_ITEMS = 752 if DATASET == 'netflix' else 472

def load_training_data(dataset='netflix'):
    """
    Carga el dataset de training.
    
    Args:
        dataset: 'netflix' o 'goodreads'
    
    Returns:
        df: pandas DataFrame con columnas [user_id, user_group, items, ratings]
    """
    path = f'data/train/{dataset}8_train.df'
    # TODO: Implementar carga con pandas
    # df = pd.read_pickle(path)
    # return df
    pass

def load_test_data(dataset='netflix'):
    """
    Carga el dataset de test (cold-start users).
    
    Args:
        dataset: 'netflix' o 'goodreads'
    
    Returns:
        test_users: lista de diccionarios con keys [group, items, ratings]
    """
    path = f'data/test_users/{dataset}8_test.json'
    # TODO: Implementar carga con json
    # with open(path, 'r') as f:
    #     return json.load(f)
    pass

def load_group_centroids(dataset='netflix'):
    """
    Carga centroides de grupos (OPCIONAL).
    
    Args:
        dataset: 'netflix' o 'goodreads'
    
    Returns:
        mu: DataFrame de 8xNUM_ITEMS con ratings promedio por grupo
    """
    path = f'data/groups/mu_{dataset}8.csv'
    # TODO (Opcional): Implementar si quieren usar para baselines
    # mu = pd.read_csv(path, header=None)
    # return mu
    pass
```

**✓ Criterio de éxito:** Poder cargar y acceder a los datos sin errores.

---

#### **1.2. Análisis Exploratorio:**

**Archivo:** `notebooks/01_exploracion_dataset.ipynb`

```python
# 🎯 IMPLEMENTAR: Análisis completo del dataset

# === Sección 1: Estadísticas Básicas ===
# TODO:
# - Imprimir número de usuarios, items, interacciones
# - Calcular longitud promedio/min/max de secuencias
# - Calcular distribución de ratings

# === Sección 2: Visualizaciones ===
# TODO: Crear al menos 3 gráficos:

# 1. Histograma de longitud de secuencias
import matplotlib.pyplot as plt
# plt.hist(...) 

# 2. Distribución de ratings (barplot)
# plt.bar(...)

# 3. Top-20 películas más populares
# Contar frecuencia de cada item, ordenar, graficar

# BONUS: Distribución de ratings por grupo
```

**✓ Criterio de éxito:** Notebook ejecutable con análisis y gráficos claros.

---

#### **1.3. Preprocesamiento:**

**Archivo:** `src/data/preprocessing.py`

```python
# 🎯 IMPLEMENTAR: Función de preprocesamiento
# El código de referencia está en el TP - pueden copiarlo y adaptarlo

import numpy as np

def create_dt_dataset(df_train):
    """
    Convierte DataFrame raw a formato Decision Transformer.
    
    REFERENCIA: Ver código completo en TRABAJO_PRACTICO_DECISION_TRANSFORMER.md
    
    Args:
        df_train: DataFrame con [user_id, user_group, items, ratings]
    
    Returns:
        trajectories: List[Dict] con formato específico
    """
    trajectories = []
    
    for idx, row in df_train.iterrows():
        # TODO: Extraer items, ratings, group
        
        # TODO: Calcular returns-to-go (R̂)
        # Hint: Iterar hacia atrás desde el final
        # returns[t] = ratings[t] + returns[t+1]
        
        # TODO: Crear diccionario con formato correcto
        trajectory = {
            'items': ...,
            'ratings': ...,
            'returns_to_go': ...,
            'timesteps': ...,
            'user_group': ...
        }
        
        trajectories.append(trajectory)
    
    return trajectories


def validate_preprocessing(trajectories):
    """
    Valida que el preprocesamiento sea correcto.
    """
    # TODO: Verificar que:
    # - Todas las trayectorias tienen las keys correctas
    # - len(items) == len(ratings) == len(returns_to_go)
    # - returns_to_go[0] == sum(ratings)
    # - returns_to_go[-1] == ratings[-1]
    pass
```

**✓ Criterio de éxito:** Generar 16,000 trayectorias con formato correcto y validaciones pasando.

---

## 🚦 PARTE 2: IMPLEMENTACIÓN DEL MODELO

### ⚠️ OPCIÓN 1: USAR CÓDIGO DE REFERENCIA (Recomendado)

El TP incluye **código completo** del Decision Transformer. Los grupos pueden:

1. **Copiar** el código tal cual del documento `TRABAJO_PRACTICO_DECISION_TRANSFORMER.md`
2. **Pegar** en `src/models/decision_transformer.py`
3. **Leer y entender** cada parte (revisar comentarios)
4. **Ejecutar** para verificar que funciona

**Archivos a crear:**
- `src/models/decision_transformer.py` (copiar código del TP)
- `src/models/__init__.py` (vacío o con imports)

**✓ Criterio de éxito:** El modelo se instancia sin errores:

```python
from src.models.decision_transformer import DecisionTransformer

model = DecisionTransformer(
    num_items=752,
    num_groups=8,
    hidden_dim=128,
    n_layers=3,
    n_heads=4
)

print(f"Parámetros totales: {sum(p.numel() for p in model.parameters())}")
# Debería ser ~10-20M parámetros
```

---

### 🌟 OPCIÓN 2: IMPLEMENTAR DESDE CERO (Opcional - Bonus)

Para grupos que quieren más desafío:

**Tareas:**

1. **Entender la arquitectura** (ver filminas de notas de orador)
2. **Implementar cada componente:**
   - Embeddings (items, rtg, timesteps, groups)
   - Transformer encoder con causal masking
   - Prediction head para items
   - Forward pass completo

**Referencias útiles:**
- Paper original: Decision Transformer (Chen et al., 2021)
- Tutorial PyTorch: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
- Código minGPT: https://github.com/karpathy/minGPT

**✓ Criterio de éxito:** Mismo que Opción 1 + entendimiento profundo.

---

### ✅ QUÉ IMPLEMENTAR SIEMPRE

#### **2.2. Dataset y DataLoader:**

**Archivo:** `src/data/dataset.py`

```python
# 🎯 IMPLEMENTAR: PyTorch Dataset customizado
# Código de referencia en el TP - adaptar

from torch.utils.data import Dataset
import torch
import numpy as np

class RecommendationDataset(Dataset):
    """
    Dataset para entrenar Decision Transformer.
    """
    def __init__(self, trajectories, context_length=20):
        """
        Args:
            trajectories: Lista de dicts con formato de create_dt_dataset()
            context_length: Ventana de contexto (cuántos timesteps usar)
        """
        # TODO: Guardar trajectories y context_length
        pass
    
    def __len__(self):
        # TODO: Retornar número de trayectorias
        pass
    
    def __getitem__(self, idx):
        """
        Retorna un sample para training.
        
        Returns:
            Dict con keys:
                - states: (context_length,) LongTensor de item IDs
                - actions: (context_length,) LongTensor de item IDs  
                - rtg: (context_length, 1) FloatTensor de returns-to-go
                - timesteps: (context_length,) LongTensor de posiciones
                - groups: () LongTensor del grupo del usuario
                - targets: (context_length,) LongTensor - next items a predecir
        """
        # TODO: Ver código de referencia en el TP
        # Hint: Extraer ventana de la trayectoria
        # Hint: Targets son los items shifted (próximo item a predecir)
        pass
```

**✓ Criterio de éxito:** Poder crear DataLoader:

```python
dataset = RecommendationDataset(trajectories, context_length=20)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Verificar un batch
batch = next(iter(loader))
print(f"Keys: {batch.keys()}")
print(f"States shape: {batch['states'].shape}")  # (64, 20)
```

---

#### **2.3. Training Loop:**

**Archivo:** `src/training/trainer.py`

```python
# 🎯 IMPLEMENTAR: Loop de entrenamiento
# Código de referencia en el TP

import torch
import torch.nn.functional as F

def train_decision_transformer(model, train_loader, val_loader, 
                               optimizer, device, num_epochs=50):
    """
    Entrena el Decision Transformer.
    
    Args:
        model: Instancia de DecisionTransformer
        train_loader: DataLoader de training
        val_loader: DataLoader de validación
        optimizer: torch.optim.Optimizer (ej: Adam)
        device: 'cuda' o 'cpu'
        num_epochs: Número de épocas
    
    Returns:
        model: Modelo entrenado
        history: Dict con losses por época
    """
    model.to(device)
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(num_epochs):
        # === TRAINING ===
        model.train()
        total_train_loss = 0
        
        for batch in train_loader:
            # TODO: Mover batch a device
            # states = batch['states'].to(device)
            # actions = ...
            # rtg = ...
            # timesteps = ...
            # groups = ...
            # targets = ...
            
            # TODO: Forward pass
            # logits = model(states, actions, rtg, timesteps, groups)
            
            # TODO: Compute loss (cross-entropy)
            # Hint: Reshape logits y targets para cross_entropy
            # loss = F.cross_entropy(...)
            
            # TODO: Backprop
            # optimizer.zero_grad()
            # loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # === VALIDATION ===
        model.eval()
        with torch.no_grad():
            total_val_loss = 0
            for batch in val_loader:
                # TODO: Similar a training pero sin backprop
                pass
            avg_val_loss = total_val_loss / len(val_loader)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}')
    
    return model, history
```

**✓ Criterio de éxito:** El loss disminuye durante training (no tiene que converger perfectamente).

---

## 🚦 PARTE 3: BASELINES Y EVALUACIÓN

### ✅ QUÉ IMPLEMENTAR

#### **3.1. Baseline: Popularity**

**Archivo:** `src/models/baselines.py`

```python
# 🎯 IMPLEMENTAR: Baselines simples (código casi completo en el TP)

import numpy as np

class PopularityRecommender:
    """
    Recomienda items más populares (no personalizados).
    """
    def __init__(self):
        self.item_counts = None
        self.popular_items = None
    
    def fit(self, trajectories):
        """
        Args:
            trajectories: Lista de trayectorias (formato DT)
        """
        # TODO: Contar frecuencia de cada item en el dataset
        # Hint: Concatenar todos los 'items' de todas las trayectorias
        # all_items = np.concatenate([traj['items'] for traj in trajectories])
        # self.item_counts = np.bincount(all_items, minlength=752)
        
        # TODO: Ordenar por frecuencia (más popular primero)
        # self.popular_items = np.argsort(self.item_counts)[::-1]
        pass
    
    def recommend(self, user_history, k=10):
        """
        Recomienda top-k items populares no vistos.
        
        Args:
            user_history: lista de item IDs ya vistos
            k: número de recomendaciones
        
        Returns:
            recommendations: lista de k item IDs
        """
        # TODO: Filtrar items ya vistos y retornar top-k
        pass
```

**✓ Criterio de éxito:** Puede generar recomendaciones (aunque sean malas).

---

#### **3.2. Baseline: Behavior Cloning (Opcional)**

Implementar un transformer SIN conditioning en R̂ (solo predice P(a|s)).

**Nota:** Esto es opcional. Si no tienen tiempo, comparen solo contra Popularity.

---

#### **3.3. Métricas de Evaluación:**

**Archivo:** `src/evaluation/metrics.py`

```python
# 🎯 IMPLEMENTAR: Funciones de métricas
# Código de referencia disponible en el TP

import torch
import numpy as np

def hit_rate_at_k(predictions, targets, k=10):
    """
    Calcula Hit Rate @K.
    
    Args:
        predictions: (batch, num_items) - scores para cada item
        targets: (batch,) - item verdadero
        k: top-K items
    
    Returns:
        hit_rate: float entre 0 y 1
    """
    # TODO: Ver código de referencia en el TP
    # Hint: Usar torch.topk para obtener top-k predicciones
    # Hint: Verificar si target está en top-k
    pass

def ndcg_at_k(predictions, targets, k=10):
    """
    Normalized Discounted Cumulative Gain @K.
    """
    # TODO: Ver fórmula en el TP
    # NDCG = DCG / IDCG
    pass

def mrr(predictions, targets):
    """
    Mean Reciprocal Rank.
    """
    # TODO: MRR = promedio de 1/rank del item verdadero
    pass
```

**✓ Criterio de éxito:** Las métricas dan valores entre 0 y 1 y son consistentes.

---

#### **3.4. Evaluation Loop:**

**Archivo:** `src/evaluation/evaluate.py`

```python
# 🎯 IMPLEMENTAR: Evaluación del modelo
# Código muy detallado en el TP - seguir esa guía

@torch.no_grad()
def evaluate_model(model, test_data, device, target_return=None, k_list=[5, 10, 20]):
    """
    Evalúa el modelo en test set (cold-start users).
    
    Ver código completo en TRABAJO_PRACTICO_DECISION_TRANSFORMER.md
    """
    model.eval()
    
    # TODO: Seguir lógica del TP:
    # 1. Para cada usuario de test
    # 2. Simular sesión: empezar con history vacío
    # 3. Ir "recomendando" items y observando ratings
    # 4. Calcular métricas
    
    pass
```

---

## 🚦 PARTE 4: EXPERIMENTOS

### ✅ QUÉ IMPLEMENTAR

#### **4.1. Experimento: Effect of Return**

**Archivo:** `notebooks/04_return_conditioning_experiments.ipynb`

```python
# 🎯 IMPLEMENTAR: Experimentar con diferentes R̂

# === Calcular percentiles de returns en training ===
# train_returns = [traj['returns_to_go'][0] for traj in trajectories]
# percentiles = {
#     'p25': np.percentile(train_returns, 25),
#     'p50': np.percentile(train_returns, 50),
#     'p75': np.percentile(train_returns, 75),
#     'p90': np.percentile(train_returns, 90),
#     'max': np.max(train_returns)
# }

# === Evaluar modelo con cada return objetivo ===
# results = {}
# for name, rtg_value in percentiles.items():
#     metrics = evaluate_model(model, test_data, device, target_return=rtg_value)
#     results[name] = metrics

# === Graficar Return vs Performance ===
# plt.plot(rtg_values, hr10_values, ...)
```

**✓ Criterio de éxito:** Gráfico que muestra cómo cambia Hit Rate con diferentes R̂.

---

#### **4.2. Análisis por Grupo:**

```python
# 🎯 IMPLEMENTAR: Performance por grupo de usuarios

# === Agrupar test users por grupo ===
# for group_id in range(8):
#     users_in_group = [u for u in test_data if u['group'] == group_id]
#     metrics = evaluate_model(model, users_in_group, device)
#     print(f'Group {group_id}: HR@10={metrics["HR@10"]:.4f}')
```

---

## 🚦 PARTE 5: REPORTE

### ✅ QUÉ ENTREGAR

**Archivo:** `REPORTE.pdf` (3-5 páginas)

**Estructura:**

1. **Introducción** (0.5 pág)
   - Contexto del problema
   - Objetivos del TP

2. **Dataset y Preprocesamiento** (1 pág)
   - Estadísticas clave
   - Gráficos más importantes
   - Explicación del preprocesamiento

3. **Implementación** (1 pág)
   - Arquitectura del modelo (diagrama simple)
   - Hiperparámetros usados
   - Detalles de training

4. **Resultados** (1.5 pág)
   - Tabla comparativa: DT vs Baselines
   - Gráficos de experiments
   - Análisis de cold-start

5. **Conclusiones** (0.5 pág)
   - Lecciones aprendidas
   - Ventajas/limitaciones observadas

---

## 📝 CHECKLIST FINAL

Antes de entregar, verificar que tienen:

### Parte 1:
- [ ] `src/data/load_data.py` funcional
- [ ] `src/data/preprocessing.py` con `create_dt_dataset()` implementada
- [ ] `notebooks/01_exploracion_dataset.ipynb` ejecutado con gráficos
- [ ] Dataset procesado guardado en `data/processed/`

### Parte 2:
- [ ] `src/models/decision_transformer.py` (copiado/adaptado del TP)
- [ ] `src/data/dataset.py` con `RecommendationDataset` implementado
- [ ] `src/training/trainer.py` con función de training
- [ ] `notebooks/02_training.ipynb` con logs y gráficos de loss
- [ ] Modelo entrenado guardado en `results/checkpoints/`

### Parte 3:
- [ ] `src/models/baselines.py` con al menos PopularityRecommender
- [ ] `src/evaluation/metrics.py` con hit_rate, ndcg, mrr
- [ ] `src/evaluation/evaluate.py` con función de evaluación
- [ ] `notebooks/03_evaluation.ipynb` con tabla de resultados

### Parte 4:
- [ ] `notebooks/04_return_conditioning_experiments.ipynb` ejecutado
- [ ] Gráficos de Return vs Performance
- [ ] Análisis por grupo

### Parte 5:
- [ ] `REPORTE.pdf` (3-5 páginas)
- [ ] `README.md` con instrucciones de uso
- [ ] `requirements.txt` con dependencias

---

## 💡 CONSEJOS FINALES

### Para aprobar (60-70%):
- Implementar Partes 1, 2, 3 correctamente
- Modelo entrena y mejora (loss baja)
- Reporte básico completo

### Para destacar (80-90%):
- Todo lo anterior +
- Parte 4 completa con análisis detallado
- Comparación con múltiples baselines
- Visualizaciones claras y profesionales

### Para excelencia (95-100%):
- Todo lo anterior +
- Implementación propia del transformer (no copiar)
- Experimentos adicionales creativos
- Análisis profundo de resultados
- Código bien documentado y organizado

---

## ❓ PREGUNTAS FRECUENTES

**P: ¿Puedo copiar el código del TP directamente?**
R: Sí para el modelo (Parte 2). Para el resto, úsenlo como referencia pero implementen ustedes.

**P: ¿Tengo que implementar todo desde cero?**
R: No. El código del modelo está completo para copiar. Lo demás es más simple.

**P: No me da tiempo para todo, ¿qué priorizar?**
R: Partes 1, 2, 3 son esenciales. Parte 4 y 5 se pueden reducir.

**P: ¿Cómo debug si algo no funciona?**
R: 
1. Verificar shapes de tensores (print)
2. Empezar con batch_size pequeño (ej: 8)
3. Verificar que datos no tienen NaN
4. Comparar con código de referencia

**P: ¿Qué specs de hardware necesito?**
R: Google Colab gratuito es suficiente. O laptop con 8GB RAM.

---

**¡Buena suerte!** 🚀

