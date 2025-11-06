# TRABAJO PRÁCTICO: DECISION TRANSFORMER PARA RECOMENDACIONES
## Basado en RLT4Rec - Reinforcement Learning Transformer for Item Recommendation

---

## 📋 INFORMACIÓN GENERAL

**Curso:** Reinforcement Learning  
**Tema:** Offline RL y Decision Transformers  
**Dataset:** Netflix o Goodreads (RLT4Rec) - 16,000 usuarios  
**Paper de referencia:** [RLT4Rec: Reinforcement Learning Transformer for User Cold Start and Item Recommendation](https://arxiv.org/abs/2412.07403)  
**Duración estimada:** 2 semanas

### **🎬 Opción 1: Netflix (Películas)** - Recomendado
- 16,000 usuarios, 752 películas, ~1.8M ratings
- Dominio familiar para la mayoría

### **📚 Opción 2: Goodreads (Libros)** - Alternativa
- 16,000 usuarios, 472 libros, ~1.8M ratings  
- Misma estructura, diferente dominio

**Cada grupo puede elegir el dataset que prefiera.** El código es idéntico, solo cambian paths y número de items.  

---

## ⚠️ IMPORTANTE: LEER PRIMERO

Este documento contiene:
- ✅ **Descripción detallada del dataset** con ejemplos
- ✅ **Código de REFERENCIA completo** (para estudiar y usar como guía)
- ✅ **Especificaciones de cada parte del TP**

**📘 Para una guía paso a paso de QUÉ IMPLEMENTAR, leer:**
### 👉 **[GUIA_IMPLEMENTACION_GRUPOS.md](./GUIA_IMPLEMENTACION_GRUPOS.md)** 👈

Esa guía clarifica:
- ✓ Qué código pueden copiar/adaptar
- ✓ Qué deben implementar desde cero
- ✓ Checklist de entregables
- ✓ Consejos y troubleshooting

---

## 🎯 OBJETIVOS DE APRENDIZAJE

Al completar este trabajo práctico, los grupos serán capaces de:

1. **Formular** un problema de recomendación secuencial como un MDP
2. **Implementar** un Decision Transformer desde cero usando PyTorch
3. **Entrenar** el modelo con datos offline de interacciones usuario-item
4. **Evaluar** políticas de recomendación usando métricas estándar
5. **Experimentar** con conditioning en return-to-go para controlar comportamiento
6. **Comparar** Decision Transformer con métodos baseline (BC, Popularity, etc.)
7. **Analizar** el problema de cold-start en sistemas de recomendación

---

## 📊 CONTEXTO DEL PROBLEMA

### **Sistema de Recomendación como MDP**

**Estado (s_t):**
- Historia de items vistos: últimos K items que el usuario interactuó
- Features del usuario: grupo (cluster) al que pertenece (0-7)
- Contexto temporal: timestep en la sesión

**Acción (a_t):**
- Qué película recomendar siguiente (1 de 752 items posibles)

**Recompensa (r_t):**
- Rating del usuario: 1 (malo) a 5 (excelente)
- Alternativa: binario (1 si rating ≥ 4, 0 sino)

**Return-to-go (R̂_t):**
- Suma de ratings futuros desde timestep t hasta el final de la sesión
- R̂_t = Σ_{t'=t}^T r_{t'}

**Objetivo:**
Aprender una política π(a | s, R̂) que, condicionada en un return-to-go objetivo, genere secuencias de recomendaciones que maximicen la satisfacción del usuario.

---

## 📁 DATASETS PROPORCIONADOS

### **Elige uno de estos dos datasets:**

#### **Opción 1: Netflix (Películas)** 🎬

```
data/
├── train/
│   └── netflix8_train.df          # 16,000 usuarios, 752 películas
├── test_users/
│   └── netflix8_test.json         # 1,600 usuarios cold-start
└── groups/
    ├── mu_netflix8.csv            # Centroides de 8 grupos
    ├── sigma_netflix8.csv         # Desviaciones estándar
    └── num_netflix8.csv           # Usuarios por grupo
```

#### **Opción 2: Goodreads (Libros)** 📚

```
data/
├── train/
│   └── goodreads8_train.df        # 16,000 usuarios, 472 libros
├── test_users/
│   └── goodreads8_test.json       # 1,600 usuarios cold-start
└── groups/
    ├── mu_goodreads8.csv          # Centroides de 8 grupos
    ├── sigma_goodreads8.csv       # Desviaciones estándar
    └── num_goodreads8.csv         # Usuarios por grupo
```

**💡 Nota:** El formato y estructura son **idénticos** en ambos datasets. Solo cambian:
- Número de items (752 vs 472)
- Dominio (películas vs libros)

---

### **📊 Estadísticas de los Datasets:**

#### **🎬 Netflix (Películas):**
```
Training:
  ✓ 16,000 usuarios (2,000 por cada uno de 8 grupos)
  ✓ 752 películas únicas (item IDs: 0-751)
  ✓ Secuencias: 25-200 ratings (promedio: 112)
  ✓ Total interacciones: ~1,797,612 ratings
  ✓ Distribución de ratings:
      Rating 1: 13.9% | Rating 2: 17.9% | Rating 3: 22.2%
      Rating 4: 24.6% | Rating 5: 21.4%

Test:
  ✓ 1,600 usuarios cold-start (200 por grupo)
  ✓ Cada usuario tiene rating de TODAS las 752 películas
```

#### **📚 Goodreads (Libros):**
```
Training:
  ✓ 16,000 usuarios (2,000 por cada uno de 8 grupos)
  ✓ 472 libros únicos (item IDs: 0-471)
  ✓ Secuencias: 25-200 ratings (promedio: 112)
  ✓ Total interacciones: ~1,796,862 ratings
  ✓ Distribución similar a Netflix

Test:
  ✓ 1,600 usuarios cold-start (200 por grupo)
  ✓ Cada usuario tiene rating de TODOS los 472 libros
```

**🔄 Comparación:**
| Característica | Netflix | Goodreads |
|---------------|---------|-----------|
| Items | 752 películas | 472 libros |
| Usuarios train | 16,000 | 16,000 |
| Usuarios test | 1,600 | 1,600 |
| Ratings totales | ~1.8M | ~1.8M |
| Complejidad | Media | Ligeramente menor (menos items) |

---

### **🔍 EXPLICACIÓN DETALLADA DEL DATASET**

#### **1. Archivo de Training: `netflix8_train.df`**

Este es un **pandas DataFrame serializado** (formato pickle) que contiene las interacciones históricas de usuarios con películas.

**Estructura:**

| user_id | user_group | items | ratings |
|---------|------------|-------|---------|
| 0 | 2 | [472, 97, 122, ...] | [4.0, 3.0, 4.0, ...] |
| 1 | 5 | [301, 89, 456, ...] | [3.0, 5.0, 5.0, ...] |
| 2 | 1 | [12, 703, 88, ...] | [5.0, 4.0, 3.0, ...] |
| ... | ... | ... | ... |

**Columnas:**

- **`user_id`** (int): Identificador único del usuario (0 a 15,999)
- **`user_group`** (int): Grupo/cluster al que pertenece el usuario (0 a 7)
  - Los usuarios fueron agrupados usando clustering (ej: K-means) basado en sus preferencias
  - Usuarios del mismo grupo tienen patrones de consumo similares
- **`items`** (numpy array): Secuencia ordenada de películas que el usuario vio/calificó
  - Cada elemento es un item_id entre 0 y 751
  - El orden representa la secuencia temporal de interacciones
- **`ratings`** (numpy array): Calificación que el usuario dio a cada película
  - Valores entre 1.0 y 5.0 (discretos: 1.0, 2.0, 3.0, 4.0, 5.0)
  - Mismo largo que `items` (alineado por índice)

**Ejemplo concreto:**

```python
# Usuario 0
user_id: 0
user_group: 2
items: array([472,  97, 122, 654, 709, 467, 574, 544, 478, 338, ...])  # 112 items total
ratings: array([4., 3., 4., 3., 5., 4., 2., 1., 4., 5., ...])          # 112 ratings

# Interpretación:
# - El usuario 0 pertenece al grupo 2
# - Primero vio la película 472 y le dio rating 4
# - Luego vio la película 97 y le dio rating 3
# - Después vio la película 122 y le dio rating 4
# - ... y así sucesivamente por 112 interacciones
```

**¿Cómo se usa esto?**

Esta secuencia representa una **trayectoria** del usuario. En términos de RL:
- Cada timestep t correspone a una interacción
- **Estado (s_t):** Las películas vistas hasta el momento (history)
- **Acción (a_t):** La película recomendada en t (= items[t])
- **Recompensa (r_t):** El rating dado (= ratings[t])

---

#### **2. Archivo de Test: `netflix8_test.json`**

Este es un archivo JSON que contiene usuarios **cold-start** (nuevos, sin historial de training).

**Estructura:**

```python
[
  {
    'group': 0,
    'iter': 0,  # ignorar - es un índice de iteración del experimento original
    'items': [0, 1, 2, 3, 4, ..., 751],  # TODOS los 752 items en orden
    'ratings': [4, 1, 3, 5, 5, ..., 3]   # Rating para cada item
  },
  {
    'group': 0,
    'iter': 1,
    'items': [0, 1, 2, 3, 4, ..., 751],
    'ratings': [5, 4, 3, 3, 5, ..., 4]
  },
  ...  # 1,600 usuarios total
]
```

**Campos:**

- **`group`** (int): Grupo al que fue asignado este usuario (0-7)
  - Basado en similitud con centroides de training (ver `mu_netflix8.csv`)
- **`iter`** (int): Índice de iteración - **IGNORAR** (solo para tracking en el paper)
- **`items`** (list): Lista de 752 elementos: [0, 1, 2, ..., 751]
  - Siempre el mismo orden (todos los items disponibles)
- **`ratings`** (list): Rating que el usuario daría a cada película
  - ratings[0] = rating para item 0
  - ratings[1] = rating para item 1
  - ... ratings[751] = rating para item 751

**Ejemplo concreto:**

```python
# Usuario de test #0
{
  'group': 0,
  'iter': 0,
  'items': [0, 1, 2, 3, 4, 5, ..., 751],
  'ratings': [4, 1, 3, 5, 5, 4, ..., 3]
}

# Interpretación:
# - Usuario pertenece al grupo 0
# - Si le recomiendas película 0 → rating esperado: 4
# - Si le recomiendas película 1 → rating esperado: 1 (mala recomendación)
# - Si le recomiendas película 3 → rating esperado: 5 (excelente)
# ... etc
```

**¿Por qué este formato?**

En test, queremos evaluar qué tan buenas son nuestras recomendaciones. Como es un ambiente offline, no podemos dejar que el usuario "explore" - ya tenemos su rating para todas las películas. Esto nos permite:

1. Simular una sesión: empezar con historial vacío, ir recomendando películas
2. Observar el rating que obtendríamos (lookup en la lista)
3. Evaluar métricas: ¿recomendamos las películas con ratings altos?

**Importante:** Estos usuarios NO aparecieron en training → problema de **cold-start**.

---

#### **3. Archivos de Grupos: `data/groups/`**

Estos archivos contienen información sobre los 8 clusters de usuarios.

**`mu_netflix8.csv`** (Centroides de grupos):
- Matriz de 8 × 752
- Fila i = centroide del grupo i
- Columna j = rating promedio que usuarios del grupo i dan a película j

```python
# Ejemplo (simplificado):
#        Item0  Item1  Item2  ...  Item751
# Grp 0   3.53   3.55   2.77  ...    4.20
# Grp 1   2.86   2.63   2.66  ...    4.01
# ...
```

**Uso:** Puedes usar estos centroides para:
- Inicializar representación de usuarios cold-start
- Entender qué prefiere cada grupo
- Baseline: recomendar items con rating alto en el centroid del grupo

**`sigma_netflix8.csv`** (Desviaciones estándar):
- Matriz de 8 × 752
- Mide la variabilidad de ratings dentro de cada grupo

**`num_netflix8.csv`** (Conteo de usuarios):
- Matriz de 8 × 752
- Número de usuarios del grupo que calificaron cada película

**Estos archivos son OPCIONALES para el TP básico.** Se pueden usar para análisis avanzado o baselines sofisticados.

---

### **🎯 RESUMEN: ¿Qué tengo disponible?**

| Archivo | Uso principal | Necesario |
|---------|---------------|-----------|
| `netflix8_train.df` | Entrenar el Decision Transformer | ✅ Sí |
| `netflix8_test.json` | Evaluar en cold-start users | ✅ Sí |
| `mu_netflix8.csv` | Centroides (opcional para baselines) | ⚠️ Opcional |
| `sigma_netflix8.csv` | Análisis avanzado | ❌ No |
| `num_netflix8.csv` | Análisis avanzado | ❌ No |

---

### **💡 Cómo Cargar los Datos (Código de Ejemplo):**

```python
import pandas as pd
import numpy as np
import json

# ============================================
# 🎯 CONFIGURACIÓN: ELEGIR DATASET
# ============================================
# Descomentar el dataset que quieras usar:

DATASET = 'netflix'    # Opción 1: Películas (752 items)
# DATASET = 'goodreads'  # Opción 2: Libros (472 items)

# Configurar paths según el dataset elegido
if DATASET == 'netflix':
    NUM_ITEMS = 752
    train_path = 'data/train/netflix8_train.df'
    test_path = 'data/test_users/netflix8_test.json'
    centroids_path = 'data/groups/mu_netflix8.csv'
    item_name = 'películas'
elif DATASET == 'goodreads':
    NUM_ITEMS = 472
    train_path = 'data/train/goodreads8_train.df'
    test_path = 'data/test_users/goodreads8_test.json'
    centroids_path = 'data/groups/mu_goodreads8.csv'
    item_name = 'libros'

print(f"📊 Dataset seleccionado: {DATASET.upper()}")
print(f"📦 Número de items: {NUM_ITEMS} {item_name}")
print("="*60)

# === CARGAR TRAINING DATA ===
df_train = pd.read_pickle(train_path)

print(f"\nNúmero de usuarios: {len(df_train)}")
print(f"Columnas: {df_train.columns.tolist()}")

# Ver un usuario ejemplo
user_0 = df_train.iloc[0]
print(f"\nUsuario 0:")
print(f"  Grupo: {user_0['user_group']}")
print(f"  # de ratings: {len(user_0['items'])}")
print(f"  Primeros 5 {item_name} vistos: {user_0['items'][:5]}")
print(f"  Primeros 5 ratings: {user_0['ratings'][:5]}")

# === CARGAR TEST DATA ===
with open(test_path, 'r') as f:
    test_users = json.load(f)

print(f"\nNúmero de usuarios de test: {len(test_users)}")
print(f"\nUsuario de test 0:")
print(f"  Grupo: {test_users[0]['group']}")
print(f"  Rating para item 0: {test_users[0]['ratings'][0]}")
print(f"  Rating para item 100: {test_users[0]['ratings'][100]}")

# === CARGAR CENTROIDES (OPCIONAL) ===
mu = pd.read_csv(centroids_path, header=None)
print(f"\nCentroides shape: {mu.shape}")  # (8, NUM_ITEMS)
print(f"Rating promedio del grupo 0 para item 0: {mu.iloc[0, 0]:.2f}")
```

**🔄 Para cambiar de dataset:** Solo modifica la variable `DATASET` al inicio del código.

---

## 🔧 ESTRUCTURA DEL TRABAJO PRÁCTICO

### **PARTE 1: Exploración y Preparación**

#### **Tareas:**

**1.1. Carga y Exploración de Datos**

```python
import pandas as pd
import numpy as np
import json

# Cargar training data
df_train = pd.read_pickle('data/train/netflix8_train.df')

# Cargar test data
with open('data/test_users/netflix8_test.json', 'r') as f:
    test_users = json.load(f)

# Cargar información de grupos
mu = pd.read_csv('data/groups/mu_netflix8.csv', header=None)
sigma = pd.read_csv('data/groups/sigma_netflix8.csv', header=None)
```

**Análisis requerido:**
- Distribución de longitud de secuencias
- Distribución de ratings por grupo de usuario
- Items más populares vs long-tail
- Sparsity del dataset

**1.2. Visualización**

Crear al menos 3 gráficos:
- Histograma de longitud de secuencias
- Distribución de ratings (general y por grupo)
- Top-20 películas más populares vs least popular

**1.3. Preprocesamiento para Decision Transformer**

🎯 **PARA IMPLEMENTAR:** Convertir datos raw a formato adecuado para el modelo.

**Objetivo:** Crear una función que tome el DataFrame de training y lo convierta en una lista de trayectorias con el formato necesario para Decision Transformer.

**🔍 Código de Referencia (Leer y Entender):**

```python
# ===================================================================
# CÓDIGO DE REFERENCIA - Muestra el formato esperado
# Los grupos deben implementar esto (o similar) en data_preprocessing.py
# ===================================================================

def create_dt_dataset(df_train):
    """
    Convierte el dataset raw a formato Decision Transformer.
    
    Args:
        df_train: DataFrame con columnas [user_id, user_group, items, ratings]
    
    Returns:
        trajectories: List[Dict] donde cada dict representa una trayectoria
                      con las siguientes keys:
            - 'items': numpy array de item IDs (secuencia completa)
            - 'ratings': numpy array de ratings (secuencia completa)
            - 'returns_to_go': numpy array con suma de rewards futuros desde cada timestep
            - 'timesteps': numpy array con índices temporales [0, 1, 2, ..., T-1]
            - 'user_group': int con el grupo del usuario (0-7)
    
    Ejemplo de salida:
        [
          {
            'items': array([472, 97, 122, ...]),      # 112 elementos
            'ratings': array([4., 3., 4., ...]),       # 112 elementos
            'returns_to_go': array([450., 446., ...]), # 112 elementos (suma acumulada hacia adelante)
            'timesteps': array([0, 1, 2, ...]),        # 112 elementos
            'user_group': 2
          },
          ... # 16,000 trayectorias total
        ]
    """
    trajectories = []
    
    # Iterar sobre cada usuario
    for idx, row in df_train.iterrows():
        items = row['items']        # numpy array de item IDs
        ratings = row['ratings']    # numpy array de ratings
        group = row['user_group']   # int (0-7)
        
        # === PASO CLAVE: Calcular returns-to-go ===
        # R̂_t = suma de rewards desde t hasta el final
        # R̂_t = r_t + r_{t+1} + ... + r_T
        
        returns = np.zeros(len(ratings))
        
        # Último timestep: R̂_T = r_T
        returns[-1] = ratings[-1]
        
        # Iterar hacia atrás: R̂_t = r_t + R̂_{t+1}
        for t in range(len(ratings)-2, -1, -1):
            returns[t] = ratings[t] + returns[t+1]
        
        # Ejemplo:
        # ratings =  [4, 3, 5, 2, 1]
        # returns = [15, 11, 8, 3, 1]  # 15=4+3+5+2+1, 11=3+5+2+1, etc.
        
        # Crear diccionario con toda la información
        trajectory = {
            'items': items,                        # Secuencia de películas
            'ratings': ratings,                    # Ratings correspondientes
            'returns_to_go': returns,              # R̂ para cada timestep
            'timesteps': np.arange(len(items)),    # [0, 1, 2, ..., T-1]
            'user_group': group                    # Cluster del usuario
        }
        
        trajectories.append(trajectory)
    
    return trajectories


# === EJEMPLO DE USO ===
# import pandas as pd
# df_train = pd.read_pickle('data/train/netflix8_train.df')
# trajectories = create_dt_dataset(df_train)
# print(f"Total trayectorias: {len(trajectories)}")  # 16,000
# print(f"Primera trayectoria keys: {trajectories[0].keys()}")
# print(f"Returns-to-go del usuario 0: {trajectories[0]['returns_to_go'][:10]}")
```

---

**🎯 Tareas para los Grupos:**

1. **Implementar la función `create_dt_dataset()`** en el archivo `data_preprocessing.py`
   - Puede ser igual al código de referencia o con mejoras propias
   - Debe generar el mismo formato de output (diccionarios con las keys especificadas)

2. **Validar el preprocesamiento:**
   - Verificar que `len(items) == len(ratings) == len(returns_to_go) == len(timesteps)`
   - Verificar que `returns_to_go[0]` es efectivamente la suma total de ratings
   - Verificar que `returns_to_go[-1] == ratings[-1]`

3. **Crear estadísticas adicionales:**
   - Distribución de longitudes de trayectorias
   - Distribución de returns-to-go iniciales (R̂_0)
   - Cuartiles de returns: p25, p50, p75, p95

4. **Guardar el dataset procesado:**
   ```python
   import pickle
   with open('data/processed/trajectories_train.pkl', 'wb') as f:
       pickle.dump(trajectories, f)
   ```

---

**✅ Entregable Parte 1:**
- ✓ Notebook: `01_exploracion_dataset.ipynb` con análisis y visualizaciones
- ✓ Script: `data_preprocessing.py` con la función implementada
- ✓ Reporte breve (1-2 páginas) con:
  - Estadísticas clave del dataset
  - Al menos 3 gráficos informativos
  - Explicación del preprocesamiento realizado

---

### **PARTE 2: Implementación del Decision Transformer**

#### **2.1. Arquitectura del Modelo**

Implementar Decision Transformer completo:

```python
import torch
import torch.nn as nn

class DecisionTransformer(nn.Module):
    def __init__(
        self,
        num_items=752,
        num_groups=8,
        hidden_dim=128,
        n_layers=3,
        n_heads=4,
        context_length=20,
        max_timestep=200,
        dropout=0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.context_length = context_length
        
        # === EMBEDDINGS ===
        
        # Item embedding (para history y acciones)
        self.item_embedding = nn.Embedding(num_items, hidden_dim)
        
        # User group embedding
        self.group_embedding = nn.Embedding(num_groups, hidden_dim)
        
        # Return-to-go embedding (escalar continuo)
        self.rtg_embedding = nn.Linear(1, hidden_dim)
        
        # Timestep embedding (positional encoding)
        self.timestep_embedding = nn.Embedding(max_timestep, hidden_dim)
        
        # === TRANSFORMER ===
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )
        
        # === PREDICTION HEAD ===
        
        # Predecir qué item recomendar
        self.predict_item = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_items)
        )
        
        # Layer normalization
        self.ln = nn.LayerNorm(hidden_dim)
        
    def forward(self, 
                states,        # (batch, seq_len) - item IDs vistos
                actions,       # (batch, seq_len) - item IDs recomendados
                returns_to_go, # (batch, seq_len, 1) - R̂ values
                timesteps,     # (batch, seq_len) - posiciones temporales
                user_groups,   # (batch,) - cluster del usuario
                attention_mask=None):
        """
        Args:
            states: IDs de items en history
            actions: IDs de items recomendados (targets)
            returns_to_go: R̂ para cada timestep
            timesteps: posiciones temporales
            user_groups: cluster del usuario
            
        Returns:
            item_logits: (batch, seq_len, num_items) - probabilidades sobre items
        """
        batch_size, seq_len = states.shape
        
        # === EMBED INPUTS ===
        
        # States (history)
        state_emb = self.item_embedding(states)  # (B, L, H)
        
        # Actions (ya recomendados, para autoregression)
        action_emb = self.item_embedding(actions)  # (B, L, H)
        
        # Returns-to-go
        rtg_emb = self.rtg_embedding(returns_to_go)  # (B, L, H)
        
        # Timesteps
        time_emb = self.timestep_embedding(timesteps)  # (B, L, H)
        
        # User group (broadcast a toda la secuencia)
        group_emb = self.group_embedding(user_groups).unsqueeze(1)  # (B, 1, H)
        group_emb = group_emb.expand(-1, seq_len, -1)  # (B, L, H)
        
        # === INTERLEAVE EMBEDDINGS ===
        # Formato: [rtg_0, state_0, action_0, rtg_1, state_1, action_1, ...]
        
        # Para simplicidad, usamos sum de embeddings + positional
        # (En la versión completa, se pueden interleave explícitamente)
        h = state_emb + rtg_emb + time_emb + group_emb
        h = self.ln(h)
        
        # === CAUSAL MASK ===
        # Asegurar que cada timestep solo ve el pasado
        if attention_mask is None:
            attention_mask = self._generate_causal_mask(seq_len).to(h.device)
        
        # === TRANSFORMER ===
        h = self.transformer(h, mask=attention_mask)  # (B, L, H)
        
        # === PREDICT NEXT ITEM ===
        item_logits = self.predict_item(h)  # (B, L, num_items)
        
        return item_logits
    
    def _generate_causal_mask(self, seq_len):
        """Generate causal mask for autoregressive generation."""
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
        return mask
```

**Componentes a implementar:**
- ✅ Embeddings (items, groups, rtg, timesteps)
- ✅ Transformer encoder con causal masking
- ✅ Prediction head para items
- ✅ Forward pass completo

**2.2. Training Loop**

```python
def train_decision_transformer(
    model, 
    train_loader, 
    optimizer, 
    device,
    num_epochs=50
):
    """
    Entrena el Decision Transformer.
    
    Loss: Cross-entropy entre item predicho y item verdadero
    """
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        for batch in train_loader:
            states = batch['states'].to(device)      # (B, L)
            actions = batch['actions'].to(device)    # (B, L)
            rtg = batch['rtg'].to(device)            # (B, L, 1)
            timesteps = batch['timesteps'].to(device) # (B, L)
            groups = batch['groups'].to(device)      # (B,)
            targets = batch['targets'].to(device)    # (B, L) - next items
            
            # Forward pass
            logits = model(states, actions, rtg, timesteps, groups)
            
            # Compute loss
            loss = F.cross_entropy(
                logits.reshape(-1, model.num_items),
                targets.reshape(-1),
                ignore_index=-1  # para padding
            )
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')
    
    return model
```

**2.3. DataLoader**

Implementar PyTorch Dataset y DataLoader:

```python
from torch.utils.data import Dataset, DataLoader

class RecommendationDataset(Dataset):
    def __init__(self, trajectories, context_length=20):
        self.trajectories = trajectories
        self.context_length = context_length
        
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        
        # Extraer secuencia completa
        items = traj['items']
        ratings = traj['ratings']
        rtg = traj['returns_to_go']
        timesteps = traj['timesteps']
        group = traj['user_group']
        
        # Tomar una ventana de context_length
        # (o toda la secuencia si es más corta)
        seq_len = min(len(items), self.context_length)
        
        # Random start point (para data augmentation)
        if len(items) > self.context_length:
            start_idx = np.random.randint(0, len(items) - self.context_length + 1)
        else:
            start_idx = 0
        
        end_idx = start_idx + seq_len
        
        # States: items vistos (history)
        # Para t, state = items[:t]
        states = items[start_idx:end_idx]
        
        # Actions: items que fueron "recomendados" (mismo que states shifted)
        actions = items[start_idx:end_idx]
        
        # Targets: próximo item a predecir
        targets = np.zeros(seq_len, dtype=np.int64)
        targets[:-1] = items[start_idx+1:end_idx]
        targets[-1] = -1  # padding para último timestep
        
        # Returns-to-go
        rtg_seq = rtg[start_idx:end_idx].reshape(-1, 1)
        
        # Timesteps
        time_seq = timesteps[start_idx:end_idx]
        
        return {
            'states': torch.tensor(states, dtype=torch.long),
            'actions': torch.tensor(actions, dtype=torch.long),
            'rtg': torch.tensor(rtg_seq, dtype=torch.float32),
            'timesteps': torch.tensor(time_seq, dtype=torch.long),
            'groups': torch.tensor(group, dtype=torch.long),
            'targets': torch.tensor(targets, dtype=torch.long)
        }
```

**Entregable Parte 2:**
- Código: `models/decision_transformer.py`
- Código: `train.py`
- Código: `dataset.py`
- Notebook: `02_training.ipynb` con logs y gráficos de loss

---

### **PARTE 3: Baselines y Evaluación**

#### **3.1. Implementar Baselines**

Comparar Decision Transformer con al menos 2 baselines:

**Baseline 1: Popularity**
```python
class PopularityRecommender:
    def __init__(self):
        self.item_counts = None
        self.popular_items = None
    
    def fit(self, train_data):
        # Contar frecuencia de cada item
        all_items = np.concatenate([traj['items'] for traj in train_data])
        self.item_counts = np.bincount(all_items, minlength=752)
        self.popular_items = np.argsort(self.item_counts)[::-1]
    
    def recommend(self, user_history, k=10):
        # Recomendar los k items más populares que no estén en history
        recommendations = []
        for item in self.popular_items:
            if item not in user_history:
                recommendations.append(item)
            if len(recommendations) == k:
                break
        return recommendations
```

**Baseline 2: Behavior Cloning (Transformer sin R̂)**
```python
class BehaviorCloningTransformer(nn.Module):
    """
    Transformer que predice P(a_t | s_t, history)
    Sin conditioning en return-to-go.
    """
    def __init__(self, num_items, hidden_dim, n_layers, n_heads):
        super().__init__()
        # Similar a DecisionTransformer pero SIN rtg_embedding
        # Solo usa: item_embedding + timestep_embedding
        ...
```

**Baseline 3 (Opcional): User-based Collaborative Filtering**
```python
class CollaborativeFiltering:
    def __init__(self):
        self.user_item_matrix = None
    
    def fit(self, train_data):
        # Construir matriz usuario-item
        # Usar similaridad coseno entre usuarios
        ...
    
    def recommend(self, user_group, user_history, k=10):
        # Recomendar basado en usuarios similares del mismo grupo
        ...
```

#### **3.2. Métricas de Evaluación**

Implementar métricas estándar:

```python
def hit_rate_at_k(predictions, targets, k=10):
    """
    Calcula Hit Rate @K.
    
    Args:
        predictions: (batch, num_items) - scores para cada item
        targets: (batch,) - item verdadero
        k: top-K items a considerar
    
    Returns:
        hit_rate: proporción de veces que target está en top-K
    """
    top_k = torch.topk(predictions, k, dim=1).indices
    hits = (top_k == targets.unsqueeze(1)).any(dim=1).float()
    return hits.mean().item()

def ndcg_at_k(predictions, targets, k=10):
    """
    Normalized Discounted Cumulative Gain @K.
    """
    top_k_indices = torch.topk(predictions, k, dim=1).indices
    
    # relevance = 1 si item está en top-k y es el target, 0 sino
    relevance = (top_k_indices == targets.unsqueeze(1)).float()
    
    # DCG = Σ (relevance / log2(rank+1))
    ranks = torch.arange(1, k+1, device=predictions.device).float()
    dcg = (relevance / torch.log2(ranks + 1)).sum(dim=1)
    
    # IDCG (ideal DCG) = 1 / log2(2) si target en posición 1
    idcg = 1.0 / np.log2(2)
    
    ndcg = dcg / idcg
    return ndcg.mean().item()

def mrr(predictions, targets):
    """
    Mean Reciprocal Rank.
    """
    # Ordenar items por score
    sorted_indices = torch.argsort(predictions, dim=1, descending=True)
    
    # Encontrar rank del target
    ranks = (sorted_indices == targets.unsqueeze(1)).nonzero()[:, 1] + 1
    
    rr = 1.0 / ranks.float()
    return rr.mean().item()
```

#### **3.3. Evaluation Loop**

```python
@torch.no_grad()
def evaluate_model(model, test_data, device, target_return=None, k_list=[5, 10, 20]):
    """
    Evalúa el modelo en test set (cold-start users).
    
    Args:
        model: Decision Transformer entrenado
        test_data: lista de usuarios de test
        target_return: R̂ objetivo para conditioning (si None, usa max del training)
        k_list: lista de K para métricas @K
    
    Returns:
        metrics: dict con Hit Rate, NDCG, MRR para cada K
    """
    model.eval()
    
    metrics = {f'HR@{k}': [] for k in k_list}
    metrics.update({f'NDCG@{k}': [] for k in k_list})
    metrics['MRR'] = []
    
    for user in test_data:
        group = user['group']
        items = user['items']
        ratings = user['ratings']
        
        # Simular sesión: tomar primeros N items como history, predecir siguiente
        context_len = 20
        
        for t in range(context_len, len(items)):
            # History
            history_items = items[t-context_len:t]
            history_ratings = ratings[t-context_len:t]
            
            # Calcular return-to-go
            if target_return is None:
                rtg = sum(history_ratings)  # o usar promedio del training
            else:
                rtg = target_return
            
            # Preparar inputs
            states = torch.tensor(history_items, dtype=torch.long).unsqueeze(0).to(device)
            actions = torch.tensor(history_items, dtype=torch.long).unsqueeze(0).to(device)
            rtg_input = torch.full((1, context_len, 1), rtg, dtype=torch.float32).to(device)
            timesteps = torch.arange(context_len, dtype=torch.long).unsqueeze(0).to(device)
            groups = torch.tensor([group], dtype=torch.long).to(device)
            
            # Predecir
            logits = model(states, actions, rtg_input, timesteps, groups)
            predictions = logits[0, -1, :]  # última posición
            
            # Target
            target_item = items[t]
            
            # Calcular métricas
            for k in k_list:
                hr = hit_rate_at_k(predictions.unsqueeze(0), 
                                   torch.tensor([target_item]).to(device), k)
                metrics[f'HR@{k}'].append(hr)
                
                ndcg = ndcg_at_k(predictions.unsqueeze(0), 
                                torch.tensor([target_item]).to(device), k)
                metrics[f'NDCG@{k}'].append(ndcg)
            
            mrr_val = mrr(predictions.unsqueeze(0), 
                         torch.tensor([target_item]).to(device))
            metrics['MRR'].append(mrr_val)
    
    # Promediar métricas
    return {key: np.mean(val) for key, val in metrics.items()}
```

**Entregable Parte 3:**
- Código: `baselines.py`
- Código: `evaluation.py`
- Notebook: `03_evaluation.ipynb` con tabla comparativa de resultados

---

### **PARTE 4: Experimentos con Return Conditioning**

#### **4.1. Effect of Target Return**

Evaluar el modelo con diferentes valores de R̂:

```python
# Calcular estadísticas de returns en training
train_returns = [traj['returns_to_go'][0] for traj in train_trajectories]
percentiles = {
    'p25': np.percentile(train_returns, 25),
    'p50': np.percentile(train_returns, 50),
    'p75': np.percentile(train_returns, 75),
    'p90': np.percentile(train_returns, 90),
    'max': np.max(train_returns)
}

# Evaluar con cada return objetivo
results = {}
for name, rtg_value in percentiles.items():
    metrics = evaluate_model(model, test_data, device, target_return=rtg_value)
    results[name] = metrics
    print(f'{name} (R̂={rtg_value:.2f}): HR@10={metrics["HR@10"]:.4f}, NDCG@10={metrics["NDCG@10"]:.4f}')

# Graficar: Return objetivo vs Performance
import matplotlib.pyplot as plt

rtg_values = list(percentiles.values())
hr10_values = [results[name]['HR@10'] for name in percentiles.keys()]

plt.figure(figsize=(10, 6))
plt.plot(rtg_values, hr10_values, marker='o', linewidth=2)
plt.xlabel('Target Return-to-go', fontsize=12)
plt.ylabel('Hit Rate @10', fontsize=12)
plt.title('Effect of Return Conditioning on Recommendation Performance')
plt.grid(True, alpha=0.3)
plt.show()
```

**Preguntas a responder:**
- ¿Especificar R̂ más alto mejora las recomendaciones?
- ¿Hay un punto de saturación o disminución?
- ¿El modelo puede extrapolar más allá del max return visto?

#### **4.2. Cold-Start Performance**

Analizar performance en usuarios nuevos (test set):

```python
# Agrupar test users por grupo
results_by_group = {}
for group_id in range(8):
    users_in_group = [u for u in test_data if u['group'] == group_id]
    
    if users_in_group:
        metrics = evaluate_model(model, users_in_group, device, target_return=percentiles['p75'])
        results_by_group[group_id] = metrics
        print(f'Group {group_id}: HR@10={metrics["HR@10"]:.4f}')

# Comparar con baseline que usa centroides de grupos
# (información de mu_netflix8.csv)
```

**Preguntas a responder:**
- ¿El modelo funciona bien para cold-start users?
- ¿Hay grupos donde funciona mejor/peor?
- ¿Usar el centroid del grupo ayuda?

**Entregable Parte 4:**
- Notebook: `04_return_conditioning_experiments.ipynb`
- Gráficos de Return vs Performance
- Análisis por grupo de usuarios

---

### **PARTE 5: Reporte Final**

#### **Contenido del Reporte (3-5 páginas):**

**1. Introducción**
- Contexto del problema
- Formulación como MDP
- Objetivos del trabajo

**2. Dataset y Preprocesamiento**
- Estadísticas del dataset
- Decisiones de preprocesamiento
- Visualizaciones clave

**3. Implementación**
- Arquitectura del Decision Transformer
- Detalles de training (hyperparams, optimización)
- Desafíos encontrados

**4. Resultados**
- Tabla comparativa: DT vs Baselines
- Gráficos de performance
- Análisis de return conditioning
- Performance en cold-start

**5. Análisis y Discusión**
- ¿Qué aprendió el modelo?
- ¿Cuándo funciona bien/mal?
- Limitaciones observadas
- Posibles mejoras

**6. Conclusiones**
- Lecciones aprendidas
- Ventajas de Decision Transformer para recomendaciones
- Trabajo futuro

**Formato:**
- PDF
- Incluir código snippets importantes
- Gráficos de alta calidad
- Referencias a papers relevantes

---

## 🎯 CRITERIOS DE EVALUACIÓN

| Componente | Criterios |
|-----------|-----------|
| **Parte 1: Exploración** | - Análisis completo del dataset<br>- Visualizaciones claras y relevantes<br>- Preprocesamiento correcto |
| **Parte 2: Implementación** | - Arquitectura funcional<br>- Training loop correcto<br>- DataLoader implementado |
| **Parte 3: Baselines** | - 2+ baselines implementados<br>- Métricas correctas<br>- Evaluation completa |
| **Parte 4: Experimentos** | - Return conditioning analizado<br>- Cold-start analysis completo |
| **Parte 5: Reporte** | - Claridad y profundidad del análisis |

### **Trabajo Adicional (Opcional):**
- Implementar attention visualization
- Análisis de embeddings de items con t-SNE
- Implementar variante con multi-objective conditioning
- Comparación con más baselines (ej: Matrix Factorization)

---

## 📚 RECURSOS Y REFERENCIAS

### **Papers:**
1. Chen et al. (2021) - Decision Transformer: Reinforcement Learning via Sequence Modeling
2. Rajapakse & Leith (2024) - RLT4Rec: RL Transformer for User Cold Start and Item Recommendation
3. Vaswani et al. (2017) - Attention is All You Need
4. Levine et al. (2020) - Offline RL: Tutorial, Review, and Perspectives

### **Código de Referencia:**
- [Decision Transformer GitHub](https://github.com/kzl/decision-transformer)
- [RLT4Rec Repository](https://github.com/dilina-r/RLT4Rec)
- [MinGPT](https://github.com/karpathy/minGPT) - Simple transformer implementation

### **Tutoriales:**
- [Transformers from Scratch](https://peterbloem.nl/blog/transformers)
- [PyTorch Transformer Tutorial](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)

---

## 🚀 ENTREGABLES FINALES

### **Estructura de Carpetas:**

```
trabajo_practico_dt_recsys/
│
├── data/                          # (proporcionado)
│   ├── train/netflix8_train.df
│   ├── test_users/netflix8_test.json
│   └── groups/*.csv
│
├── notebooks/
│   ├── 01_exploracion_dataset.ipynb
│   ├── 02_training.ipynb
│   ├── 03_evaluation.ipynb
│   └── 04_return_conditioning_experiments.ipynb
│
├── src/
│   ├── models/
│   │   ├── decision_transformer.py
│   │   └── baselines.py
│   ├── data/
│   │   ├── dataset.py
│   │   └── preprocessing.py
│   ├── training/
│   │   └── trainer.py
│   └── evaluation/
│       └── metrics.py
│
├── scripts/
│   ├── train.py
│   └── evaluate.py
│
├── results/
│   ├── figures/
│   ├── logs/
│   └── checkpoints/
│
├── requirements.txt
├── README.md
└── REPORTE.pdf
```

### **Archivo README.md debe incluir:**
- Instrucciones de instalación
- Cómo reproducir los experimentos
- Descripción de cada notebook/script
- Resultados principales (tabla resumida)

### **requirements.txt:**
```
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
tqdm>=4.65.0
jupyter>=1.0.0
```

---

## ⏰ CRONOGRAMA SUGERIDO

| Semana | Tareas | Hitos |
|--------|--------|-------|
| **1** | Parte 1: Exploración y preprocesamiento | Dataset listo, visualizaciones completas |
| **2** | Parte 2: Implementación del modelo | Modelo entrenando, loss convergiendo |
| **3** | Parte 3 y 4: Baselines y experimentos | Evaluaciones completas, gráficos finales |
| **4** | Parte 5: Reporte y pulido final | Entrega completa |

---

## 💡 CONSEJOS Y TIPS

### **Para la Implementación:**
1. Empieza con un modelo pequeño (hidden_dim=64, n_layers=2) para debugging
2. Usa batch_size pequeño al principio para verificar que funciona
3. Guarda checkpoints frecuentemente
4. Loggea todo (loss, métricas, hiperparámetros)

### **Para el Training:**
5. Si el loss no baja, revisa:
   - ¿Los embeddings están correctos?
   - ¿La causal mask está bien?
   - ¿El learning rate es apropiado? (prueba 1e-4)
6. Usa gradient clipping (max_norm=1.0)
7. Considera learning rate warmup

### **Para la Evaluación:**
8. Evalúa en validation set durante training para detectar overfitting
9. No uses los mismos items para training y test en una sesión
10. Asegúrate de que las métricas están implementadas correctamente

### **Para el Reporte:**
11. Incluye ejemplos cualitativos (ej: "para este usuario con R̂ alto, el modelo recomendó...")
12. Discute casos de falla (¿cuándo el modelo falla?)
13. Compara con trabajos relacionados (papers)

---

## ❓ PREGUNTAS FRECUENTES

**Q: ¿Puedo usar librerías de transformers pre-entrenados (ej: HuggingFace)?**
A: No para el modelo principal. El objetivo es implementar desde cero. Pero puedes usarlas para comparación (bonus).

**Q: ¿Qué tamaño de modelo es razonable?**
A: hidden_dim=128, n_layers=3, n_heads=4 es un buen punto de partida. Con el dataset (1.8M ejemplos), esto debería entrenar en 1-2 horas en una GPU.

**Q: ¿Puedo usar Google Colab?**
A: Sí, perfectamente. El dataset cabe en memoria y el modelo no es muy grande.

**Q: ¿Cómo manejo el cold-start exactamente?**
A: Los test users no tienen history de training. Usa solo su `group` y el centroid del cluster como información inicial, junto con el R̂ objetivo.

**Q: ¿Debo discretizar los ratings?**
A: No necesariamente. Puedes usar ratings continuos (1-5) directamente. O binarizarlos (≥4 = positivo) si prefieres.

**Q: ¿Qué hago si no me da tiempo para todos los experimentos?**
A: Prioriza: Parte 1, 2, 3 son esenciales. Parte 4 y 5 se pueden reducir si falta tiempo.

---

## 📧 CONTACTO Y CONSULTAS

**Docente:** [Nombre]  
**Email:** [email]  
**Horario de consultas:** [horario]  

**Fecha de entrega:** [Fecha]  
**Modalidad de entrega:** [Plataforma]

---

**¡Buena suerte con el trabajo práctico!** 🚀

