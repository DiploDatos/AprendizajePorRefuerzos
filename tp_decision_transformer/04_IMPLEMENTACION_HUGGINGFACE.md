# 🤗 IMPLEMENTACIÓN CON HUGGINGFACE TRANSFORMERS

## 📌 ALTERNATIVA PARA GRUPOS SIN EXPERIENCIA EN PYTORCH

---

## 🎯 ¿Para quién es este documento?

Este documento es una **alternativa** para grupos que:
- ✅ No tienen experiencia profunda con PyTorch
- ✅ Conocen o quieren usar HuggingFace Transformers
- ✅ Prefieren usar componentes pre-construidos de alto nivel

**⚠️ IMPORTANTE:**
- Esto es **opcional** - la implementación en `03_REFERENCIA_COMPLETA.md` sigue siendo válida
- Ambas implementaciones logran el mismo objetivo
- HuggingFace simplifica mucho el código pero oculta detalles internos

---

## 📦 DEPENDENCIAS

Agregar a `requirements.txt`:

```txt
torch>=2.0.0
transformers>=4.30.0
pandas>=1.5.0
numpy>=1.24.0
tqdm>=4.65.0
matplotlib>=3.7.0
```

Instalar:
```bash
pip install transformers torch pandas numpy tqdm matplotlib
```

---

## 🏗️ ARQUITECTURA CON HUGGINGFACE

La idea es usar **GPT-2** de HuggingFace como backbone del transformer y adaptarlo para Decision Transformer:

```
Input Sequence:
[group_emb_0, state_0, action_0, rtg_0, group_emb_1, state_1, action_1, rtg_1, ...]
                            ↓
                    GPT2Model (HuggingFace)
                            ↓
                 Transformer Encoding
                            ↓
              [h_0, h_1, h_2, ..., h_T]
                            ↓
                    Action Head (Linear)
                            ↓
              [logits_0, logits_1, ..., logits_T]
                            ↓
                    Predict Next Item
```

---

## 🔧 CÓDIGO COMPLETO

### **Parte 1: Imports y Configuración**

```python
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pickle

# Configuración del dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# === CONFIGURACIÓN DEL DATASET ===
# Cambiar aquí para elegir Netflix o Goodreads
DATASET = 'netflix'  # o 'goodreads'

if DATASET == 'netflix':
    NUM_ITEMS = 752
    train_path = 'data/train/netflix8_train.df'
    test_path = 'data/test_users/netflix8_test.json'
else:
    NUM_ITEMS = 472
    train_path = 'data/train/goodreads8_train.df'
    test_path = 'data/test_users/goodreads8_test.json'

NUM_GROUPS = 8
```

---

### **Parte 2: Decision Transformer con HuggingFace**

```python
class DecisionTransformerHF(nn.Module):
    """
    Decision Transformer usando GPT-2 de HuggingFace como backbone.
    
    Ventajas:
    - Código más simple (no implementar atención desde cero)
    - Usa componentes probados y optimizados
    - Fácil de modificar y experimentar
    
    Arquitectura:
    1. Embeddings: group, state (rating), action (item), rtg
    2. GPT-2 Transformer (de HuggingFace)
    3. Action head: predice siguiente item
    """
    
    def __init__(
        self,
        num_items=752,
        num_groups=8,
        hidden_dim=128,
        n_layers=3,
        n_heads=4,
        max_seq_len=50,
        dropout=0.1
    ):
        super().__init__()
        
        self.num_items = num_items
        self.num_groups = num_groups
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        
        # === EMBEDDINGS ===
        
        # Group embedding: representa el cluster del usuario (0-7)
        self.group_embedding = nn.Embedding(num_groups, hidden_dim)
        
        # Action embedding: representa el item (película/libro)
        # Embedding de alta dimensión para capturar similitud entre items
        self.action_embedding = nn.Embedding(num_items, hidden_dim)
        
        # State embedding: convierte rating (1-5) a vector
        # Usamos linear en vez de embedding porque ratings pueden ser continuos
        self.state_embedding = nn.Linear(1, hidden_dim)
        
        # RTG embedding: convierte return-to-go a vector
        self.rtg_embedding = nn.Linear(1, hidden_dim)
        
        # Timestep embedding: codifica posición temporal
        self.timestep_embedding = nn.Embedding(max_seq_len, hidden_dim)
        
        # === GPT-2 TRANSFORMER (HuggingFace) ===
        
        # Configuración del GPT-2
        config = GPT2Config(
            vocab_size=1,  # No usamos vocabulario de palabras
            n_positions=max_seq_len * 4,  # 4 tokens por paso (group, state, action, rtg)
            n_embd=hidden_dim,
            n_layer=n_layers,
            n_head=n_heads,
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
            use_cache=False
        )
        
        # Transformer de HuggingFace (¡esto reemplaza ~200 líneas de código!)
        self.transformer = GPT2Model(config)
        
        # === OUTPUT HEAD ===
        
        # Layer normalization final
        self.ln_f = nn.LayerNorm(hidden_dim)
        
        # Action prediction head: hidden_dim → num_items (scores)
        self.action_head = nn.Linear(hidden_dim, num_items)
        
        # Inicialización de pesos
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Inicialización de pesos (como en GPT-2)"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, states, actions, rtgs, timesteps, groups, attention_mask=None):
        """
        Forward pass del Decision Transformer.
        
        Args:
            states: (batch, seq_len, 1) - ratings de items vistos
            actions: (batch, seq_len) - IDs de items vistos
            rtgs: (batch, seq_len, 1) - return-to-go en cada paso
            timesteps: (batch, seq_len) - timestep de cada posición
            groups: (batch,) - grupo del usuario
            attention_mask: (batch, seq_len) - máscara de padding (opcional)
        
        Returns:
            action_logits: (batch, seq_len, num_items) - scores para predecir siguiente item
        """
        batch_size, seq_len = actions.shape
        
        # === CREAR EMBEDDINGS ===
        
        # Group embedding: expandir para toda la secuencia
        # (batch,) → (batch, 1, hidden_dim) → (batch, seq_len, hidden_dim)
        group_emb = self.group_embedding(groups).unsqueeze(1)
        group_emb = group_emb.expand(batch_size, seq_len, self.hidden_dim)
        
        # State embedding: ratings
        state_emb = self.state_embedding(states)  # (batch, seq_len, hidden_dim)
        
        # Action embedding: items
        action_emb = self.action_embedding(actions)  # (batch, seq_len, hidden_dim)
        
        # RTG embedding: return-to-go
        rtg_emb = self.rtg_embedding(rtgs)  # (batch, seq_len, hidden_dim)
        
        # Timestep embedding: posición temporal
        timestep_emb = self.timestep_embedding(timesteps)  # (batch, seq_len, hidden_dim)
        
        # === INTERLEAVE: [group, state, action, rtg] para cada timestep ===
        
        # Crear secuencia alternada: group_0, state_0, action_0, rtg_0, group_1, ...
        # Shape final: (batch, seq_len * 4, hidden_dim)
        stacked_inputs = torch.stack(
            [group_emb, state_emb, action_emb, rtg_emb], dim=2
        ).reshape(batch_size, seq_len * 4, self.hidden_dim)
        
        # Agregar timestep embedding (se repite 4 veces por paso)
        # (batch, seq_len, hidden_dim) → (batch, seq_len * 4, hidden_dim)
        timestep_emb_expanded = timestep_emb.repeat_interleave(4, dim=1)
        stacked_inputs = stacked_inputs + timestep_emb_expanded
        
        # === CREAR ATTENTION MASK ===
        
        if attention_mask is not None:
            # Expandir máscara para los 4 tokens por paso
            # (batch, seq_len) → (batch, seq_len * 4)
            attention_mask = attention_mask.repeat_interleave(4, dim=1)
        
        # === PASAR POR TRANSFORMER (HuggingFace hace la magia) ===
        
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=attention_mask
        )
        
        # Obtener hidden states: (batch, seq_len * 4, hidden_dim)
        hidden_states = transformer_outputs.last_hidden_state
        
        # === EXTRACT ACTION PREDICTIONS ===
        
        # Queremos predecir acciones, así que tomamos los hidden states
        # en las posiciones de "action" (cada 4 tokens, offset 2)
        # Posiciones: 2, 6, 10, 14, ... (group=0, state=1, action=2, rtg=3)
        action_hidden = hidden_states[:, 2::4, :]  # (batch, seq_len, hidden_dim)
        
        # Layer norm
        action_hidden = self.ln_f(action_hidden)
        
        # Predecir siguiente item
        action_logits = self.action_head(action_hidden)  # (batch, seq_len, num_items)
        
        return action_logits
```

---

### **Parte 3: Dataset y DataLoader**

```python
class DTDataset(Dataset):
    """
    Dataset para Decision Transformer.
    Mismo que en la implementación PyTorch pura.
    """
    
    def __init__(self, data, max_len=20):
        """
        Args:
            data: lista de diccionarios con:
                  {'states', 'actions', 'rtgs', 'timesteps', 'groups', 'attention_mask'}
            max_len: longitud máxima de secuencia (padding)
        """
        self.data = data
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        episode = self.data[idx]
        
        # Extraer datos
        states = episode['states']
        actions = episode['actions']
        rtgs = episode['rtgs']
        timesteps = episode['timesteps']
        group = episode['group']
        
        seq_len = len(actions)
        
        # Padding si es necesario
        if seq_len < self.max_len:
            pad_len = self.max_len - seq_len
            
            states = np.concatenate([states, np.zeros((pad_len, 1))])
            actions = np.concatenate([actions, np.zeros(pad_len)])
            rtgs = np.concatenate([rtgs, np.zeros((pad_len, 1))])
            timesteps = np.concatenate([timesteps, np.zeros(pad_len)])
            
            # Attention mask: 1 = válido, 0 = padding
            attention_mask = np.concatenate([
                np.ones(seq_len),
                np.zeros(pad_len)
            ])
        else:
            attention_mask = np.ones(seq_len)
        
        return {
            'states': torch.FloatTensor(states),
            'actions': torch.LongTensor(actions),
            'rtgs': torch.FloatTensor(rtgs),
            'timesteps': torch.LongTensor(timesteps),
            'groups': torch.LongTensor([group]),
            'attention_mask': torch.FloatTensor(attention_mask)
        }


def create_dt_dataset(df_train, max_len=20):
    """
    Convierte DataFrame a formato Decision Transformer.
    """
    data = []
    
    for idx, row in tqdm(df_train.iterrows(), total=len(df_train), desc="Creating dataset"):
        items = row['items']
        ratings = row['ratings']
        group = row['user_group']
        
        seq_len = len(items)
        
        # Calcular return-to-go (suma acumulada inversa)
        returns = np.array(ratings, dtype=np.float32)
        rtgs = np.cumsum(returns[::-1])[::-1]  # suma acumulada desde el final
        
        # Preparar tensores
        states = ratings.reshape(-1, 1).astype(np.float32)
        actions = np.array(items, dtype=np.int64)
        rtgs = rtgs.reshape(-1, 1).astype(np.float32)
        timesteps = np.arange(seq_len, dtype=np.int64)
        
        # Truncar si es muy largo
        if seq_len > max_len:
            states = states[-max_len:]
            actions = actions[-max_len:]
            rtgs = rtgs[-max_len:]
            timesteps = timesteps[-max_len:]
        
        data.append({
            'states': states,
            'actions': actions,
            'rtgs': rtgs,
            'timesteps': timesteps,
            'group': group
        })
    
    return data
```

---

### **Parte 4: Training Loop**

```python
def train_decision_transformer_hf(
    model,
    train_loader,
    num_epochs=10,
    learning_rate=1e-4,
    weight_decay=1e-4,
    device='cuda'
):
    """
    Entrena el Decision Transformer con HuggingFace.
    
    Args:
        model: DecisionTransformerHF
        train_loader: DataLoader con datos de entrenamiento
        num_epochs: número de épocas
        learning_rate: learning rate
        weight_decay: regularización L2
        device: 'cuda' o 'cpu'
    """
    model = model.to(device)
    model.train()
    
    # Optimizer: AdamW (recomendado para transformers)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Loss: Cross-Entropy (classification sobre items)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch in pbar:
            # Mover a device
            states = batch['states'].to(device)
            actions = batch['actions'].to(device)
            rtgs = batch['rtgs'].to(device)
            timesteps = batch['timesteps'].to(device)
            groups = batch['groups'].squeeze(1).to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass
            action_logits = model(
                states=states,
                actions=actions,
                rtgs=rtgs,
                timesteps=timesteps,
                groups=groups,
                attention_mask=attention_mask
            )  # (batch, seq_len, num_items)
            
            # Preparar targets: queremos predecir la SIGUIENTE acción
            # Predicción en t → target en t+1
            # Descartamos última predicción (no hay target)
            logits = action_logits[:, :-1, :].reshape(-1, model.num_items)
            targets = actions[:, 1:].reshape(-1)
            
            # Crear máscara para ignorar padding
            mask = attention_mask[:, 1:].reshape(-1).bool()
            
            # Calcular loss solo en posiciones válidas
            loss = criterion(logits[mask], targets[mask])
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (evitar explosión de gradientes)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Logging
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        print(f'Epoch {epoch+1} - Average Loss: {avg_loss:.4f}')
    
    return model
```

---

### **Parte 5: Evaluación (Igual que PyTorch puro)**

```python
def hit_rate_at_k(predictions, targets, k=10):
    """Hit Rate @K"""
    top_k = torch.topk(predictions, k, dim=1).indices
    hits = (top_k == targets.unsqueeze(1)).any(dim=1).float()
    return hits.mean().item()


def ndcg_at_k(predictions, targets, k=10):
    """NDCG @K"""
    top_k_indices = torch.topk(predictions, k, dim=1).indices
    relevance = (top_k_indices == targets.unsqueeze(1)).float()
    ranks = torch.arange(1, k+1, device=predictions.device).float()
    dcg = (relevance / torch.log2(ranks + 1)).sum(dim=1)
    idcg = 1.0 / np.log2(2)
    ndcg = dcg / idcg
    return ndcg.mean().item()


def mrr(predictions, targets):
    """Mean Reciprocal Rank"""
    sorted_indices = torch.argsort(predictions, dim=1, descending=True)
    ranks = (sorted_indices == targets.unsqueeze(1)).nonzero()[:, 1] + 1
    rr = 1.0 / ranks.float()
    return rr.mean().item()


@torch.no_grad()
def evaluate_cold_start(model, test_data, device, target_return=None, k_list=[5, 10, 20]):
    """
    Evalúa en cold-start users.
    
    Args:
        model: DecisionTransformerHF entrenado
        test_data: lista de usuarios de test (netflix8_test.json)
        device: 'cuda' o 'cpu'
        target_return: R̂ objetivo (default: máximo posible)
        k_list: lista de K para HR@K y NDCG@K
    """
    model.eval()
    
    metrics = {f'HR@{k}': [] for k in k_list}
    metrics.update({f'NDCG@{k}': [] for k in k_list})
    metrics['MRR'] = []
    
    for user in tqdm(test_data, desc="Evaluating cold-start"):
        group = user['group']
        items = user['items']  # todos los 752 items
        ratings = user['ratings']  # ratings del usuario para cada item
        
        # Target return: si no se especifica, usar máximo posible
        if target_return is None:
            rtg = sum(ratings)
        else:
            rtg = target_return
        
        # Simular sesión: empezar con grupo, sin historia
        for t in range(len(items)):
            # Construir historia hasta el momento
            if t == 0:
                # Primera recomendación: solo grupo + RTG
                # Usamos un placeholder para state/action (no hay historia)
                states = torch.zeros((1, 1, 1), dtype=torch.float32).to(device)
                actions = torch.zeros((1, 1), dtype=torch.long).to(device)
                rtg_input = torch.full((1, 1, 1), rtg, dtype=torch.float32).to(device)
                timesteps = torch.zeros((1, 1), dtype=torch.long).to(device)
            else:
                # Usar historia real
                history_items = [items[i] for i in range(t)]
                history_ratings = [ratings[items[i]] for i in range(t)]
                
                states = torch.FloatTensor(history_ratings).reshape(1, -1, 1).to(device)
                actions = torch.LongTensor(history_items).reshape(1, -1).to(device)
                rtg_input = torch.full((1, t, 1), rtg, dtype=torch.float32).to(device)
                timesteps = torch.arange(t, dtype=torch.long).reshape(1, -1).to(device)
            
            groups = torch.tensor([group], dtype=torch.long).to(device)
            
            # Predecir
            logits = model(states, actions, rtg_input, timesteps, groups)
            predictions = logits[0, -1, :]  # última posición
            
            # Target: item verdadero en posición t
            target_item = items[t]
            
            # Calcular métricas
            for k in k_list:
                hr = hit_rate_at_k(predictions.unsqueeze(0), 
                                   torch.tensor([target_item]).to(device), k)
                metrics[f'HR@{k}'].append(hr)
                
                ndcg = ndcg_at_k(predictions.unsqueeze(0), 
                                torch.tensor([target_item]).to(device), k)
                metrics[f'NDCG@{k}'].append(ndcg)
            
            mrr_score = mrr(predictions.unsqueeze(0), 
                           torch.tensor([target_item]).to(device))
            metrics['MRR'].append(mrr_score)
            
            # Actualizar RTG para siguiente paso
            rtg -= ratings[target_item]
    
    # Promediar métricas
    results = {k: np.mean(v) for k, v in metrics.items()}
    return results
```

---

### **Parte 6: Script Completo de Entrenamiento**

```python
def main():
    """Script completo para entrenar y evaluar"""
    
    print("="*60)
    print("DECISION TRANSFORMER - IMPLEMENTACIÓN HUGGINGFACE")
    print("="*60)
    
    # === 1. CARGAR DATOS ===
    print("\n[1/5] Cargando datos...")
    
    with open(train_path, 'rb') as f:
        df_train = pickle.load(f)
    
    import json
    with open(test_path, 'r') as f:
        test_data = json.load(f)
    
    print(f"Train users: {len(df_train)}")
    print(f"Test users: {len(test_data)}")
    print(f"Num items: {NUM_ITEMS}")
    
    # === 2. CREAR DATASET ===
    print("\n[2/5] Creando dataset...")
    
    max_len = 20  # longitud máxima de secuencia
    train_data = create_dt_dataset(df_train, max_len=max_len)
    train_dataset = DTDataset(train_data, max_len=max_len)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4
    )
    
    print(f"Dataset size: {len(train_dataset)}")
    
    # === 3. CREAR MODELO ===
    print("\n[3/5] Creando modelo...")
    
    model = DecisionTransformerHF(
        num_items=NUM_ITEMS,
        num_groups=NUM_GROUPS,
        hidden_dim=128,
        n_layers=3,
        n_heads=4,
        max_seq_len=max_len,
        dropout=0.1
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # === 4. ENTRENAR ===
    print("\n[4/5] Entrenando modelo...")
    
    model = train_decision_transformer_hf(
        model=model,
        train_loader=train_loader,
        num_epochs=10,
        learning_rate=1e-4,
        weight_decay=1e-4,
        device=device
    )
    
    # Guardar modelo
    torch.save(model.state_dict(), f'decision_transformer_hf_{DATASET}.pt')
    print(f"Modelo guardado: decision_transformer_hf_{DATASET}.pt")
    
    # === 5. EVALUAR ===
    print("\n[5/5] Evaluando en cold-start...")
    
    results = evaluate_cold_start(
        model=model,
        test_data=test_data,
        device=device,
        target_return=None,  # usar máximo posible
        k_list=[5, 10, 20]
    )
    
    print("\n" + "="*60)
    print("RESULTADOS FINALES")
    print("="*60)
    for metric, value in results.items():
        print(f"{metric:12s}: {value:.4f}")
    print("="*60)


if __name__ == '__main__':
    main()
```

---

## 📊 COMPARACIÓN: PyTorch Puro vs HuggingFace

| Aspecto | PyTorch Puro | HuggingFace |
|---------|--------------|-------------|
| **Líneas de código** | ~300 líneas | ~200 líneas |
| **Dificultad** | Alta (implementar atención) | Media (usar GPT-2) |
| **Flexibilidad** | Total control | Menos control interno |
| **Performance** | Similar | Similar |
| **Debugging** | Más difícil | Más fácil |
| **Aprendizaje** | Entiende todo | Caja negra parcial |

---

## 🎯 VENTAJAS DE HUGGINGFACE

✅ **Código más corto y legible**
✅ **Menos bugs potenciales** (componentes probados)
✅ **Optimizaciones automáticas** (flash attention, etc.)
✅ **Fácil de modificar** (cambiar GPT-2 por otros modelos)
✅ **Documentación extensa**

---

## ⚠️ DESVENTAJAS

❌ **Menos entendimiento interno** (no ves la implementación de atención)
❌ **Dependencia externa** (HuggingFace puede cambiar API)
❌ **Overhead** (biblioteca grande)
❌ **Menos personalizable** (adaptado a NLP originalmente)

---

## 🚀 ¿CUÁL USAR?

### **Usa PyTorch Puro (`03_REFERENCIA_COMPLETA.md`) si:**
- Quieres entender TODO el proceso
- Necesitas máxima flexibilidad
- Quieres aprender arquitecturas de transformers desde cero
- Tienes experiencia con PyTorch

### **Usa HuggingFace (este documento) si:**
- Eres nuevo en PyTorch
- Ya conoces HuggingFace de otros proyectos
- Quieres prototipar rápido
- Prefieres usar componentes battle-tested

---

## 💡 TIPS PARA HUGGINGFACE

### **1. Cargar modelo pre-entrenado (opcional)**

```python
# En vez de entrenar desde cero, puedes partir de GPT-2 pre-entrenado
from transformers import GPT2Model

# Cargar GPT-2 pre-entrenado de HuggingFace Hub
pretrained_gpt2 = GPT2Model.from_pretrained('gpt2')

# Adaptar config a tu problema
config = pretrained_gpt2.config
config.n_embd = 128  # ajustar dimensión

# Usar pesos pre-entrenados (transfer learning)
self.transformer = GPT2Model(config)
self.transformer.load_state_dict(pretrained_gpt2.state_dict(), strict=False)
```

### **2. Usar Trainer de HuggingFace**

```python
from transformers import Trainer, TrainingArguments

# Configurar training
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,
    per_device_train_batch_size=64,
    learning_rate=1e-4,
    logging_steps=100,
    save_steps=1000
)

# Crear Trainer (simplifica loop de training)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

# Entrenar
trainer.train()
```

### **3. Usar otros modelos de HuggingFace**

```python
# Cambiar GPT-2 por otros transformers:
from transformers import BertModel  # Bidireccional
from transformers import DistilBertModel  # Más rápido
from transformers import RobertaModel  # Mejor para clasificación

# Simplemente reemplaza GPT2Model por el que prefieras
self.transformer = BertModel(config)
```

---

## 📚 RECURSOS ADICIONALES

- [HuggingFace Transformers Docs](https://huggingface.co/docs/transformers)
- [GPT-2 Model Card](https://huggingface.co/gpt2)
- [Tutorial: Fine-tuning GPT-2](https://huggingface.co/learn/nlp-course/chapter7/6)
- [Transformers from Scratch vs HuggingFace](https://peterbloem.nl/blog/transformers)

---

## ❓ FAQ ESPECÍFICO DE HUGGINGFACE

**Q: ¿Puedo combinar PyTorch puro y HuggingFace?**
A: Sí, puedes usar HuggingFace solo para el transformer y PyTorch para embeddings/heads personalizados.

**Q: ¿El rendimiento es peor que PyTorch puro?**
A: No, HuggingFace está optimizado y puede ser incluso más rápido.

**Q: ¿Puedo usar GPT-2 pre-entrenado directamente?**
A: No directamente (está entrenado para texto), pero puedes hacer transfer learning adaptando las capas.

**Q: ¿Es válido usar esto para el TP?**
A: Sí, es una implementación completamente válida del Decision Transformer.

---

## ✅ CHECKLIST DE IMPLEMENTACIÓN

- [ ] Instalar HuggingFace Transformers
- [ ] Copiar código de `DecisionTransformerHF`
- [ ] Crear dataset con `create_dt_dataset()`
- [ ] Entrenar con `train_decision_transformer_hf()`
- [ ] Evaluar con `evaluate_cold_start()`
- [ ] Comparar con baselines
- [ ] Escribir reporte con análisis

---

**🎓 ¡Buena suerte con la implementación!**

*Este documento es una alternativa opcional. La implementación en `03_REFERENCIA_COMPLETA.md` sigue siendo válida y recomendada para aprender los detalles internos de transformers.*

