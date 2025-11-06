"""
Configuración de Dataset para el Trabajo Práctico
==================================================

Este archivo facilita el cambio entre los datasets disponibles (Netflix o Goodreads).
Modifica la variable DATASET abajo y el resto de tu código se ajustará automáticamente.

Uso:
    from config_dataset import DATASET, NUM_ITEMS, get_paths
    
    paths = get_paths()
    print(f"Usando dataset: {DATASET}")
    print(f"Número de items: {NUM_ITEMS}")
    print(f"Train path: {paths['train']}")
"""

# ============================================
# 🎯 CONFIGURACIÓN: MODIFICA ESTA LÍNEA
# ============================================

DATASET = 'netflix'    # Opciones: 'netflix' o 'goodreads'

# ============================================
# Configuración automática (NO MODIFICAR)
# ============================================

# Configuración de items por dataset
DATASET_CONFIG = {
    'netflix': {
        'num_items': 752,
        'domain': 'películas',
        'emoji': '🎬'
    },
    'goodreads': {
        'num_items': 472,
        'domain': 'libros',
        'emoji': '📚'
    }
}

# Validar que el dataset elegido existe
if DATASET not in DATASET_CONFIG:
    raise ValueError(
        f"Dataset '{DATASET}' no válido. "
        f"Opciones: {list(DATASET_CONFIG.keys())}"
    )

# Exportar configuración
NUM_ITEMS = DATASET_CONFIG[DATASET]['num_items']
DOMAIN = DATASET_CONFIG[DATASET]['domain']
EMOJI = DATASET_CONFIG[DATASET]['emoji']

def get_paths():
    """
    Retorna los paths de los archivos del dataset seleccionado.
    
    Returns:
        dict con keys: 'train', 'test', 'mu', 'sigma', 'num'
    """
    return {
        'train': f'data/train/{DATASET}8_train.df',
        'test': f'data/test_users/{DATASET}8_test.json',
        'mu': f'data/groups/mu_{DATASET}8.csv',
        'sigma': f'data/groups/sigma_{DATASET}8.csv',
        'num': f'data/groups/num_{DATASET}8.csv'
    }

def print_config():
    """Imprime la configuración actual del dataset."""
    print("=" * 70)
    print(f"{EMOJI} CONFIGURACIÓN DE DATASET")
    print("=" * 70)
    print(f"Dataset seleccionado: {DATASET.upper()}")
    print(f"Dominio: {DOMAIN}")
    print(f"Número de items: {NUM_ITEMS}")
    print("\nPaths configurados:")
    for key, path in get_paths().items():
        print(f"  {key:10s}: {path}")
    print("=" * 70)

# Imprimir configuración al importar (útil para debugging)
if __name__ == "__main__":
    print_config()
    
    # Ejemplo de uso
    print("\n✅ Ejemplo de uso:")
    print("""
    from config_dataset import DATASET, NUM_ITEMS, get_paths
    
    # En tu código:
    paths = get_paths()
    df_train = pd.read_pickle(paths['train'])
    
    model = DecisionTransformer(
        num_items=NUM_ITEMS,  # Se ajusta automáticamente
        ...
    )
    """)

