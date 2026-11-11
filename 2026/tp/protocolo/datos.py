"""Carga de datos, particiones y episodios del TP.

Los archivos de train contienen, por usuario, los ítems que valoró en el orden
registrado. Los de test contienen la valoración de *todos* los ítems del
catálogo; su orden (0, 1, 2, ...) no es un orden de consumo.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config_dataset import DATA_ROOT, DATASET_CONFIG  # noqa: E402

N_GROUPS = 8
RELEVANCE_THRESHOLD = 4.0


@dataclass(frozen=True)
class User:
    """Usuario con las valoraciones conocidas por el protocolo.

    `items` y `ratings` están alineados. En train son los ítems registrados,
    en su orden. En test son todos los ítems del catálogo.
    """

    user_id: int
    group: int
    items: np.ndarray
    ratings: np.ndarray


def num_items(dataset: str) -> int:
    return DATASET_CONFIG[dataset]["num_items"]


def load_train(dataset: str, data_root: Path = DATA_ROOT) -> list[User]:
    df = pd.read_pickle(data_root / "train" / f"{dataset}8_train.df")
    return [
        User(
            user_id=int(row.user_id),
            group=int(row.user_group),
            items=np.asarray(row.items, dtype=np.int64),
            ratings=np.asarray(row.ratings, dtype=np.float64),
        )
        for row in df.itertuples(index=False)
    ]


def load_test(dataset: str, data_root: Path = DATA_ROOT) -> list[User]:
    path = data_root / "test_users" / f"{dataset}8_test.json"
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    return [
        User(
            user_id=index,
            group=int(entry["group"]),
            items=np.asarray(entry["items"], dtype=np.int64),
            ratings=np.asarray(entry["ratings"], dtype=np.float64),
        )
        for index, entry in enumerate(raw)
    ]


def split_by_user(
    users: list[User], val_fraction: float = 0.1, seed: int = 0
) -> tuple[list[User], list[User]]:
    """Separa usuarios de train y validación, estratificando por grupo.

    La partición es por usuario: ninguna valoración de validación llega a
    entrenamiento.
    """
    rng = np.random.default_rng(seed)
    train, val = [], []
    for group in sorted({u.group for u in users}):
        members = [u for u in users if u.group == group]
        order = rng.permutation(len(members))
        n_val = int(round(val_fraction * len(members)))
        val_idx = set(order[:n_val].tolist())
        for i, user in enumerate(members):
            (val if i in val_idx else train).append(user)
    return train, val


@dataclass(frozen=True)
class Episodes:
    """Episodios de longitud fija, cortados de las trayectorias registradas."""

    items: np.ndarray  # (N, L) int64
    ratings: np.ndarray  # (N, L) float64
    groups: np.ndarray  # (N,) int64
    user_ids: np.ndarray  # (N,) int64

    def returns_to_go(self) -> np.ndarray:
        """R̂_t = suma de ratings desde t hasta el final del episodio."""
        return np.flip(np.cumsum(np.flip(self.ratings, axis=1), axis=1), axis=1)


def make_episodes(users: list[User], length: int) -> Episodes:
    """Corta cada trayectoria en tramos consecutivos de `length` pasos.

    Descarta el resto final de cada usuario. Ningún episodio mezcla usuarios.
    """
    items, ratings, groups, user_ids = [], [], [], []
    for user in users:
        n_chunks = len(user.items) // length
        for c in range(n_chunks):
            sl = slice(c * length, (c + 1) * length)
            items.append(user.items[sl])
            ratings.append(user.ratings[sl])
            groups.append(user.group)
            user_ids.append(user.user_id)
    return Episodes(
        items=np.asarray(items, dtype=np.int64).reshape(-1, length),
        ratings=np.asarray(ratings, dtype=np.float64).reshape(-1, length),
        groups=np.asarray(groups, dtype=np.int64),
        user_ids=np.asarray(user_ids, dtype=np.int64),
    )
