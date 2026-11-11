"""Simulador de sesiones de recomendación.

Una sesión dura `horizon` pasos. En cada paso se muestra un ítem que el usuario
todavía no vio y se revela su valoración, que es la recompensa. Los primeros
`warmup` ítems los elige la política de registro (uniforme al azar, como en los
datos de train) y no cuentan para las métricas de la política evaluada.

La política sólo recibe copias de lo ya observado. Nunca accede a las
valoraciones ocultas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .datos import User


@dataclass(frozen=True)
class SessionConfig:
    horizon: int = 20
    warmup: int = 0
    group_visible: bool = True

    def __post_init__(self) -> None:
        if not 0 <= self.warmup < self.horizon:
            raise ValueError("Se requiere 0 <= warmup < horizon")

    @property
    def controlled_steps(self) -> int:
        return self.horizon - self.warmup


class Policy(Protocol):
    """Interfaz común para baselines y modelos.

    `reset` se llama al iniciar cada sesión. `scores` devuelve un puntaje por
    ítem del catálogo (mayor es mejor). El protocolo enmascara los ítems ya
    mostrados y los que no tienen valoración conocida, y elige el máximo.
    """

    def reset(self, group: int | None, horizon: int) -> None: ...

    def scores(self, items_seen: np.ndarray, ratings_seen: np.ndarray) -> np.ndarray: ...


class Session:
    def __init__(self, user: User, n_items: int, config: SessionConfig, rng: np.random.Generator):
        self._hidden = np.full(n_items, np.nan)
        self._hidden[user.items] = user.ratings
        self._known = ~np.isnan(self._hidden)
        if self._known.sum() < config.horizon:
            raise ValueError(f"El usuario {user.user_id} tiene menos de {config.horizon} ítems valorados")
        self.config = config
        self.group = user.group
        self._shown = np.zeros(n_items, dtype=bool)
        self.items: list[int] = []
        self.ratings: list[float] = []
        warmup_items = rng.choice(np.flatnonzero(self._known), size=config.warmup, replace=False)
        for item in warmup_items:
            self._show(int(item))

    @property
    def done(self) -> bool:
        return len(self.items) >= self.config.horizon

    @property
    def visible_group(self) -> int | None:
        return self.group if self.config.group_visible else None

    def available_mask(self) -> np.ndarray:
        return self._known & ~self._shown

    def observed(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array(self.items, dtype=np.int64), np.array(self.ratings, dtype=np.float64)

    def step(self, item: int) -> float:
        if self.done:
            raise RuntimeError("La sesión terminó")
        if not self.available_mask()[item]:
            raise ValueError(f"El ítem {item} ya se mostró o no tiene valoración conocida")
        return self._show(item)

    def _show(self, item: int) -> float:
        rating = float(self._hidden[item])
        self._shown[item] = True
        self.items.append(item)
        self.ratings.append(rating)
        return rating

    # Sólo para métricas: no exponer a la política.
    def _available_hidden_ratings(self) -> np.ndarray:
        return self._hidden[self.available_mask()]

    def _hidden_ratings(self) -> np.ndarray:
        return self._hidden


def choose(scores: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> int:
    """Ítem disponible de mayor puntaje; desempata al azar."""
    if scores.shape != mask.shape:
        raise ValueError(f"scores tiene forma {scores.shape}; se esperaba {mask.shape}")
    masked = np.where(mask, scores, -np.inf)
    best = masked.max()
    if not np.isfinite(best):
        raise ValueError("La política devolvió puntajes no finitos para todos los ítems disponibles")
    return int(rng.choice(np.flatnonzero(masked == best)))
