"""Líneas base del protocolo común.

Todas se ajustan sólo con usuarios de train. Los archivos `groups/*.csv` son
parámetros del generador sintético: sirven para analizar, no para construir
políticas.
"""

from __future__ import annotations

import numpy as np

from .datos import N_GROUPS, User


class RandomPolicy:
    """Política de registro: uniforme sobre los ítems disponibles."""

    def __init__(self, n_items: int, seed: int = 0):
        self.n_items = n_items
        self._rng = np.random.default_rng(seed)

    def reset(self, group: int | None, horizon: int) -> None:
        pass

    def scores(self, items_seen: np.ndarray, ratings_seen: np.ndarray) -> np.ndarray:
        return self._rng.random(self.n_items)


def _shrunk_means(users: list[User], n_items: int, shrinkage: float) -> tuple[np.ndarray, np.ndarray]:
    """Medias por ítem (global y por grupo), contraídas hacia la media del grupo.

    Devuelve (global[n_items], by_group[N_GROUPS, n_items]).
    """
    sums = np.zeros((N_GROUPS, n_items))
    counts = np.zeros((N_GROUPS, n_items))
    for u in users:
        np.add.at(sums[u.group], u.items, u.ratings)
        np.add.at(counts[u.group], u.items, 1)
    global_prior = sums.sum() / counts.sum()
    g_sum, g_cnt = sums.sum(0), counts.sum(0)
    global_means = (g_sum + shrinkage * global_prior) / (g_cnt + shrinkage)
    group_prior = (sums.sum(1) / np.maximum(counts.sum(1), 1))[:, None]
    group_means = (sums + shrinkage * group_prior) / (counts + shrinkage)
    return global_means, group_means


class PopularityPolicy:
    """Ítems ordenados por cantidad de apariciones en los logs de train.

    Con logs uniformes al azar, la cantidad de apariciones no depende de la
    calidad del ítem y esta línea base rinde como `random`. Con logs de un
    sistema anterior, refleja lo que ese sistema mostraba.
    """

    def __init__(self, train: list[User], n_items: int):
        self._counts = np.zeros(n_items)
        for u in train:
            np.add.at(self._counts, u.items, 1)

    def reset(self, group: int | None, horizon: int) -> None:
        pass

    def scores(self, items_seen: np.ndarray, ratings_seen: np.ndarray) -> np.ndarray:
        return self._counts


class ItemMeanPolicy:
    """Ítems ordenados por su valoración media en train.

    Con `by_group=True` usa la media del grupo del usuario cuando el grupo es
    visible; si no lo es, usa la media global.
    """

    def __init__(self, train: list[User], n_items: int, by_group: bool, shrinkage: float = 5.0):
        self.by_group = by_group
        self._global, self._by_group = _shrunk_means(train, n_items, shrinkage)
        self._current = self._global

    def reset(self, group: int | None, horizon: int) -> None:
        self._current = self._by_group[group] if (self.by_group and group is not None) else self._global

    def scores(self, items_seen: np.ndarray, ratings_seen: np.ndarray) -> np.ndarray:
        return self._current


class GroupPosteriorPolicy:
    """Infiere el grupo con las valoraciones observadas y recomienda en consecuencia.

    Modelo: rating ~ Normal(media[g, i], var[g, i]). Con grupo visible, la
    posterior es el grupo dado. Sin grupo, parte de un prior uniforme y se
    actualiza con cada valoración. Recomienda el ítem de mayor rating esperado
    bajo la posterior, sin explorar.
    """

    def __init__(self, train: list[User], n_items: int, shrinkage: float = 5.0, min_var: float = 0.05):
        _, self._means = _shrunk_means(train, n_items, shrinkage)
        sq = np.zeros((N_GROUPS, n_items))
        counts = np.zeros((N_GROUPS, n_items))
        for u in train:
            np.add.at(sq[u.group], u.items, (u.ratings - self._means[u.group, u.items]) ** 2)
            np.add.at(counts[u.group], u.items, 1)
        pooled = (sq.sum(1) / np.maximum(counts.sum(1), 1))[:, None]
        self._vars = np.maximum((sq + shrinkage * pooled) / (counts + shrinkage), min_var)
        self._known_group: int | None = None

    def reset(self, group: int | None, horizon: int) -> None:
        self._known_group = group

    def posterior(self, items_seen: np.ndarray, ratings_seen: np.ndarray) -> np.ndarray:
        if self._known_group is not None:
            p = np.zeros(N_GROUPS)
            p[self._known_group] = 1.0
            return p
        m = self._means[:, items_seen]
        v = self._vars[:, items_seen]
        loglik = -0.5 * (np.log(2 * np.pi * v) + (ratings_seen - m) ** 2 / v)
        logp = loglik.sum(1)
        p = np.exp(logp - logp.max())
        return p / p.sum()

    def scores(self, items_seen: np.ndarray, ratings_seen: np.ndarray) -> np.ndarray:
        return self.posterior(items_seen, ratings_seen) @ self._means
