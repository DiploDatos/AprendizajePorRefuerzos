"""Logs de un sistema de recomendación anterior.

Los datos de train vienen de una política de registro uniforme. En un sistema
real, los logs los produce el recomendador que estaba en producción, que
muestra más los ítems que cree buenos. Este módulo simula esos logs sin
inventar valoraciones: corre el sistema anterior en el simulador sobre cada
usuario de train, con el catálogo restringido a los ítems que ese usuario
valoró. Es un dataset de entrenamiento derivado, no un cambio en la
evaluación: los modelos se siguen evaluando en test con el protocolo común.

La idea viene del replay de Li et al. (WSDM 2011): con logs uniformes, los
ítems valorados de cada usuario son un subconjunto al azar del catálogo y su
valoración no condiciona qué ítems hay. Pero no es equivalente a correr el
sistema anterior con el catálogo completo: elige entre 25 y 200 ítems, no
entre todos. Por eso sus logs son algo peores que los que produciría en
producción, y casi aleatorios para usuarios con pocos ítems valorados.

Sistema anterior, por sesión:
- con probabilidad ε elige uniforme entre los ítems disponibles;
- si no, muestrea con softmax(media del ítem en el grupo / τ).

Usa la media del grupo, que un sistema en producción conoce. La media de cada
usuario se calcula sin sus propias valoraciones (leave-one-out), para que el
sistema no elija según el rating que está por revelar.

Cada sesión toma un ε de `epsilons` al azar. Con varios valores se obtiene un
dataset de calidad mixta, como los "medium-replay" de D4RL.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .datos import N_GROUPS, Episodes, User, make_episodes
from .simulador import Session, SessionConfig, choose

# nombre -> (valores de ε por sesión, temperatura τ)
PRESETS: dict[str, tuple[tuple[float, ...], float]] = {
    "mixto": ((0.0, 0.25, 0.5, 1.0), 0.25),
    "experto": ((0.1,), 0.1),
}
LOG_CHOICES = ("random", *PRESETS)


def logs_cache(dataset: str, logs: str, split: str) -> Path | None:
    """Archivo de caché por defecto, relativo a `2026/tp`."""
    return None if logs == "random" else Path("outputs/logs") / f"{dataset}_{logs}_{split}.npz"


def _group_sums(users: list[User], n_items: int) -> tuple[np.ndarray, np.ndarray]:
    sums = np.zeros((N_GROUPS, n_items))
    counts = np.zeros((N_GROUPS, n_items))
    for u in users:
        np.add.at(sums[u.group], u.items, u.ratings)
        np.add.at(counts[u.group], u.items, 1)
    return sums, counts


def _shrunk(sums: np.ndarray, counts: np.ndarray, shrinkage: float) -> np.ndarray:
    prior = sums.sum() / counts.sum()
    return (sums + shrinkage * prior) / (counts + shrinkage)


class PreviousSystem:
    """Política estocástica del sistema anterior (ε-uniforme + softmax).

    El muestreo softmax se implementa con el truco de Gumbel-max, para usar el
    mismo `choose` (argmax enmascarado) que el resto del protocolo.
    """

    def __init__(self, group_means: np.ndarray, global_means: np.ndarray, epsilon: float,
                 temperature: float, rng: np.random.Generator):
        self._group_means = group_means
        self._global_means = global_means
        self.epsilon = epsilon
        self.temperature = temperature
        self._rng = rng
        self._current = global_means

    def reset(self, group: int | None, horizon: int) -> None:
        self._current = self._group_means[group] if group is not None else self._global_means

    def scores(self, items_seen: np.ndarray, ratings_seen: np.ndarray) -> np.ndarray:
        n = self._current.shape[0]
        if self._rng.random() < self.epsilon:
            return self._rng.random(n)
        return self._current / self.temperature + self._rng.gumbel(size=n)

    @classmethod
    def fit(cls, users: list[User], n_items: int, epsilon: float, temperature: float,
            rng: np.random.Generator, shrinkage: float = 5.0) -> "PreviousSystem":
        sums, counts = _group_sums(users, n_items)
        group_means = _shrunk(sums, counts, shrinkage)
        global_means = _shrunk(sums.sum(0, keepdims=True), counts.sum(0, keepdims=True), shrinkage)[0]
        return cls(group_means, global_means, epsilon, temperature, rng)


@dataclass(frozen=True)
class LoggedData:
    episodes: Episodes
    epsilons: np.ndarray  # (N,) ε de la sesión que generó cada episodio

    def logged_users(self) -> list[User]:
        """Episodios como usuarios, para ajustar baselines con estos logs.

        Un mismo usuario puede aparecer en varias sesiones, y un ítem puede
        repetirse entre sesiones. Cada sesión cuenta como un usuario.
        """
        e = self.episodes
        return [User(int(e.user_ids[i]), int(e.groups[i]), e.items[i], e.ratings[i])
                for i in range(len(e.groups))]


def generate_logs(
    users: list[User],
    reference_users: list[User],
    n_items: int,
    epsilons: tuple[float, ...],
    temperature: float = 0.25,
    horizon: int = 20,
    seed: int = 0,
    shrinkage: float = 5.0,
) -> LoggedData:
    """Genera `len(user.items) // horizon` sesiones del sistema anterior por usuario.

    `reference_users` son los usuarios con los que se ajustó el sistema
    anterior. Si un usuario de `users` está entre ellos, sus valoraciones se
    excluyen de las medias que se usan para recomendarle.
    """
    sums, counts = _group_sums(reference_users, n_items)
    reference_ids = {u.user_id for u in reference_users}
    rng = np.random.default_rng(seed)
    config = SessionConfig(horizon=horizon, warmup=0, group_visible=True)

    items, ratings, groups, user_ids, eps_out = [], [], [], [], []
    for user in users:
        s, c = sums.copy(), counts.copy()
        if user.user_id in reference_ids:
            s[user.group, user.items] -= user.ratings
            c[user.group, user.items] -= 1
        group_means = _shrunk(s, c, shrinkage)
        global_means = _shrunk(s.sum(0, keepdims=True), c.sum(0, keepdims=True), shrinkage)[0]
        for _ in range(len(user.items) // horizon):
            eps = float(epsilons[rng.integers(len(epsilons))])
            policy = PreviousSystem(group_means, global_means, eps, temperature, rng)
            session = Session(user, n_items, config, rng)
            policy.reset(session.visible_group, horizon)
            while not session.done:
                items_seen, ratings_seen = session.observed()
                session.step(choose(policy.scores(items_seen, ratings_seen), session.available_mask(), rng))
            seen_items, seen_ratings = session.observed()
            items.append(seen_items)
            ratings.append(seen_ratings)
            groups.append(user.group)
            user_ids.append(user.user_id)
            eps_out.append(eps)

    episodes = Episodes(
        items=np.asarray(items, dtype=np.int64).reshape(-1, horizon),
        ratings=np.asarray(ratings, dtype=np.float64).reshape(-1, horizon),
        groups=np.asarray(groups, dtype=np.int64),
        user_ids=np.asarray(user_ids, dtype=np.int64),
    )
    return LoggedData(episodes, np.asarray(eps_out))


def logged_episodes(
    logs: str,
    users: list[User],
    reference_users: list[User],
    n_items: int,
    horizon: int = 20,
    seed: int = 0,
    cache: Path | None = None,
) -> LoggedData:
    """Episodios de `users` bajo los logs `logs` ("random" o un preset).

    "random" son los logs originales cortados en tramos consecutivos. Si se da
    `cache`, guarda o reutiliza los episodios en ese archivo `.npz`.
    """
    if logs == "random":
        ep = make_episodes(users, horizon)
        return LoggedData(ep, np.ones(len(ep.groups)))
    if cache is not None and cache.exists():
        z = np.load(cache)
        return LoggedData(Episodes(z["items"], z["ratings"], z["groups"], z["user_ids"]), z["epsilons"])
    epsilons, temperature = PRESETS[logs]
    data = generate_logs(users, reference_users, n_items, epsilons, temperature, horizon, seed)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        e = data.episodes
        np.savez_compressed(cache, items=e.items, ratings=e.ratings, groups=e.groups,
                            user_ids=e.user_ids, epsilons=data.epsilons)
    return data
