"""Métricas de sesión y de ranking."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .datos import RELEVANCE_THRESHOLD


def session_bounds(available_ratings: np.ndarray, steps: int) -> tuple[float, float]:
    """Retorno esperado de la política aleatoria y retorno del oráculo.

    Ambos se calculan sobre los ítems disponibles al empezar la parte
    controlada de la sesión, con `steps` recomendaciones sin repetición.
    """
    random_expected = float(available_ratings.mean() * steps)
    oracle = float(np.sort(available_ratings)[-steps:].sum())
    return random_expected, oracle


def normalized_score(ret: float, random_expected: float, oracle: float) -> float:
    """0 = como la política aleatoria, 1 = como el oráculo."""
    gap = oracle - random_expected
    return float((ret - random_expected) / gap) if gap > 0 else float("nan")


def ranking_metrics(
    scores: np.ndarray, hidden: np.ndarray, mask: np.ndarray, k: int = 10
) -> dict[str, float]:
    """NDCG@k y precisión@k con relevancia binaria (rating >= umbral).

    Rankea sólo los ítems de `mask`. Es una evaluación de ranking estático: no
    es el retorno de una sesión.
    """
    candidates = np.flatnonzero(mask)
    order = candidates[np.argsort(-scores[candidates], kind="stable")][:k]
    relevant = hidden[order] >= RELEVANCE_THRESHOLD
    discounts = 1.0 / np.log2(np.arange(2, len(order) + 2))
    dcg = float((relevant * discounts).sum())
    n_rel = int((hidden[candidates] >= RELEVANCE_THRESHOLD).sum())
    idcg = float(discounts[: min(k, n_rel)].sum())
    return {
        f"ndcg@{k}": dcg / idcg if idcg > 0 else float("nan"),
        f"precision@{k}": float(relevant.mean()),
    }


def summarize(results: pd.DataFrame, by: str | list[str] | None = None) -> pd.DataFrame:
    """Media e intervalo de 95 % por usuario.

    Promedia primero las semillas de cada usuario, para que el tamaño de
    muestra sea la cantidad de usuarios y no de sesiones.
    """
    keys = [] if by is None else ([by] if isinstance(by, str) else list(by))
    metric_cols = [
        c for c in results.columns
        if c not in {"user_id", "group", "seed", *keys} and pd.api.types.is_numeric_dtype(results[c])
    ]
    per_user = results.groupby(["user_id", *keys], as_index=False)[metric_cols].mean()
    grouped = per_user.groupby(keys) if keys else per_user.assign(_all=0).groupby("_all")
    mean = grouped[metric_cols].mean()
    half = 1.96 * grouped[metric_cols].std(ddof=1) / np.sqrt(grouped[metric_cols].count())
    out = pd.concat({"mean": mean, "ci95": half}, axis=1).swaplevel(axis=1).sort_index(axis=1)
    out.insert(0, ("n_users", ""), grouped.size())
    return out
