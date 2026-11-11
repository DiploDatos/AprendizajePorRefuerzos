"""Ejecución del protocolo común de evaluación.

Uso, desde `2026/tp`:

    python -m protocolo.evaluar --dataset netflix --warmup 0 5 --output outputs/baselines

Evalúa las líneas base sobre los usuarios de test y escribe una fila por
sesión (`sesiones.csv`) y los resúmenes por condición y por grupo.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from .baselines import GroupPosteriorPolicy, ItemMeanPolicy, PopularityPolicy, RandomPolicy
from .datos import RELEVANCE_THRESHOLD, User, load_test, load_train, num_items, split_by_user
from .logs import LOG_CHOICES, PRESETS, PreviousSystem, logged_episodes, logs_cache
from .metricas import normalized_score, ranking_metrics, session_bounds, summarize
from .simulador import Policy, Session, SessionConfig, choose


def run_session(
    policy: Policy, user: User, n_items: int, config: SessionConfig, seed: int, user_index: int
) -> dict[str, float]:
    rng = np.random.default_rng([seed, user_index])
    session = Session(user, n_items, config, rng)
    policy.reset(session.visible_group, config.horizon)

    steps = config.controlled_steps
    random_expected, oracle = session_bounds(session._available_hidden_ratings(), steps)

    controlled: list[float] = []
    first_ranking: dict[str, float] = {}
    while not session.done:
        items_seen, ratings_seen = session.observed()
        scores = np.asarray(policy.scores(items_seen, ratings_seen), dtype=np.float64)
        mask = session.available_mask()
        if not first_ranking:
            first_ranking = ranking_metrics(scores, session._hidden_ratings(), mask, k=10)
        controlled.append(session.step(choose(scores, mask, rng)))

    ret = float(np.sum(controlled))
    return {
        "user_id": user.user_id,
        "group": user.group,
        "seed": seed,
        "return": ret,
        "full_return": float(np.sum(session.ratings)),
        "mean_rating": ret / steps,
        "hit_rate": float(np.mean(np.array(controlled) >= RELEVANCE_THRESHOLD)),
        "normalized": normalized_score(ret, random_expected, oracle),
        "random_expected": random_expected,
        "oracle": oracle,
        **first_ranking,
    }


def evaluate(
    policy: Policy,
    users: list[User],
    n_items: int,
    config: SessionConfig,
    seeds: Iterable[int] = (0, 1, 2),
) -> pd.DataFrame:
    rows = [
        run_session(policy, user, n_items, config, seed, index)
        for seed in seeds
        for index, user in enumerate(users)
    ]
    return pd.DataFrame(rows)


def baseline_factories(train: list[User], n_items: int) -> dict[str, Callable[[], Policy]]:
    return {
        "random": lambda: RandomPolicy(n_items, seed=0),
        "popularity": lambda: PopularityPolicy(train, n_items),
        "item_mean": lambda: ItemMeanPolicy(train, n_items, by_group=False),
        "group_mean": lambda: ItemMeanPolicy(train, n_items, by_group=True),
        "group_posterior": lambda: GroupPosteriorPolicy(train, n_items),
    }


def evaluate_grid(
    factories: dict[str, Callable[[], Policy]],
    users: list[User],
    n_items: int,
    horizon: int,
    warmups: Iterable[int],
    group_visibility: Iterable[bool],
    seeds: Iterable[int],
) -> pd.DataFrame:
    frames = []
    for name, make in factories.items():
        for warmup in warmups:
            for visible in group_visibility:
                config = SessionConfig(horizon=horizon, warmup=warmup, group_visible=visible)
                df = evaluate(make(), users, n_items, config, seeds)
                frames.append(df.assign(policy=name, warmup=warmup, group_visible=visible))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["netflix", "goodreads"], default="netflix")
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--warmup", type=int, nargs="+", default=[0, 5])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--split", choices=["test", "val"], default="test",
                        help="val: usuarios de validación con su catálogo observado")
    parser.add_argument("--logs", choices=LOG_CHOICES, default="random",
                        help="Logs con los que se ajustan las líneas base (ver logs.py)")
    parser.add_argument("--output", type=Path, default=Path("outputs/baselines"))
    args = parser.parse_args()

    n_items = num_items(args.dataset)
    train, val = split_by_user(load_train(args.dataset))
    users = load_test(args.dataset) if args.split == "test" else val
    factories = baseline_factories(train, n_items)
    if args.logs != "random":
        logged = logged_episodes(args.logs, train, train, n_items, args.horizon, seed=0,
                                 cache=logs_cache(args.dataset, args.logs, "train"))
        factories = baseline_factories(logged.logged_users(), n_items)
        _, temperature = PRESETS[args.logs]
        factories["sistema_anterior_eps0"] = lambda: PreviousSystem.fit(
            train, n_items, 0.0, temperature, np.random.default_rng(0))
    results = evaluate_grid(factories, users, n_items, args.horizon, args.warmup, (True, False), args.seeds)
    write_reports(results, args.output)


def write_reports(results: pd.DataFrame, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output / "sesiones.csv", index=False)
    conditions = ["policy", "warmup", "group_visible"]
    overall = summarize(results, by=conditions)
    overall.to_csv(output / "resumen.csv")
    summarize(results, by=[*conditions, "group"]).to_csv(output / "resumen_por_grupo.csv")
    cols = [c for c in ["return", "normalized", "hit_rate", "ndcg@10"] if c in overall.columns.get_level_values(0)]
    with pd.option_context("display.width", 200, "display.precision", 3):
        print(overall[cols].xs("mean", axis=1, level=1))


if __name__ == "__main__":
    main()
