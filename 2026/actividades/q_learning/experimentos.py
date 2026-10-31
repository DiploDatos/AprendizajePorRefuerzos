"""Infraestructura de la actividad formativa de Q-learning. La actualización se inyecta desde el notebook."""
import json
import platform
from pathlib import Path

import gymnasium as gym
import numpy as np


def greedy(q_row, rng):
    """Desempate aleatorio: evita favorecer por accidente a la acción cero."""
    return int(rng.choice(np.flatnonzero(q_row == q_row.max())))


def make_env(env_id, slippery=False):
    if env_id == "FrozenLake-v1":
        return gym.make(env_id, is_slippery=slippery, max_episode_steps=100)
    if env_id == "Taxi-v3":
        return gym.make(env_id, max_episode_steps=200)
    raise ValueError("Esta actividad fija FrozenLake-v1 y Taxi-v3.")


def train(update, env_id, seed, steps=100000, slippery=False, bonus=0.0,
          alpha=0.3, gamma=0.95, min_epsilon=0.1):
    """Presupuesto exacto en transiciones, independiente del largo del episodio."""
    env = make_env(env_id, slippery)
    rng = np.random.default_rng(seed)
    q = np.zeros((env.observation_space.n, env.action_space.n))
    state, _ = env.reset(seed=seed)
    history, original, modified, length = [], 0.0, 0.0, 0
    for t in range(steps):
        epsilon = min_epsilon + (1 - min_epsilon) * np.exp(-6 * t / steps)
        action = int(rng.integers(env.action_space.n)) if rng.random() < epsilon else greedy(q[state], rng)
        next_state, reward, terminated, truncated, _ = env.step(action)
        shaped = reward + (bonus if not terminated else 0.0)
        q[state, action] = update(q, state, action, shaped, next_state,
                                  terminated, truncated, alpha, gamma)
        original += reward
        modified += shaped
        length += 1
        state = next_state
        if terminated or truncated or t == steps - 1:
            history.append({"steps": t + 1, "length": length,
                            "original_return": float(original),
                            "modified_return": float(modified),
                            "terminated": bool(terminated), "truncated": bool(truncated),
                            "budget_cut": bool(not (terminated or truncated))})
            state, _ = env.reset()  # Continuar el RNG del entorno; no resembrar cada episodio.
            original, modified, length = 0.0, 0.0, 0
    env.close()
    return q, history


def evaluate(env_id, seed, q=None, episodes=100, slippery=False, split="validation"):
    if split not in ("validation", "test"):
        raise ValueError("split debe ser validation o test")
    # Reset seeds reservadas, distintas entre particiones y de las usadas en train.
    offset = 1000000 if split == "validation" else 2000000
    rng = np.random.default_rng(offset + seed)
    env = make_env(env_id, slippery)
    records = []
    for episode in range(episodes):
        state, _ = env.reset(seed=offset + episode)
        total, length, success = 0.0, 0, False
        while True:
            action = int(rng.integers(env.action_space.n)) if q is None else greedy(q[state], rng)
            state, reward, terminated, truncated, _ = env.step(action)
            total += reward
            length += 1
            success = bool(terminated and reward > 0)
            if terminated or truncated:
                break
        records.append({"return": float(total), "length": length,
                        "success": success, "truncated": bool(truncated),
                        "reset_seed": offset + episode})
    env.close()
    return {"return": float(np.mean([r["return"] for r in records])),
            "success": float(np.mean([r["success"] for r in records])),
            "length": float(np.mean([r["length"] for r in records])),
            "truncated": float(np.mean([r["truncated"] for r in records])),
            "episodes": records}


def run_suite(update, steps=100000, seeds=(11, 22, 33, 44, 55), episodes=100,
              split="validation", output=None):
    scenarios = [("lake_determinista", "FrozenLake-v1", False, 0.0),
                 ("lake_resbaladizo", "FrozenLake-v1", True, 0.0),
                 ("taxi", "Taxi-v3", False, 0.0),
                 ("lake_bonus", "FrozenLake-v1", True, 0.05)]
    result = {"versions": {"python": platform.python_version(), "numpy": np.__version__,
                           "gymnasium": gym.__version__},
              "config": {"steps": steps, "seeds": list(seeds), "eval_episodes": episodes,
                         "split": split, "alpha": 0.3, "gamma": 0.95,
                         "epsilon": "0.1 + 0.9 * exp(-6*t/steps)"}, "scenarios": {}}
    for name, env_id, slippery, bonus in scenarios:
        runs = []
        for seed in seeds:
            q, history = train(update, env_id, seed, steps, slippery, bonus)
            score = evaluate(env_id, seed, q, episodes, slippery, split)
            random_score = evaluate(env_id, seed, None, episodes, slippery, split)
            runs.append({"seed": seed, "trained": score, "random": random_score,
                         "train_history": history})
            if output:
                Path(output).mkdir(parents=True, exist_ok=True)
                np.savez(Path(output) / (name + "_" + str(seed) + ".npz"), q=q)
        summary = {}
        for policy in ("trained", "random"):
            summary[policy] = {metric: {"mean": float(np.mean([r[policy][metric] for r in runs])),
                                       "seed_std": float(np.std([r[policy][metric] for r in runs], ddof=1))
                                       if len(runs) > 1 else None}
                               for metric in ("return", "success", "length", "truncated")}
        result["scenarios"][name] = {"env": env_id, "slippery": slippery,
                                     "training_bonus": bonus, "runs": runs, "summary": summary}
        print(name, json.dumps(summary, ensure_ascii=False))
    if output:
        (Path(output) / "resultados.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def numerical_checks(update):
    q = np.array([[0., 0., 0., 0.], [8., 1., 0., 0.]])
    assert np.isclose(update(q, 0, 0, 0., 1, False, False, 1., .9), 7.2), "max != argmax"
    q[1, 0] = 10.
    assert np.isclose(update(q, 0, 0, 1., 1, True, False, 1., .9), 1.), "No bootstrap terminal"
    assert np.isclose(update(q, 0, 0, 1., 1, False, True, 1., .9), 10.), "Bootstrap al truncar"
    print("Tres contraejemplos numéricos verificados.")


def trace_policy(q, seed=2000000, slippery=True):
    """Rollout FrozenLake sin aprendizaje ni shaping, útil para interpretar loops."""
    env = make_env("FrozenLake-v1", slippery)
    state, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    trace = []
    while True:
        action = greedy(q[state], rng)
        next_state, reward, terminated, truncated, _ = env.step(action)
        trace.append({"state": int(state), "action": action, "next_state": int(next_state),
                      "reward": float(reward), "terminated": bool(terminated), "truncated": bool(truncated)})
        state = next_state
        if terminated or truncated:
            break
    env.close()
    return trace
