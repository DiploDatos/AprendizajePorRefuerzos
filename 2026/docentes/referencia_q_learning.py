"""Solución docente. Ejecutar desde cualquier carpeta con el Python de la actividad Q-learning."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "actividades" / "q_learning"))
from experimentos import numerical_checks, run_suite


def q_update(q, state, action, reward, next_state, terminated, truncated, alpha, gamma):
    target = reward + gamma * (not terminated) * q[next_state].max()
    return q[state, action] + alpha * (target - q[state, action])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33, 44, 55])
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--output", default="/tmp/q-learning-referencia")
    args = parser.parse_args()
    numerical_checks(q_update)
    run_suite(q_update, args.steps, args.seeds, args.episodes, args.split, args.output)
