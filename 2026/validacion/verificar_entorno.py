"""Prueba corta del entorno del curso, sin descargas de modelos ni entrenamiento del TP."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf", action="store_true", help="Probar también el extra tp-hf")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import gymnasium as gym
    import numpy as np
    import pandas as pd
    import torch
    from stable_baselines3 import PPO

    torch.set_num_threads(2)
    torch.manual_seed(2026)
    versions = {name: importlib.metadata.version(name) for name in (
        "gymnasium", "numpy", "matplotlib", "torch", "stable-baselines3",
        "pandas", "scikit-learn", "ipykernel", "jupyterlab")}
    checks = []
    for name in ("FrozenLake-v1", "Taxi-v3", "CartPole-v1"):
        env = gym.make(name, render_mode="rgb_array")
        try:
            observation, info = env.reset(seed=2026)
            env.action_space.seed(2026)
            for _ in range(8):
                observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
                if terminated or truncated:
                    observation, info = env.reset()
            frame = env.render()
            assert frame.ndim == 3 and frame.shape[2] == 3, name
        finally:
            env.close()
        checks.append(name + ": reset, step y render RGB")

    # Un update real detecta incompatibilidades SB3/Gymnasium/PyTorch.
    agent = PPO("MlpPolicy", "CartPole-v1", n_steps=32, batch_size=32,
                n_epochs=1, device="cpu", seed=2026, verbose=0)
    try:
        before = {k: v.detach().clone() for k, v in agent.policy.state_dict().items()}
        agent.learn(total_timesteps=64)
        assert agent.num_timesteps == 64
        assert any(not torch.equal(before[k], v) for k, v in agent.policy.state_dict().items())
    finally:
        agent.get_env().close()
    checks.append("PPO/SB3: entrenamiento CPU y cambio de pesos")

    data_root = ROOT / "archive/2025/tp_decision_transformer/data"
    datasets = {}
    for name, num_items in (("netflix", 752), ("goodreads", 472)):
        # Sólo leer los archivos de datos proporcionados en este repositorio.
        train = pd.read_pickle(data_root / "train" / (name + "8_train.df"))
        test = json.loads((data_root / "test_users" / (name + "8_test.json")).read_text())
        assert isinstance(train, pd.DataFrame) and not train.empty and test
        groups = {}
        for kind in ("mu", "sigma", "num"):
            frame = pd.read_csv(data_root / "groups" / (kind + "_" + name + "8.csv"), header=None)
            assert frame.shape == (8, num_items), (name, kind, frame.shape)
            groups[kind] = list(frame.shape)
        datasets[name] = {"train_shape": list(train.shape), "test_users": len(test), "groups": groups}
    checks.append("Datos del TP: pickle, JSON y CSV de Netflix/Goodreads")

    if args.hf:
        from transformers import DecisionTransformerConfig, DecisionTransformerModel, GPT2Config, GPT2Model
        versions["transformers"] = importlib.metadata.version("transformers")
        versions["accelerate"] = importlib.metadata.version("accelerate")
        # Configuración aleatoria pequeña: sin bajar pesos, cuentas ni tokens.
        model = DecisionTransformerModel(DecisionTransformerConfig(
            state_dim=4, act_dim=2, hidden_size=16, n_layer=1, n_head=2,
            max_ep_len=16, resid_pdrop=0., embd_pdrop=0., attn_pdrop=0.))
        model.eval()
        states = torch.randn(2, 4, 4)
        actions = torch.randn(2, 4, 2)
        inputs = dict(states=states, actions=actions, rewards=torch.zeros(2, 4, 1),
                      returns_to_go=torch.ones(2, 4, 1), timesteps=torch.arange(4).repeat(2, 1),
                      attention_mask=torch.ones(2, 4))
        prediction = model(**inputs).action_preds
        assert prediction.shape == actions.shape
        # Una modificación futura no puede cambiar predicciones del prefijo.
        changed = dict(inputs, states=states.clone(), actions=actions.clone())
        changed["states"][:, -1] += 100
        changed["actions"][:, -1] -= 100
        assert torch.allclose(prediction[:, :-1], model(**changed).action_preds[:, :-1], atol=1e-5)
        loss = (prediction - actions).square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        assert any(x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
                   for x in model.parameters())
        backbone = GPT2Model(GPT2Config(n_embd=16, n_layer=1, n_head=2, vocab_size=32))
        assert backbone(torch.tensor([[1, 2, 3]])).last_hidden_state.shape == (1, 3, 16)
        checks.append("Transformers: DT causal + backward y GPT2 local, sin descargas")

    subprocess.run([sys.executable, str(ROOT / "2026/validacion/verificar_material.py")],
                   cwd=ROOT, check=True)
    checks.append("Invariantes de Q-learning, REINFORCE/RLOO/GRPO y notebooks")
    report = {"python": platform.python_version(), "platform": platform.platform(),
              "machine": platform.machine(), "versions": versions,
              "checks": checks, "datasets": datasets,
              "scope": "Compatibilidad del entorno; no valida el protocolo ni entrena el TP completo"}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
