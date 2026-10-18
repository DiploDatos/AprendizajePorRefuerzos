"""Checks de invariantes matemáticos, artefactos y protocolo; no entrena de nuevo."""
import ast
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "actividades" / "q_learning"))
sys.path.insert(0, str(ROOT / "clase2"))
sys.path.insert(0, str(ROOT / "docentes"))
import torch
import numpy as np
import mini_rlhf as demo
import llm_policy_gradients as pg
from referencia_q_learning import q_update
from experimentos import numerical_checks, train, evaluate, trace_policy


def main():
    numerical_checks(q_update)
    # El presupuesto es en pasos, incluso si corta un episodio; evaluación no modifica Q.
    q, history = train(q_update, "FrozenLake-v1", 11, steps=127)
    assert history[-1]["steps"] == 127
    assert sum(h["length"] for h in history) == 127
    snapshot = q.copy()
    validation = evaluate("FrozenLake-v1", 11, q, episodes=4, split="validation")
    test = evaluate("FrozenLake-v1", 11, q, episodes=4, split="test")
    assert np.array_equal(q, snapshot)
    assert set(r["reset_seed"] for r in validation["episodes"]).isdisjoint(
        set(r["reset_seed"] for r in test["episodes"]))
    assert len(trace_policy(q)) <= 100
    # Pares A/B balanceados; no aceptar feedback simulado como archivo humano.
    pairs = demo.simulated_pairs()
    assert sum(p["choice"] == "a" for p in pairs) == sum(p["choice"] == "b" for p in pairs)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "prefs.json"
        path.write_text(json.dumps(pairs))
        try:
            demo.load_preferences(path)
        except ValueError:
            pass
        else:
            raise AssertionError("Aceptó datos simulados como humanos")
        for p in pairs:
            p["source"] = "human"  # Solo verifica formato, no constituye anotación humana real.
        path.write_text(json.dumps(pairs))
        assert demo.load_preferences(path)[1] == "HUMAN"
        pairs[0]["a"] = "texto modificado"
        path.write_text(json.dumps(pairs))
        try:
            demo.load_preferences(path)
        except ValueError:
            pass
        else:
            raise AssertionError("Aceptó modificación de pares")
    # KL = 0 entre modelos iguales; se excluyen controles y tokens después de EOS.
    demo.seed_all(19)
    model = demo.TinyLM()
    context = torch.tensor([demo.TOKENS["<saludo>"]] * 24)
    labels, _, kl, _ = demo.sample(model, model, context)
    assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-5)
    for row in labels.tolist():
        if demo.EOS in row:
            assert all(t == demo.PAD for t in row[row.index(demo.EOS) + 1:])
        assert all(t in (demo.EOS, demo.PAD) or t >= len(demo.SPECIAL) for t in row)
    # Teacher forcing debe puntuar la distribución exacta usada al generar.
    with torch.no_grad():
        lp, _, mask = pg.token_statistics(model, context, labels)
        demo.seed_all(19)
        sampled, sample_lp, _, _ = demo.sample(model, model, context)
        scored_lp, _, _ = pg.token_statistics(model, context, sampled)
    assert torch.allclose(scored_lp.sum(1), sample_lp, atol=2e-5)
    row = torch.tensor([[demo.TOKENS["hola"], demo.EOS, demo.TOKENS["vete"], demo.PAD]])
    assert pg.completion_mask(row).tolist() == [[True, True, False, False]]
    grouped = torch.tensor([[0., 0., 1., 1.], [1., 1., 1., 1.]])
    assert torch.allclose(pg.group_advantages(grouped, "rloo")[0], torch.tensor([-2/3., -2/3., 2/3., 2/3.]))
    assert pg.group_advantages(grouped, "grpo").tolist() == [[-1., -1., 1., 1.], [0., 0., 0., 0.]]
    # Enumerar dos acciones Bernoulli verifica el baseline leave-one-out.
    prob, estimate = .3, 0.
    for a in (0, 1):
        for b in (0, 1):
            mass = (prob if a else 1-prob) * (prob if b else 1-prob)
            advantages = pg.group_advantages(torch.tensor([[float(a), float(b)]]), "rloo")[0]
            estimate += mass * float((advantages * torch.tensor([a-prob, b-prob])).mean())
    assert abs(estimate - prob*(1-prob)) < 1e-7
    ratios = torch.tensor([1.5, .5, 1.5, .5], requires_grad=True)
    surrogate, clipped = pg.clipped_surrogate(ratios, torch.tensor([1., 1., -1., -1.]))
    (-surrogate.sum()).backward()
    assert clipped.tolist() == [True, False, False, True]
    assert ratios.grad.tolist() == [0., -1., 1., 0.]
    p, q = torch.tensor([.8, .2]), torch.tensor([.5, .5])
    k3 = pg.sampled_k3(p.log(), q.log())
    assert (k3 >= 0).all()
    assert torch.allclose((p*k3).sum(), (p*(p.log()-q.log())).sum())
    # Un contrato válido puede ser una respuesta poco útil: el proxy no mide utilidad.
    contract_c, contract_y = demo.batch_rows([{"context": "ayuda", "text": "claro vete"},
                                             {"context": "ayuda", "text": "claro claro"}], torch.device("cpu"))
    assert pg.contract_reward(contract_c, contract_y).tolist() == [1., 0.]
    pgout = ROOT / "clase2" / "outputs" / "policy_gradients_notebook"
    if (pgout / "policy_gradients.pt").exists():
        artifact_pg = torch.load(pgout / "policy_gradients.pt", map_location="cpu", weights_only=True)
        reference_pg, rm_pg = demo.TinyLM(), demo.RewardModel()
        reference_pg.load_state_dict(artifact_pg["reference"])
        rm_pg.load_state_dict(artifact_pg["reward_model"])
        metrics_pg = json.loads((pgout / "metrics.json").read_text())
        for method in ("reinforce", "rloo", "grpo"):
            policy_pg = demo.TinyLM()
            policy_pg.load_state_dict(artifact_pg["policies"][method])
            score_pg = pg.evaluate(policy_pg, reference_pg, rm_pg, artifact_pg["reward_center"],
                                   artifact_pg["reward_scale"], torch.device("cpu"))
            assert score_pg["samples"] == metrics_pg["runs"][method]["after"]["samples"]
            assert abs(score_pg["reward_proxy"] - metrics_pg["runs"][method]["after"]["reward_proxy"]) < 1e-5
    # El checkpoint vuelve a producir las métricas de evaluación guardadas.
    out = ROOT / "clase2" / "outputs" / "notebook"
    if (out / "mini_rlhf.pt").exists():
        artifact = torch.load(out / "mini_rlhf.pt", map_location="cpu", weights_only=True)
        policy, reference, rm = demo.TinyLM(), demo.TinyLM(), demo.RewardModel()
        policy.load_state_dict(artifact["policy"])
        reference.load_state_dict(artifact["reference"])
        rm.load_state_dict(artifact["reward_model"])
        score = demo.evaluate(policy, reference, rm, artifact["reward_center"], artifact["reward_scale"], torch.device("cpu"))
        saved = json.loads((out / "metrics.json").read_text())
        assert abs(score["reward_proxy"] - saved["after"]["reward_proxy"]) < 1e-5
        assert score["samples"] == saved["after"]["samples"]
    for path in ROOT.rglob("*.ipynb"):
        notebook = json.loads(path.read_text())
        assert notebook["nbformat"] == 4
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), str(path), "exec")
    for name in ("mini_rlhf.py", "diagnostico_jetson.py", "llm_policy_gradients.py"):
        ast.parse((ROOT / "clase2" / name).read_text(), feature_version=(3, 6))
    print("OK: Bellman, presupuesto, evaluación inmutable, splits, etiquetas, EOS, KL, RLOO, clipping direccional, verificador, checkpoints y notebooks.")


if __name__ == "__main__":
    main()
