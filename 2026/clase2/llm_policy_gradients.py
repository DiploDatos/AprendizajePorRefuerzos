"""Laboratorio de gradientes de política para generación autoregresiva pequeña.

Implementación docente propia: REINFORCE, RLOO y GRPO original simplificado.
Reutiliza TinyLM y preferencias; no requiere transformers ni pesos externos.
"""
import argparse
import copy
import json
import platform
import time
from pathlib import Path

import torch
from torch.nn import functional as F

import mini_rlhf as tiny


def completion_mask(labels):
    """Incluye el primer EOS; excluye PAD y cualquier acción posterior al EOS."""
    ended_before = torch.cat([torch.zeros_like(labels[:, :1]),
                              (labels == tiny.EOS).long().cumsum(dim=1)[:, :-1]], dim=1)
    return (labels != tiny.PAD) & (ended_before == 0)


def token_statistics(model, contexts, labels):
    # Shift causal: BOS predice la primera acción, y_t predice y_{t+1}.
    inputs = torch.cat([torch.full_like(labels[:, :1], tiny.BOS), labels[:, :-1]], dim=1)
    logits = model(contexts, inputs)
    allowed = torch.ones(len(tiny.VOCAB), dtype=torch.bool, device=labels.device)
    allowed[:len(tiny.SPECIAL)] = False
    allowed[tiny.EOS] = True
    logits = logits.masked_fill(~allowed, -1e9)
    logps = F.log_softmax(logits, dim=-1)
    selected = logps.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    entropy = -(logps.exp() * logps).sum(dim=-1)
    mask = completion_mask(labels)
    return selected * mask.float(), entropy * mask.float(), mask


def group_advantages(rewards, method):
    """Filas = prompts; columnas = muestras independientes de ese prompt."""
    if rewards.ndim != 2 or rewards.shape[1] < 2:
        raise ValueError("Se requieren al menos dos respuestas por prompt")
    if method == "rloo":
        baseline = (rewards.sum(dim=1, keepdim=True) - rewards) / (rewards.shape[1] - 1)
        return rewards - baseline
    if method == "grpo":
        centered = rewards - rewards.mean(dim=1, keepdim=True)
        std = centered.square().mean(dim=1, keepdim=True).sqrt()
        return centered / (std + 1e-8)
    raise ValueError("Método de grupo desconocido")


def clipped_surrogate(ratio, advantage, epsilon=.2):
    raw = ratio * advantage
    capped = ratio.clamp(1 - epsilon, 1 + epsilon) * advantage
    active = ((advantage > 0) & (ratio > 1 + epsilon)) | (
        (advantage < 0) & (ratio < 1 - epsilon))
    return torch.minimum(raw, capped), active


def sampled_k3(logps, ref_logps):
    # exp(x) - x - 1; expm1 evita cancelación alrededor de cero.
    x = ref_logps - logps
    return torch.expm1(x) - x


def contract_reward(contexts, labels):
    """Verificador exacto de un contrato de formato, NO de calidad semántica.

    Debe empezar por un prefijo, tener 1..6 palabras sin repetición y emitir EOS.
    Una respuesta que cumple puede ser inútil: discutir ese contraejemplo.
    """
    prefixes = {tiny.TOKENS["<saludo>"]: [tiny.TOKENS["hola"]],
                tiny.TOKENS["<ayuda>"]: [tiny.TOKENS["claro"]],
                tiny.TOKENS["<gracias>"]: [tiny.TOKENS["de"], tiny.TOKENS["nada"]]}
    mask = completion_mask(labels)
    result = []
    for context, row, valid in zip(contexts.tolist(), labels.tolist(), mask.tolist()):
        actions = [t for t, keep in zip(row, valid) if keep]
        words = [t for t in actions if t != tiny.EOS]
        prefix = prefixes[context]
        result.append(float(words[:len(prefix)] == prefix and 1 <= len(words) <= 6
                            and len(set(words)) == len(words) and tiny.EOS in actions))
    return torch.tensor(result, device=labels.device)


def reward_values(contexts, labels, rm, center, scale, source):
    if source == "learned":
        return (rm(contexts, labels) - center) / scale
    if source == "verifier":
        return contract_reward(contexts, labels)
    raise ValueError("Fuente de recompensa desconocida")


def train_policy(model, reference, rm, center, scale, device, method,
                 steps=60, group_size=8, beta=.1, updates=3, seed=3030,
                 reward_source="learned"):
    if method not in ("reinforce", "rloo", "grpo"):
        raise ValueError("Método desconocido")
    if steps < 1 or group_size < 2 or beta < 0 or updates < 1:
        raise ValueError("Pasos/updates positivos, grupo >= 2 y beta >= 0")
    tiny.seed_all(seed)
    model.eval()  # No dropout: muestreo y evaluación de logprobs deben coincidir.
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    contexts = torch.tensor([tiny.TOKENS["<" + c + ">"] for c in tiny.RESPONSES], device=device)
    contexts = contexts.repeat_interleave(group_size)
    baseline, history = 0., []
    for step in range(steps):
        # old es la política al muestrear ESTE rollout; ref sigue siendo el SFT inicial.
        with torch.no_grad():
            labels, sampled_lp, _, _ = tiny.sample(model, reference, contexts)
            old_logps, _, mask = token_statistics(model, contexts, labels)
            ref_logps, _, _ = token_statistics(reference, contexts, labels)
            assert torch.allclose(old_logps.sum(1), sampled_lp, atol=2e-5)
            rewards = reward_values(contexts, labels, rm, center, scale, reward_source)
            seq_kl = (old_logps - ref_logps).sum(1)
            if method == "grpo":
                advantage = group_advantages(rewards.view(3, group_size), "grpo").reshape(-1)
            else:
                returns = rewards - beta * seq_kl
                advantage = (returns - baseline if method == "reinforce" else
                             group_advantages(returns.view(3, group_size), "rloo").reshape(-1))
            reward_groups = rewards.view(3, group_size)
            zero_groups = ((reward_groups.max(1)[0] - reward_groups.min(1)[0]) == 0).float().mean()
        # REINFORCE/RLOO hacen una actualización on-policy, sin reutilizar el batch.
        inner_updates = updates if method == "grpo" else 1
        max_deviation, active_fraction = 0., 0.
        for _ in range(inner_updates):
            logps, entropy, _ = token_statistics(model, contexts, labels)
            if method == "grpo":
                ratio = (logps - old_logps).exp()
                surrogate, active = clipped_surrogate(ratio, advantage[:, None])
                per_token = -surrogate + beta * sampled_k3(logps, ref_logps)
                # GRPO original: media por respuesta y después por batch.
                loss = ((per_token * mask.float()).sum(1) / mask.sum(1)).mean()
                active_fraction = float((active & mask).float().sum().item() / mask.sum().item())
                max_deviation = max(max_deviation, float((ratio[mask] - 1).abs().max().item()))
            else:
                loss = -(advantage * logps.sum(1)).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        if method == "reinforce":
            baseline = .9 * baseline + .1 * float(returns.mean().item())
        if step % 5 == 0 or step == steps - 1:
            history.append({"step": step, "reward": float(rewards.mean().item()),
                            "sampled_sequence_kl": float(seq_kl.mean().item()),
                            "mean_tokens": float(mask.float().sum(1).mean().item()),
                            "entropy_per_token": float(entropy.sum().item() / mask.sum().item()),
                            "equal_reward_group_fraction": float(zero_groups.item()),
                            "active_clip_fraction": active_fraction,
                            "max_ratio_deviation": max_deviation, "loss": float(loss.item())})
    return history


def evaluate(model, reference, rm, center, scale, device, seed=9900, n=180):
    result = tiny.evaluate(model, reference, rm, center, scale, device, seed, n)
    tiny.seed_all(seed)
    contexts = torch.tensor([tiny.TOKENS["<" + list(tiny.RESPONSES)[i % 3] + ">"]
                             for i in range(n)], device=device)
    with torch.no_grad():
        labels, _, _, _ = tiny.sample(model, reference, contexts)
        result["contract_pass_rate"] = float(contract_reward(contexts, labels).mean().item())
    result["unique_text_fraction"] = len(set(s["text"] for s in result["samples"])) / n
    return result


def prepare(device, preference_path=None, seed=2026):
    model, sft_history = tiny.sft_train(device, seed=seed)
    reference = copy.deepcopy(model).eval()
    for p in reference.parameters():
        p.requires_grad_(False)
    pairs, provenance = tiny.load_preferences(preference_path)
    tiny.seed_all(seed + 1)
    rm, center, scale, reward_history, accuracy = tiny.reward_train(pairs, device)
    return model, reference, rm, center, scale, {
        "feedback_source": provenance, "sft_history": sft_history,
        "reward_history": reward_history, "preference_validation_accuracy": accuracy}


def compare(model, reference, rm, center, scale, device, steps=60, group_size=8,
            beta=.1, updates=3, reward_source="learned", seed=3030):
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    reference_initial = {k: v.detach().clone() for k, v in reference.state_dict().items()}
    result = {"before": evaluate(model, reference, rm, center, scale, device), "runs": {}}
    policies = {}
    for method in ("reinforce", "rloo", "grpo"):
        policy = copy.deepcopy(model)
        start = time.perf_counter()
        history = train_policy(policy, reference, rm, center, scale, device, method,
                               steps, group_size, beta, updates, seed, reward_source)
        if device.type == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        result["runs"][method] = {"history": history,
                                   "after": evaluate(policy, reference, rm, center, scale, device),
                                   "seconds_training": seconds,
                                   "sampled_completions": steps * 3 * group_size,
                                   "optimizer_updates": steps * (updates if method == "grpo" else 1)}
        assert any(not torch.equal(initial[k], v) for k, v in policy.state_dict().items())
        policies[method] = policy
    assert all(torch.equal(reference_initial[k], v) for k, v in reference.state_dict().items())
    assert all(torch.equal(initial[k], v) for k, v in model.state_dict().items())
    result["config"] = {"device": str(device), "steps": steps, "group_size": group_size,
                         "beta": beta, "grpo_updates_per_rollout": updates, "seed": seed,
                         "reward_source": reward_source, "loss_type": "original_grpo",
                         "max_tokens": tiny.MAX_TOKENS}
    result["versions"] = {"python": platform.python_version(), "torch": torch.__version__}
    result["human_quality_verified"] = False
    return result, policies


def save(result, policies, reference, rm, center, scale, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    def cpu_state(model):
        return {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"policies": {k: cpu_state(v) for k, v in policies.items()},
                "reference": cpu_state(reference), "reward_model": cpu_state(rm),
                "reward_center": center.cpu(), "reward_scale": scale.cpu(),
                "config": result["config"]}, str(output / "policy_gradients.pt"))
    # Cegado del resultado principal GRPO, reutilizando el protocolo de la demo RLHF.
    blind_result = {"config": result["config"], "before": result["before"],
                    "after": result["runs"]["grpo"]["after"]}
    tiny.save_artifacts(blind_result, policies["grpo"], reference, rm, center, scale, output / "blind_grpo")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/policy_gradients")
    parser.add_argument("--preferences")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--reward-source", choices=["learned", "verifier"], default="learned")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--beta", type=float, default=.1)
    parser.add_argument("--updates", type=int, default=3)
    args = parser.parse_args()
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    torch.set_num_threads(2)
    model, reference, rm, center, scale, preparation = prepare(device, args.preferences)
    result, policies = compare(model, reference, rm, center, scale, device,
                               args.steps, args.group_size, args.beta, args.updates, args.reward_source)
    result["preparation"] = preparation
    save(result, policies, reference, rm, center, scale, args.output)
    for method, run in result["runs"].items():
        print(method, "proxy", round(run["after"]["reward_proxy"], 3),
              "contrato", round(run["after"]["contract_pass_rate"], 3),
              "updates", run["optimizer_updates"])
    print("Origen de preferencias:", preparation["feedback_source"], "; recompensa:", args.reward_source)


if __name__ == "__main__":
    main()
