"""Mini modelo de lenguaje: SFT -> preferencias -> reward model -> REINFORCE + KL.

Compatible en sintaxis con Python 3.6+ y operaciones básicas de PyTorch.
Sin descargas ni servicios externos. No es un LLM preentrenado.
"""
import argparse
import copy
import json
import platform
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


RESPONSES = {
    "saludo": ["hola puedo ayudarte", "hola que necesitas", "no me importa", "vete ahora",
               "hola hola hola hola", "puedo ayudarte paso a paso cuando quieras"],
    "ayuda": ["claro vamos paso a paso", "claro puedo ayudarte", "no me importa", "hazlo solo",
              "paso paso paso paso", "claro vamos paso a paso cuando quieras ahora"],
    "gracias": ["de nada seguimos cuando quieras", "de nada", "no me importa", "vete ahora",
                "nada nada nada nada", "de nada puedo ayudarte paso a paso cuando quieras"],
}
PROMPTS = {"saludo": "Hola, ¿podemos empezar?",
           "ayuda": "No entiendo el ejercicio. ¿Me ayudás?",
           "gracias": "Gracias por la explicación."}
SPECIAL = ["<pad>", "<bos>", "<eos>", "<saludo>", "<ayuda>", "<gracias>"]
VOCAB = SPECIAL + sorted(set(" ".join(sum(RESPONSES.values(), [])).split()))
TOKENS = {word: i for i, word in enumerate(VOCAB)}
PAD, BOS, EOS = 0, 1, 2
MAX_TOKENS = 10


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encode(text):
    words = text.split()
    unknown = sorted(set(words) - set(TOKENS))
    if unknown:
        raise ValueError("Vocabulario cerrado: palabras no admitidas " + str(unknown))
    if len(words) + 1 > MAX_TOKENS:
        raise ValueError("Respuesta demasiado larga para esta demo")
    return [TOKENS[w] for w in words] + [EOS]


def decode(ids):
    words = []
    for token in ids:
        if int(token) == EOS:
            break
        if int(token) >= len(SPECIAL):
            words.append(VOCAB[int(token)])
    return " ".join(words)


def batch_rows(rows, device):
    contexts = torch.tensor([TOKENS["<" + r["context"] + ">"] for r in rows], device=device)
    labels = torch.full((len(rows), MAX_TOKENS), PAD, dtype=torch.long, device=device)
    for i, row in enumerate(rows):
        ids = encode(row["text"])
        labels[i, :len(ids)] = torch.tensor(ids, device=device)
    return contexts, labels


class TinyLM(nn.Module):
    def __init__(self):
        super(TinyLM, self).__init__()
        self.embedding = nn.Embedding(len(VOCAB), 24)
        self.context = nn.Embedding(len(VOCAB), 48)
        self.gru = nn.GRU(24, 48, batch_first=True)
        self.head = nn.Linear(48, len(VOCAB))

    def forward(self, contexts, tokens):
        h = self.context(contexts).unsqueeze(0)
        out, _ = self.gru(self.embedding(tokens), h)
        return self.head(out)


def sft_train(device, steps=250, seed=2026):
    seed_all(seed)
    model = TinyLM().to(device)
    rows = [{"context": c, "text": text} for c, texts in RESPONSES.items() for text in texts]
    contexts, labels = batch_rows(rows, device)
    inputs = torch.cat([torch.full((len(rows), 1), BOS, device=device, dtype=torch.long),
                        labels[:, :-1]], dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.008)
    history = []
    model.train()
    for step in range(steps):
        loss = F.cross_entropy(model(contexts, inputs).reshape(-1, len(VOCAB)),
                               labels.reshape(-1), ignore_index=PAD)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 25 == 0 or step == steps - 1:
            history.append({"step": step, "loss": float(loss.item())})
    return model, history


def make_pairs():
    """24 pares, orden A/B balanceado. No hay etiquetas humanas incorporadas."""
    pairs = []
    comparisons = [(0, 2), (1, 3), (0, 4), (1, 5), (0, 3), (1, 2), (0, 5), (1, 4)]
    for context, texts in RESPONSES.items():
        for j, (good, other) in enumerate(comparisons):
            flip = (j % 2 == 1)
            pairs.append({"id": context + "_" + str(j), "context": context,
                          "prompt": PROMPTS[context],
                          "a": texts[other] if flip else texts[good],
                          "b": texts[good] if flip else texts[other],
                          "choice": None, "source": "unlabeled",
                          "split": "validation" if j >= 6 else "train"})
    return pairs


def simulated_pairs():
    pairs = make_pairs()
    for p in pairs:
        j = int(p["id"].rsplit("_", 1)[1])
        p["choice"] = "b" if j % 2 else "a"
        p["source"] = "simulated"
    return pairs


def load_preferences(path=None):
    if path is None:
        return simulated_pairs(), "SIMULATED"
    pairs = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {p["id"]: p for p in make_pairs()}
    if len(pairs) != len(expected) or len(set(p["id"] for p in pairs)) != len(expected):
        raise ValueError("Etiquetar exactamente los 24 pares exportados; sin IDs repetidos")
    for p in pairs:
        template = expected.get(p["id"])
        if template is None or any(p.get(k) != template[k] for k in ("context", "a", "b", "split")):
            raise ValueError("Los pares/splits deben coincidir con la plantilla exportada")
        if p.get("source") != "human" or p.get("choice") not in ("a", "b", "tie", "skip"):
            raise ValueError("Cada etiqueta debe tener source=human y choice=a/b/tie/skip")
    return pairs, "HUMAN"


def reward_features(contexts, labels):
    # Bolsa de palabras + longitud + repetición. Pierde orden: limitación deliberada.
    mask = (labels != PAD) & (labels != EOS)
    counts = F.one_hot(labels, len(VOCAB)).float() * mask.unsqueeze(-1).float()
    counts = counts.sum(dim=1)
    lengths = mask.float().sum(dim=1, keepdim=True).clamp(min=1)
    bow = counts / lengths
    repeated = ((counts - 1).clamp(min=0).sum(dim=1, keepdim=True) / lengths)
    context_features = F.one_hot(contexts - TOKENS["<saludo>"], 3).float()
    return torch.cat([bow, lengths / MAX_TOKENS, repeated, context_features], dim=1)


class RewardModel(nn.Module):
    def __init__(self):
        super(RewardModel, self).__init__()
        self.net = nn.Sequential(nn.Linear(len(VOCAB) + 5, 32), nn.Tanh(), nn.Linear(32, 1))

    def forward(self, contexts, labels):
        return self.net(reward_features(contexts, labels)).squeeze(-1)


def preference_tensors(pairs, device, split):
    rows = [p for p in pairs if p["split"] == split and p["choice"] != "skip"]
    if len(rows) < 3 or not any(p["choice"] in ("a", "b") for p in rows):
        raise ValueError("Se requieren al menos 3 pares utilizables y una preferencia en " + split)
    contexts, a = batch_rows([{"context": p["context"], "text": p["a"]} for p in rows], device)
    _, b = batch_rows([{"context": p["context"], "text": p["b"]} for p in rows], device)
    targets = torch.tensor([1. if p["choice"] == "a" else 0. if p["choice"] == "b" else .5
                            for p in rows], device=device)
    return contexts, a, b, targets


def reward_train(pairs, device, steps=200):
    model = RewardModel().to(device)
    c, a, b, targets = preference_tensors(pairs, device, "train")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    history = []
    for step in range(steps):
        score_a, score_b = model(c, a), model(c, b)
        loss = F.binary_cross_entropy_with_logits(score_a - score_b, targets)
        loss = loss + .001 * (score_a.square().mean() + score_b.square().mean())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 25 == 0 or step == steps - 1:
            history.append({"step": step, "loss": float(loss.item())})
    cval, aval, bval, tval = preference_tensors(pairs, device, "validation")
    with torch.no_grad():
        scores = torch.cat([model(c, a), model(c, b)])
        center, scale = scores.mean(), scores.std().clamp(min=.1)
        margins = model(cval, aval) - model(cval, bval)
        decisive = tval != .5
        accuracy = float(((margins[decisive] > 0) == (tval[decisive] > .5)).float().mean().item())
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, center, scale, history, accuracy


def sample(model, reference, contexts, max_tokens=MAX_TOKENS):
    """Muestreo on-policy; logprob incluye EOS y excluye pasos posteriores."""
    device, batch = contexts.device, len(contexts)
    token = torch.full((batch, 1), BOS, dtype=torch.long, device=device)
    h = model.context(contexts).unsqueeze(0)
    h_ref = reference.context(contexts).unsqueeze(0).detach()
    alive = torch.ones(batch, dtype=torch.bool, device=device)
    labels, log_probs, ratios, entropies = [], [], [], []
    for _ in range(max_tokens):
        out, h = model.gru(model.embedding(token), h)
        logits = model.head(out[:, 0])
        with torch.no_grad():
            out_ref, h_ref = reference.gru(reference.embedding(token), h_ref)
            ref_logits = reference.head(out_ref[:, 0])
        # El espacio de acción excluye controles/contextos, pero admite EOS.
        allowed = torch.ones(len(VOCAB), dtype=torch.bool, device=device)
        allowed[:len(SPECIAL)] = False
        allowed[EOS] = True
        logits = logits.masked_fill(~allowed, -1e9)
        ref_logits = ref_logits.masked_fill(~allowed, -1e9)
        dist = torch.distributions.Categorical(logits=logits)
        sampled = dist.sample()
        lp = dist.log_prob(sampled)
        ref_lp = F.log_softmax(ref_logits, dim=-1).gather(1, sampled.unsqueeze(1)).squeeze(1)
        labels.append(torch.where(alive, sampled, torch.zeros_like(sampled)))
        log_probs.append(lp * alive.float())
        ratios.append((lp - ref_lp) * alive.float())
        entropies.append(dist.entropy() * alive.float())
        alive = alive & (sampled != EOS)
        token = sampled.unsqueeze(1)
    return (torch.stack(labels, dim=1), torch.stack(log_probs, dim=1).sum(dim=1),
            torch.stack(ratios, dim=1).sum(dim=1), torch.stack(entropies, dim=1).sum(dim=1))


def evaluate(model, reference, rm, center, scale, device, seed=9900, n=90):
    seed_all(seed)
    contexts = torch.tensor([TOKENS["<" + list(RESPONSES)[i % 3] + ">"] for i in range(n)], device=device)
    with torch.no_grad():
        labels, _, kl, _ = sample(model, reference, contexts)
        rewards = (rm(contexts, labels) - center) / scale
    texts = [decode(row) for row in labels.cpu().tolist()]
    lengths = [len(t.split()) for t in texts]
    result = {"reward_proxy": float(rewards.mean().item()), "sampled_sequence_kl": float(kl.mean().item()),
              "mean_words": sum(lengths) / n,
              "eos_rate": float((labels == EOS).any(dim=1).float().mean().item()),
              "repetition_rate": sum(len(t.split()) != len(set(t.split())) for t in texts) / n,
              "sample_count": n,
              "samples": [{"context": list(RESPONSES)[i % 3], "text": t} for i, t in enumerate(texts)]}
    return result


def rl_train(model, reference, rm, center, scale, device, steps=100, beta=.1, seed=3030):
    seed_all(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    history, baseline = [], 0.0
    contexts = torch.tensor([TOKENS["<" + list(RESPONSES)[i % 3] + ">"] for i in range(36)], device=device)
    for step in range(steps):
        labels, lp, kl, entropy = sample(model, reference, contexts)
        with torch.no_grad():
            reward = (rm(contexts, labels) - center) / scale
            objective_sample = reward - beta * kl.detach()
            advantage = objective_sample - baseline  # Baseline de batches anteriores.
        loss = -(advantage * lp).mean()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        baseline = .9 * baseline + .1 * float(objective_sample.mean().item())
        if step % 10 == 0 or step == steps - 1:
            history.append({"step": step, "reward_proxy": float(reward.mean().item()),
                            "sampled_sequence_kl": float(kl.mean().item()),
                            "sequence_entropy": float(entropy.mean().item()), "loss": float(loss.item())})
    return history


def save_artifacts(result, model, reference, rm, center, scale, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # CPU copies: no modificar el device del modelo activo en el notebook.
    torch.save({"policy": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "reference": {k: v.detach().cpu() for k, v in reference.state_dict().items()},
                "reward_model": {k: v.detach().cpu() for k, v in rm.state_dict().items()},
                "reward_center": center.cpu(), "reward_scale": scale.cpu(),
                "config": result["config"]}, str(output / "mini_rlhf.pt"))
    (output / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    blind, key = [], {}
    rng = random.Random(result["config"]["seed"] + 17)
    before, after = result["before"], result["after"]
    for i in range(12):
        a, b = before["samples"][i], after["samples"][i]
        flip = rng.random() < .5
        blind.append({"id": i, "context": a["context"], "prompt": PROMPTS[a["context"]],
                      "a": b["text"] if flip else a["text"], "b": a["text"] if flip else b["text"],
                      "choice": None, "reason": ""})
        key[str(i)] = {"a": "after" if flip else "before", "b": "before" if flip else "after"}
    (output / "evaluacion_ciega.json").write_text(json.dumps(blind, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "clave_evaluacion_docente.json").write_text(json.dumps(key, indent=2), encoding="utf-8")


def run_demo(output, preference_path=None, device_name="cpu", sft_steps=250,
             reward_steps=200, rl_steps=100, beta=.1, seed=2026):
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA no disponible; verificar wheel/JetPack o elegir cpu")
    torch.set_num_threads(2)
    started = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pairs, provenance = load_preferences(preference_path)
    print("Preferencias:", provenance, "Dispositivo:", device)
    model, sft_history = sft_train(device, sft_steps, seed)
    reference = copy.deepcopy(model).eval()
    for p in reference.parameters():
        p.requires_grad_(False)
    reference_weights = {k: v.detach().clone() for k, v in reference.state_dict().items()}
    seed_all(seed + 1)
    rm, center, scale, reward_history, accuracy = reward_train(pairs, device, reward_steps)
    before = evaluate(model, reference, rm, center, scale, device)
    weights_before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    rl_history = rl_train(model, reference, rm, center, scale, device, rl_steps, beta, seed + 2)
    after = evaluate(model, reference, rm, center, scale, device)
    assert all(torch.equal(reference_weights[k], v) for k, v in reference.state_dict().items()), "Referencia alterada"
    assert any(not torch.equal(weights_before[k], v) for k, v in model.state_dict().items()), "No hubo fine-tuning"
    result = {"feedback_source": provenance, "human_quality_verified": False,
              "versions": {"python": platform.python_version(), "torch": torch.__version__},
              "config": {"device": str(device), "seed": seed, "beta": beta,
                         "sft_steps": sft_steps, "reward_steps": reward_steps, "rl_steps": rl_steps,
                         "max_tokens": MAX_TOKENS, "vocab": VOCAB},
              "parameters": sum(p.numel() for p in model.parameters()),
              "reward_parameters": sum(p.numel() for p in rm.parameters()),
              "preference_validation_accuracy": accuracy,
              "sft_history": sft_history, "reward_history": reward_history, "rl_history": rl_history,
              "before": before, "after": after, "seconds": time.perf_counter() - started}
    save_artifacts(result, model, reference, rm, center, scale, output)
    print("Parámetros:", result["parameters"], "Segundos:", round(result["seconds"], 2))
    print("Recompensa proxy antes/después:", before["reward_proxy"], after["reward_proxy"])
    print("No inferir calidad humana a partir de esa recompensa.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/mini_rlhf")
    parser.add_argument("--preferences", help="Archivo de 24 pares etiquetados por personas")
    parser.add_argument("--export-pairs", help="Solo exportar plantilla sin etiquetas")
    parser.add_argument("--label-pairs", help="Etiquetar plantilla interactivamente por consola")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--beta", type=float, default=.1)
    parser.add_argument("--sft-steps", type=int, default=250)
    parser.add_argument("--reward-steps", type=int, default=200)
    parser.add_argument("--rl-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.export_pairs:
        Path(args.export_pairs).write_text(json.dumps(make_pairs(), ensure_ascii=False, indent=2), encoding="utf-8")
        print("Plantilla exportada:", args.export_pairs)
        return
    if args.label_pairs:
        path = Path(args.label_pairs)
        pairs = json.loads(path.read_text(encoding="utf-8")) if path.exists() else make_pairs()
        for p in pairs:
            print("\n", p["prompt"], "\nA:", p["a"], "\nB:", p["b"])
            choice = ""
            while choice not in ("a", "b", "tie", "skip"):
                choice = input("Preferís a / b / tie / skip: ").strip().lower()
            p["choice"], p["source"] = choice, "human"
        path.write_text(json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Etiquetas guardadas:", path)
        return
    if min(args.sft_steps, args.reward_steps, args.rl_steps) < 1 or args.beta < 0:
        parser.error("Pasos positivos y beta no negativo")
    run_demo(args.output, args.preferences, args.device, args.sft_steps,
             args.reward_steps, args.rl_steps, args.beta, args.seed)


if __name__ == "__main__":
    main()
