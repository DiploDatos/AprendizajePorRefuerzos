import numpy as np
import pytest

from protocolo.baselines import GroupPosteriorPolicy, ItemMeanPolicy, RandomPolicy
from protocolo.datos import User, make_episodes, split_by_user
from protocolo.evaluar import evaluate, run_session
from protocolo.metricas import normalized_score, ranking_metrics, session_bounds, summarize
from protocolo.simulador import Session, SessionConfig, choose

N_ITEMS = 30


def full_user(user_id=0, group=0, seed=0):
    rng = np.random.default_rng(seed)
    return User(user_id, group, np.arange(N_ITEMS), rng.integers(1, 6, N_ITEMS).astype(float))


class OraclePolicy:
    """Usa las valoraciones ocultas: sólo existe para probar las cotas."""

    def __init__(self, user):
        self.true = np.full(N_ITEMS, -np.inf)
        self.true[user.items] = user.ratings

    def reset(self, group, horizon):
        pass

    def scores(self, items_seen, ratings_seen):
        return self.true


class RecordingPolicy:
    def __init__(self):
        self.calls = []
        self.groups = []

    def reset(self, group, horizon):
        self.groups.append(group)

    def scores(self, items_seen, ratings_seen):
        self.calls.append((items_seen.copy(), ratings_seen.copy()))
        items_seen[:] = -1  # mutar la copia no debe afectar la sesión
        return np.zeros(N_ITEMS)


def test_session_reveals_rating_only_when_shown_and_forbids_repeats():
    user = full_user()
    s = Session(user, N_ITEMS, SessionConfig(horizon=5), np.random.default_rng(0))
    assert s.observed()[0].size == 0
    assert s.step(3) == user.ratings[3]
    with pytest.raises(ValueError):
        s.step(3)


def test_session_rejects_items_without_known_rating():
    user = User(0, 0, np.arange(10), np.full(10, 3.0))
    s = Session(user, N_ITEMS, SessionConfig(horizon=5), np.random.default_rng(0))
    with pytest.raises(ValueError):
        s.step(20)
    assert not s.available_mask()[10:].any()


def test_warmup_items_are_distinct_shown_and_reproducible():
    cfg = SessionConfig(horizon=10, warmup=4)
    a = Session(full_user(), N_ITEMS, cfg, np.random.default_rng([1, 2]))
    b = Session(full_user(), N_ITEMS, cfg, np.random.default_rng([1, 2]))
    assert a.items == b.items and len(set(a.items)) == 4
    assert not a.available_mask()[a.items].any()


def test_session_ends_at_horizon():
    s = Session(full_user(), N_ITEMS, SessionConfig(horizon=3), np.random.default_rng(0))
    for item in range(3):
        s.step(item)
    assert s.done
    with pytest.raises(RuntimeError):
        s.step(10)


def test_group_hidden_when_not_visible():
    policy = RecordingPolicy()
    run_session(policy, full_user(group=5), N_ITEMS, SessionConfig(horizon=3, group_visible=False), 0, 0)
    run_session(policy, full_user(group=5), N_ITEMS, SessionConfig(horizon=3, group_visible=True), 0, 0)
    assert policy.groups == [None, 5]


def test_policy_sees_only_past_and_cannot_corrupt_session():
    policy = RecordingPolicy()
    row = run_session(policy, full_user(), N_ITEMS, SessionConfig(horizon=4, warmup=1), 0, 0)
    sizes = [len(items) for items, _ in policy.calls]
    assert sizes == [1, 2, 3]
    assert np.isfinite(row["return"])


def test_oracle_scores_one_and_bounds_are_consistent():
    user = full_user()
    row = run_session(OraclePolicy(user), user, N_ITEMS, SessionConfig(horizon=6, warmup=2), 0, 0)
    assert row["normalized"] == pytest.approx(1.0)
    assert row["random_expected"] < row["oracle"]
    assert row["ndcg@10"] == pytest.approx(1.0)


def test_session_bounds_and_normalization():
    rand, orac = session_bounds(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), steps=2)
    assert rand == pytest.approx(6.0) and orac == pytest.approx(9.0)
    assert normalized_score(6.0, rand, orac) == pytest.approx(0.0)
    assert normalized_score(9.0, rand, orac) == pytest.approx(1.0)


def test_ranking_metrics_ignore_masked_items():
    hidden = np.array([5.0, 1.0, 5.0, 1.0])
    mask = np.array([False, True, True, True])
    scores = np.array([100.0, 0.0, 1.0, 0.0])  # el ítem 0 está enmascarado
    m = ranking_metrics(scores, hidden, mask, k=1)
    assert m["ndcg@1"] == pytest.approx(1.0) and m["precision@1"] == pytest.approx(1.0)


def test_choose_respects_mask_and_breaks_ties_randomly():
    mask = np.array([False, True, True])
    picks = {choose(np.array([9.0, 1.0, 1.0]), mask, np.random.default_rng(s)) for s in range(20)}
    assert picks == {1, 2}


def test_split_is_by_user_and_stratified():
    users = [User(i, i % 2, np.arange(3), np.ones(3)) for i in range(40)]
    train, val = split_by_user(users, val_fraction=0.25, seed=0)
    assert {u.user_id for u in train}.isdisjoint({u.user_id for u in val})
    assert len(val) == 10 and sum(u.group for u in val) == 5


def test_episodes_do_not_mix_users_and_rtg_is_suffix_sum():
    users = [User(0, 1, np.arange(7), np.arange(1, 8, dtype=float)), User(1, 2, np.arange(2), np.ones(2))]
    ep = make_episodes(users, length=3)
    assert ep.items.shape == (2, 3) and list(ep.user_ids) == [0, 0]
    assert list(ep.returns_to_go()[0]) == [6.0, 5.0, 3.0]


def _two_group_train():
    rng = np.random.default_rng(0)
    users = []
    for uid in range(200):
        g = uid % 2
        items = rng.permutation(N_ITEMS)[:20]
        # grupo 0 prefiere ítems bajos; grupo 1, altos
        base = np.where(items < N_ITEMS // 2, 5.0, 1.0) if g == 0 else np.where(items < N_ITEMS // 2, 1.0, 5.0)
        users.append(User(uid, g, items, base))
    return users


def test_group_mean_uses_group_only_when_visible():
    pol = ItemMeanPolicy(_two_group_train(), N_ITEMS, by_group=True)
    pol.reset(0, 10)
    s0 = pol.scores(np.array([]), np.array([]))
    pol.reset(None, 10)
    s_none = pol.scores(np.array([]), np.array([]))
    assert s0[: N_ITEMS // 2].mean() > s0[N_ITEMS // 2 :].mean()
    assert abs(s_none[: N_ITEMS // 2].mean() - s_none[N_ITEMS // 2 :].mean()) < 0.5


def test_group_posterior_identifies_group_from_feedback():
    pol = GroupPosteriorPolicy(_two_group_train(), N_ITEMS)
    pol.reset(None, 10)
    p = pol.posterior(np.array([0, 1, 20]), np.array([1.0, 1.0, 5.0]))  # comportamiento del grupo 1
    assert p.argmax() == 1 and p.max() > 0.99


def test_summarize_counts_users_not_sessions():
    users = [full_user(i, i % 2, seed=i) for i in range(6)]
    df = evaluate(RandomPolicy(N_ITEMS), users, N_ITEMS, SessionConfig(horizon=5), seeds=(0, 1, 2))
    out = summarize(df)
    assert int(out[("n_users", "")].iloc[0]) == 6
