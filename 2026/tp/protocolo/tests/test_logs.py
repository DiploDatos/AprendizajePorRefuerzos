import numpy as np

from protocolo.baselines import PopularityPolicy
from protocolo.datos import User
from protocolo.logs import generate_logs

N_ITEMS = 40


def _users(n=60, seed=0):
    """Ítems 0-19 buenos (5) y 20-39 malos (1); cada usuario valoró 30 al azar."""
    rng = np.random.default_rng(seed)
    quality = np.where(np.arange(N_ITEMS) < 20, 5.0, 1.0)
    users = []
    for i in range(n):
        items = rng.choice(N_ITEMS, 30, replace=False)
        users.append(User(i, 0, items, quality[items]))
    return users


def test_logs_only_replay_real_ratings_without_repeats():
    users = _users()
    data = generate_logs(users, users, N_ITEMS, epsilons=(0.0, 1.0), horizon=10, seed=0)
    by_id = {u.user_id: dict(zip(u.items.tolist(), u.ratings.tolist())) for u in users}
    e = data.episodes
    assert len(e.groups) == sum(len(u.items) // 10 for u in users)
    for items, ratings, uid in zip(e.items, e.ratings, e.user_ids):
        assert len(set(items.tolist())) == len(items)
        assert all(by_id[uid][i] == r for i, r in zip(items.tolist(), ratings.tolist()))


def test_greedy_sessions_beat_uniform_sessions():
    users = _users()
    data = generate_logs(users, users, N_ITEMS, epsilons=(0.0, 1.0), temperature=0.25, horizon=10, seed=0)
    returns = data.episodes.ratings.sum(1)
    assert returns[data.epsilons == 0.0].mean() > returns[data.epsilons == 1.0].mean() + 10


def test_previous_system_ignores_the_users_own_ratings():
    # Sólo el usuario 0 valoró los ítems 0-9, con 5. Sin sus propias valoraciones,
    # esos ítems no tienen evidencia y el sistema no debe preferirlos.
    target = User(0, 0, np.arange(20), np.r_[np.full(10, 5.0), np.full(10, 3.0)])
    others = [User(i, 0, np.arange(10, 20), np.full(10, 3.0)) for i in range(1, 30)]
    users = [target, *others]
    data = generate_logs([target], users, 20, epsilons=(0.0,), temperature=0.05, horizon=2, seed=0)
    first = data.episodes.items[:, 0]
    assert (first < 10).mean() < 0.9  # con fuga, elegiría siempre un ítem 0-9


def test_popularity_counts_appearances_not_ratings():
    users = [User(0, 0, np.array([0, 1]), np.array([1.0, 5.0])), User(1, 0, np.array([0]), np.array([1.0]))]
    s = PopularityPolicy(users, 3).scores(np.array([]), np.array([]))
    assert s.tolist() == [2.0, 1.0, 0.0]
