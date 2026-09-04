"""
Agent-based model for moral heterogeneity, endogenous institutions, and
elite overproduction.

Moral types: 0 = altruistic, 1 = extractive, 2 = norm-following.
Regimes: 0 = inclusive/altruistic, 1 = extractive.
"""

from collections import defaultdict

import numpy as np
from numpy.random import default_rng


class Params:
    # Population
    N: int = 200
    L: int = 100
    c: float = 1 / 4
    omega_0: float = 2 / 3

    # Production
    q: float = 1 / 3
    t_c: int = 3

    # Grid
    l: int = 30
    a: int = 30
    S_max: float = 50.0

    # Social behaviour
    sigma_a: float = 1 / 3
    sigma: float = 1 / 6
    sigma_hat: float = 1 / 9

    # Institutional
    u: float = 0.75
    g: int = 3

    # Elite
    p_e: float = 0.01
    plus_rate: float = 0.05
    mu: float = 0.50

    # Congress
    eta: float = 0.50
    z: float = 1.96
    e: float = 0.30
    tau: float = 0.05
    gamma_c: float = 0.50

    # Elite--congress hierarchy (Proposition 3)
    elite_override: bool = True

    # Elite oversupply / disintegration
    theta: float = 0.10
    stochastic_disint: bool = True
    disint_scale: float = 10.0
    p_max: float = 0.30

    # Threshold smoothing
    smooth_thresholds: bool = True
    k_regime: float = 40.0
    k_mu: float = 40.0

    # Adaptive moral-type switching
    adaptive: bool = True
    psi: float = 0.02
    beta_fermi: float = 0.5
    same_faction_bias: float = 0.7

    # Simulation
    T: int = 500


def gini(x: np.ndarray) -> float:
    x = x[x > 0].astype(float)
    if len(x) < 2 or np.sum(x) <= 0:
        return 0.0
    x = np.sort(x)
    n = len(x)
    idx = np.arange(1, n + 1)
    return float((2 * np.dot(idx, x) - (n + 1) * np.sum(x)) / (n * np.sum(x)))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def draw_type(rng, d_A: float, d_X: float) -> int:
    r = rng.random()
    return 0 if r < d_A else (1 if r < d_A + d_X else 2)


def _respawn(i, rng, E, alive, age, pos_x, pos_y, faction, mtype,
             prod_queue, p, type_probs):
    E[i] = p.omega_0
    alive[i] = True
    age[i] = 0
    pos_x[i] = rng.integers(0, p.l)
    pos_y[i] = rng.integers(0, p.a)
    faction[i] = rng.integers(0, p.g)
    mtype[i] = int(rng.choice(3, p=type_probs))
    prod_queue[i] = []


def _regime_from_population(alpha: float, p: Params, rng) -> int:
    if p.smooth_thresholds:
        return 0 if rng.random() < sigmoid(p.k_regime * (alpha - p.u)) else 1
    return 0 if alpha >= p.u else 1


def _classify_body(share_alt: float, p: Params, rng, fallback: int) -> int:
    if p.smooth_thresholds:
        p_alt = sigmoid(p.k_mu * (share_alt - p.mu))
        p_ext = sigmoid(p.k_mu * ((1.0 - share_alt) - p.mu))
        r = rng.random()
        if r < p_alt:
            return 0
        if r < p_alt + (1.0 - p_alt) * p_ext:
            return 1
        return fallback
    return 0 if share_alt > p.mu else 1 if share_alt < 1.0 - p.mu else fallback


def _effective_types(mtype: np.ndarray, regime: int, t: int) -> np.ndarray:
    eff = mtype.copy()
    eff[mtype == 2] = 2 if t == 0 else regime
    return eff


def _eligible_pool(indices, E, p: Params):
    if len(indices) == 0:
        return indices
    wealth = E[indices]
    cutoff = float(np.quantile(wealth, 1.0 - p.eta)) if len(indices) > 1 else float(wealth[0])
    eligible = indices[wealth >= cutoff]
    return eligible if len(eligible) > 0 else indices


def _top_fraction_indices(alive_idx, E, fraction: float, rng) -> np.ndarray:
    if len(alive_idx) == 0 or fraction <= 0:
        return np.array([], dtype=int)
    n_select = int(np.rint(fraction * len(alive_idx)))
    n_select = min(max(n_select, 1), len(alive_idx))
    scores = E[alive_idx] + rng.random(len(alive_idx)) * 1e-12
    order = np.argsort(-scores)
    return alive_idx[order[:n_select]]


def _form_congress(alive_idx, faction, E, p: Params, rng) -> np.ndarray:
    N_alive = len(alive_idx)
    if N_alive == 0:
        return np.array([], dtype=int)

    n0 = int(np.floor((p.z ** 2) * 0.25 / (p.e ** 2)))
    C = int(np.rint((n0 * N_alive) / (N_alive + n0))) if n0 > 0 else 0
    C = min(max(C, 1), N_alive)

    faction_members = []
    quotas = []
    for f in range(p.g):
        fi = alive_idx[faction[alive_idx] == f]
        faction_members.append(fi)
        quotas.append(len(fi) / N_alive * C)

    seats = np.floor(quotas).astype(int)
    remaining = C - int(seats.sum())
    if remaining > 0:
        remainders = np.array(quotas) - seats
        order = np.argsort(-(remainders + rng.random(p.g) * 1e-12))
        for f in order[:remaining]:
            seats[f] += 1

    members = []
    for f, n_seats in enumerate(seats):
        if n_seats <= 0 or len(faction_members[f]) == 0:
            continue
        eligible = _eligible_pool(faction_members[f], E, p)
        replace = n_seats > len(eligible)
        members.extend(rng.choice(eligible, n_seats, replace=replace).tolist())

    return np.array(members, dtype=int)


def _congress_nature(cong_idx, mtype, regime: int, p: Params, rng) -> int:
    if len(cong_idx) == 0:
        return regime
    share_alt = float(np.sum(mtype[cong_idx] == 0)) / len(cong_idx)
    return _classify_body(share_alt, p, rng, regime)


def _elite_nature(elite_idx, mtype, regime: int, p: Params, rng) -> int:
    if len(elite_idx) == 0:
        return regime
    share_alt = float(np.sum(mtype[elite_idx] == 0)) / len(elite_idx)
    return _classify_body(share_alt, p, rng, regime)


def _apply_elite_override(cong_idx, alive_idx, mtype, faction, E, p: Params, rng) -> np.ndarray:
    if len(cong_idx) == 0:
        return cong_idx

    needed = max(
        0,
        int(np.floor(len(cong_idx) / 2)) + 1 - int(np.sum(mtype[cong_idx] == 0)),
    )
    if needed == 0:
        return cong_idx

    new_cong = cong_idx.copy()
    replace_positions = np.where(mtype[new_cong] == 1)[0]
    rng.shuffle(replace_positions)

    for pos in replace_positions:
        if needed == 0:
            break
        old_member = new_cong[pos]
        not_current = ~np.isin(alive_idx, new_cong)
        candidates = alive_idx[
            not_current
            & (mtype[alive_idx] == 0)
            & (faction[alive_idx] == faction[old_member])
        ]
        candidates = _eligible_pool(candidates, E, p)
        if len(candidates) == 0:
            candidates = alive_idx[not_current & (mtype[alive_idx] == 0)]
            candidates = _eligible_pool(candidates, E, p)
        if len(candidates) == 0:
            break
        new_cong[pos] = int(rng.choice(candidates))
        needed -= 1

    return new_cong


def _resolve_interactions(cell_map, eff, mtype, E, faction, p: Params, rng) -> None:
    acted = set()

    # Theft subphase.
    for ags in cell_map.values():
        if len(ags) < 2:
            continue
        pre_E = {int(i): float(E[i]) for i in ags}
        claims = defaultdict(list)
        for i in ags:
            if eff[i] != 1:
                continue
            for j in ags:
                if i != j and pre_E[j] > 0.0 and pre_E[j] < pre_E[i]:
                    claims[int(j)].append(int(i))

        assigned = defaultdict(list)
        for victim, thieves in claims.items():
            max_w = max(pre_E[th] for th in thieves)
            winners = [th for th in thieves if pre_E[th] == max_w]
            thief = int(rng.choice(winners))
            assigned[thief].append(victim)

        for thief, victims in assigned.items():
            if thief in acted:
                continue
            for victim in victims:
                if victim in acted or E[victim] <= 0.0:
                    continue
                E[thief] += E[victim]
                E[victim] = 0.0
                acted.add(victim)
            acted.add(thief)

    # Donation subphase.
    for ags in cell_map.values():
        if len(ags) < 2:
            continue
        donors = [int(i) for i in rng.permutation(ags) if i not in acted and eff[i] == 0]
        for i in donors:
            possible = []
            for j in ags:
                j = int(j)
                if j == i or j in acted or E[j] >= p.c:
                    continue
                amount = p.sigma_a if mtype[i] == 0 else (
                    p.sigma if faction[i] == faction[j] else p.sigma_hat
                )
                if E[i] >= amount and E[j] + amount >= p.c:
                    possible.append((float(E[j]), j, amount))
            if not possible:
                continue
            min_stock = min(row[0] for row in possible)
            tied = [row for row in possible if row[0] == min_stock]
            _, recipient, amount = tied[int(rng.integers(len(tied)))]
            E[i] -= amount
            E[recipient] += amount
            acted.add(i)


def _adapt_preferences(alive_idx, mtype, faction, E, p: Params, rng) -> None:
    N_alive = len(alive_idx)
    if not p.adaptive or p.psi <= 0.0 or N_alive <= 1:
        return

    considering = alive_idx[rng.random(N_alive) < p.psi]
    if len(considering) == 0:
        return

    peers = np.empty(len(considering), dtype=int)
    same_bias = rng.random(len(considering)) < p.same_faction_bias
    for k, i in enumerate(considering):
        pool = None
        if same_bias[k]:
            pool = alive_idx[(faction[alive_idx] == faction[i]) & (alive_idx != i)]
        if pool is None or len(pool) == 0:
            pool = alive_idx[alive_idx != i]
        peers[k] = pool[rng.integers(len(pool))]

    diff = E[peers] - E[considering]
    do_switch = rng.random(len(considering)) < sigmoid(p.beta_fermi * diff)
    if np.any(do_switch):
        mtype[considering[do_switch]] = mtype[peers[do_switch]]


def run(d_A: float, d_X: float, seed: int, p: Params = None) -> dict:
    if p is None:
        p = Params()

    rng = default_rng(seed)
    N = p.N

    E = np.full(N, p.omega_0)
    alive = np.ones(N, dtype=bool)
    age = rng.integers(0, p.L // 2, N)
    pos_x = rng.integers(0, p.l, N)
    pos_y = rng.integers(0, p.a, N)
    faction = rng.integers(0, p.g, N)
    mtype = np.array([draw_type(rng, d_A, d_X) for _ in range(N)], dtype=np.int8)
    prod_queue = [[] for _ in range(N)]
    grid = rng.uniform(0, p.S_max, (p.l, p.a))

    Y_t = np.full(p.T, np.nan)
    G_t = np.full(p.T, np.nan)
    alpha_t = np.full(p.T, np.nan)
    regime_t = np.full(p.T, np.nan)
    congress_t = np.full(p.T, np.nan)
    disint_t = np.full(p.T, np.nan)
    poverty_t = np.full(p.T, np.nan)
    deaths_t = np.full(p.T, np.nan)
    mortality_t = np.full(p.T, np.nan)
    N_t = np.zeros(p.T, dtype=np.int32)

    dx = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1], dtype=np.int32)
    dy = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1], dtype=np.int32)

    for t in range(p.T):
        alive_idx = np.where(alive)[0]
        if len(alive_idx) == 0:
            break

        N_start = len(alive_idx)
        poverty_t[t] = int(np.sum(E[alive_idx] < p.c))

        # 1. Harvest receipt.
        for i in alive_idx:
            delivered = sum(amt for (dt, amt) in prod_queue[i] if dt == t)
            prod_queue[i] = [(dt, amt) for (dt, amt) in prod_queue[i] if dt != t]
            E[i] += delivered

        # 2. Collective update.
        alpha_start = float(np.sum(mtype[alive_idx] == 0)) / N_start
        regime = _regime_from_population(alpha_start, p, rng)
        regime_t[t] = regime

        E_alive = E[alive_idx]
        elite_idx = _top_fraction_indices(alive_idx, E, p.p_e, rng)
        elite_nat = _elite_nature(elite_idx, mtype, regime, p, rng)

        cong_idx = _form_congress(alive_idx, faction, E, p, rng)
        cong_nat = _congress_nature(cong_idx, mtype, regime, p, rng)
        if p.elite_override and cong_nat == 1 and elite_nat == 0 and len(elite_idx) > 0:
            cong_idx = _apply_elite_override(cong_idx, alive_idx, mtype, faction, E, p, rng)
            cong_nat = _congress_nature(cong_idx, mtype, regime, p, rng)

        top_idx = _top_fraction_indices(alive_idx, E, p.theta, rng)
        if p.stochastic_disint and len(cong_idx) > 0:
            excess = max(0, len(top_idx) - len(cong_idx))
            p_disint = min(p.p_max, excess / (p.disint_scale * len(cong_idx)))
            disint = bool(rng.random() < p_disint)
        elif len(cong_idx) > 0:
            disint = len(top_idx) > len(cong_idx)
        else:
            disint = False
        if disint:
            ns = min(len(cong_idx), len(top_idx))
            if ns > 0:
                cong_idx = rng.choice(top_idx, ns, replace=False)
                cong_nat = _congress_nature(cong_idx, mtype, regime, p, rng)
        disint_t[t] = int(disint)
        congress_t[t] = cong_nat

        # 3. Movement and collection.
        ax = pos_x[alive_idx]
        ay = pos_y[alive_idx]
        nx_all = (ax[:, None] + dx[None, :]) % p.l
        ny_all = (ay[:, None] + dy[None, :]) % p.a
        neigh_sp = grid[nx_all, ny_all] + rng.random((len(alive_idx), 9)) * 1e-9
        best = np.argmax(neigh_sp, axis=1)
        pos_x[alive_idx] = nx_all[np.arange(len(alive_idx)), best]
        pos_y[alive_idx] = ny_all[np.arange(len(alive_idx)), best]

        cell_cnt = np.zeros((p.l, p.a), dtype=np.int32)
        np.add.at(cell_cnt, (pos_x[alive_idx], pos_y[alive_idx]), 1)
        share_grid = np.where(cell_cnt > 0, grid / np.maximum(cell_cnt, 1), 0.0)
        E[alive_idx] += share_grid[pos_x[alive_idx], pos_y[alive_idx]]
        grid[cell_cnt > 0] = 0.0

        # 4. Interaction.
        eff = _effective_types(mtype, regime, t)
        cell_map = defaultdict(list)
        for i in alive_idx:
            cell_map[(pos_x[i], pos_y[i])].append(int(i))
        _resolve_interactions(cell_map, eff, mtype, E, faction, p, rng)

        # 5. Levy, taxation and redistribution.
        alive_idx = np.where(alive)[0]
        N_alive = len(alive_idx)
        levy = float(np.sum(E[alive_idx])) * p.plus_rate
        E[alive_idx] *= 1.0 - p.plus_rate
        if elite_nat == 0 and N_alive > 0:
            E[alive_idx] += levy / N_alive
        elif len(elite_idx) > 0:
            E[elite_idx] += levy / len(elite_idx)

        tax = float(np.sum(E[alive_idx])) * p.tau
        E[alive_idx] *= 1.0 - p.tau
        if cong_nat == 0 and N_alive > 0:
            for f in range(p.g):
                fi = alive_idx[faction[alive_idx] == f]
                if len(fi) > 0:
                    E[fi] += (len(fi) / N_alive * tax) / len(fi)
        else:
            if len(cong_idx) > 0:
                E[cong_idx] += tax * p.gamma_c / len(cong_idx)
            if len(elite_idx) > 0:
                E[elite_idx] += tax * (1.0 - p.gamma_c) / len(elite_idx)

        # 6. Subsistence, mortality and replacement.
        type_counts = np.bincount(mtype[alive_idx].astype(int), minlength=3).astype(float)
        type_probs = type_counts / max(type_counts.sum(), 1.0)

        E[alive_idx] -= p.c
        starved = np.where((E < 0.0) & alive)[0]
        deaths = len(starved)
        for i in starved:
            grid[pos_x[i], pos_y[i]] += max(0.0, E[i] + p.c)
            _respawn(i, rng, E, alive, age, pos_x, pos_y, faction, mtype,
                     prod_queue, p, type_probs)

        alive_idx = np.where(alive)[0]
        age[alive_idx] += 1
        old = np.where((age >= p.L) & alive)[0]
        deaths += len(old)
        for i in old:
            grid[pos_x[i], pos_y[i]] += max(0.0, E[i])
            _respawn(i, rng, E, alive, age, pos_x, pos_y, faction, mtype,
                     prod_queue, p, type_probs)

        alive_idx = np.where(alive)[0]
        N_alive = len(alive_idx)
        if N_alive == 0:
            break

        # 7. Planting decision.
        for i in alive_idx:
            if E[i] >= (p.t_c + 1) * p.c + p.q:
                output = int(np.floor(p.t_c * p.q))
                E[i] -= p.q
                prod_queue[i].append((t + p.t_c, float(output)))

        # 8. Preference adaptation.
        _adapt_preferences(alive_idx, mtype, faction, E, p, rng)

        # 9. Indicator calculation.
        alive_idx = np.where(alive)[0]
        N_alive = len(alive_idx)
        Ea = E[alive_idx]
        Y_t[t] = float(np.mean(Ea))
        G_t[t] = gini(Ea)
        alpha_t[t] = float(np.sum(mtype[alive_idx] == 0)) / N_alive
        deaths_t[t] = deaths
        mortality_t[t] = deaths / N_start
        N_t[t] = N_alive

    return {
        "Y": Y_t,
        "G": G_t,
        "alpha": alpha_t,
        "regime": regime_t,
        "congress": congress_t,
        "disint": disint_t,
        "poverty": poverty_t,
        "deaths": deaths_t,
        "mortality": mortality_t,
        "N": N_t,
    }
