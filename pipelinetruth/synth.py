"""Synthetic CRM generator with deliberate real-world messiness.

The generator knows the true data-generating process (a logistic win model with account, owner,
lead-source and engagement effects, quarter-end seasonality, and a mid-2024 drift event), so the
rest of the system can be judged against a world where "honest limits" are verifiable.

Mess injected on purpose: missing amounts / lead sources / industries, non-positive and 20x
outlier amounts, stale CRM close dates, duplicate accounts under name variants, duplicate
opportunity rows, close dates before creation dates, and orphan activity rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config

SEGMENTS = ["SMB", "MidMarket", "Enterprise"]
SEG_P = [0.55, 0.37, 0.08]
SEG_LOG_AMOUNT = {"SMB": (np.log(8_000), 0.5), "MidMarket": (np.log(35_000), 0.5), "Enterprise": (np.log(150_000), 0.6)}
SEG_EFFECT = {"SMB": 0.2, "MidMarket": 0.0, "Enterprise": -0.4}
SEG_CYCLE = {"SMB": 35, "MidMarket": 70, "Enterprise": 120}
SOURCES = ["Inbound", "Outbound", "Partner", "Referral", "Event"]
SOURCE_P = [0.30, 0.25, 0.15, 0.15, 0.15]
SOURCE_EFFECT = {"Inbound": 0.1, "Outbound": -0.4, "Partner": 0.5, "Referral": 0.7, "Event": -0.1}
PARTNER_EFFECT_AFTER_DRIFT = -0.5  # concept drift: partner-sourced deals stop converting
ACTIVITY_MULT_AFTER_DRIFT = 0.7    # covariate drift: CRM logging change reduces logged activity
INDUSTRIES = ["Software", "Finance", "Health", "Retail", "Manufacturing", "Education"]
REGIONS = ["NA", "EMEA", "APAC", "LATAM"]
STAGES = ["Qualify", "Develop", "Propose", "Negotiate"]
_ADJ = ["Northwind", "Blue", "Apex", "Summit", "Cobalt", "Harbor", "Ember", "Lumen", "Vertex", "Orchid",
        "Granite", "Nimbus", "Atlas", "Cedar", "Falcon", "Helix", "Ionic", "Juniper", "Kestrel", "Lattice",
        "Meridian", "Nova", "Onyx", "Pioneer", "Quartz", "Redwood", "Sierra", "Tidal", "Umber", "Zenith"]
_NOUN = ["Dynamics", "Systems", "Logistics", "Foods", "Analytics", "Robotics", "Media", "Energy", "Capital",
         "Health", "Networks", "Materials", "Labs", "Freight", "Retail", "Software", "Textiles", "Optics",
         "Bio", "Motors", "Cloud", "Metals", "Learning", "Travel", "Security", "Water", "Power", "Design",
         "Pharma", "Marine"]
_TAIL = ["", "Group", "Partners", "Holdings", "Global"]
_SUFFIX = ["Inc", "Corp", "LLC"]
_VARIANT = {"Inc": ["INC.", "Incorporated", "inc"], "Corp": ["Corporation", "Corp.", "CORP"], "LLC": ["L.L.C.", "llc", "LLC."]}


def _account_names(rng: np.random.Generator, n: int) -> list[tuple[str, str]]:
    # (adjective, noun, tail) is unique, so the canonical key (name minus legal suffix) is unique too
    combos = [(a, b, t) for a in _ADJ for b in _NOUN for t in _TAIL]
    idx = rng.choice(len(combos), size=n, replace=False)
    return [(f"{combos[i][0]} {combos[i][1]} {combos[i][2]}".strip(), str(rng.choice(_SUFFIX))) for i in idx]


def _logistic(x):
    return 1 / (1 + np.exp(-x))


def generate(cfg: Config | None = None, n_opps: int | None = None, seed: int | None = None) -> dict[str, pd.DataFrame]:
    """Return raw CRM tables: accounts, opportunities, activities, stage_history."""
    cfg = cfg or Config()
    n = n_opps or cfg.n_opps
    rng = np.random.default_rng(cfg.seed if seed is None else seed)

    # ---------------------------------------------------------------- accounts (with duplicates)
    n_acc = max(60, int(n * 0.26))
    names = _account_names(rng, n_acc)
    quality = rng.normal(0, 0.5, n_acc)
    acc_rows, dup_of, canon_of_idx = [], {}, {}
    next_id = 1
    for i, (base, suf) in enumerate(names):
        industry = None if rng.random() < 0.05 else str(rng.choice(INDUSTRIES))
        region = str(rng.choice(REGIONS, p=[0.5, 0.25, 0.15, 0.10]))
        acc_rows.append((next_id, f"{base} {suf}", industry, region))
        canon_of_idx[i] = next_id
        next_id += 1
        if rng.random() < 0.07:  # duplicate account record under a name variant
            variant = str(rng.choice(_VARIANT[suf]))
            nm = f"{base.upper() if rng.random() < 0.3 else base} {variant}"
            acc_rows.append((next_id, nm + (" " if rng.random() < 0.3 else ""), industry, region))
            dup_of[i] = next_id
            next_id += 1
    accounts = pd.DataFrame(acc_rows, columns=["account_id", "account_name", "industry", "region"])

    # ---------------------------------------------------------------- opportunity skeleton
    days_total = (cfg.as_of - cfg.start_date).days
    day_grid = np.arange(days_total + 1)
    dates = cfg.start_date + pd.to_timedelta(day_grid, unit="D")
    month_season = np.array([1.25, 1.0, 1.15, 1.0, 0.95, 1.2, 0.85, 0.9, 1.15, 1.0, 1.0, 0.8])
    growth = 1 + 0.15 * day_grid / 365
    w = month_season[dates.month - 1] * growth
    w = w / w.sum()
    created_idx = rng.choice(day_grid, size=n, p=w)
    created = (cfg.start_date + pd.to_timedelta(created_idx, unit="D")).to_numpy()

    seg = rng.choice(SEGMENTS, size=n, p=SEG_P)
    src = rng.choice(SOURCES, size=n, p=SOURCE_P)
    mu = np.array([SEG_LOG_AMOUNT[s][0] for s in seg])
    sd = np.array([SEG_LOG_AMOUNT[s][1] for s in seg])
    log_amt = rng.normal(mu, sd)
    amount = np.exp(log_amt).round(2)
    owner = rng.integers(1, 41, n)
    owner_skill = rng.normal(0, 0.4, 40)[owner - 1]
    eng = rng.normal(0, 1, n)

    acct_idx = rng.choice(n_acc, size=n, p=(lambda x: x / x.sum())(np.exp(rng.normal(0, 0.9, n_acc))))
    acct_id = np.array([canon_of_idx[i] for i in acct_idx])
    has_dup = np.array([i in dup_of for i in acct_idx])
    use_dup = has_dup & (rng.random(n) < 0.5)
    acct_id = np.where(use_dup, [dup_of.get(i, 0) for i in acct_idx], acct_id)

    after = created >= np.datetime64(cfg.drift_start)
    src_eff = np.array([SOURCE_EFFECT[s] for s in src])
    src_eff = np.where(after & (src == "Partner"), PARTNER_EFFECT_AFTER_DRIFT, src_eff)
    cq = pd.DatetimeIndex(created).quarter.to_numpy()
    logit = (-1.1 + np.array([SEG_EFFECT[s] for s in seg]) + src_eff - 0.25 * (log_amt - mu)
             + 0.7 * eng + owner_skill + quality[acct_idx] + 0.25 * (cq == 4))
    p_win = _logistic(logit)
    won = rng.random(n) < p_win

    # ---------------------------------------------------------------- timing
    mean_cycle = np.array([SEG_CYCLE[s] for s in seg], dtype=float) * np.where(won, 1.0, 0.8)
    cycle = rng.gamma(4.0, mean_cycle / 4.0)
    cycle = np.clip(cycle, cfg.snapshot_days + 1, cfg.max_cycle_days).round().astype(int)
    closed = pd.DatetimeIndex(created) + pd.to_timedelta(cycle, unit="D")
    # quarter-end pull: some won deals slip/accelerate to land in the last week of the quarter
    q_end = closed.to_period("Q").end_time.normalize()
    gap = (q_end - closed).days
    pull = won & (gap <= 25) & (rng.random(n) < 0.45)
    closed = closed.where(~pull, q_end - pd.to_timedelta(rng.integers(0, 7, n), unit="D"))
    cycle = (closed - pd.DatetimeIndex(created)).days.to_numpy()
    cycle = np.clip(cycle, cfg.snapshot_days + 1, cfg.max_cycle_days)
    closed = pd.DatetimeIndex(created) + pd.to_timedelta(cycle, unit="D")

    is_open = closed > cfg.as_of
    status = np.where(is_open, "open", np.where(won, "won", "lost"))

    # CRM expected close date: noisy truth, but 12% are stale defaults (created + 30d), 5% missing
    exp_close = closed + pd.to_timedelta(rng.normal(0, 20, n).round(), unit="D")
    stale = rng.random(n) < 0.12
    exp_close = exp_close.where(~stale, pd.DatetimeIndex(created) + pd.Timedelta(days=30))
    exp_close_s = pd.Series(exp_close)
    exp_close_s[rng.random(n) < 0.05] = pd.NaT

    closed_s = pd.Series(closed)
    closed_s[is_open] = pd.NaT

    # ---------------------------------------------------------------- injected mess
    amount_s = pd.Series(amount)
    amount_s[rng.random(n) < 0.05] = np.nan
    amount_s[rng.random(n) < 0.005] = 0.0
    out = rng.random(n) < 0.01
    amount_s[out] = amount_s[out] * 20
    src_s = pd.Series(src, dtype=object)
    src_s[rng.random(n) < 0.04] = None
    bad_close = (~is_open) & (rng.random(n) < 0.005)
    closed_s[bad_close] = pd.Series(created)[bad_close] - pd.to_timedelta(rng.integers(1, 10, bad_close.sum()), unit="D")

    opps = pd.DataFrame({
        "opp_id": [f"OPP-{i:06d}" for i in range(1, n + 1)],
        "account_id": acct_id, "owner_id": owner, "segment": seg, "lead_source": src_s,
        "amount": amount_s, "created_date": pd.to_datetime(created),
        "expected_close_date": exp_close_s, "status": status, "closed_date": closed_s,
    })
    opps = opps.sort_values("created_date").reset_index(drop=True)
    # latent arrays are indexed by original opp number; re-align them to the date-sorted order
    base_idx = opps["opp_id"].str[4:].astype(int).to_numpy() - 1
    eng_s, cycle_s, won_s = eng[base_idx], cycle[base_idx], won[base_idx]
    after_s = after[base_idx]

    # ---------------------------------------------------------------- activities
    m = np.where(after_s, ACTIVITY_MULT_AFTER_DRIFT, 1.0)
    horizon = np.minimum(cycle_s, (cfg.as_of - opps["created_date"]).dt.days.to_numpy())
    n_early = rng.poisson(np.exp(1.9 + 0.45 * eng_s) * m)
    n_late = rng.poisson(np.exp(0.6 + 0.3 * eng_s) * m * np.maximum(horizon - cfg.snapshot_days, 0) / 7)
    rows = []
    for cnt, lo, hi in ((n_early, 0, cfg.snapshot_days), (n_late, cfg.snapshot_days, None)):
        idx = np.repeat(np.arange(n), cnt)
        top = (np.full(n, hi) if hi is not None else np.maximum(horizon, cfg.snapshot_days + 1))[idx]
        day = lo + (rng.random(len(idx)) * (top - lo)).astype(int)
        rows.append(pd.DataFrame({"i": idx, "day": day}))
    act = pd.concat(rows, ignore_index=True)
    act = act[act["day"] <= horizon[act["i"].to_numpy()]].reset_index(drop=True)
    e_row = eng_s[act["i"].to_numpy()]
    n_contacts = 1 + rng.poisson(1.0 * np.exp(0.3 * e_row))
    kind = np.where(rng.random(len(act)) < np.clip(0.25 + 0.05 * e_row, 0.05, 0.6), "meeting",
                    np.where(rng.random(len(act)) < 0.5, "email", "call"))
    activities = pd.DataFrame({
        "activity_id": np.arange(1, len(act) + 1),
        "opp_id": opps["opp_id"].to_numpy()[act["i"].to_numpy()],
        "activity_date": opps["created_date"].to_numpy()[act["i"].to_numpy()] + pd.to_timedelta(act["day"].to_numpy(), unit="D"),
        "activity_type": kind,
        "contact_no": (rng.random(len(act)) * n_contacts).astype(int) + 1,
    })
    orphans = activities.sample(40, random_state=1).copy()
    orphans["opp_id"] = "OPP-9" + orphans["opp_id"].str[5:]
    orphans["activity_id"] = np.arange(len(activities) + 1, len(activities) + 41)
    activities = pd.concat([activities, orphans], ignore_index=True)

    # ---------------------------------------------------------------- stage history
    u = 0.7 * eng_s + rng.normal(0, 1, n)
    s14 = 1 + (u > -0.3).astype(int) + (u > 0.7) + (u > 1.7)
    created_arr = opps["created_date"].to_numpy()
    st_rows = [pd.DataFrame({"opp_id": opps["opp_id"], "stage": "Qualify", "entered_date": created_arr})]
    reach = np.ones(n, dtype=bool)
    for k_ in (2, 3, 4):
        early = s14 >= k_
        d_early = np.minimum(1 + ((k_ - 1) * cfg.snapshot_days / s14).astype(int), cfg.snapshot_days - 1)
        d_late = cfg.snapshot_days + ((cycle_s - cfg.snapshot_days) * (k_ - 1) / 4).astype(int)
        # late stage progression: wins always progress, losses stall with prob 0.4 per stage
        progress = early | won_s | (rng.random(n) < 0.6)
        reach = reach & progress
        day = np.where(early, d_early, d_late)
        full_date = created_arr + pd.to_timedelta(day, unit="D").to_numpy()
        keep = reach & (full_date <= np.where(opps["closed_date"].isna(), np.datetime64(cfg.as_of), opps["closed_date"].to_numpy()))
        st_rows.append(pd.DataFrame({"opp_id": opps["opp_id"][keep], "stage": STAGES[k_ - 1],
                                     "entered_date": full_date[keep]}))
    stage_history = pd.concat(st_rows, ignore_index=True)

    # duplicate opportunity rows (double-submitted records)
    dups = opps.sample(frac=0.004, random_state=2)
    opps = pd.concat([opps, dups], ignore_index=True)

    return {"accounts": accounts, "opportunities": opps, "activities": activities, "stage_history": stage_history}


def write_raw(tables: dict[str, pd.DataFrame], raw_dir) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(raw_dir / f"{name}.csv", index=False)
