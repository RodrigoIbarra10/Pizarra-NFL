"""
Construcción de la base de datos de variables (features) para el modelo NFL.

Fuentes (nflverse):
  - games.csv                : calendario, resultados, descanso, líneas, clima
  - play_by_play_{y}.parquet : EPA, éxito, yardas, sacks, INT, fumbles por jugada

Todas las variables "pre-partido" se calculan SOLO con información anterior al
partido (shift(1)) para evitar fuga de información (data leakage).
"""
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"
FIRST_PBP = 2012

# Franquicias que se mudaron -> una sola identidad para rachas/tendencias
TEAM_MAP = {"OAK": "LV", "SD": "LAC", "STL": "LA"}


# --------------------------------------------------------------------------- #
# 1. Carga
# --------------------------------------------------------------------------- #
def load_games() -> pd.DataFrame:
    g = pd.read_csv(DATA / "games.csv")
    for c in ("home_team", "away_team"):
        g[c] = g[c].replace(TEAM_MAP)
    g["gameday"] = pd.to_datetime(g["gameday"])
    g = g.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    g["played"] = g["home_score"].notna()
    return g


def load_pbp_team_games() -> pd.DataFrame:
    """Agrega play-by-play a nivel equipo-partido (ofensiva)."""
    cols = ["game_id", "posteam", "defteam", "play_type", "epa", "wpa", "success",
            "yards_gained", "pass", "rush", "sack", "interception",
            "fumble_lost", "touchdown", "qb_dropback", "down", "third_down_converted",
            "third_down_failed", "yardline_100", "drive"]
    out = []
    for y in range(FIRST_PBP, 2100):
        f = DATA / f"pbp_{y}.parquet"
        if not f.exists():
            break
        p = pd.read_parquet(f, columns=cols)
        p = p[p["posteam"].notna() & p["epa"].notna()
              & p["play_type"].isin(["pass", "run"])]
        p["is_pass"] = (p["pass"] == 1).astype(int)
        p["is_rush"] = (p["rush"] == 1).astype(int)
        p["pass_epa"] = np.where(p.is_pass == 1, p.epa, np.nan)
        p["rush_epa"] = np.where(p.is_rush == 1, p.epa, np.nan)
        p["third"] = (p["down"] == 3).astype(int)
        p["rz"] = (p["yardline_100"] <= 20).astype(int)
        agg = p.groupby(["game_id", "posteam", "defteam"]).agg(
            plays=("epa", "size"),
            epa_play=("epa", "mean"),
            pass_epa=("pass_epa", "mean"),
            rush_epa=("rush_epa", "mean"),
            success=("success", "mean"),
            yards=("yards_gained", "sum"),
            pass_rate=("is_pass", "mean"),
            sacks_taken=("sack", "sum"),
            ints_thrown=("interception", "sum"),
            fumbles_lost=("fumble_lost", "sum"),
            third_conv=("third_down_converted", "sum"),
            third_att=("third", "sum"),
        ).reset_index()
        out.append(agg)
    tg = pd.concat(out, ignore_index=True)
    tg["posteam"] = tg["posteam"].replace(TEAM_MAP)
    tg["defteam"] = tg["defteam"].replace(TEAM_MAP)
    tg["giveaways"] = tg["ints_thrown"] + tg["fumbles_lost"]
    tg["third_pct"] = tg["third_conv"] / tg["third_att"].replace(0, np.nan)
    return tg


# --------------------------------------------------------------------------- #
# 2. Elo (estilo FiveThirtyEight: margen de victoria + regresión a la media)
# --------------------------------------------------------------------------- #
def compute_elo(g: pd.DataFrame, k=20.0, hfa=48.0, revert=1 / 3) -> pd.DataFrame:
    elo, last_season = {}, None
    pre_h, pre_a = [], []
    for r in g.itertuples(index=False):
        if r.season != last_season:  # inicio temporada: regresar a 1505
            elo = {t: v * (1 - revert) + 1505 * revert for t, v in elo.items()}
            last_season = r.season
        eh, ea = elo.get(r.home_team, 1505.0), elo.get(r.away_team, 1505.0)
        pre_h.append(eh); pre_a.append(ea)
        if not r.played:
            continue
        adv = 0.0 if r.location == "Neutral" else hfa
        dr = eh + adv - ea
        exp_h = 1 / (1 + 10 ** (-dr / 400))
        mov = r.home_score - r.away_score
        res = 1.0 if mov > 0 else 0.0 if mov < 0 else 0.5
        mult = np.log(abs(mov) + 1) * 2.2 / ((dr if mov > 0 else -dr) * 0.001 + 2.2)
        delta = k * mult * (res - exp_h)
        elo[r.home_team] = eh + delta
        elo[r.away_team] = ea - delta
    g = g.copy()
    g["home_elo"], g["away_elo"] = pre_h, pre_a
    return g


# --------------------------------------------------------------------------- #
# 2b. Calificación de QB (EPA/dropback con decaimiento + contracción bayesiana)
# --------------------------------------------------------------------------- #
QB_PRIOR, QB_K, QB_DECAY = -0.05, 200.0, 0.93


QB_STATE = {}   # calificación actual de cada QB (para sustituir titulares lesionados)


def qb_rating_now(qb_id):
    S, N = QB_STATE.get(qb_id, (0.0, 0.0))
    return (S + QB_PRIOR * QB_K) / (N + QB_K)


def qb_ratings(g: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for y in range(FIRST_PBP - 0, 2100):
        f = DATA / f"pbp_{y}.parquet"
        if not f.exists():
            break
        p = pd.read_parquet(f, columns=["game_id", "passer_id", "qb_epa", "qb_dropback"])
        p = p[(p.qb_dropback == 1) & p.passer_id.notna() & p.qb_epa.notna()]
        rows.append(p.groupby(["game_id", "passer_id"]).agg(n=("qb_epa", "size"),
                                                            s=("qb_epa", "sum")).reset_index())
    q = pd.concat(rows).merge(g[["game_id", "gameday"]], on="game_id").sort_values(["gameday", "game_id"])
    state = {}                      # qb -> (S, N)
    by_game = {gid: d for gid, d in q.groupby("game_id")}
    home_r, away_r = [], []
    rating = lambda qb: (lambda S, N: (S + QB_PRIOR * QB_K) / (N + QB_K))(*state.get(qb, (0.0, 0.0)))
    for r in g.itertuples(index=False):
        home_r.append(rating(r.home_qb_id) if isinstance(r.home_qb_id, str) else np.nan)
        away_r.append(rating(r.away_qb_id) if isinstance(r.away_qb_id, str) else np.nan)
        d = by_game.get(r.game_id)
        if d is not None:
            for qb, n, s_ in zip(d.passer_id, d.n, d.s):
                S, N = state.get(qb, (0.0, 0.0))
                state[qb] = (S * QB_DECAY + s_, N * QB_DECAY + n)
    g = g.copy()
    g["home_qb_rating"], g["away_qb_rating"] = home_r, away_r
    QB_STATE.clear(); QB_STATE.update(state)
    return g


# --------------------------------------------------------------------------- #
# 3. Tabla larga equipo-partido + variables pre-partido
# --------------------------------------------------------------------------- #
def team_long(g: pd.DataFrame, tg: pd.DataFrame) -> pd.DataFrame:
    base = ["game_id", "season", "week", "game_type", "gameday", "played"]
    h = g[base + ["home_team", "away_team", "home_score", "away_score", "home_rest", "location"]].copy()
    h.columns = base + ["team", "opp", "pf", "pa", "rest", "location"]
    h["is_home"] = (h["location"] != "Neutral").astype(int)
    a = g[base + ["away_team", "home_team", "away_score", "home_score", "away_rest", "location"]].copy()
    a.columns = base + ["team", "opp", "pf", "pa", "rest", "location"]
    a["is_home"] = 0
    a["is_away"] = (a["location"] != "Neutral").astype(int)
    h["is_away"] = 0
    L = pd.concat([h, a], ignore_index=True)

    off = tg.rename(columns={"posteam": "team", "defteam": "opp"})
    off_cols = ["epa_play", "pass_epa", "rush_epa", "success", "yards", "pass_rate",
                "sacks_taken", "ints_thrown", "giveaways", "third_pct", "plays"]
    L = L.merge(off[["game_id", "team"] + off_cols], on=["game_id", "team"], how="left")
    d = off[["game_id", "team"] + off_cols].rename(
        columns={"team": "opp", **{c: f"d_{c}" for c in off_cols}})
    L = L.merge(d, on=["game_id", "opp"], how="left")  # lo que permitió la defensa
    L = L.rename(columns={"d_sacks_taken": "sacks_made", "d_ints_thrown": "ints_made",
                          "d_giveaways": "takeaways"})

    L["margin"] = L["pf"] - L["pa"]
    L["win"] = np.where(L.played, (L.margin > 0).astype(float) + 0.5 * (L.margin == 0), np.nan)
    L["loss"] = np.where(L.played, (L.margin < 0).astype(float), np.nan)
    L["tie"] = np.where(L.played, (L.margin == 0).astype(float), np.nan)
    L["to_margin"] = L["takeaways"] - L["giveaways"]
    L = L.sort_values(["team", "gameday", "game_id"]).reset_index(drop=True)

    grp = L.groupby("team", group_keys=False)
    sgrp = L.groupby(["team", "season"], group_keys=False)

    # --- Rendimiento temporada actual (antes del partido) ---
    for c in ["win", "loss", "tie"]:
        L[f"s_{c}s"] = sgrp[c].transform(lambda s: s.shift(1).fillna(0).cumsum())
    gp = L["s_wins"] + L["s_losss"] + L["s_ties"]
    L["s_games"] = gp
    L["s_winpct"] = np.where(gp > 0, (L["s_wins"] + 0.5 * L["s_ties"]) / gp.replace(0, 1), np.nan)

    # Promedios temporada-a-la-fecha (ofensiva/defensiva)
    std_cols = {"pf": "s_pf", "pa": "s_pa", "yards": "s_yds", "d_yards": "s_yds_allowed",
                "epa_play": "s_off_epa", "d_epa_play": "s_def_epa", "sacks_made": "s_sacks",
                "ints_made": "s_ints", "to_margin": "s_to_margin",
                "success": "s_off_sr", "d_success": "s_def_sr"}
    for c, n in std_cols.items():
        L[n] = sgrp[c].transform(lambda s: s.shift(1).expanding().mean())
    L["s_pf_total"] = sgrp["pf"].transform(lambda s: s.shift(1).fillna(0).cumsum())

    # Récord local / visitante (temporada)
    for loc in ["is_home", "is_away"]:
        w = L["win"] * L[loc]
        n = L[loc] * L["played"].astype(int)
        L[f"_w_{loc}"], L[f"_n_{loc}"] = w, n
        sg = L.groupby(["team", "season"])
        cw = sg[f"_w_{loc}"].transform(lambda s: s.shift(1).fillna(0).cumsum())
        cn = sg[f"_n_{loc}"].transform(lambda s: s.shift(1).fillna(0).cumsum())
        tag = "home" if loc == "is_home" else "away"
        L[f"s_{tag}_wins"] = cw
        L[f"s_{tag}_games"] = cn
        L[f"s_{tag}_winpct"] = np.where(cn > 0, cw / cn.replace(0, 1), np.nan)
    L = L.drop(columns=[c for c in L.columns if c.startswith("_")])

    # --- Tendencias últimos 3/5/10 (cruzan temporadas) ---
    for n in (3, 5, 10):
        for c, nm in [("margin", "mov"), ("pf", "pf"), ("pa", "pa"), ("win", "win"),
                      ("epa_play", "off_epa"), ("d_epa_play", "def_epa")]:
            L[f"l{n}_{nm}"] = grp[c].transform(lambda s: s.shift(1).rolling(n, min_periods=1).mean())

    # --- EWMA (forma ponderada, cruza temporadas) ---
    ew_cols = {"epa_play": "ew_off_epa", "d_epa_play": "ew_def_epa", "pass_epa": "ew_off_pass",
               "rush_epa": "ew_off_rush", "d_pass_epa": "ew_def_pass", "d_rush_epa": "ew_def_rush",
               "success": "ew_off_sr", "d_success": "ew_def_sr", "pf": "ew_pf", "pa": "ew_pa",
               "margin": "ew_mov", "sacks_made": "ew_sacks", "sacks_taken": "ew_sacks_taken",
               "to_margin": "ew_to", "yards": "ew_yds", "d_yards": "ew_yds_allowed",
               "plays": "ew_plays", "pass_rate": "ew_pass_rate"}
    for c, n in ew_cols.items():
        L[n] = grp[c].transform(lambda s: s.shift(1).ewm(span=10, min_periods=1).mean())

    # --- Rachas ---
    def streak(s):
        out, cur = [], 0
        for v in s.shift(1):
            out.append(cur)
            if pd.isna(v):
                continue
            if v == 1:
                cur = cur + 1 if cur > 0 else 1
            elif v == 0:
                cur = cur - 1 if cur < 0 else -1
            else:
                cur = 0
        return pd.Series(out, index=s.index)
    L["streak"] = sgrp["win"].transform(streak)  # + victorias seguidas, - derrotas seguidas

    # --- Descanso / calendario ---
    L["prev_gameday"] = grp["gameday"].shift(1)
    L["days_rest"] = (L["gameday"] - L["prev_gameday"]).dt.days.clip(upper=21)
    L["rest"] = L["rest"].fillna(L["days_rest"]).fillna(7)
    L["short_week"] = (L["rest"] <= 5).astype(int)
    L["off_bye"] = (L["rest"] >= 12).astype(int)
    away_flag = (1 - L["is_home"])
    L["consec_road"] = _consec(L, away_flag)

    # --- Temporada anterior / playoffs ---
    reg = L[L.game_type == "REG"].groupby(["team", "season"]).agg(
        prev_winpct=("win", "mean"), prev_mov=("margin", "mean")).reset_index()
    po = L[(L.game_type != "REG") & L.played].groupby(["team", "season"]).agg(
        prev_po_games=("win", "size"), prev_po_wins=("win", "sum")).reset_index()
    prev = reg.merge(po, on=["team", "season"], how="left").fillna({"prev_po_games": 0, "prev_po_wins": 0})
    prev["prev_playoffs"] = (prev["prev_po_games"] > 0).astype(int)
    prev["season"] += 1
    L = L.merge(prev, on=["team", "season"], how="left")
    return L


def _consec(L, away_flag):
    """Partidos consecutivos como visitante (incluyendo el actual)."""
    res = pd.Series(0, index=L.index)
    for _, idx in L.groupby("team").groups.items():
        cnt = 0
        for i in idx:
            cnt = cnt + 1 if away_flag[i] == 1 else 0
            res[i] = cnt
    return res


def head_to_head(g: pd.DataFrame, years=10) -> pd.DataFrame:
    """Historial directo: % victorias del local y margen medio en últimos enfrentamientos."""
    rec = {}
    h2h_w, h2h_m, h2h_n = [], [], []
    for r in g.itertuples(index=False):
        key = tuple(sorted([r.home_team, r.away_team]))
        hist = [x for x in rec.get(key, []) if x[0] >= r.season - years]
        if hist:
            # orientado al equipo local de este partido
            ms = [m if t == r.home_team else -m for _, t, m in hist]
            h2h_w.append(np.mean([m > 0 for m in ms])); h2h_m.append(np.mean(ms)); h2h_n.append(len(ms))
        else:
            h2h_w.append(np.nan); h2h_m.append(np.nan); h2h_n.append(0)
        if r.played:
            rec.setdefault(key, []).append((r.season, r.home_team, r.home_score - r.away_score))
    g = g.copy()
    g["h2h_home_winpct"], g["h2h_home_mov"], g["h2h_n"] = h2h_w, h2h_m, h2h_n
    return g


# --------------------------------------------------------------------------- #
# 4. Tabla final a nivel partido (local vs visitante)
# --------------------------------------------------------------------------- #
TEAM_FEATS = [
    "s_games", "s_wins", "s_losss", "s_winpct", "s_pf", "s_pa", "s_pf_total", "s_yds", "s_yds_allowed",
    "s_off_epa", "s_def_epa", "s_off_sr", "s_def_sr", "s_sacks", "s_ints", "s_to_margin",
    "s_home_winpct", "s_away_winpct",
    "l3_mov", "l5_mov", "l10_mov", "l3_win", "l5_win", "l10_win", "l3_pf", "l5_pf", "l10_pf",
    "l3_pa", "l5_pa", "l10_pa", "l3_off_epa", "l5_off_epa", "l10_off_epa",
    "l3_def_epa", "l5_def_epa", "l10_def_epa",
    "ew_off_epa", "ew_def_epa", "ew_off_pass", "ew_off_rush", "ew_def_pass", "ew_def_rush",
    "ew_off_sr", "ew_def_sr", "ew_pf", "ew_pa", "ew_mov", "ew_sacks", "ew_sacks_taken", "ew_to",
    "ew_yds", "ew_yds_allowed", "ew_plays", "ew_pass_rate",
    "streak", "rest", "short_week", "off_bye", "consec_road",
    "prev_winpct", "prev_mov", "prev_playoffs", "prev_po_wins",
]


def build_dataset() -> pd.DataFrame:
    g = load_games()
    g = compute_elo(g)
    g = head_to_head(g)
    g = qb_ratings(g)
    tg = load_pbp_team_games()
    L = team_long(g, tg)

    keep = ["game_id", "team"] + TEAM_FEATS
    H = L[keep].rename(columns={c: f"home_{c}" for c in TEAM_FEATS}).rename(columns={"team": "home_team"})
    A = L[keep].rename(columns={c: f"away_{c}" for c in TEAM_FEATS}).rename(columns={"team": "away_team"})
    g = g.drop(columns=["home_rest", "away_rest"])
    df = g.merge(H, on=["game_id", "home_team"], how="left").merge(A, on=["game_id", "away_team"], how="left")

    # Diferencias local - visitante
    diffs = pd.DataFrame({f"diff_{c}": df[f"home_{c}"] - df[f"away_{c}"] for c in TEAM_FEATS})
    df = pd.concat([df, diffs], axis=1).copy()
    df["elo_diff"] = df["home_elo"] - df["away_elo"]
    df["qb_diff"] = df["home_qb_rating"] - df["away_qb_rating"]
    # QB titular vs. nivel reciente del ataque aéreo del equipo (detecta cambios por lesión)
    df["home_qb_vs_team"] = df["home_qb_rating"] - df["home_ew_off_pass"]
    df["away_qb_vs_team"] = df["away_qb_rating"] - df["away_ew_off_pass"]
    df["neutral"] = (df["location"] == "Neutral").astype(int)
    df["dome"] = df["roof"].isin(["dome", "closed"]).astype(int)
    df["wind"] = df["wind"].fillna(0)
    df["temp"] = df["temp"].fillna(65)
    df["playoff"] = (df["game_type"] != "REG").astype(int)
    df["home_margin"] = df["home_score"] - df["away_score"]
    df["game_total"] = df["home_score"] + df["away_score"]
    df["home_win"] = np.where(df.played, (df.home_margin > 0).astype(float), np.nan)
    return df[df.season >= FIRST_PBP].reset_index(drop=True), L


if __name__ == "__main__":
    df, L = build_dataset()
    df.to_parquet(DATA / "model_dataset.parquet")
    L.to_parquet(DATA / "team_games.parquet")
    print(df.shape, "partidos;", df.played.sum(), "jugados")
