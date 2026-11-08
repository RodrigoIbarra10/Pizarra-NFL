"""
Proyecciones de jugadores para apuestas de props.

Proyección = forma del jugador (EWMA, cruza temporadas)
           x ajuste por defensa rival (lo que permite a esa posición vs. promedio liga)
           x ajuste por guion del partido (puntos proyectados del equipo vs. su promedio)
Distribución: Normal (yardas, con desviación proporcional a la media estimada en backtest)
              Poisson (TDs, intercepciones, recepciones)
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm, poisson

DATA = Path(__file__).parent / "data"
TEAM_MAP = {"OAK": "LV", "SD": "LAC", "STL": "LA"}

STATS = {
    "QB": ["attempts", "completions", "passing_yards", "passing_tds", "passing_interceptions",
           "rushing_yards"],
    "RB": ["carries", "rushing_yards", "rushing_tds", "receptions", "receiving_yards"],
    "WR": ["targets", "receptions", "receiving_yards", "receiving_tds"],
    "TE": ["targets", "receptions", "receiving_yards", "receiving_tds"],
}
COUNT_STATS = {"passing_tds", "passing_interceptions", "rushing_tds", "receiving_tds"}
# exponente del ajuste por guion de partido (puntos proyectados del equipo)
SCRIPT_EXP = {"passing_tds": 0.8, "rushing_tds": 0.8, "receiving_tds": 0.8, "passing_yards": 0.3,
              "receiving_yards": 0.3, "rushing_yards": 0.3}
SPAN = 8
# Priors por intento (nivel QB suplente) y tasa de TD por toque (liga)
QB_PRIOR = {"completions": 0.60, "passing_yards": 6.0, "passing_tds": 0.032, "passing_interceptions": 0.03}
QB_K = 120.0
TD_PER_REC, TD_PER_CARRY = 0.062, 0.030


def load_players() -> pd.DataFrame:
    fs = sorted(DATA.glob("pw_*.parquet"))
    p = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    p = p[p.position.isin(STATS)].copy()
    for c in ("team", "opponent_team"):
        p[c] = p[c].replace(TEAM_MAP)
    g = pd.read_csv(DATA / "games.csv", usecols=["game_id", "gameday"])
    p = p.merge(g, on="game_id", how="left")
    p["gameday"] = pd.to_datetime(p["gameday"])
    return p.sort_values(["player_id", "gameday"]).reset_index(drop=True)


def all_stats():
    return sorted({s for v in STATS.values() for s in v})


def player_form(p: pd.DataFrame) -> pd.DataFrame:
    """EWMA por jugador ANTES de cada partido (shift) y también incluyendo el último."""
    grp = p.groupby("player_id")
    for s in all_stats():
        p[f"f_{s}"] = grp[s].transform(lambda x: x.shift(1).ewm(span=SPAN, min_periods=1).mean())
        p[f"now_{s}"] = grp[s].transform(lambda x: x.ewm(span=SPAN, min_periods=1).mean())
    p["n_games_prior"] = grp.cumcount()
    # Eficiencia por intento del QB (contraída a nivel suplente si tiene poca muestra)
    qb = p.position == "QB"
    p["cum_att"] = grp["attempts"].cumsum().clip(upper=600)
    for s, prior in QB_PRIOR.items():
        ew_s = grp[s].transform(lambda x: x.ewm(span=16, min_periods=1).mean())
        ew_a = grp["attempts"].transform(lambda x: x.ewm(span=16, min_periods=1).mean())
        raw = (ew_s / ew_a.replace(0, np.nan)).fillna(prior)
        p[f"rate_{s}"] = np.where(qb, (raw * p.cum_att + prior * QB_K) / (p.cum_att + QB_K), np.nan)
    # Intentos de pase esperados por equipo (EWMA)
    ta = p[qb].groupby(["team", "game_id", "gameday"]).attempts.sum().reset_index().sort_values("gameday")
    ta["team_att"] = ta.groupby("team").attempts.transform(lambda x: x.ewm(span=8).mean())
    p.attrs["team_att"] = ta.groupby("team").team_att.last().to_dict()
    return p


def defense_factors(p: pd.DataFrame) -> pd.DataFrame:
    """Lo que cada defensa permite por posición, relativo al promedio de la liga (EWMA)."""
    allowed = p.groupby(["game_id", "gameday", "opponent_team", "position"])[all_stats()].sum().reset_index()
    allowed = allowed.sort_values("gameday")
    lg = allowed.groupby(["position"])[all_stats()].transform(lambda x: x.rolling(400, min_periods=20).mean())
    out = allowed[["game_id", "gameday", "opponent_team", "position"]].copy()
    grp = allowed.groupby(["opponent_team", "position"])
    for s in all_stats():
        ew_pre = grp[s].transform(lambda x: x.shift(1).ewm(span=12, min_periods=3).mean())
        ew_now = grp[s].transform(lambda x: x.ewm(span=12, min_periods=3).mean())
        base = lg[s].replace(0, np.nan)
        out[f"dpre_{s}"] = (ew_pre / base).clip(0.6, 1.5)
        out[f"dnow_{s}"] = (ew_now / base).clip(0.6, 1.5)
    return out


def _project(form, dfac, script, stat):
    d = np.sqrt(dfac.fillna(1.0))                     # amortiguado
    sc = script.fillna(1.0) ** SCRIPT_EXP.get(stat, 0.15)
    return form * d * sc


# --------------------------------------------------------------------------- #
def backtest_props(p, dfac, seasons=(2024, 2025, 2026)):
    """Compara la proyección contra el promedio simple de la temporada (baseline)."""
    x = p.merge(dfac[["game_id", "opponent_team", "position"] + [c for c in dfac if c.startswith("dpre_")]],
                on=["game_id", "opponent_team", "position"], how="left")
    x = x[x.season.isin(seasons) & (x.n_games_prior >= 4)]
    rows, sd_ratio = [], {}
    for pos, stats in STATS.items():
        xp = x[x.position == pos]
        if pos == "QB":
            xp = xp[xp.attempts >= 15]
        elif pos == "RB":
            xp = xp[xp.f_carries >= 6]
        else:
            xp = xp[xp.f_targets >= 3]
        for s in stats:
            proj = _project(xp[f"f_{s}"], xp[f"dpre_{s}"], pd.Series(1.0, index=xp.index), s)
            base = xp.groupby(["player_id", "season"])[s].transform(lambda v: v.shift(1).expanding().mean()) \
                     .fillna(xp[f"f_{s}"])
            ok = proj.notna() & base.notna()
            mae_m = (xp.loc[ok, s] - proj[ok]).abs().mean()
            mae_b = (xp.loc[ok, s] - base[ok]).abs().mean()
            resid = xp.loc[ok, s] - proj[ok]
            sd_ratio[(pos, s)] = float(resid.std() / max(proj[ok].mean(), 1e-6))
            rows.append({"pos": pos, "stat": s, "n": int(ok.sum()), "MAE_modelo": mae_m,
                         "MAE_prom_temporada": mae_b, "mejora_%": 100 * (1 - mae_m / mae_b)})
    return pd.DataFrame(rows), sd_ratio


def project_games(games: pd.DataFrame, p: pd.DataFrame, dfac: pd.DataFrame, team_ppg: dict,
                  sd_ratio: dict) -> pd.DataFrame:
    """games: filas con away_team, home_team, pred_home_pts, pred_away_pts, home_qb_id, away_qb_id."""
    last = p.sort_values("gameday").groupby("player_id").tail(1)
    latest_def = dfac.sort_values("gameday").groupby(["opponent_team", "position"]).tail(1)
    season_now = p.season.max()
    out = []
    for g in games.itertuples(index=False):
        for team, opp, pts, qb in [(g.home_team, g.away_team, g.pred_home_pts, g.home_qb_id),
                                   (g.away_team, g.home_team, g.pred_away_pts, g.away_qb_id)]:
            roster = last[(last.team == team) & (last.season == season_now)]
            script = pts / team_ppg.get(team, pts)
            for pos, stats in STATS.items():
                r = roster[roster.position == pos]
                if pos == "QB":
                    if isinstance(qb, str) and (last.player_id == qb).any():
                        r = last[last.player_id == qb]          # titular anunciado (aunque sea nuevo en el equipo)
                    else:
                        r = r.sort_values("now_attempts", ascending=False).head(1)
                elif pos == "RB":
                    r = r[r.now_carries >= 5].sort_values("now_carries", ascending=False).head(2)
                else:
                    r = r[r.now_targets >= 3].sort_values("now_targets", ascending=False).head(3 if pos == "WR" else 1)
                dd = latest_def[(latest_def.opponent_team == opp) & (latest_def.position == pos)]
                for pl in r.itertuples(index=False):
                    rec = {"game": f"{g.away_team} @ {g.home_team}", "team": team, "opp": opp,
                           "player_id": pl.player_id,
                           "player": pl.player_display_name, "pos": pos}
                    for s in stats:
                        fac = dd[f"dnow_{s}"].iloc[0] if len(dd) else 1.0
                        rec[s] = float(_project(pd.Series([getattr(pl, f"now_{s}")]),
                                                pd.Series([fac]), pd.Series([script]), s).iloc[0])
                    if pos == "QB":
                        att = props_team_att(p, team) * (script ** 0.1)
                        rec["attempts"] = att
                        for s_, _ in QB_PRIOR.items():
                            fac = dd[f"dnow_{s_}"].iloc[0] if len(dd) else 1.0
                            rec[s_] = float(att * getattr(pl, f"rate_{s_}") * np.sqrt(fac)
                                            * script ** SCRIPT_EXP.get(s_, 0.0))
                    else:
                        if "rushing_tds" in rec:
                            rec["rushing_tds"] = 0.5 * rec["rushing_tds"] + 0.5 * rec["carries"] * TD_PER_CARRY
                        rec["receiving_tds"] = 0.5 * rec.get("receiving_tds", 0) + 0.5 * rec["receptions"] * TD_PER_REC
                        lam = rec.get("rushing_tds", 0) + rec.get("receiving_tds", 0)
                        rec["p_anytime_td"] = 1 - np.exp(-lam)
                    out.append(rec)
    return pd.DataFrame(out)


REC_STATS = ["targets", "receptions", "receiving_yards", "receiving_tds"]
RUSH_STATS = ["carries", "rushing_yards", "rushing_tds"]


def apply_injuries(pp: pd.DataFrame, status: dict, absorb=0.7) -> pd.DataFrame:
    """Quita jugadores Out/IR/Doubtful y reparte su volumen entre compañeros proyectados."""
    pp = pp.copy()
    pp["status"] = pp.player_id.map(status).fillna("")
    out_mask = pp.status.isin(["Out", "Inactive", "Injured Reserve", "Doubtful", "Suspension",
                               "Physically Unable to Perform"]) & (pp.pos != "QB")
    removed = pp[out_mask]
    pp = pp[~out_mask]
    for team, d in removed.groupby("team"):
        mates = (pp.team == team) & (pp.pos != "QB")
        vt = d.targets.fillna(0).sum()
        if vt > 0 and pp.loc[mates, "targets"].fillna(0).sum() > 0:
            f = 1 + absorb * vt / pp.loc[mates, "targets"].fillna(0).sum()
            pp.loc[mates, REC_STATS] = pp.loc[mates, REC_STATS] * f
        vc = d.carries.fillna(0).sum()
        rb = mates & (pp.pos == "RB")
        if vc > 0 and pp.loc[rb, "carries"].fillna(0).sum() > 0:
            f = 1 + absorb * vc / pp.loc[rb, "carries"].fillna(0).sum()
            pp.loc[rb, RUSH_STATS] = pp.loc[rb, RUSH_STATS] * f
    nq = pp.pos != "QB"
    pp.loc[nq, "p_anytime_td"] = 1 - np.exp(-(pp.loc[nq, "rushing_tds"].fillna(0) + pp.loc[nq, "receiving_tds"].fillna(0)))
    pp.attrs["removed"] = removed[["player", "team", "pos", "status"]].to_dict("records")
    return pp


def props_team_att(p, team):
    return p.attrs.get("team_att", {}).get(team, 33.0)


def prob_over(proj, line, pos, stat, sd_ratio):
    """Probabilidad de superar una línea de prop."""
    if stat in COUNT_STATS or stat in ("receptions", "completions", "targets", "carries", "attempts"):
        if stat in COUNT_STATS:
            return float(1 - poisson.cdf(np.floor(line), proj))
    sd = sd_ratio.get((pos, stat), 0.5) * proj
    return float(1 - norm.cdf(line, loc=proj, scale=max(sd, 1e-6)))


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    p = player_form(load_players())
    dfac = defense_factors(p)
    res, sd = backtest_props(p, dfac)
    print(res.round(2).to_string(index=False))
