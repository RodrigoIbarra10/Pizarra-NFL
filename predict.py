"""
Uso:
    python predict.py --update        # descarga datos frescos (nflverse + ESPN + clima) y predice
    python predict.py                 # usa datos locales de nflverse, pero trae ESPN/clima en vivo
    python predict.py --no-live       # sin ESPN/clima
    python predict.py --season 2026 --week 10
Salidas en ./output/: predicciones_*.csv, props_*.csv, reporte_*.md, semana.json
"""
import argparse, json, subprocess, warnings
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm

import features, model, props, injuries, live

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
DATA, OUT = ROOT / "data", ROOT / "output"
REL = "https://github.com/nflverse/nflverse-data/releases/download"
SEVERE = 0.9


def update_data(season):
    files = {"games.csv": "schedules/games.csv",
             f"pbp_{season}.parquet": f"pbp/play_by_play_{season}.parquet",
             f"pw_{season}.parquet": f"stats_player/stats_player_week_{season}.parquet",
             f"tw_{season}.parquet": f"stats_team/stats_team_week_{season}.parquet",
             f"injuries_{season}.parquet": f"injuries/injuries_{season}.parquet",
             f"snap_counts_{season}.parquet": f"snap_counts/snap_counts_{season}.parquet",
             f"depth_charts_{season}.parquet": f"depth_charts/depth_charts_{season}.parquet",
             "players.parquet": "players/players.parquet"}
    for f, path in files.items():
        print("descargando", f)
        r = subprocess.run(["curl", "-sSLf", "-o", str(DATA / f), f"{REL}/{path}"])
        if r.returncode:
            print("  aviso: no se pudo descargar", f)


def american(p):
    p = np.clip(p, 0.01, 0.99)
    return np.where(p >= 0.5, -100 * p / (1 - p), 100 * (1 - p) / p).round(0)


def ml_to_prob(ml):
    ml = np.asarray(ml, dtype=float)
    return np.where(ml < 0, -ml / (-ml + 100), 100 / (ml + 100))


# --------------------------------------------------------------------------- #
def replace_injured_qbs(target, status_by_id, season):
    """Si el QB titular del calendario está Out/IR/Doubtful, usar el siguiente del depth chart."""
    notes = []
    f = DATA / f"depth_charts_{season}.parquet"
    dc = pd.read_parquet(f) if f.exists() else pd.DataFrame()
    if not dc.empty:
        dc["team"] = dc["team"].replace(features.TEAM_MAP)
        dc = dc[dc.pos_abb == "QB"]
        dc = dc[dc.dt == dc.groupby("team").dt.transform("max")].sort_values(["team", "pos_rank"])
    t = target.copy()
    for i, r in t.iterrows():
        for side in ("home", "away"):
            qb, team = r[f"{side}_qb_id"], r[f"{side}_team"]
            st = status_by_id.get(qb)
            if st is None or injuries.STATUS_W.get(st, 0) < SEVERE or dc.empty:
                continue
            cands = dc[(dc.team == team) & (dc.gsis_id != qb)]
            cands = cands[~cands.gsis_id.map(lambda x: injuries.STATUS_W.get(status_by_id.get(x), 0) >= SEVERE)]
            if cands.empty:
                continue
            new = cands.iloc[0]
            notes.append(f"{team}: {r[f'{side}_qb_name']} ({st}) → {new.player_name}")
            t.at[i, f"{side}_qb_id"], t.at[i, f"{side}_qb_name"] = new.gsis_id, new.player_name
            t.at[i, f"{side}_qb_rating"] = features.qb_rating_now(new.gsis_id)
        t.at[i, "qb_diff"] = t.at[i, "home_qb_rating"] - t.at[i, "away_qb_rating"]
        t.at[i, "home_qb_vs_team"] = t.at[i, "home_qb_rating"] - t.at[i, "home_ew_off_pass"]
        t.at[i, "away_qb_vs_team"] = t.at[i, "away_qb_rating"] - t.at[i, "away_ew_off_pass"]
    return t, notes


def depth_chart_qbs(target, lineups):
    """Si el QB disponible #1 del depth chart de hoy difiere del QB del calendario, usar el del depth chart."""
    notes, t = [], target.copy()
    for i, r in t.iterrows():
        if r.get("played"):
            continue
        for side in ("home", "away"):
            qb = next((x for x in lineups.get(r[f"{side}_team"], []) if x["pos"] == "QB"), None)
            if qb is None or not isinstance(qb.get("gsis"), str) or qb["gsis"] == r[f"{side}_qb_id"]:
                continue
            notes.append(f"{r[f'{side}_team']}: {r[f'{side}_qb_name']} → {qb['player']} (depth chart de hoy)")
            t.at[i, f"{side}_qb_id"], t.at[i, f"{side}_qb_name"] = qb["gsis"], qb["player"]
            t.at[i, f"{side}_qb_rating"] = features.qb_rating_now(qb["gsis"])
        t.at[i, "qb_diff"] = t.at[i, "home_qb_rating"] - t.at[i, "away_qb_rating"]
        t.at[i, "home_qb_vs_team"] = t.at[i, "home_qb_rating"] - t.at[i, "home_ew_off_pass"]
        t.at[i, "away_qb_vs_team"] = t.at[i, "away_qb_rating"] - t.at[i, "away_ew_off_pass"]
    return t, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int)
    ap.add_argument("--week", type=int)
    ap.add_argument("--update", action="store_true")
    ap.add_argument("--no-live", action="store_true")
    a = ap.parse_args()

    if a.update:
        update_data(a.season or pd.Timestamp.today().year)
    df, _ = features.build_dataset()
    played, upcoming = df[df.played], df[~df.played]
    season = a.season or int(upcoming.season.min())
    week = a.week or int(upcoming[upcoming.season == season].week.min())
    target = df[(df.season == season) & (df.week == week)].copy()
    print(f"Entrenando con {len(played)} partidos; prediciendo {season} semana {week} ({len(target)} partidos)")

    # ---------------- Datos en vivo: ESPN + clima ----------------
    lv, hist = (pd.DataFrame(), pd.DataFrame())
    if not a.no_live:
        try:
            lv, hist = live.get_live(season, week)
        except Exception as e:
            print("Aviso: sin datos en vivo ->", e)
    if not lv.empty:
        target = target.merge(lv, on=["home_team", "away_team"], how="left")
        outdoor = target.indoor.eq(False)
        target.loc[outdoor & target.w_temp.notna(), "temp"] = target.w_temp
        target.loc[outdoor & target.w_wind.notna(), "wind"] = target.w_wind
        target.loc[target.indoor.eq(True), ["temp", "wind", "dome"]] = [65, 0, 1]

    # ---------------- Lesiones ----------------
    abbr_by_name = {}
    if not lv.empty:
        abbr_by_name = {**dict(zip(lv.home_name, lv.home_team)), **dict(zip(lv.away_name, lv.away_team))}
    espn_inj, off_inj = injuries.current_injuries(season, week)
    status, cleared = injuries.merge_status(espn_inj, off_inj, abbr_by_name, return_active=True)
    status_by_id = dict(zip(status.gsis_id.dropna(), status.loc[status.gsis_id.notna(), "status"]))
    target, qb_notes = replace_injured_qbs(target, status_by_id, season)
    lineups = injuries.probable_starters(season, set(target.home_team) | set(target.away_team), status)
    target, qb_notes2 = depth_chart_qbs(target, lineups)
    qb_notes += qb_notes2

    # ---------------- Modelo ----------------
    m = model.NFLModel().fit(played)
    pr = m.predict(target)
    pr = pr.merge(target.drop(columns=[c for c in pr.columns if c != "game_id"], errors="ignore"), on="game_id")

    # Ajuste por lesiones (coeficientes calibrados con 10 años de reportes)
    co = injuries.load_coefs()
    vals = injuries.player_values(injuries.load_snaps(), features.load_games())
    vals = vals[vals.game_id.isin(pr.game_id)]
    lost = injuries.current_losses(status, vals)
    lost["side"] = np.where(lost.grp.isin(injuries.OFF_GROUPS), "off", "def")
    agg = lost.groupby(["team", "side"]).loss.sum().unstack(fill_value=0).reindex(columns=["off", "def"], fill_value=0)
    get = lambda t, s: float(agg.loc[t, s]) if t in agg.index else 0.0
    d_m, d_h, d_a, key_inj = [], [], [], []
    for r in pr.itertuples():
        ho, hd, ao, ad = get(r.home_team, "off"), get(r.home_team, "def"), get(r.away_team, "off"), get(r.away_team, "def")
        d_m.append(co["margin_off"] * (ao - ho) + co["margin_def"] * (ad - hd))
        d_h.append(-co["pts_off"] * ho + co["pts_def"] * ad)
        d_a.append(-co["pts_off"] * ao + co["pts_def"] * hd)
        k = lost[lost.team.isin([r.home_team, r.away_team]) & (lost.loss >= 0.3)].sort_values("loss", ascending=False)
        key_inj.append([{"team": x.team, "player": x.player if isinstance(x.player, str) else x["name"],
                         "pos": x.grp, "status": x.status, "snaps": round(float(x.value), 2)}
                        for _, x in k.iterrows()])
    pr["inj_margin_adj"], pr["inj_key"] = d_m, key_inj
    pr["pred_margin_base"], pr["pred_total_base"], pr["p_home_base"] = pr.pred_margin, pr.pred_total, pr.p_home
    pr["pred_margin"] = pr.pred_margin + pr.inj_margin_adj
    pr["pred_total"] = pr.pred_total + np.array(d_h) + np.array(d_a)
    pr["p_home"] = norm.cdf(norm.ppf(pr.p_home.clip(0.01, 0.99)) + pr.inj_margin_adj / pr.sd_margin)
    pr["pred_home_pts"] = (pr.pred_total + pr.pred_margin) / 2
    pr["pred_away_pts"] = (pr.pred_total - pr.pred_margin) / 2

    # ---------------- Mercado: usar línea en vivo si existe ----------------
    if "spread_close" in pr:
        pr["spread_line"] = pr.spread_close.fillna(pr.spread_line)
        pr["total_line"] = pr.total_close.fillna(pr.total_line)
    pr["favorito"] = np.where(pr.p_home >= 0.5, pr.home_team, pr.away_team)
    pr["prob_favorito"] = np.maximum(pr.p_home, 1 - pr.p_home)
    pr["moneyline_justa_local"] = american(pr.p_home)
    pr["p_mercado_local"] = model.market_prob(pr.spread_line)
    if "home_ml_close" in pr:
        ph, pa = ml_to_prob(pr.home_ml_close), ml_to_prob(pr.away_ml_close)
        pr["p_mercado_local"] = np.where(np.isnan(ph), pr.p_mercado_local, ph / (ph + pa))  # sin vig
    pr["edge_spread"] = pr.pred_margin - pr.spread_line
    pr["p_local_cubre"] = 1 - norm.cdf(pr.spread_line, pr.pred_margin, pr.sd_margin)
    pr["edge_total"] = pr.pred_total - pr.total_line
    pr["p_over"] = 1 - norm.cdf(pr.total_line, pr.pred_total, pr.sd_total)

    # ---------------- Props ----------------
    p = props.player_form(props.load_players())
    dfac = props.defense_factors(p)
    _, sd_ratio = props.backtest_props(p, dfac, seasons=(season - 1, season))
    pp = props.project_games(pr, p, dfac, _team_ppg(played, season), sd_ratio)
    pp = props.apply_injuries(pp, status_by_id)
    removed = pp.attrs.get("removed", [])

    OUT.mkdir(exist_ok=True)
    tag = f"{season}_w{week:02d}"
    pr.drop(columns=["inj_key"]).to_csv(OUT / f"predicciones_{tag}.csv", index=False)
    pp.to_csv(OUT / f"props_{tag}.csv", index=False)
    (OUT / f"reporte_{tag}.md").write_text(render_md(pr, pp, season, week, qb_notes))
    js = to_json(pr, pp, hist, status, season, week, qb_notes, removed, co, sd_ratio)
    js['lineups'] = {k: [{kk: vv for kk, vv in r.items() if kk != 'gsis'} for r in v] for k, v in lineups.items()}
    js['cleared'] = cleared[['team', 'name', 'pos']].fillna('').to_dict('records')
    (OUT / "semana.json").write_text(json.dumps(_sanitize(js), ensure_ascii=False, default=_clean, allow_nan=False))
    print(render_md(pr, pp, season, week, qb_notes)[:3000])


def _sanitize(o):
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if np.isnan(o) else float(o)
    return o


def _clean(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    return str(o)


def _nz(v, n=1):
    try:
        v = float(v)
        return None if np.isnan(v) else round(v, n)
    except (TypeError, ValueError):
        return None


def _team_ppg(played, season):
    s = played[played.season == season]
    h = s.groupby("home_team").home_score.agg(["sum", "count"])
    a = s.groupby("away_team").away_score.agg(["sum", "count"])
    t = h.add(a, fill_value=0)
    return (t["sum"] / t["count"]).to_dict()


def to_json(pr, pp, hist, status, season, week, qb_notes, removed, co, sd_ratio=None):
    bt = pd.read_csv(DATA / "backtest_predictions.csv")
    rep = model.report(bt).round(4).to_dict("records")
    _, ats, _ = model.score_report(bt)
    games = []
    for r in pr.sort_values(["gameday", "game_id"]).itertuples():
        g = {k: getattr(r, k, None) for k in ["game_id", "away_team", "home_team", "away_qb_name", "home_qb_name",
                                                 "venue", "city", "country", "kickoff_utc", "status", "book"]}
        g.update({k: _nz(getattr(r, k, None), 3) for k in [
            "p_home", "p_home_base", "p_logit", "p_gbm", "p_margin", "p_mercado_local", "pred_margin",
            "pred_total", "pred_home_pts", "pred_away_pts", "pred_margin_base", "pred_total_base",
            "inj_margin_adj", "spread_line", "total_line", "spread_open", "spread_close", "total_open",
            "total_close", "home_ml_open", "home_ml_close", "away_ml_open", "away_ml_close", "edge_spread",
            "p_local_cubre", "edge_total", "p_over", "w_temp", "w_wind", "w_gust", "w_precip_prob", "w_precip",
            "w_snow", "home_elo", "away_elo", "home_qb_rating", "away_qb_rating", "home_ew_off_epa",
            "home_ew_def_epa", "away_ew_off_epa", "away_ew_def_epa", "home_s_wins", "home_s_losss",
            "away_s_wins", "away_s_losss", "home_streak", "away_streak", "home_rest", "away_rest",
            "h2h_home_winpct", "h2h_n", "home_score_live", "away_score_live", "moneyline_justa_local"]})
        g["indoor"] = bool(getattr(r, "indoor", False)) if not pd.isna(getattr(r, "indoor", np.nan)) else None
        g["weather"] = live.WMO.get(int(r.w_code), "") if not pd.isna(getattr(r, "w_code", np.nan)) else ""
        g["injuries"] = r.inj_key
        g["home_logo"], g["away_logo"] = getattr(r, "home_logo", None), getattr(r, "away_logo", None)
        if not hist.empty:
            h = hist[(hist.home_team == r.home_team) & (hist.away_team == r.away_team)]
            g["line_history"] = h[["ts", "spread_close", "total_close"]].to_dict("records")
        g["props"] = [{k: (_nz(v, 2) if isinstance(v, (float, np.floating)) else v) for k, v in row.items()
                       if k not in ("game", "opp", "player_id")}
                      for row in pp[pp.game == f"{r.away_team} @ {r.home_team}"].to_dict("records")]
        games.append(g)
    inj_list = status[status.team.isin(set(pr.home_team) | set(pr.away_team))][
        ["team", "name", "pos", "status", "detail", "src"]].sort_values(["team", "status"])
    return {"season": season, "week": week,
            "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "games": games, "qb_changes": qb_notes, "props_removed": removed,
            "injury_list": inj_list.fillna("").to_dict("records"),
            "backtest": rep, "ats": ats.round(1).to_dict("records"),
            "injury_coefs": {k: v for k, v in co.items() if k != "n_partidos"},
            "sd_ratio": {f"{k[0]}|{k[1]}": round(v, 3) for k, v in (sd_ratio or {}).items()}}


def render_md(pr, pp, season, week, qb_notes):
    L = [f"# Predicciones NFL {season} — Semana {week}\n"]
    if qb_notes:
        L += ["**Cambios de QB por lesión:** " + "; ".join(qb_notes) + "\n"]
    L += ["| Partido | QBs | Favorito | Prob. | Marcador | Spread modelo | Línea (apertura→actual) | Total modelo | O/U | Clima | Ajuste lesiones |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in pr.sort_values("gameday").itertuples():
        so = getattr(r, "spread_open", np.nan)
        line = f"{r.home_team} {-r.spread_line:+.1f}" + (f" (abrió {-so:+.1f})" if not pd.isna(so) else "")
        wx = "Techado" if getattr(r, "indoor", False) is True else (
            f"{r.w_temp:.0f}°F, viento {r.w_wind:.0f} mph, lluvia {r.w_precip_prob:.0f}%" if not pd.isna(getattr(r, "w_temp", np.nan)) else "")
        L.append(f"| {r.away_team} @ {r.home_team} | {r.away_qb_name} vs {r.home_qb_name} | **{r.favorito}** | "
                 f"{100*r.prob_favorito:.0f}% | {r.away_team} {r.pred_away_pts:.0f} - {r.pred_home_pts:.0f} {r.home_team} | "
                 f"{r.home_team} {-r.pred_margin:+.1f} | {line} | {r.pred_total:.1f} | {r.total_line} | {wx} | "
                 f"{r.inj_margin_adj:+.1f} |")
    return "\n".join(L)


if __name__ == "__main__":
    main()
