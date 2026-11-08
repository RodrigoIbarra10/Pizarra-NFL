"""
Impacto de lesiones.

1) Valor de cada jugador = % de snaps promedio en los 4 partidos previos del equipo
   (0 en los partidos que no jugó -> un jugador en IR desde hace semanas ya vale ~0
   y no se cuenta doble, porque las métricas del equipo ya reflejan su ausencia).
2) "Pérdida" por grupo de posición = suma(valor x peso del estatus).
3) Cuántos puntos vale cada grupo se ESTIMA con datos: regresión del error del modelo
   (margen real - margen predicho, walk-forward 2016-2025) contra las pérdidas.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

DATA = Path(__file__).parent / "data"
TEAM_MAP = {"OAK": "LV", "SD": "LAC", "STL": "LA"}

GROUPS = {"RB": ["RB", "FB"], "WR": ["WR"], "TE": ["TE"], "OL": ["T", "G", "C", "OT", "OG", "OL"],
          "DL": ["DE", "DT", "NT", "DL"], "LB": ["LB", "OLB", "ILB", "MLB"],
          "DB": ["CB", "S", "FS", "SS", "DB"]}
POS2GRP = {p: g for g, ps in GROUPS.items() for p in ps}
OFF_GROUPS, DEF_GROUPS = ["RB", "WR", "TE", "OL"], ["DL", "LB", "DB"]
STATUS_W = {"Out": 1.0, "Inactive": 1.0, "Injured Reserve": 1.0, "Doubtful": 0.9, "Questionable": 0.2,
            "Suspension": 1.0, "Physically Unable to Perform": 1.0}
COEF_FILE = DATA / "injury_coefs.json"
# Respaldo (puntos por titular completo) si no se ha calibrado
DEFAULT = {"margin_off": 0.75, "margin_def": 0.95, "pts_off": 0.5, "pts_def": 0.5}


def _players():
    p = pd.read_parquet(DATA / "players.parquet", columns=["gsis_id", "pfr_id", "espn_id", "display_name", "position"])
    p["espn_id"] = p["espn_id"].astype("string").str.replace(r"\.0$", "", regex=True)
    return p


def load_snaps():
    fs = sorted(DATA.glob("snap_counts_*.parquet"))
    s = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    s["team"] = s["team"].replace(TEAM_MAP)
    s["grp"] = s["position"].map(POS2GRP)
    s["pct"] = s[["offense_pct", "defense_pct"]].max(axis=1)
    return s[s.grp.notna()]


def player_values(snaps, games):
    """Valor pre-partido de cada jugador para cada partido de su equipo (y para el próximo)."""
    g = games[["game_id", "gameday", "home_team", "away_team"]]
    tg = pd.concat([g.rename(columns={"home_team": "team"})[["game_id", "gameday", "team"]],
                    g.rename(columns={"away_team": "team"})[["game_id", "gameday", "team"]]])
    tg = tg.sort_values(["team", "gameday"])
    out = []
    for team, d in tg.groupby("team"):
        ids = d.game_id.tolist()
        s = snaps[snaps.team == team]
        mat = s.pivot_table(index="game_id", columns="pfr_player_id", values="pct", aggfunc="max")
        mat = mat.reindex(ids).fillna(0.0)
        played_mask = mat.sum(axis=1) > 0                       # partidos ya jugados con snaps
        roll = mat.where(played_mask, np.nan).shift(1).rolling(4, min_periods=1).mean().ffill()
        st = roll.stack()
        st = st[st > 0.15].rename("value").reset_index()
        st.columns = ["game_id", "pfr_player_id", "value"]
        st["team"] = team
        out.append(st)
    v = pd.concat(out, ignore_index=True)
    pos = snaps.drop_duplicates("pfr_player_id", keep="last")[["pfr_player_id", "grp", "player"]]
    return v.merge(pos, on="pfr_player_id", how="left")


def historic_losses(games):
    """Pérdidas por grupo para cada partido histórico, usando el reporte oficial de lesiones."""
    fs = [f for f in sorted(DATA.glob("injuries_*.parquet"))]
    inj = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    inj["team"] = inj["team"].replace(TEAM_MAP)
    inj = inj[inj.report_status.isin(STATUS_W)]
    inj["w"] = inj.report_status.map(STATUS_W)
    pl = _players()
    inj = inj.merge(pl[["gsis_id", "pfr_id"]], on="gsis_id", how="left").rename(columns={"pfr_id": "pfr_player_id"})
    # partido del equipo en esa semana
    gg = games[["game_id", "season", "week", "home_team", "away_team"]]
    tw = pd.concat([gg.rename(columns={"home_team": "team"})[["game_id", "season", "week", "team"]],
                    gg.rename(columns={"away_team": "team"})[["game_id", "season", "week", "team"]]])
    inj = inj.merge(tw, on=["season", "week", "team"], how="inner")
    vals = player_values(load_snaps(), games)
    x = inj.merge(vals, on=["game_id", "team", "pfr_player_id"], how="inner")
    x["loss"] = x["value"] * x["w"]
    return x.groupby(["game_id", "team", "grp"]).loss.sum().unstack(fill_value=0).reset_index()


def game_matrix(games, losses):
    """Une pérdidas de local y visitante a cada partido."""
    cols = list(GROUPS)
    L = losses.reindex(columns=["game_id", "team"] + cols, fill_value=0)
    h = L.rename(columns={"team": "home_team", **{c: f"h_{c}" for c in cols}})
    a = L.rename(columns={"team": "away_team", **{c: f"a_{c}" for c in cols}})
    m = games[["game_id", "home_team", "away_team"]].merge(h, on=["game_id", "home_team"], how="left") \
        .merge(a, on=["game_id", "away_team"], how="left").fillna(0)
    return m


def calibrate(bt_file=DATA / "backtest_predictions.csv"):
    import features
    games = features.load_games()
    bt = pd.read_csv(bt_file)
    bt = bt[bt.season.between(2016, 2025)]
    m = game_matrix(games, historic_losses(games)).merge(bt, on="game_id")
    m["res_margin"] = m.home_margin - m.pred_margin
    m["res_home"] = (m.game_total + m.home_margin) / 2 - m.pred_home_pts
    m["res_away"] = (m.game_total - m.home_margin) / 2 - m.pred_away_pts
    G = list(GROUPS)
    Xm = pd.DataFrame({g: m[f"a_{g}"] - m[f"h_{g}"] for g in G})      # + = visitante más golpeado

    # puntos anotados por un equipo ~ - pérdidas ofensivas propias + pérdidas defensivas rivales
    Xo = pd.concat([pd.DataFrame({**{f"off_{g}": m[f"h_{g}"] for g in OFF_GROUPS},
                                  **{f"def_{g}": m[f"a_{g}"] for g in DEF_GROUPS}}),
                    pd.DataFrame({**{f"off_{g}": m[f"a_{g}"] for g in OFF_GROUPS},
                                  **{f"def_{g}": m[f"h_{g}"] for g in DEF_GROUPS}})], ignore_index=True)
    yo = pd.concat([m.res_home, m.res_away], ignore_index=True)

    # Modelo agrupado (más estable que por posición): 2 coeficientes
    Om = sum(m[f"a_{g}"] - m[f"h_{g}"] for g in OFF_GROUPS)
    Dm = sum(m[f"a_{g}"] - m[f"h_{g}"] for g in DEF_GROUPS)
    Xm = pd.DataFrame({"off": Om, "def": Dm})
    tr = m.season <= 2022
    r_tr = LinearRegression(fit_intercept=False).fit(Xm[tr], m.res_margin[tr])
    mae0 = float(m.res_margin[~tr].abs().mean())
    mae1 = float((m.res_margin[~tr] - r_tr.predict(Xm[~tr])).abs().mean())
    rm = LinearRegression(fit_intercept=False).fit(Xm, m.res_margin)
    # puntos de un equipo: - pérdidas ofensivas propias, + pérdidas defensivas del rival
    Xo = pd.DataFrame({"off": pd.concat([sum(m[f"h_{g}"] for g in OFF_GROUPS), sum(m[f"a_{g}"] for g in OFF_GROUPS)]),
                       "def": pd.concat([sum(m[f"a_{g}"] for g in DEF_GROUPS), sum(m[f"h_{g}"] for g in DEF_GROUPS)])})
    yo = pd.concat([m.res_home, m.res_away])
    ro = LinearRegression(fit_intercept=False).fit(Xo, yo)
    coefs = {"margin_off": float(max(rm.coef_[0], 0)), "margin_def": float(max(rm.coef_[1], 0)),
             "pts_off": float(max(-ro.coef_[0], 0)), "pts_def": float(max(ro.coef_[1], 0)),
             "validacion_2023_2025": {"MAE_margen_sin_lesiones": mae0, "MAE_margen_con_lesiones": mae1,
                                      "partidos": int((~tr).sum())},
             "n_partidos": int(len(m))}
    COEF_FILE.write_text(json.dumps(coefs, indent=1))
    return coefs


def load_coefs():
    return json.loads(COEF_FILE.read_text()) if COEF_FILE.exists() else DEFAULT


# --------------------------------------------------------------------------- #
# Lesiones actuales (ESPN + reporte oficial nflverse)
# --------------------------------------------------------------------------- #
ESPN_ABBR = {"WSH": "WAS", "LAR": "LA", "JAX": "JAX"}


def current_injuries(season, week):
    rows = []
    f = DATA / "espn_inj.json"
    if f.exists():
        d = json.loads(f.read_text())
        for tm in d.get("injuries", []):
            for i in tm.get("injuries", []):
                a = i.get("athlete", {})
                ab = (a.get("team") or {}).get("abbreviation") or tm.get("abbreviation")
                href = " ".join(l.get("href", "") for l in a.get("links", []))
                import re
                mm = re.search(r"/id/(\d+)", href)
                rows.append({"espn_id": mm.group(1) if mm else None, "name": a.get("displayName"),
                             "team_name": tm.get("displayName"), "team_espn": ab,
                             "pos": (a.get("position") or {}).get("abbreviation"),
                             "status": ("Inactive" if (((i.get("details") or {}).get("fantasyStatus") or {})
                                        .get("abbreviation") == "INACTIVE") else i.get("status")),
                             "detail": i.get("shortComment") or "",
                             "date": i.get("date"), "src": "ESPN"})
    e = pd.DataFrame(rows)
    pl = _players()
    if not e.empty:
        pl1 = pl.dropna(subset=["espn_id"]).drop_duplicates("espn_id")
        e = e.merge(pl1[["espn_id", "gsis_id", "pfr_id"]], on="espn_id", how="left")
    # reporte oficial de la semana
    fi = DATA / f"injuries_{season}.parquet"
    o = pd.DataFrame()
    if fi.exists():
        o = pd.read_parquet(fi)
        o = o[(o.week == week) & o.report_status.notna()]
        o = o.rename(columns={"full_name": "name", "position": "pos", "report_status": "status",
                              "report_primary_injury": "detail"})[["gsis_id", "name", "team", "pos", "status", "detail"]]
        o["src"] = "Oficial"
        o = o.merge(pl.dropna(subset=["gsis_id"]).drop_duplicates("gsis_id")[["gsis_id", "pfr_id"]],
                    on="gsis_id", how="left")
    return e, o


def team_name_map(games):
    return None


def merge_status(espn, official, abbr_by_name, return_active=False):
    """Une ambas fuentes. ESPN es más reciente (inactivos del día del partido y jugadores ya
    habilitados = "Active"), así que tiene prioridad; el reporte oficial cubre a quien ESPN no lista."""
    if not espn.empty:
        espn = espn.copy()
        espn["team"] = espn.team_name.map(abbr_by_name).fillna(espn.team_espn.replace({"WSH": "WAS", "LAR": "LA"}))
    allr = pd.concat([espn, official], ignore_index=True)
    allr["key"] = allr.gsis_id.fillna(allr.name)
    allr["prio"] = (allr.src == "ESPN").astype(int)
    allr = allr.sort_values("prio", ascending=False).drop_duplicates(["key"])
    # Jugadores que el reporte oficial tenía en duda y ESPN ya marca como activos
    cleared = allr[(allr.status == "Active") & allr.key.isin(set(official.gsis_id.dropna()) if not official.empty else set())]
    allr = allr[allr.status.isin(STATUS_W)].copy()
    allr["w"] = allr.status.map(STATUS_W)
    return (allr, cleared) if return_active else allr


def current_losses(status, values_next):
    """values_next: valor de cada jugador para el próximo partido (team, pfr_player_id, value, grp)."""
    x = status.merge(values_next, left_on=["team", "pfr_id"], right_on=["team", "pfr_player_id"], how="inner")
    x["loss"] = x.value * x.w
    return x


# --------------------------------------------------------------------------- #
# Titulares probables (depth chart + lesiones/inactivos)
# --------------------------------------------------------------------------- #
OFF_SLOTS = ["QB", "RB", "WR", "TE", "LT", "LG", "C", "RG", "RT"]
POS_ES = {"QB": "QB", "RB": "RB", "WR": "WR", "TE": "TE", "LT": "Tackle izq.", "LG": "Guard izq.", "C": "Centro",
          "RG": "Guard der.", "RT": "Tackle der.", "FB": "FB"}


def probable_starters(season, teams, status):
    f = DATA / f"depth_charts_{season}.parquet"
    if not f.exists():
        return {}
    d = pd.read_parquet(f)
    d["team"] = d["team"].replace(TEAM_MAP)
    d = d[d.team.isin(teams)]
    d = d[d.dt == d.groupby("team").dt.transform("max")]
    d = d[d.pos_grp != "Special Teams"]
    st = dict(zip(status.gsis_id.dropna(), status.loc[status.gsis_id.notna(), "status"]))
    out = {}
    for team, t in d.groupby("team"):
        rows = []
        for (grp, slot), s in t.sort_values("pos_rank").groupby(["pos_grp", "pos_slot"], sort=False):
            s = s.sort_values("pos_rank")
            first = s.iloc[0]
            if first.pos_abb == "FB":
                continue
            side = "off" if "D" not in grp.split()[-1] else "def"
            avail = s[~s.gsis_id.map(lambda x: STATUS_W.get(st.get(x), 0) >= 0.9)]
            who = avail.iloc[0] if len(avail) else first
            fs = st.get(first.gsis_id)
            rows.append({"side": side, "slot": int(slot), "pos": first.pos_abb,
                         "player": who.player_name, "gsis": who.gsis_id,
                         "status": st.get(who.gsis_id) if who.gsis_id != first.gsis_id or fs == "Questionable" else None,
                         "replaces": first.player_name if who.gsis_id != first.gsis_id else None,
                         "replaced_status": fs if who.gsis_id != first.gsis_id else None})
        order = {p: i for i, p in enumerate(OFF_SLOTS)}
        rows.sort(key=lambda r: (r["side"] != "off", order.get(r["pos"], 50), r["slot"]))
        out[team] = rows
    return out
