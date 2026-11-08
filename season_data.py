"""
Datos de la temporada actual para la página:
  - Resultados semana a semana con estadísticas de equipo y de jugadores (box score)
  - Tabla de posiciones por división y líderes de la temporada
  - Calendario de las semanas siguientes
Salida: output/temporada.json
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
DATA, OUT = ROOT / "data", ROOT / "output"
TEAM_MAP = {"OAK": "LV", "SD": "LAC", "STL": "LA"}

DIVISIONS = {
    "AFC Este": ["BUF", "MIA", "NE", "NYJ"], "AFC Norte": ["BAL", "CIN", "CLE", "PIT"],
    "AFC Sur": ["HOU", "IND", "JAX", "TEN"], "AFC Oeste": ["DEN", "KC", "LV", "LAC"],
    "NFC Este": ["DAL", "NYG", "PHI", "WAS"], "NFC Norte": ["CHI", "DET", "GB", "MIN"],
    "NFC Sur": ["ATL", "CAR", "NO", "TB"], "NFC Oeste": ["ARI", "LA", "SF", "SEA"],
}
TEAM_NAMES = {
    "ARI": "Cardinals", "ATL": "Falcons", "BAL": "Ravens", "BUF": "Bills", "CAR": "Panthers", "CHI": "Bears",
    "CIN": "Bengals", "CLE": "Browns", "DAL": "Cowboys", "DEN": "Broncos", "DET": "Lions", "GB": "Packers",
    "HOU": "Texans", "IND": "Colts", "JAX": "Jaguars", "KC": "Chiefs", "LV": "Raiders", "LAC": "Chargers",
    "LA": "Rams", "MIA": "Dolphins", "MIN": "Vikings", "NE": "Patriots", "NO": "Saints", "NYG": "Giants",
    "NYJ": "Jets", "PHI": "Eagles", "PIT": "Steelers", "SF": "49ers", "SEA": "Seahawks", "TB": "Buccaneers",
    "TEN": "Titans", "WAS": "Commanders"}


def r1(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), 1)


def i0(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else int(round(float(v)))


def build(season=None, last_week=None):
    g = pd.read_csv(DATA / "games.csv")
    for c in ("home_team", "away_team"):
        g[c] = g[c].replace(TEAM_MAP)
    season = season or int(g[g.home_score.isna()].season.min())
    s = g[(g.season == season) & (g.game_type == "REG")]
    played = s[s.home_score.notna()]
    if not last_week:   # última semana COMPLETA (todos sus partidos jugados)
        done = s.groupby("week").home_score.apply(lambda x: x.notna().all())
        last_week = int(done[done].index.max())
    played = played[played.week <= last_week]

    tw = pd.read_parquet(DATA / f"tw_{season}.parquet")
    pw = pd.read_parquet(DATA / f"pw_{season}.parquet")
    for df in (tw, pw):
        for c in ("team", "opponent_team"):
            df[c] = df[c].replace(TEAM_MAP)

    tw["turnovers"] = tw.passing_interceptions + tw.fumbles_lost_total.fillna(
        tw.rushing_fumbles_lost + tw.receiving_fumbles_lost + tw.sack_fumbles_lost)
    tw["first_downs"] = tw.passing_first_downs + tw.rushing_first_downs
    tw["total_yards"] = tw.passing_yards - tw.sack_yards_lost.fillna(0) + tw.rushing_yards

    games = []
    for r in played.sort_values(["week", "gameday", "gametime"]).itertuples():
        box = {}
        for side, team in (("away", r.away_team), ("home", r.home_team)):
            t = tw[(tw.game_id == r.game_id) & (tw.team == team)]
            if t.empty:
                continue
            t = t.iloc[0]
            box[side] = {
                "yds": i0(t.total_yards), "pass": i0(t.passing_yards), "rush": i0(t.rushing_yards),
                "fd": i0(t.first_downs), "to": i0(t.turnovers), "sacks": r1(t.def_sacks),
                "pen": f"{i0(t.penalties)}-{i0(t.penalty_yards)}",
                "comp": f"{i0(t.completions)}/{i0(t.attempts)}", "car": i0(t.carries),
                "epa": r1((t.passing_epa or 0) + (t.rushing_epa or 0)),
            }
        pl = pw[pw.game_id == r.game_id]
        qbs = pl[pl.attempts >= 5].sort_values("attempts", ascending=False)
        rush = pl[pl.carries >= 3].sort_values("rushing_yards", ascending=False)
        rec = pl[pl.targets >= 2].sort_values("receiving_yards", ascending=False)
        players = {
            "pass": [[x.player_display_name, x.team, f"{i0(x.completions)}/{i0(x.attempts)}", i0(x.passing_yards),
                      i0(x.passing_tds), i0(x.passing_interceptions), i0(x.sacks_suffered)] for x in qbs.itertuples()],
            "rush": [[x.player_display_name, x.team, x.position, i0(x.carries), i0(x.rushing_yards),
                      r1(x.rushing_yards / x.carries) if x.carries else None, i0(x.rushing_tds)] for x in rush.itertuples()],
            "rec": [[x.player_display_name, x.team, x.position, i0(x.receptions), i0(x.targets),
                     i0(x.receiving_yards), i0(x.receiving_tds)] for x in rec.itertuples()],
            "def": [[x.player_display_name, x.team, x.position, i0(x.def_tackles_solo + x.def_tackle_assists),
                     r1(x.def_sacks), i0(x.def_interceptions)]
                    for x in pl[(pl.def_sacks > 0) | (pl.def_interceptions > 0)]
                    .sort_values(["def_sacks", "def_interceptions"], ascending=False).head(8).itertuples()],
        }
        hs, as_ = int(r.home_score), int(r.away_score)
        games.append({
            "id": r.game_id, "week": int(r.week), "date": r.gameday, "day": r.weekday, "time": r.gametime,
            "away": r.away_team, "home": r.home_team, "as": as_, "hs": hs, "ot": bool(r.overtime),
            "stadium": r.stadium, "neutral": r.location == "Neutral",
            "spread": r1(r.spread_line), "total": r1(r.total_line),
            # resultado vs. apuestas
            "cover": None if pd.isna(r.spread_line) else
            ("home" if hs - as_ > r.spread_line else "away" if hs - as_ < r.spread_line else "push"),
            "ou": None if pd.isna(r.total_line) else
            ("over" if hs + as_ > r.total_line else "under" if hs + as_ < r.total_line else "push"),
            "box": box, "players": players,
        })

    # ---- Posiciones ----
    rows = []
    for side, opp in (("home", "away"), ("away", "home")):
        d = played[[f"{side}_team", f"{side}_score", f"{opp}_score", f"{opp}_team"]].copy()
        d.columns = ["team", "pf", "pa", "opp"]
        d["side"] = side
        rows.append(d)
    L = pd.concat(rows)
    L["w"] = (L.pf > L.pa).astype(int); L["l"] = (L.pf < L.pa).astype(int); L["t"] = (L.pf == L.pa).astype(int)
    div_of = {t: d for d, ts in DIVISIONS.items() for t in ts}
    L["div_game"] = L.apply(lambda x: div_of.get(x.team) == div_of.get(x.opp), axis=1)
    order = played.sort_values(["week", "gameday"])
    standings = []
    for div, teams in DIVISIONS.items():
        tbl = []
        for t in teams:
            d = L[L.team == t]
            seq = []
            for x in order.itertuples():
                if t in (x.home_team, x.away_team):
                    my, op = (x.home_score, x.away_score) if x.home_team == t else (x.away_score, x.home_score)
                    seq.append("G" if my > op else "P" if my < op else "E")
            streak = ""
            if seq:
                k = 1
                while k < len(seq) and seq[-k - 1] == seq[-1]:
                    k += 1
                streak = f"{seq[-1]}{k}"
            dd = d[d.div_game]
            tbl.append({"team": t, "name": TEAM_NAMES.get(t, t), "w": int(d.w.sum()), "l": int(d.l.sum()),
                        "t": int(d.t.sum()), "pf": int(d.pf.sum()), "pa": int(d.pa.sum()),
                        "home": f"{int(d[d.side=='home'].w.sum())}-{int(d[d.side=='home'].l.sum())}",
                        "away": f"{int(d[d.side=='away'].w.sum())}-{int(d[d.side=='away'].l.sum())}",
                        "div": f"{int(dd.w.sum())}-{int(dd.l.sum())}", "streak": streak, "last5": seq[-5:]})
        tbl.sort(key=lambda x: (-(x["w"] + 0.5 * x["t"]) / max(x["w"] + x["l"] + x["t"], 1), -(x["pf"] - x["pa"])))
        standings.append({"div": div, "teams": tbl})

    # ---- Líderes de la temporada ----
    pws = pw[pw.week <= last_week]
    agg = pws.groupby(["player_id", "player_display_name", "position"]).agg(
        team=("team", "last"), gp=("game_id", "nunique"),
        pass_yds=("passing_yards", "sum"), pass_td=("passing_tds", "sum"), ints=("passing_interceptions", "sum"),
        att=("attempts", "sum"), comp=("completions", "sum"),
        rush_yds=("rushing_yards", "sum"), car=("carries", "sum"), rush_td=("rushing_tds", "sum"),
        rec=("receptions", "sum"), tgt=("targets", "sum"), rec_yds=("receiving_yards", "sum"), rec_td=("receiving_tds", "sum"),
        sacks=("def_sacks", "sum"), def_int=("def_interceptions", "sum"),
        tkl=("def_tackles_solo", "sum"), ast=("def_tackle_assists", "sum")).reset_index()
    agg["tot_td"] = agg.rush_td + agg.rec_td

    def top(col, n=10, extra=()):
        d = agg.sort_values(col, ascending=False).head(n)
        return [[x.player_display_name, x.team, x.position, int(x.gp), r1(getattr(x, col))] +
                [r1(getattr(x, e)) for e in extra] for x in d.itertuples()]
    leaders = {
        "pass_yds": top("pass_yds", extra=("pass_td", "ints", "comp", "att")),
        "rush_yds": top("rush_yds", extra=("car", "rush_td")),
        "rec_yds": top("rec_yds", extra=("rec", "tgt", "rec_td")),
        "tot_td": top("tot_td", extra=("rush_td", "rec_td")),
        "sacks": top("sacks"), "def_int": top("def_int"),
    }

    # ---- Calendario de las semanas siguientes ----
    nxt = s[(s.week > last_week + 1)].sort_values(["week", "gameday", "gametime"])
    schedule = [{"week": int(x.week), "date": x.gameday, "day": x.weekday, "time": x.gametime,
                 "away": x.away_team, "home": x.home_team, "stadium": x.stadium,
                 "neutral": x.location == "Neutral", "div": bool(x.div_game),
                 "spread": r1(x.spread_line), "total": r1(x.total_line)} for x in nxt.itertuples()]
    byes = {}
    for wk in sorted(set(s.week)):
        playing = set(s[s.week == wk].home_team) | set(s[s.week == wk].away_team)
        byes[int(wk)] = sorted(set(TEAM_NAMES) - playing)

    out = {"season": season, "last_week": last_week, "games": games, "standings": standings,
           "leaders": leaders, "schedule": schedule, "byes": byes, "team_names": TEAM_NAMES}
    OUT.mkdir(exist_ok=True)
    (OUT / "temporada.json").write_text(json.dumps(out, ensure_ascii=False, allow_nan=False, default=str))
    return out


if __name__ == "__main__":
    d = build()
    print(len(d["games"]), "partidos;", len(d["schedule"]), "en calendario;",
          (OUT / "temporada.json").stat().st_size // 1024, "KB")
