"""
Datos en vivo para la semana a predecir:
  - ESPN scoreboard : hora, estadio, techado, líneas DraftKings de APERTURA y ACTUALES
  - ESPN injuries   : estatus de lesiones de los 32 equipos (incluye IR)
  - Open-Meteo      : pronóstico del clima a la hora del kickoff (sin API key)
  - Historial de líneas: cada ejecución guarda una foto en data/lines_history.csv
"""
import json, subprocess, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"
ESPN = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
ABBR = {"LAR": "LA", "WSH": "WAS"}

# Coordenadas por ciudad del estadio (incluye sedes internacionales)
CITY_COORDS = {
    "Glendale": (33.528, -112.263), "Atlanta": (33.755, -84.401), "Baltimore": (39.278, -76.623),
    "Orchard Park": (42.774, -78.787), "Charlotte": (35.226, -80.853), "Chicago": (41.862, -87.617),
    "Cincinnati": (39.095, -84.516), "Cleveland": (41.506, -81.700), "Arlington": (32.748, -97.093),
    "Denver": (39.744, -105.020), "Detroit": (42.340, -83.046), "Green Bay": (44.501, -88.062),
    "Houston": (29.685, -95.411), "Indianapolis": (39.760, -86.164), "Jacksonville": (30.324, -81.637),
    "Kansas City": (39.049, -94.484), "Inglewood": (33.953, -118.339), "Las Vegas": (36.091, -115.184),
    "Paradise": (36.091, -115.184), "Miami Gardens": (25.958, -80.239), "Minneapolis": (44.974, -93.258),
    "Foxborough": (42.091, -71.264), "New Orleans": (29.951, -90.081), "East Rutherford": (40.814, -74.074),
    "Philadelphia": (39.901, -75.168), "Pittsburgh": (40.447, -80.016), "Santa Clara": (37.403, -121.970),
    "Seattle": (47.595, -122.332), "Tampa": (27.976, -82.503), "Nashville": (36.166, -86.771),
    "Landover": (38.908, -76.865), "London": (51.556, -0.280), "Munich": (48.219, 11.625),
    "Berlin": (52.515, 13.239), "Frankfurt": (50.069, 8.646), "Madrid": (40.453, -3.688),
    "Mexico City": (19.303, -99.150), "São Paulo": (-23.545, -46.474), "Sao Paulo": (-23.545, -46.474),
    "Rio de Janeiro": (-22.912, -43.230), "Dublin": (53.361, -6.251), "Melbourne": (-37.820, 144.983),
    "Paris": (48.924, 2.360),
}
COVERED = {"SoFi Stadium", "Allegiant Stadium", "U.S. Bank Stadium", "Ford Field", "Caesars Superdome",
           "Mercedes-Benz Stadium", "AT&T Stadium", "Lucas Oil Stadium", "NRG Stadium",
           "State Farm Stadium", "Santiago Bernabéu"}


def _get(url, out=None, tries=5):
    for i in range(tries):
        r = subprocess.run(["curl", "-sSf", "--compressed", "-m", "25", url], capture_output=True)
        if r.returncode == 0:
            if out:
                Path(out).write_bytes(r.stdout)
            return json.loads(r.stdout)
        time.sleep(2 ** i)
    raise RuntimeError(f"No se pudo descargar {url}: {r.stderr.decode()[:200]}")


def fetch_espn(season, week, seasontype=2):
    sb = _get(f"{ESPN}/scoreboard?week={week}&seasontype={seasontype}&dates={season}", DATA / "espn_sb.json")
    try:
        _get(f"{ESPN}/injuries", DATA / "espn_inj.json")
    except RuntimeError as e:
        print("Aviso lesiones ESPN:", e)
    return sb


def _num(s):
    try:
        return float(str(s).replace("o", "").replace("u", "").replace("+", ""))
    except (TypeError, ValueError):
        return np.nan


def parse_scoreboard(sb):
    rows = []
    for e in sb.get("events", []):
        c = e["competitions"][0]
        teams = {x["homeAway"]: x for x in c["competitors"]}
        h = ABBR.get(teams["home"]["team"]["abbreviation"], teams["home"]["team"]["abbreviation"])
        a = ABBR.get(teams["away"]["team"]["abbreviation"], teams["away"]["team"]["abbreviation"])
        v = c.get("venue", {})
        r = {"espn_id": e.get("id"), "home_team": h, "away_team": a, "kickoff_utc": e["date"],
             "venue": v.get("fullName"), "city": (v.get("address") or {}).get("city"),
             "country": (v.get("address") or {}).get("country"),
             "indoor": bool(v.get("indoor")) or v.get("fullName") in COVERED,
             "espn_neutral": bool(c.get("neutralSite")), "status": e["status"]["type"]["description"],
             "home_score_live": _num(teams["home"].get("score")), "away_score_live": _num(teams["away"].get("score")),
             "home_name": teams["home"]["team"]["displayName"], "away_name": teams["away"]["team"]["displayName"],
             "home_logo": teams["home"]["team"].get("logo"), "away_logo": teams["away"]["team"].get("logo")}
        odds = (c.get("odds") or [{}])[0]
        r["book"] = (odds.get("provider") or {}).get("name")
        ps, tot, ml = odds.get("pointSpread", {}), odds.get("total", {}), odds.get("moneyline", {})
        # spread en convención nflverse: positivo = local favorito  ->  -(línea del local)
        for k in ("open", "close"):
            hl = _num(((ps.get("home") or {}).get(k) or {}).get("line"))
            r[f"spread_{k}"] = -hl if not np.isnan(hl) else np.nan
            r[f"total_{k}"] = _num(((tot.get("over") or {}).get(k) or {}).get("line"))
            r[f"home_ml_{k}"] = _num(((ml.get("home") or {}).get(k) or {}).get("odds"))
            r[f"away_ml_{k}"] = _num(((ml.get("away") or {}).get(k) or {}).get("odds"))
        if np.isnan(r["spread_close"]) and odds.get("spread") is not None:
            r["spread_close"] = -float(odds["spread"])
        if np.isnan(r["total_close"]) and odds.get("overUnder") is not None:
            r["total_close"] = float(odds["overUnder"])
        rows.append(r)
    return pd.DataFrame(rows)


def weather(df):
    """Pronóstico Open-Meteo en la hora del kickoff para cada partido."""
    out = []
    for r in df.itertuples(index=False):
        w = {"home_team": r.home_team, "w_temp": np.nan, "w_wind": np.nan, "w_gust": np.nan,
             "w_precip_prob": np.nan, "w_precip": np.nan, "w_snow": np.nan, "w_code": np.nan}
        ll = CITY_COORDS.get(r.city)
        if ll is not None:
            ko = pd.Timestamp(r.kickoff_utc).tz_convert("UTC").floor("h")
            url = ("https://api.open-meteo.com/v1/forecast?latitude={}&longitude={}"
                   "&hourly=temperature_2m,wind_speed_10m,wind_gusts_10m,precipitation_probability,"
                   "precipitation,snowfall,weather_code&temperature_unit=fahrenheit&wind_speed_unit=mph"
                   "&precipitation_unit=inch&timezone=UTC&start_date={}&end_date={}").format(
                ll[0], ll[1], ko.date(), ko.date())
            try:
                j = _get(url)["hourly"]
                i = j["time"].index(ko.strftime("%Y-%m-%dT%H:00"))
                w.update(w_temp=j["temperature_2m"][i], w_wind=j["wind_speed_10m"][i],
                         w_gust=j["wind_gusts_10m"][i], w_precip_prob=j["precipitation_probability"][i],
                         w_precip=j["precipitation"][i], w_snow=j["snowfall"][i], w_code=j["weather_code"][i])
            except Exception as ex:  # fuera del horizonte de pronóstico, conexión, etc.
                print("Aviso clima", r.home_team, str(ex)[:80])
        if np.isnan(w["w_temp"]):           # respaldo: clima del resumen de ESPN
            try:
                sm = DATA / "espn_sum" / f"{r.espn_id}.json"
                if not sm.exists():
                    _get(f"{ESPN}/summary?event={r.espn_id}", sm)
                gw = json.loads(sm.read_text()).get("gameInfo", {}).get("weather") or {}
                if gw.get("temperature") is not None:
                    w.update(w_temp=float(gw["temperature"]), w_gust=float(gw.get("gust") or np.nan),
                             w_wind=float(gw.get("gust") or 0) * 0.7, w_precip_prob=float(gw.get("precipitation") or 0))
                    print("  clima de respaldo ESPN para", r.home_team)
            except Exception as ex:
                print("  sin clima para", r.home_team, str(ex)[:60])
        out.append(w)
    return pd.DataFrame(out)


WMO = {0: "Despejado", 1: "Mayormente despejado", 2: "Parcialmente nublado", 3: "Nublado", 45: "Niebla",
       48: "Niebla", 51: "Llovizna", 53: "Llovizna", 55: "Llovizna intensa", 61: "Lluvia ligera",
       63: "Lluvia", 65: "Lluvia fuerte", 71: "Nieve ligera", 73: "Nieve", 75: "Nieve fuerte",
       80: "Chubascos", 81: "Chubascos", 82: "Chubascos fuertes", 95: "Tormenta", 96: "Tormenta", 99: "Tormenta"}


def snapshot_lines(df, season, week):
    f = DATA / "lines_history.csv"
    snap = df[["home_team", "away_team", "book", "spread_close", "total_close", "home_ml_close",
               "away_ml_close"]].copy()
    snap.insert(0, "ts", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"))
    snap.insert(1, "season", season); snap.insert(2, "week", week)
    snap = snap[snap.spread_close.notna()]          # partidos ya iniciados no tienen línea
    snap.to_csv(f, mode="a", header=not f.exists(), index=False)
    h = pd.read_csv(f)
    return h[(h.season == season) & (h.week == week)]


def get_live(season, week):
    sb = fetch_espn(season, week, 3 if week > 18 else 2)
    df = parse_scoreboard(sb)
    if df.empty:
        return df, pd.DataFrame()
    df = df.merge(weather(df), on="home_team", how="left")
    hist = snapshot_lines(df, season, week)
    return df, hist
