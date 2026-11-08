"""
Descarga todos los datos históricos de nflverse a ./data (≈300 MB).
Necesario en un clon nuevo del repositorio, antes de correr predict.py.

Uso:
    python download_data.py            # temporadas 2012 hasta la actual
    python download_data.py --season 2026   # solo una temporada (actualización rápida)
"""
import argparse, subprocess
from datetime import date
from pathlib import Path

DATA = Path(__file__).parent / "data"
REL = "https://github.com/nflverse/nflverse-data/releases/download"


def get(name, path):
    r = subprocess.run(["curl", "-sSLf", "-o", str(DATA / name), f"{REL}/{path}"])
    print(("ok   " if r.returncode == 0 else "FALLÓ"), name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, help="descargar solo esta temporada")
    a = ap.parse_args()
    DATA.mkdir(exist_ok=True)
    cur = date.today().year if date.today().month >= 8 else date.today().year - 1
    get("games.csv", "schedules/games.csv")
    get("players.parquet", "players/players.parquet")
    seasons = [a.season] if a.season else range(2012, cur + 1)
    for y in seasons:
        get(f"pbp_{y}.parquet", f"pbp/play_by_play_{y}.parquet")
        if y >= 2016:
            get(f"pw_{y}.parquet", f"stats_player/stats_player_week_{y}.parquet")
            get(f"tw_{y}.parquet", f"stats_team/stats_team_week_{y}.parquet")
            get(f"injuries_{y}.parquet", f"injuries/injuries_{y}.parquet")
            get(f"snap_counts_{y}.parquet", f"snap_counts/snap_counts_{y}.parquet")
    get(f"depth_charts_{cur if not a.season else a.season}.parquet",
        f"depth_charts/depth_charts_{cur if not a.season else a.season}.parquet")


if __name__ == "__main__":
    main()
