"""
Probabilidad de Over/Under para una línea de prop de un jugador.
Uso: python prop_line.py "Josh Allen" passing_yards 245.5 [--file output/props_2026_w09.csv]
Stats: passing_yards, passing_tds, passing_interceptions, completions, attempts, rushing_yards,
       carries, rushing_tds, receptions, receiving_yards, targets, receiving_tds
"""
import argparse, glob, warnings
import pandas as pd
import props
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("player"); ap.add_argument("stat"); ap.add_argument("line", type=float)
ap.add_argument("--file", default=sorted(glob.glob("output/props_*.csv"))[-1])
a = ap.parse_args()
d = pd.read_csv(a.file)
r = d[d.player.str.contains(a.player, case=False)]
if r.empty:
    raise SystemExit(f"No encontré a {a.player} en {a.file}")
r = r.iloc[0]
p = props.player_form(props.load_players())
_, sd = props.backtest_props(p, props.defense_factors(p), seasons=(p.season.max() - 1, p.season.max()))
proj = r[a.stat]
po = props.prob_over(proj, a.line, r.pos, a.stat, sd)
print(f"{r.player} ({r.team} vs {r.opp}) {a.stat}: proyección {proj:.1f} | línea {a.line}")
print(f"  P(Over)  = {100*po:.1f}%   cuota justa (decimal) {1/max(po,1e-3):.2f}")
print(f"  P(Under) = {100*(1-po):.1f}%   cuota justa (decimal) {1/max(1-po,1e-3):.2f}")
