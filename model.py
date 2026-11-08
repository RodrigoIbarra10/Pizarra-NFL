"""
Modelos de predicción NFL:
  1) Ganador     -> Regresión logística + Gradient Boosting (probabilidad de victoria local)
  2) Margen      -> Ridge (puntos local - visitante)
  3) Total       -> Ridge (puntos totales del partido)
  => Marcador proyectado = ((total + margen)/2 , (total - margen)/2)

Validación walk-forward: para la temporada S se entrena con temporadas < S.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score, mean_absolute_error

from features import TEAM_FEATS

DATA = Path(__file__).parent / "data"

CONTEXT = ["qb_diff", "home_qb_rating", "away_qb_rating", "home_qb_vs_team", "away_qb_vs_team",
           "elo_diff", "home_elo", "away_elo", "neutral", "div_game", "dome", "wind", "temp",
           "playoff", "week", "h2h_home_winpct", "h2h_home_mov", "h2h_n"]
DIFF = [f"diff_{c}" for c in TEAM_FEATS]
# Para el total importan los niveles (no sólo la diferencia): ofensiva de ambos + defensa de ambos
LEVEL = [f"{s}_{c}" for s in ("home", "away") for c in
         ["ew_off_epa", "ew_def_epa", "ew_pf", "ew_pa", "ew_off_sr", "ew_def_sr", "ew_plays",
          "ew_pass_rate", "s_pf", "s_pa", "l5_pf", "l5_pa", "ew_yds", "ew_yds_allowed"]]
FEATURES = CONTEXT + DIFF + LEVEL

MARGIN_SD = 13.5   # desviación típica histórica del margen NFL (se re-estima en backtest)


def _logit():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=0.02, max_iter=3000))


def _gbm():
    return HistGradientBoostingClassifier(max_iter=250, learning_rate=0.03, max_depth=3,
                                          min_samples_leaf=40, l2_regularization=1.0)


def _ridge(alpha):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=alpha))


class NFLModel:
    def fit(self, tr: pd.DataFrame):
        X, y = tr[FEATURES], tr["home_win"]
        mask = tr["home_margin"] != 0                      # empates fuera del clasificador
        self.logit = _logit().fit(X[mask], y[mask])
        self.gbm = _gbm().fit(X[mask], y[mask])
        self.margin = _ridge(300).fit(X, tr["home_margin"])
        self.total = _ridge(300).fit(X, tr["game_total"])
        res_m = tr["home_margin"] - self.margin.predict(X)
        res_t = tr["game_total"] - self.total.predict(X)
        self.sd_margin, self.sd_total = float(res_m.std()), float(res_t.std())
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[FEATURES]
        out = df[["game_id", "season", "week", "gameday", "away_team", "home_team"]].copy()
        p_log = self.logit.predict_proba(X)[:, 1]
        p_gbm = self.gbm.predict_proba(X)[:, 1]
        m = self.margin.predict(X)
        t = self.total.predict(X)
        p_mar = norm.cdf(m / self.sd_margin)
        out["p_logit"], out["p_gbm"], out["p_margin"] = p_log, p_gbm, p_mar
        out["p_home"] = 0.4 * p_log + 0.2 * p_gbm + 0.4 * p_mar      # ensamble
        out["pred_margin"], out["pred_total"] = m, t
        out["pred_home_pts"] = (t + m) / 2
        out["pred_away_pts"] = (t - m) / 2
        out["sd_margin"], out["sd_total"] = self.sd_margin, self.sd_total
        return out


# --------------------------------------------------------------------------- #
# Backtest
# --------------------------------------------------------------------------- #
def market_prob(spread):
    """spread_line de nflverse: positivo = local favorito."""
    return norm.cdf(spread / MARGIN_SD)


def backtest(df: pd.DataFrame, seasons=range(2016, 2027)) -> pd.DataFrame:
    played = df[df.played].copy()
    preds = []
    for s in seasons:
        tr, te = played[played.season < s], played[played.season == s]
        if te.empty:
            continue
        p = NFLModel().fit(tr).predict(te)
        p = p.merge(te[["game_id", "home_win", "home_margin", "game_total", "spread_line",
                        "total_line", "elo_diff"]], on="game_id")
        preds.append(p)
    return pd.concat(preds, ignore_index=True)


def report(bt: pd.DataFrame) -> pd.DataFrame:
    bt = bt[bt.home_margin != 0].copy()
    bt["p_market"] = market_prob(bt["spread_line"])
    bt["p_elo"] = 1 / (1 + 10 ** (-(bt["elo_diff"] + 48) / 400))
    rows = []
    for name, col in [("Ensamble", "p_home"), ("Logística", "p_logit"), ("GBM", "p_gbm"),
                      ("Margen->Prob", "p_margin"), ("Sólo Elo", "p_elo"),
                      ("Mercado (spread)", "p_market"), ("Siempre local", None)]:
        p = bt[col].clip(0.01, 0.99) if col else pd.Series(0.57, index=bt.index)
        rows.append({"modelo": name,
                     "acierto_%": 100 * accuracy_score(bt.home_win, p > 0.5),
                     "log_loss": log_loss(bt.home_win, p),
                     "brier": brier_score_loss(bt.home_win, p)})
    return pd.DataFrame(rows)


def score_report(bt):
    ok = bt.dropna(subset=["spread_line", "total_line"]).copy()
    r = {
        "MAE margen modelo": mean_absolute_error(ok.home_margin, ok.pred_margin),
        "MAE margen mercado": mean_absolute_error(ok.home_margin, ok.spread_line),
        "MAE total modelo": mean_absolute_error(ok.game_total, ok.pred_total),
        "MAE total mercado": mean_absolute_error(ok.game_total, ok.total_line),
        "MAE pts local": mean_absolute_error(ok.home_margin * 0 + (ok.game_total + ok.home_margin) / 2,
                                             ok.pred_home_pts),
    }
    # Contra la línea: ¿cuándo el modelo discrepa del spread por >= X puntos?
    ok["edge"] = ok.pred_margin - ok.spread_line
    ok["cover"] = np.sign(ok.home_margin - ok.spread_line)
    ats = []
    for thr in (0, 1.5, 3, 4.5):
        sub = ok[(ok.edge.abs() >= thr) & (ok.cover != 0)]
        hit = (np.sign(sub.edge) == sub.cover).mean()
        ats.append({"edge_min_pts": thr, "partidos": len(sub), "acierto_ATS_%": 100 * hit})
    ok["tedge"] = ok.pred_total - ok.total_line
    ok["over"] = np.sign(ok.game_total - ok.total_line)
    ou = []
    for thr in (0, 1.5, 3, 4.5):
        sub = ok[(ok.tedge.abs() >= thr) & (ok.over != 0)]
        ou.append({"edge_min_pts": thr, "partidos": len(sub),
                   "acierto_OU_%": 100 * (np.sign(sub.tedge) == sub.over).mean()})
    return r, pd.DataFrame(ats), pd.DataFrame(ou)


def calibration(bt, bins=10):
    b = bt[bt.home_margin != 0].copy()
    b["bin"] = pd.cut(b.p_home, np.linspace(0, 1, bins + 1))
    return b.groupby("bin", observed=True).agg(partidos=("home_win", "size"),
                                               prob_pred=("p_home", "mean"),
                                               frec_real=("home_win", "mean")).round(3)


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    df = pd.read_parquet(DATA / "model_dataset.parquet")
    bt = backtest(df)
    bt.to_csv(DATA / "backtest_predictions.csv", index=False)
    pd.set_option("display.width", 200)
    print("=== GANADOR: walk-forward 2016-2026 ===")
    print(report(bt).round(4).to_string(index=False))
    print("\nPor temporada (ensamble):")
    print(bt[bt.home_margin != 0].groupby("season").apply(
        lambda d: pd.Series({"partidos": len(d), "acierto_%": 100 * ((d.p_home > .5) == d.home_win).mean()})
    ).round(1).T.to_string())
    r, ats, ou = score_report(bt)
    print("\n=== MARCADOR ===")
    for k, v in r.items():
        print(f"{k:22s} {v:6.2f}")
    print("\nContra el spread (ATS):\n", ats.round(1).to_string(index=False))
    print("\nOver/Under:\n", ou.round(1).to_string(index=False))
    print("\nCalibración:\n", calibration(bt).to_string())
