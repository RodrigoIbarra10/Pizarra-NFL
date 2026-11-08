"""Genera output/pizarra_nfl.html a partir de output/semana.json (correr después de predict.py)."""
from pathlib import Path
ROOT = Path(__file__).parent
data = (ROOT / "output" / "semana.json").read_text().replace("</", "<\\/")
season_f = ROOT / "output" / "temporada.json"
season = season_f.read_text().replace("</", "<\\/") if season_f.exists() else "{}"
html = (ROOT / "page_template.html").read_text().replace("/*__DATA__*/", data).replace("/*__SEASON__*/", season)
(ROOT / "output" / "pizarra_nfl.html").write_text(html)
print("ok", len(html) // 1024, "KB")
