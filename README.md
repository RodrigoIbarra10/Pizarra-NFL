# Modelo predictivo NFL (datos nflverse)

## Instalación (clon nuevo)
```bash
pip install -r requirements.txt
python download_data.py           # descarga el historial de nflverse (~300 MB) a data/
python model.py                   # genera data/backtest_predictions.csv (lo usa la página)
```

## Uso semanal
```bash
python predict.py --update && python season_data.py && python build_page.py
```
Luego publica `output/pizarra_nfl.html`.

## Comandos
```bash
python predict.py --update        # descarga datos frescos y predice la próxima semana
python predict.py --season 2026 --week 10
python model.py                   # backtest walk-forward 2016-hoy
python props.py                   # backtest de props de jugadores
python prop_line.py "Josh Allen" passing_yards 245.5   # P(over/under) de una línea
python -c "import injuries; injuries.calibrate()"   # recalibra el valor de las lesiones
python season_data.py             # resultados, posiciones, líderes y calendario de la temporada
python build_page.py              # genera output/pizarra_nfl.html (página visual)
```

## Datos en vivo (live.py, injuries.py)
- ESPN: líneas de apertura y actuales (DraftKings), estadio y techado, estatus de lesiones (incluye IR).
- Open-Meteo: pronóstico de temperatura, viento y lluvia a la hora del kickoff. Temperatura y viento entran al modelo.
- Lesiones: cada titular ausente vale según su % de snaps en los últimos 4 partidos × coeficiente calibrado
  (0.77 pts por titular ofensivo y 0.97 por defensivo, estimados con 10 años de reportes oficiales).
  Si el QB titular está Out o en IR se usa el siguiente del depth chart. En los props se quitan los lesionados
  y su volumen se reparte entre sus compañeros.
- Historial de líneas: data/lines_history.csv (una foto por ejecución).
Salidas en `output/`: `reporte_*.md`, `predicciones_*.csv`, `props_*.csv`.

## Variables (features.py) — todas calculadas sólo con datos previos al partido
| Categoría | Variables |
|---|---|
| Rendimiento | V/D/E, % victorias temporada |
| Rachas | racha actual (+V / −D) |
| Ofensiva | puntos prom./totales, yardas, EPA/jugada, EPA pase/carrera, % éxito |
| Defensiva | puntos/yardas permitidos, EPA permitido, sacks, INT, margen de pérdidas |
| Localía | récord local y visitante, neutral |
| Calendario | semana, división, techo/domo, clima (viento, temperatura) |
| Descanso | días de descanso, semana corta, post-bye, partidos seguidos de visita |
| Tendencias | últimos 3/5/10 partidos + EWMA |
| Enfrentamientos | % victorias y margen medio últimos 10 años |
| Playoffs | % victorias año anterior, clasificó a playoffs, victorias en playoffs |
| Avanzadas | Elo (margen + regresión a la media), EPA, calificación del QB titular |

## Modelos (model.py)
- Ganador: ensamble de Regresión logística + Gradient Boosting + margen→probabilidad.
- Marcador: Ridge para margen y total → puntos local = (total+margen)/2.
- Props: forma del jugador (EWMA) × defensa rival × guion de partido; QBs = intentos del equipo × eficiencia por intento.

## Resultados backtest (2016-2026, cada temporada predicha sólo con datos anteriores)
| | Acierto ganador | Log-loss |
|---|---|---|
| Modelo (ensamble) | 65.6% | 0.627 |
| Mercado (spread Vegas) | 66.4% | 0.612 |
| Sólo Elo | 64.0% | 0.635 |
| Siempre el local | 54.9% | 0.689 |

MAE margen 10.1 pts (mercado 9.8); MAE total 10.7 (mercado 10.5).
ATS con diferencia ≥3 pts vs. la línea: ~53.5% (punto de equilibrio a -110 = 52.4%). Over/Under: sin ventaja.

## Limitaciones
El ajuste por lesiones fuera del QB mejora poco (MAE 10.05 → 10.01). Las lesiones de último minuto dependen de que ESPN
o nflverse ya las publiquen. Totales (O/U): el modelo no supera al mercado.
