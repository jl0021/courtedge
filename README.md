# CourtEdge

Read-only ATP and WTA combo prediction engine for Kalshi tennis markets. It scans public
markets and logs paper trades only. It never places orders. The locked rules are in
`CLAUDE.md` and `vault/Hard_Guardrails.md`.

## Milestone 1

| Module | Purpose |
|---|---|
| `src/ingest.py` | Loads 2024–2026 ATP and WTA matches into `data/courtedge.db` and builds `player_lookup`. |
| `src/model.py` | Walk-forward ratings: Elo, surface Elo, ranking points, serve and return rates. |
| `src/sim.py` | 10,000-trial point-by-point Markov simulator, plus the exact solution of the same chain. |
| `src/risk_gate.py` | Locked leg and combo gate. |
| `src/backtest.py` | Blind 2025–2026 walk-forward backtest, Brier ceiling and calibration gate. |

## Run it

```bash
pip install numpy pandas pytest tabulate

python -m src.ingest                      # clones the data archive into data/raw/ on first run
python -m src.backtest --tune             # tune on the 2024 season only, then run the backtest
python -m src.backtest                    # backtest with the saved parameters
python -m src.backtest --engine analytic  # fast check without sampling
pytest
```

`python -m src.backtest` exits with status 0 only if the Milestone 1 gate passes: Brier score below 0.215,
and predicted and actual win rates within 3.5 points in every probability bucket.
Each run writes a note to `vault/Post_Mortems/`.

## Data

Match data comes from [Aneeshers/tennis-sackmann-archive](https://github.com/Aneeshers/tennis-sackmann-archive),
a mirror of Jeff Sackmann's tennis datasets, licensed CC BY-NC-SA 4.0 (attribution, non-commercial, share-alike).
