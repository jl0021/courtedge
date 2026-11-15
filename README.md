# CourtEdge

Read-only ATP and WTA combo prediction engine for Kalshi tennis markets. It scans public
markets and logs paper trades only. It never places orders. The locked rules are in
`CLAUDE.md` and `vault/Hard_Guardrails.md`.

## Milestone 1

| Module | Purpose |
|---|---|
| `src/ingest.py` | Loads 2024–2026 ATP and WTA matches into `data/courtedge.db`, builds `player_lookup`, then catches up to today. |
| `src/espn_feed.py` | Read-only client for ESPN's public tennis scoreboard, used for the catch-up. |
| `src/model.py` | Walk-forward ratings: Elo, surface Elo, ranking points, serve and return rates. |
| `src/sim.py` | 10,000-trial point-by-point Markov simulator, plus the exact solution of the same chain. |
| `src/risk_gate.py` | Locked leg and combo gate. |
| `src/backtest.py` | Blind 2025–2026 walk-forward backtest, Brier ceiling and calibration gate. |
| `src/server.py`, `src/web/` | Local dashboard. |

## Open the dashboard

On Linux or macOS, from the project folder:

```bash
./start.sh
```

The first run sets up the Python environment, downloads the match data (about 26 MB)
and runs the fast backtest. That takes a minute or two. After that it opens in seconds.
Your browser opens http://127.0.0.1:8770. Leave the terminal open while you use it;
Ctrl+C stops it. On Ubuntu, if the first run says it cannot create the environment,
run `sudo apt install -y python3-venv` once.

- **Performance**: the Milestone 1 gate, calibration chart, and scores by tour, season and surface.
- **Matchup simulator**: pick two players, a surface and a format; get win chance, fair price,
  set scores and total games from 10,000 simulated matches.
- **Players**: ratings by tour and surface, with last-match veto flags.
- **Guardrails**: the locked values.
- **Run fast backtest** takes a few seconds and uses exact probabilities from the same model.
  **Run full simulation** takes about 8 minutes, plays every 2025–2026 match 10,000 times,
  and writes a note to `vault/Post_Mortems/`.
- **Update data** fetches results played since the last update, then refreshes ratings, 7-day
  workload and retirement/walkover flags. The dashboard also does this each time it opens.

The dashboard never contacts Kalshi.

## Commands

Inside the environment (`source .venv/bin/activate`):

```bash
python -m src.server                      # the dashboard, same as ./start.sh
python -m src.ingest                      # load the archive, then catch up to today
python -m src.ingest --catch-up           # only fetch results newer than what is stored
python -m src.backtest --tune             # tune on the 2024 season only, then run the backtest
python -m src.backtest                    # full simulator backtest with the saved parameters
python -m src.backtest --engine analytic  # fast backtest
pytest
```

`python -m src.backtest` exits with status 0 only if the Milestone 1 gate passes: Brier score below 0.215,
and predicted and actual win rates within 3.5 points in every probability bucket.
Each command-line run writes a note to `vault/Post_Mortems/`.

## Data

The seed comes from [Aneeshers/tennis-sackmann-archive](https://github.com/Aneeshers/tennis-sackmann-archive),
a mirror of Jeff Sackmann's tennis datasets, licensed CC BY-NC-SA 4.0 (attribution, non-commercial, share-alike).
It ends in early June 2026.

Results after that come from ESPN's public tennis scoreboard. It is an undocumented endpoint, so it can
change without notice. It covers ATP tour events and WTA events down to the 125 level, but not ATP
Challengers, and it carries no surface, ranking points or serve statistics. A match is stored only when
both player names resolve at 98% confidence; the rest are skipped and counted in the dashboard header.
To fix a skipped name, add the feed's spelling to `config/name_aliases.json` and run **Update data** again.
