# CourtEdge MVP — Read-Only ATP & WTA Combo Prediction Engine

## Directory & Obsidian Vault Layout
- `src/` — Python ingestion (`ingest.py`), 10k-trial point-by-point Markov simulator (`sim.py`), risk gate (`risk_gate.py`), read-only Kalshi market scanner (`kalshi_reader.py`), walk-forward backtest (`backtest.py`), and Obsidian Markdown writer (`vault_writer.py`).
- `data/courtedge.db` — SQLite database storing surface-segmented telemetry, `player_lookup`, and paper-traded combos.
- `vault/` — Active Obsidian Vault directory (all `.md` files here render live in Obsidian):
  - `vault/Hard_Guardrails.md` — Locked risk parameters (never modified automatically).
  - `vault/Active_Playbook.md` — Heuristics refined weekly after >= 30 settled legs.
  - `vault/Daily_Slates/YYYY-MM-DD.md` — Daily ranked combo cards with YAML frontmatter and `[[Player Name]]` wikilinks.
  - `vault/Players/<Player_Name>.md` — Per-player surface stats, 7-day court fatigue, and retirement flags.
  - `vault/Winners/` & `vault/Post_Mortems/` — Auto-settled combo results and calibration audits.

## Hard Guardrails (Locked — Never Override)
1. **Read-Only MVP:** Never call Kalshi `POST` order or RFQ endpoints. Scan public markets (`GET`) and log paper trades only.
2. **Combo Hit-Probability & Edge Gate (No Moonshots):**
   - No arbitrary leg-count ceiling, **BUT every combo must have `fair_combo_prob >= 0.30` (minimum 30% chance to cash)** so a $10 bankroll ($1 flat unit) avoids long losing streaks.
   - Each individual leg must satisfy `simulated_prob - kalshi_ask >= 0.05` (+5%), and the joint combo probability must beat Kalshi's implied combo price by `>= +0.06` (+6%).
3. **Data Sources & Settlement Rules:**
   - Seed 2024–2026 ATP & WTA match history from `Aneeshers/tennis-sackmann-archive` (upstream `JeffSackmann/tennis_atp` returns 404—do not use it).
   - Enforce `>= 98%` player-name match confidence in `player_lookup` and veto any player who lost their last match by `RET` or `W/O` (the advancing winner is not vetoed).
   - Under Kalshi tennis rules, once 1 ball is played, mid-match retirements (`RET`) resolve the advancing winner as `Yes` and the retiring player as `No`.
4. **Obsidian Formatting Standard:**
   - Every generated Markdown note in `vault/` must start with clean YAML frontmatter (`date`, `tour`, `surface`, `legs`, `combo_hit_prob`, `edge_pct`, `status`) and link players with `[[First Last]]` wikilinks.
5. **30-Trade Sample Rule:** Never modify `vault/Active_Playbook.md` heuristics unless a pattern spans at least 30 settled legs and improves the Brier score.

## Verification & Workflow
- Build and verify Milestone 1 (`ingest.py`, `sim.py`, `player_lookup`, `backtest.py`) first and confirm both gates on blind 2025–2026 walk-forward validation before wiring up live Kalshi market polling: a regression **Brier Score < 0.215**, and **calibration within ±3.5%** (predicted vs. actual win rate in each of the five favourite-probability buckets from 0.5 to 1.0).
- Run `pytest` after every module change.
