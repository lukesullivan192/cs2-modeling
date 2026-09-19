# cs2-modeling

Tools for building CS2 round-outcome, round-to-round economy, and
match-outcome models from pro match demos.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

1. Download demos from HLTV:
   ```bash
   python demos/fetch_demos.py
   ```
   Saves `.dem` files into `demos/saves/`.

2. Parse demos into training data:
   ```bash
   python demos/parse_demos.py
   ```
   Writes two CSVs into `data/`:
   - `data.csv`: one row per mid-round game-state snapshot.
   - `round_data.csv`: one row per round, taken at the instant freezetime
     ends (before anyone can move, buy, or take damage) -- the shape a
     "forecast a round that hasn't been played yet" model needs.

   Re-running only parses new demos; pass `--retry-failed` to also
   re-attempt demos that previously failed, or `--force` to reparse
   everything.

3. Train/evaluate the round-winner model:
   ```bash
   python models/round_model.py train   # fits on all data, saves models/saves/round_model.json
   python models/round_model.py test    # match-grouped 80/20 train/test MAE/log-loss report
   ```
   An XGBoost classifier predicting P(CT wins) a round from each side's
   equipment value entering it.

4. Train/evaluate the round-to-round economy model:
   ```bash
   python models/economy_model.py train
   python models/economy_model.py test
   ```
   An XGBoost regressor predicting each side's next-round equipment value
   from its current economy state and `round_model`'s win probability for
   the round in between -- the piece a round-to-round match simulation
   would call each simulated round.

5. Evaluate the match-outcome dynamic program:
   ```bash
   python models/match_model.py train   # ensures round_model.json/economy_model.json exist in models/saves/
   python models/match_model.py test    # match-level accuracy, broken down by round number
   ```
   Chains `round_model` and `economy_model` into a full match simulation:
   from any round's pre-round state, it branches on who wins each
   remaining round, applies CS2's actual scoring/halftime/overtime rules
   to fill in the rest of the state, and computes P(CT wins the match).
   `test` runs this from every real round of the held-out matches and
   reports how accurate the resulting match-winner prediction is at each
   round number, plus one overall average, against a naive
   "whoever's-currently-leading" baseline.

6. Run the trained models live against an in-progress HLTV match:
   ```bash
   python live/run_live_model.py <hltv_match_url_or_id>
   ```
   e.g. `python live/run_live_model.py 2398160` or a full match URL.
   Prints each round's pre-round state and `P(CT wins the match)` once
   per round, as soon as `parse_live_round.py` finds that round's state
   (the gamestate right after freeze time ends), updating live as the
   match is played. Trains `round_model`/`economy_model` first (same as
   step 3-5's `train` modes) if
   `models/saves/round_model.json`/`economy_model.json` don't exist yet;
   otherwise loads them as-is.

   This chains three pieces under `live/`, each usable on its own too:
   - `fetch_live_match.py`: opens the match page in a real,
     Cloudflare-cleared browser and taps its own WebSocket traffic to
     capture HLTV's scorebot feed (score, live player/round state, kill
     log) -- see its module docstring for why a plain socket.io client no
     longer works.
   - `parse_live_round.py`: turns that raw feed into one row per round in
     `round_data.csv`'s exact shape (`match_id`, `round_num`, `ct_score`,
     `t_score`, `ct_equip_value`, `t_equip_value`,
     `ct_loss_bonus_streak`, `t_loss_bonus_streak`), tracking each
     round's state until its first kill locks it in as "the instant
     freeze time ends" (`python live/parse_live_round.py <match>` prints
     just this, with no model involved).
   - `run_live_model.py`: feeds each new round straight into
     `MatchSimulator.p_ct_wins_match()` from step 5.
