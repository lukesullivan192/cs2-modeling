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
   python demo_fetcher.py
   ```
   Saves `.dem` files into `demos/`.

2. Parse demos into training data:
   ```bash
   python parse_demos.py
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
   python round_model.py train   # fits on all data, saves data/round_model.json
   python round_model.py test    # match-grouped 80/20 train/test MAE/log-loss report
   ```
   An XGBoost classifier predicting P(CT wins) a round from each side's
   equipment value entering it.

4. Train/evaluate the round-to-round economy model:
   ```bash
   python economy_model.py train
   python economy_model.py test
   python economy_model.py predict --ct-equip 4200 --t-equip 4200 \
       --ct-streak 1 --t-streak 1 --ct-win-prob 0.55
   ```
   An XGBoost regressor predicting each side's next-round equipment value
   from its current economy state and `round_model`'s win probability for
   the round in between -- the piece a round-to-round match simulation
   would call each simulated round.

5. Evaluate the match-outcome dynamic program:
   ```bash
   python match_model.py train   # ensures round_model.json/economy_model.json exist
   python match_model.py test    # match-level accuracy, broken down by round number
   ```
   Chains `round_model` and `economy_model` into a full match simulation:
   from any round's pre-round state, it branches on who wins each
   remaining round, applies CS2's actual scoring/halftime/overtime rules
   to fill in the rest of the state, and computes P(CT wins the match).
   `test` runs this from every real round of the held-out matches and
   reports how accurate the resulting match-winner prediction is at each
   round number, plus one overall average, against a naive
   "whoever's-currently-leading" baseline.
