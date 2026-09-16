# cs2-modeling

Tools for building a CS2 round-outcome model from pro match demos.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

1. Download demos from HLTV:
   ```bash
   python fetch_demos.py
   ```
   Saves `.dem` files into `demos/`.

2. Parse demos into training data:
   ```bash
   python parse_demos.py
   ```
   Writes mid-round game-state snapshots to `data/data.csv`.
