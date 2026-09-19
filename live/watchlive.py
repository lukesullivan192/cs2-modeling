"""Record three live rounds of an HLTV match to a JSONL file.

Usage: python live/watchlive.py <match_id>

Same client and handlers as notebooks/live_feed_walkthrough.ipynb, with one
change: frames are only kept from the first live RoundStart onwards, and the
run stops after ROUNDS_TO_RECORD RoundEnd events. Each kept frame is written
as {"event": ..., "data": ...} with the payload exactly as received (log and
fullLog decoded from their JSON strings, nothing else touched).

The very first "log" the server sends after readyForMatch is the whole match
history, which contains many RoundStart events already in the past. It is
skipped so that recording starts on a RoundStart that happens live.
"""

import json
import os
import signal
import sys
from datetime import datetime

import socketio

SCOREBOT_URL = "https://scorebot-lb.hltv.org"
ROUNDS_TO_RECORD = 3
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_DIR, "data", "raw")


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: python live/watchlive.py <match_id>")
    match_id = sys.argv[1]

    frames = []
    state = {"backlog_seen": False, "recording": False, "rounds_done": 0}
    sio = socketio.Client()

    @sio.on("connect")
    def on_connect():
        print("connected via", sio.transport())
        sio.emit("readyForMatch", json.dumps({"token": "", "listId": match_id}))
        print("sent readyForMatch for", match_id)

    @sio.on("disconnect")
    def on_disconnect():
        print("disconnected")

    @sio.on("scoreboard")
    def on_scoreboard(data):
        if state["recording"]:
            frames.append({"event": "scoreboard", "data": data})
        print("scoreboard", data.get("ctTeamName"), data.get("counterTerroristScore"), "-",
              data.get("terroristScore"), data.get("terroristTeamName"),
              "| round", data.get("currentRound"), data.get("currentRoundState"),
              "| recording" if state["recording"] else "| waiting for RoundStart")

    @sio.on("log")
    def on_log(data):
        events = json.loads(data)
        names = [name for ev in events["log"] for name in ev]
        print("log", names[:20], "..." if len(names) > 20 else "")

        if not state["backlog_seen"]:
            state["backlog_seen"] = True   # match history, not live; skip it
            return
        if not state["recording"]:
            if "RoundStart" in names:
                state["recording"] = True
                print(">>> RoundStart seen, recording")
            else:
                return
        frames.append({"event": "log", "data": events})

        state["rounds_done"] += names.count("RoundEnd")
        if state["rounds_done"] >= ROUNDS_TO_RECORD:
            print(f">>> {state['rounds_done']} rounds recorded, stopping")
            sio.disconnect()

    @sio.on("fullLog")
    def on_full_log(data):
        if state["recording"]:
            frames.append({"event": "fullLog", "data": json.loads(data)})
        print("fullLog received")

    sio.connect(SCOREBOT_URL)
    signal.signal(signal.SIGINT, signal.default_int_handler)   # engineio's own handler can hang on Ctrl+C
    try:
        sio.wait()          # returns when on_log disconnects, or on Ctrl+C
    except KeyboardInterrupt:
        sio.eio.disconnect(abort=True)

    os.makedirs(RAW_DIR, exist_ok=True)
    out_path = os.path.join(RAW_DIR, f"live_{match_id}_{datetime.now():%Y%m%d-%H%M%S}.jsonl")
    with open(out_path, "w") as f:
        for frame in frames:
            f.write(json.dumps(frame) + "\n")
    print(f"wrote {len(frames)} frames ({state['rounds_done']} rounds) -> {out_path}")


if __name__ == "__main__":
    main()
