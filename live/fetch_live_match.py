"""Capture the HLTV scorebot feed for one live match by watching it inside a real browser.

Usage: python live/fetch_live_match.py <hltv_match_url_or_id> [--rounds N]

    python live/fetch_live_match.py https://www.hltv.org/matches/2398160/saw-youngsters-vs-revenix-hyperx-retake-season-12
    python live/fetch_live_match.py 2398160

Or import fetch_live_match() and iterate it directly:

    from live.fetch_live_match import fetch_live_match
    for event in fetch_live_match("2398160", rounds=1):
        ...

HLTV's scorebot websocket (scorebot-lb.hltv.org) now sits behind a Cloudflare
JS challenge that a plain socket.io client (see temp/record_both.py,
temp/record_all_live.py) can no longer pass. The match page's own JavaScript
opens that exact socket to drive its live scoreboard widget, so instead this
script opens the match page in a real, Cloudflare-cleared browser and taps
the traffic from inside it: a script registered via CDP before page load
wraps window.WebSocket so every socket.io frame sent over any "scorebot"
connection gets decoded and queued in the page, and this process drains that
queue about once a second.

fetch_live_match() yields each message as {"ts": <unix seconds received>,
"event": "scoreboard"|"log"|"fullLog", "data": <payload>} -- the same schema
temp/record_both.py wrote from its direct socket.io client, just handed to
the caller instead of written to a file.

The first "log" is the match history backlog, not a live event, and is
skipped, matching the old scripts. With rounds set, the generator stops
after N RoundEnd events; otherwise it runs until the caller stops iterating
or Ctrl+C.
"""

import argparse
import glob
import json
import os
import re
import shutil
import sys
import time

import seleniumbase.undetected as uc

POLL_INTERVAL = 1.0
KEPT_EVENTS = {"scoreboard", "log", "fullLog"}

# Runs in the page before any of its own scripts. Wraps WebSocket so every
# socket.io EVENT frame ("42[...]") sent over a connection to a "scorebot"
# host gets decoded and pushed onto a queue this process drains.
CAPTURE_JS = """
window.__scorebotEvents = [];
const OrigWS = window.WebSocket;
window.WebSocket = new Proxy(OrigWS, {
  construct(target, args) {
    const ws = new target(...args);
    if (String(args[0] || "").includes("scorebot")) {
      ws.addEventListener("message", function(ev) {
        try {
          const raw = ev.data;
          if (typeof raw === "string" && raw.startsWith("42")) {
            const parsed = JSON.parse(raw.slice(2));
            window.__scorebotEvents.push({ts: Date.now() / 1000, event: parsed[0], data: parsed[1]});
          }
        } catch (e) {}
      });
    }
    return ws;
  }
});
"""


def resolve_url_and_id(arg: str) -> tuple:
    m = re.search(r"/matches/(\d+)/", arg)
    if m:
        return arg, m.group(1)
    if arg.isdigit():
        return f"https://www.hltv.org/matches/{arg}/live", arg
    sys.exit(f"couldn't find an HLTV match id in {arg!r}")


def find_browser_and_driver() -> tuple:
    browser = (shutil.which("brave-browser") or shutil.which("google-chrome")
               or shutil.which("chromium-browser") or shutil.which("chromium"))
    if not browser:
        sys.exit("no Chromium-based browser found (looked for brave-browser, google-chrome, chromium)")
    name = os.path.basename(browser).replace("-browser", "").replace("-", "_")
    # Prefer the chromedriver SeleniumBase already downloaded and version-matched
    # to this exact browser; a mismatched driver reliably crashes on this page.
    candidates = glob.glob(os.path.join(os.path.dirname(uc.__file__), "..", "drivers", f"{name}*", "chromedriver"))
    driver = candidates[0] if candidates else None
    return browser, driver


def fetch_live_match(match: str, rounds: int = 0, verbose: bool = True):
    """Yields {"ts", "event", "data"} dicts for one live HLTV match's scorebot feed.

    match: HLTV match URL or numeric match id.
    rounds: stop after N RoundEnd events (0: run until the caller stops iterating).
    """
    url, _match_id = resolve_url_and_id(match)
    browser, driver_path = find_browser_and_driver()
    state = {"backlog_seen": False, "rounds_done": 0}

    if verbose:
        print(f"launching browser ({browser})...")
    driver = uc.Chrome(driver_executable_path=driver_path, browser_executable_path=browser, headless=True)
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {"source": CAPTURE_JS})
    try:
        if verbose:
            print(f"opening {url}")
        driver.get(url)
        time.sleep(6)   # let the Cloudflare challenge clear and the page's own socket connect
        if verbose:
            print("TITLE:", driver.title)

        while True:
            if driver.service.process.poll() is not None:
                if verbose:
                    print("browser process died, stopping", flush=True)
                return
            time.sleep(POLL_INTERVAL)
            events = driver.execute_script("var e = window.__scorebotEvents; window.__scorebotEvents = []; return e;") or []
            for ev in events:
                name, data = ev.get("event"), ev.get("data")
                if name not in KEPT_EVENTS:
                    continue
                if name in ("log", "fullLog") and isinstance(data, str):
                    data = json.loads(data)
                if name == "log":
                    if not state["backlog_seen"]:
                        state["backlog_seen"] = True   # match history, not live; skip it
                        if verbose:
                            print(f"skipped history log ({len(data.get('log', []))} events)", flush=True)
                        continue
                    round_names = [n for e2 in data.get("log", []) for n in e2]
                    if verbose:
                        print("log", round_names, flush=True)
                    state["rounds_done"] += round_names.count("RoundEnd")
                elif name == "scoreboard" and verbose:
                    print(f"scoreboard {data.get('ctTeamName')} {data.get('counterTerroristScore')} - "
                          f"{data.get('terroristScore')} {data.get('terroristTeamName')} | round {data.get('currentRound')} "
                          f"{data.get('currentRoundState')}", flush=True)
                yield {"ts": ev.get("ts", time.time()), "event": name, "data": data}
                if rounds and state["rounds_done"] >= rounds:
                    if verbose:
                        print(f"{state['rounds_done']} rounds recorded, stopping", flush=True)
                    return
    finally:
        try:
            driver.quit()
        except Exception:
            pass


def main() -> None:
    ap = argparse.ArgumentParser(description="Capture the HLTV scorebot feed for one live match via a real browser")
    ap.add_argument("match", help="HLTV match URL or numeric match id")
    ap.add_argument("--rounds", type=int, default=0, help="stop after N RoundEnd events (default: run until Ctrl+C)")
    args = ap.parse_args()

    count = 0
    try:
        for _event in fetch_live_match(args.match, rounds=args.rounds):
            count += 1
    except KeyboardInterrupt:
        pass
    finally:
        print(f"\n{count} events captured")


if __name__ == "__main__":
    main()
