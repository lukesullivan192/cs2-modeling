import os
import re
import time
import json
import random
import shutil
from seleniumbase import SB
from bs4 import BeautifulSoup
import patoolib


class HLTVDemoFetcher:
    def __init__(self, target_dir="demos", min_stars=0):
        # The final destination for your .dem files
        self.target_dir = os.path.abspath(target_dir)
        os.makedirs(self.target_dir, exist_ok=True)

        # HLTV star-rates each match 0-5 for significance. "Top tier" means
        # filtering on this, not just taking whatever results come back.
        self.min_stars = min_stars

        # SeleniumBase's hardcoded default download directory
        self.sb_download_dir = os.path.abspath("downloaded_files")
        os.makedirs(self.sb_download_dir, exist_ok=True)

        # Record of match IDs we've already downloaded, so re-runs never
        # re-scrape or re-download the same match. Stored alongside the
        # demos themselves.
        self.tracking_file = os.path.join(self.target_dir, ".downloaded_matches.json")
        self.downloaded_match_ids = self._load_downloaded_ids()

    def _load_downloaded_ids(self):
        if os.path.exists(self.tracking_file):
            try:
                with open(self.tracking_file, "r") as f:
                    return set(json.load(f))
            except (json.JSONDecodeError, OSError) as e:
                print(f"Warning: could not read tracking file ({e}), starting fresh.")
                return set()
        return set()

    def _save_downloaded_ids(self):
        with open(self.tracking_file, "w") as f:
            json.dump(sorted(self.downloaded_match_ids), f, indent=2)

    @staticmethod
    def _extract_match_id(url):
        """
        Pulls the numeric match ID out of an HLTV match URL, e.g.
        https://www.hltv.org/matches/2372746/spirit-vs-natus-vincere-...
        -> "2372746"
        Falls back to the full URL if the pattern doesn't match, so we
        still dedupe (just less cleanly) instead of crashing.
        """
        m = re.search(r"/matches/(\d+)/", url)
        return m.group(1) if m else url

    def fetch_top_tier_demos(self, num_matches=5):
        print(f"Setting up undetected browser. Demos will extract to: {self.target_dir}")
        print(f"Already have {len(self.downloaded_match_ids)} match(es) on record; these will be skipped.")

        # Point to the Brave browser executable on Linux
        brave_path = "/usr/bin/brave-browser"

        with SB(uc=True, xvfb=True, ad_block_on=True, headless=False, binary_location=brave_path) as sb:

            tier_1_matches = []  # list of (url, match_id) tuples, new matches only
            seen_urls = set()
            offset = 0
            max_pages_to_check = 10  # Failsafe to prevent infinite loops
            pages_checked = 0

            # Pagination Loop: Keep loading new pages until we hit our target number
            while len(tier_1_matches) < num_matches and pages_checked < max_pages_to_check:
                print(f"Navigating to HLTV results (Offset {offset})...")
                sb.get(f"https://www.hltv.org/results?offset={offset}")

                # Wait out the initial Cloudflare challenge or page load
                sb.sleep(5)

                # Optional: Click cookie acceptance if it obscures elements
                if sb.is_element_visible("button#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll"):
                    sb.click("button#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll")

                # Extract Top Tier Matches using BeautifulSoup
                soup = BeautifulSoup(sb.get_page_source(), "html.parser")
                match_elements = soup.select("div.result-con")

                if not match_elements:
                    print("No more match results found. Stopping search.")
                    break

                for el in match_elements:
                    stars = len(el.select("i.fa-star"))
                    if stars >= self.min_stars:
                        a_tag = el.find("a")
                        if a_tag and "href" in a_tag.attrs:
                            link = "https://www.hltv.org" + a_tag["href"]
                            match_id = self._extract_match_id(link)

                            if match_id in self.downloaded_match_ids:
                                # Already have this one on disk -- skip entirely.
                                continue
                            if link in seen_urls:
                                # Already queued this run -- avoid duplicates.
                                continue

                            seen_urls.add(link)
                            tier_1_matches.append((link, match_id))

                    # Stop searching once we have enough NEW matches
                    if len(tier_1_matches) >= num_matches:
                        break

                print(f"Currently have {len(tier_1_matches)}/{num_matches} new (not-yet-downloaded) matches.")
                offset += 100
                pages_checked += 1

            print(f"Proceeding to download demos for {len(tier_1_matches)} matches.")

            # Visit each match page and download the demo
            for url, match_id in tier_1_matches:
                print(f"\nScraping match: {url.split('/')[-1]}")
                sb.get(url)
                sb.sleep(3)  # Wait for page to fully render

                try:
                    # Use BeautifulSoup to scan the entire HTML regardless of scroll position
                    page_soup = BeautifulSoup(sb.get_page_source(), "html.parser")
                    demo_a_tag = page_soup.select_one('a[href^="/download/demo/"]')

                    if demo_a_tag:
                        demo_link = demo_a_tag.get("href")

                        # Convert relative path to absolute URL if necessary
                        if demo_link and demo_link.startswith("/"):
                            demo_link = "https://www.hltv.org" + demo_link

                        print(f"Found link! Triggering download...")
                        # Snapshot before triggering so we can tell which file
                        # is actually ours, instead of picking up a stale
                        # leftover from a previous failed run.
                        pre_existing = set(os.listdir(self.sb_download_dir))
                        sb.get(demo_link)

                        # Wait for the download in SeleniumBase's folder, then move & extract
                        if self._wait_for_downloads(pre_existing):
                            extracted = self.move_and_extract_demo(pre_existing)
                            if extracted:
                                # Only mark as done once the demo is actually
                                # on disk -- a failed extraction should be
                                # retried on the next run.
                                self.downloaded_match_ids.add(match_id)
                                self._save_downloaded_ids()
                    else:
                        print("No demo link available for this match in the HTML.")

                except Exception as e:
                    print(f"Error occurred while extracting demo: {e}")

                # Randomized cooldown between matches to mimic human behavior
                cooldown = random.uniform(15, 30)
                print(f"Sleeping for {cooldown:.1f} seconds to prevent rate limiting...")
                time.sleep(cooldown)

    def _wait_for_downloads(self, pre_existing, timeout=120):
        """
        Monitors the default downloaded_files folder until a *new* file
        (not present before we triggered the download) shows up and its
        .crdownload counterpart is gone. Only checking "no .crdownload
        exists" is a race: it can fire before the download even starts, or
        match a stale file left over from a previous failed run.
        """
        print("Waiting for download to finish...")
        seconds = 0
        while seconds < timeout:
            files = set(os.listdir(self.sb_download_dir))
            new_files = files - pre_existing
            new_finished = [f for f in new_files if not f.endswith(".crdownload")]
            new_in_progress = [f for f in new_files if f.endswith(".crdownload")]

            if new_finished and not new_in_progress:
                print("Download complete.")
                return True

            time.sleep(2)
            seconds += 2

        print("Warning: Download timed out.")
        return False

    def move_and_extract_demo(self, pre_existing):
        """
        Finds the newly downloaded .rar (i.e. not present before this
        download was triggered), moves it to the demos folder, and
        extracts it. Some HLTV archives contain an internal subfolder
        (e.g. "Downloads/") that patoolib preserves on extraction -- we
        flatten that afterward so every .dem file ends up directly in
        target_dir, never in a nested subfolder.

        Returns True on success, False otherwise, so the caller only marks
        the match as "downloaded" once it actually has the files.
        """
        # Only consider files that appeared after we triggered this
        # download, so a stale .rar from a previous failed run never gets
        # misattributed to the current match.
        current_files = set(os.listdir(self.sb_download_dir))
        new_files = current_files - pre_existing
        files = [os.path.join(self.sb_download_dir, f) for f in new_files if f.endswith('.rar')]
        if not files:
            print("No new .rar file found to extract.")
            return False

        newest_file = max(files, key=os.path.getmtime)
        filename = os.path.basename(newest_file)
        target_file = os.path.join(self.target_dir, filename)

        # Move the file to your desired folder
        print(f"Moving {filename} to {self.target_dir}...")
        shutil.move(newest_file, target_file)

        # Extract the archive
        print(f"Extracting {filename}...")
        try:
            patoolib.extract_archive(target_file, outdir=self.target_dir)
            os.remove(target_file)  # Clean up the .rar file
            self._flatten_demos_folder()
            print("Extraction successful. .dem files are ready.")
            return True
        except Exception as e:
            print(f"Extraction failed: {e}")
            return False

    def _flatten_demos_folder(self):
        """
        Moves any .dem files found in subdirectories of target_dir (e.g.
        a nested "Downloads/" folder from the archive) up to target_dir
        itself, then removes any now-empty subdirectories left behind.
        """
        moved = 0
        for root, _dirs, files in os.walk(self.target_dir):
            if root == self.target_dir:
                continue
            for f in files:
                if f.lower().endswith(".dem"):
                    src = os.path.join(root, f)
                    dst = os.path.join(self.target_dir, f)
                    if os.path.exists(dst):
                        # Avoid overwriting a same-named demo already there.
                        base, ext = os.path.splitext(f)
                        counter = 1
                        while os.path.exists(dst):
                            dst = os.path.join(self.target_dir, f"{base}_{counter}{ext}")
                            counter += 1
                    shutil.move(src, dst)
                    moved += 1

        # Remove now-empty subdirectories, deepest first.
        for root, _dirs, _files in os.walk(self.target_dir, topdown=False):
            if root == self.target_dir:
                continue
            if not os.listdir(root):
                os.rmdir(root)

        if moved:
            print(f"Flattened {moved} demo file(s) out of nested subfolder(s).")


if __name__ == "__main__":
    fetcher = HLTVDemoFetcher()
    # You can now safely request a larger number like 10 or 20
    fetcher.fetch_top_tier_demos(num_matches=10)