"""
Instagram video downloader using instagrapi (Instagram Private API).
Use this backend when Instaloader hits 429s: different API and built-in proxy/delays.
Set INSTAGRAM_BACKEND=instagrapi or config["backend"] = "instagrapi" to use.
Login is required (username/password or session file).
"""
import os
import time
import random
from pathlib import Path

from instagrapi import Client
from instagrapi.exceptions import LoginRequired, PrivateError, UserNotFound


class IGDownloaderInstagrapi:
    """Download profile videos using instagrapi (Private API). Same interface as downloader.IGDownloader."""

    def __init__(
        self,
        download_dir="downloads",
        session_file=None,
        username=None,
        password=None,
        initial_delay_secs=0,
        proxy=None,
    ):
        self.cl = Client()
        # Mimic human: random 1–3 s delay after each request (instagrapi best practice)
        self.cl.delay_range = [1, 3]
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)
        self._session_loaded = False
        self._session_file = session_file or os.environ.get("INSTAGRAM_SESSION_FILE")
        self._username = username or os.environ.get("INSTAGRAM_USER")
        self._password = password or os.environ.get("INSTAGRAM_PASSWORD")
        self._initial_delay_secs = initial_delay_secs

        proxy_url = proxy or os.environ.get("INSTAGRAM_PROXY") or os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
        if proxy_url and proxy_url.strip():
            self.cl.set_proxy(proxy_url.strip())
            print("Using proxy for Instagram requests (instagrapi).")

    def _ensure_session(self):
        if self._session_loaded:
            return
        if not self._username or not self._password:
            print("Instagrapi backend requires INSTAGRAM_USER and INSTAGRAM_PASSWORD (or session file).")
            self._session_loaded = True
            return
        session_path = self._session_file
        if session_path and os.path.isfile(session_path):
            try:
                self.cl.load_settings(session_path)
                self.cl.login(self._username, self._password)
                self.cl.get_timeline_feed()
                self._session_loaded = True
                print("Using saved Instagram session (instagrapi).")
                return
            except (LoginRequired, Exception) as e:
                print(f"Session load failed: {e}. Trying fresh login.")
        try:
            self.cl.login(self._username, self._password)
            if session_path:
                self.cl.dump_settings(session_path)
            self._session_loaded = True
            print("Logged in to Instagram (instagrapi).")
        except Exception as e:
            print(f"Login failed: {e}")
        self._session_loaded = True

    def download_profile_videos(self, profile_name, max_count=5):
        print(f"Starting download for profile: {profile_name} (instagrapi backend)")
        self._ensure_session()

        if self._initial_delay_secs > 0:
            print(f"Initial delay: waiting {self._initial_delay_secs}s...")
            time.sleep(self._initial_delay_secs)

        target_dir = self.download_dir / profile_name
        target_dir.mkdir(exist_ok=True)
        existing_videos = list(target_dir.glob("*.mp4"))
        if existing_videos:
            print(f"Found {len(existing_videos)} existing videos locally. Using those.")
            return [str(f) for f in existing_videos[:max_count]]

        try:
            user = self.cl.user_info_by_username(profile_name)
            user_id = str(user.pk)
            # Fetch more than needed so we have enough after filtering videos
            amount = min(max_count * 4, 50)
            medias = self.cl.user_medias(user_id, amount=amount)
            downloaded_files = []
            count = 0

            for media in medias:
                if count >= max_count:
                    break
                if media.media_type != 2:
                    continue
                try:
                    product_type = getattr(media, "product_type", "") or ""
                    if product_type == "clips":
                        path = self.cl.clip_download(media.pk, folder=target_dir)
                    else:
                        path = self.cl.video_download(media.pk, folder=target_dir)
                    if path and Path(path).exists():
                        downloaded_files.append(str(path))
                        count += 1
                        print(f"Downloaded video {count}/{max_count}: {media.code}")
                    if count < max_count:
                        time.sleep(random.uniform(2, 6))
                except Exception as e:
                    print(f"Skip media {media.code}: {e}")
                    continue

            return downloaded_files
        except UserNotFound:
            print(f"Profile not found: {profile_name}")
            return []
        except PrivateError:
            print(f"Profile is private: {profile_name}. Instagrapi requires following or login.")
            return []
        except Exception as e:
            print(f"Error downloading from Instagram (instagrapi): {e}")
            return []
