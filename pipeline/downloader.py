import instaloader
import os
import time
import random
from pathlib import Path

# Realistic Instagram app user agents (iOS/Android) to reduce fingerprinting and rate limits.
# Rotating to a single random one per run avoids looking like the default Instaloader client.
INSTAGRAM_USER_AGENTS = [
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Instagram 312.0.0.0.0 (iPhone14,2; iOS 17_0; en_US; en; scale=3.00; 1179x2556; 0)",
    "Mozilla/5.0 (Linux; Android 14; SM-S918B Build/UP1A.231005.007; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/119.0.6045.193 Mobile Safari/537.36 Instagram 312.0.0.0.0 Android (34/14; 420dpi; 1080x2340; samsung; SM-S918B; b0q; qcom; en_GB; 0)",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/22A3370 Instagram 351.0.1.35.98 (iPhone16,1; iOS 18_0_1; en_US; en-US; scale=3.00; 1179x2556; 0)",
    "Mozilla/5.0 (Linux; Android 15; Pixel 8 Pro Build/AP4A.241205.024; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/120.0.6099.144 Mobile Safari/537.36 Instagram 320.0.0.0.0 Android (35/15; 420dpi; 1080x2400; Google; Pixel 8 Pro; shiba; qcom; en_US; 0)",
]


def _normalize_proxy(url: str) -> dict:
    """Turn a single proxy URL into requests-style proxies dict."""
    url = (url or "").strip()
    if not url:
        return {}
    if not url.startswith(("http://", "https://", "socks4://", "socks5://")):
        url = "http://" + url
    return {"http": url, "https": url}


class ConservativeRateController(instaloader.RateController):
    """Reduces request rate to avoid 429 Too Many Requests.
    Fewer requests per 11-min window and extra sleep between requests.
    """
    def count_per_sliding_window(self, query_type: str) -> int:
        # Allow only 8 requests per 11-minute window (default is higher)
        return 8

    def sleep(self, secs: float) -> None:
        # Add extra delay: at least 20s, plus up to 40s random to appear less automated
        extra = random.uniform(20, 40)
        total = secs + extra
        if total > 0:
            print(f"Rate limit: sleeping {total:.0f}s before next request...")
            time.sleep(total)


class IGDownloader:
    def __init__(
        self,
        download_dir="downloads",
        session_file=None,
        username=None,
        password=None,
        initial_delay_secs=0,
        user_agent=None,
        proxy=None,
    ):
        # User-Agent: custom, env, or random Instagram-style mobile UA to reduce rate limits
        ua = (
            user_agent
            or os.environ.get("INSTAGRAM_USER_AGENT")
            or random.choice(INSTAGRAM_USER_AGENTS)
        )
        self.L = instaloader.Instaloader(
            download_pictures=False,
            download_videos=True,
            download_video_thumbnails=False,
            download_geotags=False,
            download_comments=False,
            save_metadata=True,
            compress_json=False,
            rate_controller=lambda ctx: ConservativeRateController(ctx),
            user_agent=ua,
        )
        # Proxy: config, env, or requests convention (HTTP_PROXY/HTTPS_PROXY)
        proxy_url = proxy or os.environ.get("INSTAGRAM_PROXY") or os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
        if proxy_url:
            proxies = _normalize_proxy(proxy_url)
            if proxies:
                self.L.context._session.proxies = proxies
                print("Using proxy for Instagram requests.")
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)
        self._session_loaded = False
        self._session_file = session_file or os.environ.get("INSTAGRAM_SESSION_FILE")
        self._username = username or os.environ.get("INSTAGRAM_USER")
        self._password = password or os.environ.get("INSTAGRAM_PASSWORD")
        self._initial_delay_secs = initial_delay_secs

    def _ensure_session(self):
        """Load or login session once. Logged-in access has better rate limits."""
        if self._session_loaded:
            return
        if self._session_file and os.path.isfile(self._session_file):
            try:
                self.L.load_session_from_file(self._username or "", self._session_file)
                if self.L.test_login():
                    self._session_loaded = True
                    print("Using saved Instagram session (better rate limits).")
                    return
            except Exception as e:
                print(f"Session load failed: {e}")
        if self._username and self._password:
            try:
                self.L.login(self._username, self._password)
                self._session_loaded = True
                if self._session_file:
                    self.L.save_session_to_file(self._session_file)
                print("Logged in to Instagram (better rate limits).")
            except Exception as e:
                print(f"Login failed: {e}")
        self._session_loaded = True  # avoid retrying every time

    def download_profile_videos(self, profile_name, max_count=5):
        print(f"Starting download for profile: {profile_name}")
        self._ensure_session()

        if self._initial_delay_secs > 0:
            print(f"Initial delay: waiting {self._initial_delay_secs}s (rate limit cooldown)...")
            time.sleep(self._initial_delay_secs)

        # Check if we already have files locally to avoid rate limits
        target_dir = self.download_dir / profile_name
        if target_dir.exists():
            existing_videos = list(target_dir.glob("*.mp4"))
            if existing_videos:
                print(f"Found {len(existing_videos)} existing videos locally. Using those.")
                return [str(f) for f in existing_videos[:max_count]]

        try:
            profile = instaloader.Profile.from_username(self.L.context, profile_name)
            count = 0
            downloaded_files = []

            for post in profile.get_posts():
                if post.is_video:
                    self.L.download_post(post, target=str(target_dir))
                    count += 1
                    print(f"Downloaded video {count}/{max_count}: {post.shortcode}")

                    for file in target_dir.glob(f"*{post.shortcode}*.mp4"):
                        downloaded_files.append(str(file))

                    if count >= max_count:
                        break
                    # Space out requests to reduce 429 risk (5–15 s between posts)
                    delay = random.uniform(5, 15)
                    print(f"Waiting {delay:.1f}s before next request...")
                    time.sleep(delay)
            return downloaded_files
        except Exception as e:
            print(f"Error downloading from Instagram: {e}")
            print(
                "Tip: Use a session (INSTAGRAM_SESSION_FILE + INSTAGRAM_USER) or login "
                "(INSTAGRAM_USER + INSTAGRAM_PASSWORD) for better limits. "
                "If already rate-limited, wait ~30–60 min or set initial_delay_secs."
            )
            return []

if __name__ == "__main__":
    # Test with a public account if needed
    # downloader = IGDownloader()
    # downloader.download_profile_videos("instagram", max_count=1)
    pass
