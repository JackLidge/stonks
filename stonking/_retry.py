import threading
import time

from yfinance.exceptions import YFRateLimitError


class RateLimiter:
    """Thread-safe rate limiter — enforces a minimum gap between requests across all workers."""

    def __init__(self, pause: float):
        self._pause = pause
        self._lock = threading.Lock()
        self._last = 0.0

    def acquire(self):
        if self._pause <= 0:
            return
        with self._lock:
            elapsed = time.time() - self._last
            if elapsed < self._pause:
                time.sleep(self._pause - elapsed)
            self._last = time.time()


def with_retry(fn, *args, retries: int = 4, base_delay: float = 60.0,
               rate_limiter: RateLimiter | None = None, **kwargs):
    """
    Call fn(*args, **kwargs), retrying on YFRateLimitError with exponential backoff.
    Acquires a rate limiter slot before each attempt if one is supplied.
    Default delays on rate limit: 60s, 120s, 240s, 480s.
    """
    for attempt in range(retries + 1):
        if rate_limiter:
            rate_limiter.acquire()
        try:
            return fn(*args, **kwargs)
        except YFRateLimitError:
            if attempt == retries:
                raise
            wait = base_delay * (2 ** attempt)
            ticker = args[0] if args else "?"
            print(f"\nRate limited ({ticker}) — retrying in {wait:.0f}s "
                  f"(attempt {attempt + 1}/{retries})")
            time.sleep(wait)
