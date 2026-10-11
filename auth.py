"""PIN check with lockout, plus the header checks that keep this service Tailscale-only in spirit."""
import hmac
import os
import secrets
import time
from urllib.parse import urlparse

MAX_TRIES = 5
LOCK_SECONDS = 15 * 60


def load_or_create_pin(path: str) -> str:
    """Read the PIN file; create it with 6 random digits on first run. Returns the PIN."""
    if os.path.exists(path):
        pin = open(path).read().strip()
        if pin:
            return pin
    pin = "".join(secrets.choice("0123456789") for _ in range(6))
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(pin + "\n")
    return pin


class PinGuard:
    """One user, so the lockout is global: after MAX_TRIES wrong PINs nothing is accepted for LOCK_SECONDS."""

    def __init__(self, pin: str, clock=time.monotonic):
        self._pin = pin
        self._clock = clock
        self._fails = 0
        self._locked_until = 0.0

    def locked_for(self) -> int:
        left = self._locked_until - self._clock()
        return int(left) + 1 if left > 0 else 0

    def check(self, attempt: str):
        """Returns (ok, seconds_locked). A correct PIN while locked is still refused."""
        left = self.locked_for()
        if left:
            return False, left
        if isinstance(attempt, str) and hmac.compare_digest(attempt.encode(), self._pin.encode()):
            self._fails = 0
            return True, 0
        return self.record_failure()

    def reset(self):
        self._fails = 0

    def record_failure(self):
        """A failed sign-in of any kind (wrong PIN, bad Face ID passkey) counts toward the same lockout."""
        self._fails += 1
        if self._fails >= MAX_TRIES:
            self._fails = 0
            self._locked_until = self._clock() + LOCK_SECONDS
            return False, LOCK_SECONDS
        return False, 0


def origin_ok(headers) -> bool:
    """Browsers always send Origin on a WebSocket. It must be the same site as the page we served."""
    origin = headers.get("Origin")
    if not origin:
        return True  # non-browser client; it still needs the PIN
    host = headers.get("Host", "")
    return urlparse(origin).netloc == host


def login_ok(headers, expected: str) -> bool:
    """When VOICE_ALLOWED_LOGIN is set, `tailscale serve` must have put the same login in the header."""
    if not expected:
        return True
    return headers.get("Tailscale-User-Login", "").lower() == expected.lower()


def host_allowed(host: str) -> bool:
    return host not in ("0.0.0.0", "::", "")
