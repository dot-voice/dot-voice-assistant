"""Face ID sign-in (WebAuthn passkey), so the PIN is typed once per device, not on every call. The PIN stays: it enrolls a device and is the fallback.

- RP ID / origin are fixed in config, never taken from the Host header. Renaming the Tailscale node breaks enrolled
  passkeys: enroll again with the PIN (VOICE_RP_ID changes the name).
- Login challenges are single-use and expire; failures count toward the PIN lockout (server.py).
- passkeys.json holds public keys only; it is gitignored like pin.txt.
"""
import json
import os
import threading
import time

from webauthn import (generate_authentication_options, generate_registration_options, options_to_json,
                      verify_authentication_response, verify_registration_response)
from webauthn.helpers import base64url_to_bytes, bytes_to_base64url
from webauthn.helpers.structs import (AuthenticatorSelectionCriteria, PublicKeyCredentialDescriptor, ResidentKeyRequirement,
                                      UserVerificationRequirement)

import config

HERE = os.path.dirname(os.path.abspath(__file__))
STORE = os.path.join(HERE, "passkeys.json")
RP_ID = os.environ.get("VOICE_RP_ID", config.get("rp_id"))
CHALLENGE_TTL_S = 600
MAX_PENDING = 20


def _client_challenge(credential: dict) -> str:
    raw = credential.get("response", {}).get("clientDataJSON", "")
    data = json.loads(base64url_to_bytes(raw).decode())
    return data.get("challenge", "")


class Passkeys:
    def __init__(self, path=STORE, rp_id=RP_ID, clock=time.time):
        self.path, self.rp_id, self.origin = path, rp_id, "https://" + rp_id
        self.clock = clock
        self._pending = {}   # login challenge (base64url) -> expiry
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ store
    def _load(self) -> list:
        try:
            return json.load(open(self.path)).get("credentials", [])
        except (OSError, ValueError):
            return []

    def _save(self, creds):
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"rp_id": self.rp_id, "credentials": creds}, f, indent=2)
        os.chmod(tmp, 0o600)
        os.replace(tmp, self.path)

    def count(self) -> int:
        return len(self._load())

    # ------------------------------------------------------------------ login
    def login_options(self):
        """Options for navigator.credentials.get, or None when no device is enrolled."""
        creds = self._load()
        if not creds:
            return None
        opts = generate_authentication_options(
            rp_id=self.rp_id, user_verification=UserVerificationRequirement.REQUIRED,
            allow_credentials=[PublicKeyCredentialDescriptor(id=base64url_to_bytes(c["id"])) for c in creds])
        now = self.clock()
        with self._lock:
            self._pending = {k: v for k, v in self._pending.items() if v > now}
            if len(self._pending) >= MAX_PENDING:
                self._pending.pop(min(self._pending, key=self._pending.get))
            self._pending[bytes_to_base64url(opts.challenge)] = now + CHALLENGE_TTL_S
        return json.loads(options_to_json(opts))

    def verify_login(self, credential: dict):
        """True = signed in, False = refused (counts toward the lockout), None = stale challenge (service restarted or
        the page sat open past the TTL): not the caller's fault, the page fetches a new one and they tap again."""
        try:
            challenge = _client_challenge(credential)
        except Exception:
            return False
        with self._lock:
            expiry = self._pending.pop(challenge, 0)   # single use, whatever happens next
        if expiry < self.clock():
            return None
        creds = self._load()
        match = next((c for c in creds if c["id"] == credential.get("id")), None)
        if not match:
            return False
        try:
            v = verify_authentication_response(
                credential=credential, expected_challenge=base64url_to_bytes(challenge), expected_rp_id=self.rp_id,
                expected_origin=self.origin, credential_public_key=base64url_to_bytes(match["public_key"]),
                credential_current_sign_count=match.get("sign_count", 0), require_user_verification=True)
        except Exception:
            return False
        match["sign_count"] = v.new_sign_count
        match["last_used"] = time.strftime("%Y-%m-%d %H:%M")
        self._save(creds)
        return True

    # ------------------------------------------------------------------ enroll (only on a connection that passed the PIN)
    def register_options(self):
        opts = generate_registration_options(
            rp_id=self.rp_id, rp_name="Voice Call", user_name=config.user_name(), user_id=(config.user_name() + "-voice-call").encode(),
            user_display_name=config.get("owner"),
            authenticator_selection=AuthenticatorSelectionCriteria(resident_key=ResidentKeyRequirement.PREFERRED,
                                                                   user_verification=UserVerificationRequirement.REQUIRED),
            exclude_credentials=[PublicKeyCredentialDescriptor(id=base64url_to_bytes(c["id"])) for c in self._load()])
        return json.loads(options_to_json(opts)), bytes_to_base64url(opts.challenge)

    def verify_register(self, credential: dict, challenge: str, label: str = "") -> bool:
        try:
            v = verify_registration_response(credential=credential, expected_challenge=base64url_to_bytes(challenge),
                                             expected_rp_id=self.rp_id, expected_origin=self.origin,
                                             require_user_verification=True)
        except Exception:
            return False
        creds = [c for c in self._load() if c["id"] != bytes_to_base64url(v.credential_id)]
        creds.append({"id": bytes_to_base64url(v.credential_id), "public_key": bytes_to_base64url(v.credential_public_key),
                      "sign_count": v.sign_count, "label": label[:60], "created": time.strftime("%Y-%m-%d %H:%M")})
        self._save(creds)
        return True
