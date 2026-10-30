"""Dot as the front door to every module (dotapi).

Provides:
1. Feed reader for replies to Dot (feed.jsonl tells to VOICE starting with "DOT:" or "NEEDS HER:")
2. Session state lookup (busy / live / asleep / unknown)
3. Headless module execution via Claude CLI
4. Asynchronous routing to active sessions or background headless jobs
"""
import json
import os
import shutil
import re
import subprocess
import threading
import time

import config
from diag import log_line
import updates

HERE = os.path.dirname(os.path.abspath(__file__))
FEED = os.environ.get("VOICE_FEED", config.path("feed_path"))
CURSOR = os.path.join(HERE, "data", "feed_cursor.json")
STATES = config.path("session_states")
PROJECTS = os.path.expanduser("~/.claude/projects")
SESSIONS_DIR = os.path.expanduser("~/Library/Application Support/Claude/claude-code-sessions")

try:
    ROLES = config.path("roles_json")
except Exception:
    ROLES = os.path.expanduser("~/.dot/roles.json")

try:
    _claude_bin = config.get("claude_bin")
except Exception:
    _claude_bin = None
CLAUDE = _claude_bin or shutil.which("claude") or "claude"   # launchd PATH may lack it: set claude_bin in local_config

BUSY_S = 180
STATE_MAX_AGE_S = 600
HEADLESS_TIMEOUT_S = 300
DESKTOP_PROC = "disclaimer"

PHONE_RULES = (
    "You are answering the owner by phone through Dot. Reply in English only, at most 3 short spoken sentences, "
    "no markdown, no lists, no phone numbers or emails of clients. Never send messages or email, never delete, "
    "never spend money, never change the calendar: if the request needs that, say it needs her yes and stop. "
    "Work only inside this folder."
)


def _load_cursor(cursor_path):
    try:
        with open(cursor_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return int(data.get("offset", 0))
            return int(data)
    except Exception:
        return None


def _save_cursor(cursor_path, offset):
    try:
        dir_name = os.path.dirname(os.path.abspath(cursor_path))
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        tmp = cursor_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"offset": int(offset)}, f)
        os.replace(tmp, cursor_path)
    except Exception:
        pass


def read_tells(feed_path=None, cursor_path=None):
    try:
        feed_path = feed_path or FEED
        cursor_path = cursor_path or CURSOR
        try:
            feed_size = os.path.getsize(feed_path)
        except OSError:
            return []

        if not os.path.exists(cursor_path):
            _save_cursor(cursor_path, feed_size)
            return []

        offset = _load_cursor(cursor_path)
        if offset is None or offset > feed_size:
            offset = 0

        current_offset = offset
        records = []
        with open(feed_path, "rb") as f:
            f.seek(offset)
            while True:
                line_bytes = f.readline()
                if not line_bytes:
                    break
                if not line_bytes.endswith(b"\n"):
                    break
                current_offset += len(line_bytes)
                try:
                    line_str = line_bytes.decode("utf-8", errors="replace")
                    rec = json.loads(line_str)
                except Exception:
                    continue
                if not isinstance(rec, dict):
                    continue
                if rec.get("to") != "VOICE":
                    continue
                text = rec.get("text")
                if not isinstance(text, str):
                    continue
                stripped = text.strip()
                if stripped.startswith("DOT:"):
                    needs_her = False
                    clean_text = stripped[4:].strip()
                elif stripped.startswith("NEEDS HER:"):
                    needs_her = True
                    clean_text = stripped[10:].strip()
                else:
                    continue
                frm = str(rec.get("from_role") or rec.get("from") or "ORG").upper()
                records.append({"from": frm, "text": clean_text, "needs_her": needs_her})

        _save_cursor(cursor_path, current_offset)
        return records
    except Exception:
        return []


def pump_tells(feed_path=None, cursor_path=None, post_fn=None) -> int:
    try:
        tells = read_tells(feed_path=feed_path, cursor_path=cursor_path)
        if not tells:
            return 0
        if post_fn is None:
            post_fn = updates.post
        count = 0
        for t in tells:
            try:
                post_fn(t["from"], t["text"], needs_her=t["needs_her"])
            except TypeError as e:
                if "needs_her" in str(e):
                    post_fn(t["from"], t["text"])
                else:
                    raise
            count += 1
        return count
    except Exception:
        return 0


def module_folder(role, roles_path=None):
    if not role:
        return None
    role_u = str(role).strip().upper()
    roles_path = roles_path or ROLES
    try:
        roles_path = os.path.expanduser(roles_path)
        with open(roles_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    for folder, info in data.items():
        if not isinstance(info, dict):
            continue
        if "retired_to" in info:
            continue
        session = str(info.get("session", "")).strip().upper()
        tasks = [str(t).strip().upper() for t in info.get("tasks", []) if t]
        if role_u == session or role_u in tasks:
            return folder
    return None


def project_dir(folder, projects=None):
    base = projects or PROJECTS
    clean_name = re.sub(r"[^A-Za-z0-9]", "-", os.path.realpath(folder))
    return os.path.join(base, clean_name)


def _lsof_folders():
    try:
        proc = subprocess.run(
            ["lsof", "-a", "-d", "cwd", "-c", DESKTOP_PROC, "-Fn"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        paths = []
        for line in proc.stdout.splitlines():
            if line.startswith("n"):
                paths.append(line[1:])
        return paths
    except Exception:
        return []


def _sessions_dir_folders(sessions_dir):
    if not sessions_dir:
        return []
    try:
        root_dir = os.path.expanduser(sessions_dir)
        if not os.path.isdir(root_dir):
            return []
        folders = []
        seen = 0
        for dirpath, _dirnames, filenames in os.walk(root_dir):
            for fname in filenames:
                if not fname.endswith(".json"):
                    continue
                if seen >= 2000:
                    return folders
                seen += 1
                fpath = os.path.join(dirpath, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    continue
                if not isinstance(data, dict) or data.get("isArchived"):
                    continue
                for key in ("cwd", "originCwd"):
                    val = data.get(key)
                    if isinstance(val, str) and val and (key == "cwd" or val != data.get("cwd")):
                        folders.append(val)
        return folders
    except Exception:
        return []


def desktop_folders(lister=None, sessions_dir=None):
    try:
        if lister is not None:
            res = lister()
            paths = list(res) if res is not None else []
            if sessions_dir is not None:
                paths.extend(_sessions_dir_folders(sessions_dir))
            return paths
        paths = _lsof_folders()
        sdir = SESSIONS_DIR if sessions_dir is None else sessions_dir
        paths.extend(_sessions_dir_folders(sdir))
        return paths
    except Exception:
        return []


def desktop_open(folder, lister=None, sessions_dir=None):
    if not folder:
        return False
    try:
        target = os.path.realpath(os.path.expanduser(folder))
        prefix = target if target.endswith(os.sep) else target + os.sep
        for p in desktop_folders(lister=lister, sessions_dir=sessions_dir):
            if not p:
                continue
            rp = os.path.realpath(os.path.expanduser(p))
            if rp == target or rp.startswith(prefix):
                return True
        return False
    except Exception:
        return False


def session_state(role, roles_path=None, projects=None, states_path=None, now=None, lister=None, sessions_dir=None):
    now_ts = now if now is not None else time.time()
    role_u = str(role).strip().upper()
    states_file = states_path or STATES

    # 1) states file entry newer than STATE_MAX_AGE_S
    try:
        p = os.path.expanduser(states_file)
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                sdata = json.load(f)
            if isinstance(sdata, dict):
                entry = sdata.get(role_u) or sdata.get(role)
                if isinstance(entry, dict):
                    ts = entry.get("ts")
                    if ts is not None and (now_ts - float(ts)) <= STATE_MAX_AGE_S:
                        st = str(entry.get("state", "")).strip().lower()
                        if st == "busy":
                            return "busy"
                        elif st in ("live", "idle"):
                            return "live"
                        elif st in ("offline", "asleep"):
                            return "asleep"
    except Exception:
        pass

    # 2) else folder unknown -> "unknown"; open desktop -> "live"; newest *.jsonl mtime in project_dir within BUSY_S -> "busy"; else "asleep"
    folder = module_folder(role, roles_path=roles_path)
    if not folder:
        return "unknown"
    if lister is not None:
        try:
            lsof_paths = list(lister() or [])
        except Exception:
            lsof_paths = []
        sess_paths = _sessions_dir_folders(sessions_dir) if sessions_dir is not None else []
    else:
        lsof_paths = _lsof_folders()
        sdir = SESSIONS_DIR if sessions_dir is None else sessions_dir
        sess_paths = _sessions_dir_folders(sdir)
    if not lsof_paths:
        log_line("dotapi: lsof empty, sessions_dir folders=%d" % len(sess_paths))
    if desktop_open(folder, lister=lambda: lsof_paths + sess_paths):
        return "live"
    pdir = project_dir(folder, projects=projects)
    newest_mtime = None
    try:
        if os.path.isdir(pdir):
            for name in os.listdir(pdir):
                if name.endswith(".jsonl"):
                    full_p = os.path.join(pdir, name)
                    try:
                        mt = os.path.getmtime(full_p)
                        if newest_mtime is None or mt > newest_mtime:
                            newest_mtime = mt
                    except OSError:
                        continue
    except OSError:
        pass

    if newest_mtime is not None and (now_ts - newest_mtime) <= BUSY_S:
        return "busy"
    return "asleep"


def run_headless(role, words, folder=None, model="sonnet", runner=subprocess.run, lock_dir=None, now=time.time):
    wall_start = time.time()
    try:
        current_time = now() if callable(now) else time.time()
        target_folder = folder or module_folder(role)
        data_dir = lock_dir or os.path.join(HERE, "data")
        try:
            os.makedirs(data_dir, exist_ok=True)
        except Exception:
            pass
        lock_file = os.path.join(data_dir, "headless_%s.lock" % role)

        acquired = False
        try:
            fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(current_time).encode("utf-8"))
            os.close(fd)
            acquired = True
        except FileExistsError:
            try:
                mtime = os.path.getmtime(lock_file)
                if (current_time - mtime) < HEADLESS_TIMEOUT_S:
                    return {
                        "ok": False,
                        "text": "busy",
                        "seconds": 0.0,
                        "tokens": 0,
                        "cost_usd": 0.0,
                        "model": model,
                    }
                try:
                    os.remove(lock_file)
                except OSError:
                    pass
                fd = os.open(lock_file, os.O_CREAT | os.O_WRONLY | os.O_TRUNC)
                os.write(fd, str(current_time).encode("utf-8"))
                os.close(fd)
                acquired = True
            except Exception:
                return {
                    "ok": False,
                    "text": "busy",
                    "seconds": 0.0,
                    "tokens": 0,
                    "cost_usd": 0.0,
                    "model": model,
                }
        except Exception:
            return {
                "ok": False,
                "text": "busy",
                "seconds": 0.0,
                "tokens": 0,
                "cost_usd": 0.0,
                "model": model,
            }

        try:
            cmd = [
                CLAUDE,
                "-p",
                words,
                "--output-format",
                "json",
                "--model",
                model,
                "--strict-mcp-config",
                "--allowedTools",
                "Read,Grep,Glob,Edit,Write",
                "--permission-mode",
                "acceptEdits",
                "--max-turns",
                "12",
                "--no-session-persistence",
                "--append-system-prompt",
                PHONE_RULES,
            ]
            env = {
                k: v for k, v in os.environ.items()
                if not (k.startswith("CLAUDECODE") or k.startswith("CLAUDE_CODE_"))
            }

            try:
                proc = runner(
                    cmd,
                    cwd=target_folder,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=HEADLESS_TIMEOUT_S,
                )
            except subprocess.TimeoutExpired:
                return {
                    "ok": False,
                    "text": "headless timeout",
                    "seconds": round(time.time() - wall_start, 2),
                    "tokens": 0,
                    "cost_usd": 0.0,
                    "model": model,
                }
            except Exception as e:
                return {
                    "ok": False,
                    "text": f"headless runner error: {e}",
                    "seconds": round(time.time() - wall_start, 2),
                    "tokens": 0,
                    "cost_usd": 0.0,
                    "model": model,
                }

            wall_seconds = round(time.time() - wall_start, 2)
            err_tail = ""
            if proc.returncode != 0:
                stderr = getattr(proc, "stderr", "") or ""
                err_tail = stderr.strip()[-300:]
                log_line("dotapi: headless exit=%s err=%s" % (proc.returncode, " ".join(err_tail.split())[-120:]))

            stdout = getattr(proc, "stdout", "") or ""
            first_brace = stdout.find("{")
            last_brace = stdout.rfind("}")
            if first_brace == -1 or last_brace == -1 or last_brace < first_brace:
                res = {
                    "ok": False,
                    "text": f"headless failed with exit {proc.returncode}" if proc.returncode != 0 else "no json in stdout",
                    "seconds": wall_seconds,
                    "tokens": 0,
                    "cost_usd": 0.0,
                    "model": model,
                }
                if proc.returncode != 0:
                    res["error"] = err_tail
                return res

            json_str = stdout[first_brace : last_brace + 1]
            try:
                data = json.loads(json_str)
            except Exception:
                res = {
                    "ok": False,
                    "text": f"headless failed with exit {proc.returncode}" if proc.returncode != 0 else "invalid json in stdout",
                    "seconds": wall_seconds,
                    "tokens": 0,
                    "cost_usd": 0.0,
                    "model": model,
                }
                if proc.returncode != 0:
                    res["error"] = err_tail
                return res

            is_error = bool(data.get("is_error", False))
            result_raw = data.get("result") or data.get("text") or ""
            text = " ".join(str(result_raw).split())[:400]

            usage = data.get("usage") or {}
            tokens = int(
                usage.get("input_tokens", 0)
                + usage.get("cache_creation_input_tokens", 0)
                + usage.get("cache_read_input_tokens", 0)
                + usage.get("output_tokens", 0)
            )
            cost_usd = float(data.get("total_cost_usd") or data.get("cost_usd") or 0.0)

            ok = (not is_error) and (proc.returncode == 0 or bool(text))
            res = {
                "ok": ok,
                "text": text if ok else (text or (f"headless failed with exit {proc.returncode}" if proc.returncode != 0 else "error")),
                "seconds": wall_seconds,
                "tokens": tokens,
                "cost_usd": cost_usd,
                "model": model,
            }
            if proc.returncode != 0:
                res["error"] = err_tail
            return res
        finally:
            if acquired:
                try:
                    os.remove(lock_file)
                except OSError:
                    pass
    except Exception as e:
        return {
            "ok": False,
            "text": str(e),
            "seconds": round(time.time() - wall_start, 2),
            "tokens": 0,
            "cost_usd": 0.0,
            "model": model,
        }


def ask_module(role, words, tell_fn, post_fn=None, log_fn=None, state_fn=None, headless_fn=None, spawn=None) -> str:
    state = (state_fn or session_state)(role)
    Role = str(role).title()

    if state in ("busy", "live"):
        tell_fn(role, words)
        if log_fn:
            log_fn("dotapi: role=%s path=queued state=%s" % (role, state))
        return "%s is awake, I passed it on. I will tell you when it replies." % Role

    if state == "asleep":
        def job():
            fn = headless_fn or run_headless
            r = fn(role, words)
            target_post = post_fn or updates.post
            if r.get("ok"):
                res_text = r["text"]
            else:
                try:
                    tell_fn(role, words)
                except Exception:
                    pass
                res_text = "%s could not finish it in the background, so I passed it to the %s session; it will answer when it is open." % (Role, Role)
            try:
                target_post(role, res_text)
            except Exception:
                pass
            if log_fn:
                log_fn(
                    "dotapi: role=%s path=headless ok=%s seconds=%.1f tokens=%d model=%s"
                    % (role, r.get("ok"), float(r.get("seconds", 0.0)), int(r.get("tokens", 0)), r.get("model", ""))
                )
            if r.get("ok"):
                try:
                    tell_fn(role, "Dot headless job did this for her (her words: %s): %s" % (words[:200], str(r.get("text", ""))[:300]))
                except Exception:
                    pass

        sp = spawn or (lambda f: threading.Thread(target=f, daemon=True).start())
        sp(job)
        return "%s was asleep, so I started a background job. About a minute; I will say the answer when it lands." % Role

    tell_fn("ORG", words)
    return "I passed it to ORG, it will route it."
