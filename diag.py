"""Plain-text diagnostics. Writes stage names, counts and error details, never what was said."""
import os
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ERROR_LOG = os.environ.get("VOICE_ERROR_LOG") or os.path.join(HERE, "logs", "errors.log")  # tests point this at a temp file


def log_line(message: str, path: str = None):
    path = path or os.environ.get("VOICE_ERROR_LOG") or ERROR_LOG   # read at call time: tests set it after import
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + message + "\n")
    except OSError:
        pass


def log_exception(stage: str, path: str = None):
    log_line("ERROR at %s:\n%s" % (stage, traceback.format_exc()), path)
