"""One place for the version number. Bump it on every change that goes live, so the page and the log show which build is running."""
import glob
import os
import time

VERSION = "v65"
_HERE = os.path.dirname(os.path.abspath(__file__))
# BUILT = when the newest code file changed, so the page date can never be stale again.
BUILT = time.strftime("%Y-%m-%d %H:%M", time.localtime(max(
    os.path.getmtime(f) for f in glob.glob(os.path.join(_HERE, "*.py")) + glob.glob(os.path.join(_HERE, "static", "*")))))
