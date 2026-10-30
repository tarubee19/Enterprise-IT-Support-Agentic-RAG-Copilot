"""Download the public-domain 18F/TTS Handbook (CC0) used as the realistic test corpus, pinned to a commit."""
import subprocess
from pathlib import Path

REPO = "https://github.com/18F/handbook"
COMMIT = "c220f93896b39e9e4ab0f486a1d5bf346ac06e3e"
DEST = Path(__file__).resolve().parent / "corpus" / "handbook"

if not (DEST / "pages").exists():
    DEST.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "-q", REPO, str(DEST)], check=True)
    subprocess.run(["git", "-C", str(DEST), "checkout", "-q", COMMIT], check=True)
print("corpus at", DEST)
