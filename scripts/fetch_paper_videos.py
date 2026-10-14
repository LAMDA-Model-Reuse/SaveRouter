"""Fetch public paper media into a Pages artifact; keep MP4s out of Git history."""
from pathlib import Path
import argparse
import hashlib
import json
import urllib.request
import urllib.error

parser = argparse.ArgumentParser()
parser.add_argument("output", type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "docs/paper_videos.json").read_text())
args.output.mkdir(parents=True, exist_ok=True)
base = manifest["download_base"]

for item in manifest["files"]:
    name = item["name"]
    if Path(name).name != name or not name.endswith((".mp4", ".srt", ".vtt")):
        raise ValueError("Unexpected media filename")
    target = args.output / name
    if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == item["sha256"]:
        print("Cached", name, flush=True)
        continue
    temporary = target.with_suffix(target.suffix + ".part")
    for attempt in range(3):
        # Cache-keyed GET retries are safe; never accept a truncated response.
        url = base + "/" + name + "?sha256=" + item["sha256"] + "&attempt=" + str(attempt)
        request = urllib.request.Request(url, headers={"User-Agent": "paper-video-pages/1.0", "Cache-Control": "no-cache"})
        checksum = hashlib.sha256()
        try:
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as out:
                while data := response.read(1024 * 1024):
                    checksum.update(data)
                    out.write(data)
            if temporary.stat().st_size != item["bytes"] or checksum.hexdigest() != item["sha256"]:
                raise ValueError("Downloaded media is incomplete or has a checksum mismatch")
            break
        except (urllib.error.URLError, OSError, ValueError):
            if attempt == 2:
                raise RuntimeError("Unable to fetch verified media: " + name)
            print("Retrying verified GET:", name, flush=True)
    temporary.replace(target)
    print("Verified", name, flush=True)
