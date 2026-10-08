# SAVERouter explainer videos

This directory contains the reproducible source for the English and Chinese
SAVERouter explainer videos. The renderer uses the paper method figure and main
results image already stored under `assets/`.

```bash
python -m pip install edge-tts imageio-ffmpeg numpy pillow qrcode
python media/video/make_videos.py
```

The script downloads the Noto Sans CJK font into an ignored cache, synthesizes
the two narrations, renders 1080p slides, and writes MP4 and SRT files under
`media/video/output/`.

Generated media is distributed through the GitHub Release rather than Git
history. See `VOICEOVER.md` for the editable scripts.
