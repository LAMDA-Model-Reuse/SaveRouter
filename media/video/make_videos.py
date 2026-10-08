#!/usr/bin/env python3
"""Render the English and Chinese SAVERouter explainer videos."""

from __future__ import annotations

import argparse
import asyncio
import re
import subprocess
import urllib.request
from pathlib import Path

import edge_tts
import imageio_ffmpeg
import numpy as np
import qrcode
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
WORK = HERE / "work"
OUTPUT = HERE / "output"
CACHE = HERE / ".cache"
WIDTH, HEIGHT = 1920, 1080

PAPER = "#f7f4ed"
NIGHT = "#11101b"
NIGHT_SOFT = "#1a1827"
WHITE = "#ffffff"
MUTED = "#aaa4bc"
INK = "#171521"
ACCENT = "#f4b860"
VIOLET = "#8b7bf6"

FONT_URLS = {
    "regular": "https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf",
    "bold": "https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Bold.otf",
}

SCENES = {
    "en": [
        {
            "kind": "title",
            "eyebrow": "SPARSE SUPERVISION FOR ECONOMICAL LLM ROUTING",
            "title": "Routing should\npay for itself.",
            "subtitle": "Count the cost before deployment.",
            "voice": "LLM routing can cut serving costs. But before a router saves anything, it must first pay for supervision.",
        },
        {
            "kind": "problem",
            "eyebrow": "THE MISSING COST",
            "title": "Training data\nis not free.",
            "subtitle": "Dense query-by-model feedback creates an upfront bill.",
            "voice": "Training data is not free. Every candidate model run on every historical query adds an upfront bill that conventional evaluations often leave out.",
        },
        {
            "kind": "method",
            "eyebrow": "SAVEROUTER",
            "title": "Acquire only what\nthe router needs.",
            "subtitle": "Exactly K informative model outcomes per training query.",
            "voice": "SAVERouter changes the acquisition process. For each training query, it selects exactly K informative model outcomes instead of filling the entire query-by-model matrix.",
        },
        {
            "kind": "stages",
            "eyebrow": "FROM SPARSE FEEDBACK TO ROUTING",
            "title": "Share globally.\nRefine locally.",
            "subtitle": "Structured capability estimates plus query-level corrections.",
            "voice": "It then shares capability evidence across related query groups, while learning query-level residual corrections for fine-grained routing decisions.",
        },
        {
            "kind": "metrics",
            "eyebrow": "MEASURE THE FULL ECONOMICS",
            "title": "When does routing\nbreak even?",
            "subtitle": "SA-BEP measures payback. SA-CR measures amortized cost.",
            "voice": "We evaluate the full economics with two metrics. S A B E P asks when serving-time savings repay supervision. S A C R measures amortized cost at a deployment horizon.",
        },
        {
            "kind": "results",
            "eyebrow": "FOUR ROUTING BENCHMARKS",
            "title": "Less feedback.\nEarlier payback.",
            "subtitle": "Competitive or better routing quality in the main setting.",
            "voice": "Across four routing benchmarks, SAVERouter uses about thirty-three to forty-one percent of available training feedback, while reducing break-even volume by approximately one point nine to nine point five times versus the fastest conventional router.",
        },
        {
            "kind": "cta",
            "eyebrow": "PAPER · CODE · COLAB · PYPI",
            "title": "Build a router that\npays for itself.",
            "subtitle": "lamda-model-reuse.github.io/SaveRouter",
            "voice": "Read the paper, explore payback interactively, or install SAVERouter from PyPI. Routing should pay for itself.",
        },
    ],
    "zh": [
        {
            "kind": "title",
            "eyebrow": "面向经济型大模型路由的稀疏监督",
            "title": "路由，应该\n为自己买单。",
            "subtitle": "在部署之前，把监督成本算进去。",
            "voice": "大模型路由可以降低推理成本。但在省下第一分钱之前，路由器必须先为监督数据买单。",
        },
        {
            "kind": "problem",
            "eyebrow": "被忽略的成本",
            "title": "训练数据，\n并不免费。",
            "subtitle": "密集的查询模型反馈，会产生一笔前期账单。",
            "voice": "训练数据并不免费。让每个候选模型回答每条历史查询，会产生一笔可观的前期成本，而传统评测往往忽略了它。",
        },
        {
            "kind": "method",
            "eyebrow": "SAVEROUTER",
            "title": "只获取路由器\n真正需要的反馈。",
            "subtitle": "每条训练查询，恰好选择 K 个高信息量模型结果。",
            "voice": "SAVERouter 重新设计了反馈获取过程。对每条训练查询，它只选择 K 个最有信息量的模型结果，不再填满整个查询模型矩阵。",
        },
        {
            "kind": "stages",
            "eyebrow": "从稀疏反馈到路由决策",
            "title": "全局共享，\n局部细化。",
            "subtitle": "结构化能力估计，加上查询级修正。",
            "voice": "随后，它在相关查询组之间共享能力信息，同时学习查询级残差，保留细粒度的路由决策。",
        },
        {
            "kind": "metrics",
            "eyebrow": "衡量完整经济性",
            "title": "路由何时\n能够回本？",
            "subtitle": "SA-BEP 衡量回本时间，SA-CR 衡量摊销成本。",
            "voice": "我们用两个指标衡量完整经济性。SA-BEP 回答节省何时覆盖监督成本；SA-CR 衡量指定部署规模下的摊销成本。",
        },
        {
            "kind": "results",
            "eyebrow": "四个路由基准",
            "title": "更少反馈，\n更早回本。",
            "subtitle": "在论文主设置中保持有竞争力或更好的路由质量。",
            "voice": "在四个路由基准上，SAVERouter 只使用约百分之三十三到四十一的训练反馈。相比最快的传统路由器，回本部署量降低约一点九到九点五倍。",
        },
        {
            "kind": "cta",
            "eyebrow": "论文 · 代码 · COLAB · PYPI",
            "title": "让路由真正\n为自己买单。",
            "subtitle": "lamda-model-reuse.github.io/SaveRouter",
            "voice": "阅读论文，在线探索回本曲线，或者从 PyPI 安装 SAVERouter。让路由真正为自己买单。",
        },
    ],
}

VOICES = {"en": "en-US-AriaNeural", "zh": "zh-CN-XiaoxiaoNeural"}
LANGUAGE_CODES = {"en": "eng", "zh": "zho"}


def ensure_fonts() -> dict[str, Path]:
    font_dir = CACHE / "fonts"
    font_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for weight, url in FONT_URLS.items():
        path = font_dir / f"NotoSansCJKsc-{weight.title()}.otf"
        if not path.exists():
            print(f"Downloading {path.name}...")
            urllib.request.urlretrieve(url, path)
        paths[weight] = path
    return paths


def font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size=size)


def gradient_background() -> Image.Image:
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH]
    violet = np.maximum(0.0, 1.0 - (((x - 1500) / 850) ** 2 + ((y - 230) / 620) ** 2))
    amber = np.maximum(0.0, 1.0 - (((x - 260) / 700) ** 2 + ((y - 970) / 500) ** 2))
    pixels = np.stack(
        (17 + 17 * violet + 8 * amber, 16 + 12 * violet + 5 * amber, 27 + 34 * violet + amber),
        axis=-1,
    ).astype(np.uint8)
    image = Image.fromarray(pixels, mode="RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    for x in range(0, WIDTH, 96):
        draw.line((x, 0, x, HEIGHT), fill=(255, 255, 255, 10), width=1)
    for y in range(0, HEIGHT, 96):
        draw.line((0, y, WIDTH, y), fill=(255, 255, 255, 10), width=1)
    return image


def draw_brand(draw: ImageDraw.ImageDraw, fonts: dict[str, Path]) -> None:
    x, y = 90, 68
    for width in (40, 29, 18):
        draw.rounded_rectangle((x, y, x + width, y + 5), radius=3, fill=ACCENT)
        y += 12
    draw.text((148, 67), "SAVERouter", font=font(fonts["bold"], 34), fill=WHITE)
    draw.text((1740, 70), "SinapisAI × LAMDA", font=font(fonts["regular"], 22), fill=MUTED, anchor="ra")


def fit_image(path: Path, size: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    return image


def wrap_text(draw: ImageDraw.ImageDraw, text: str, text_font: ImageFont.FreeTypeFont, max_width: int) -> str:
    cjk = bool(re.search(r"[\u3400-\u9fff]", text))
    tokens = list(text) if cjk else text.split()
    separator = "" if cjk else " "
    lines = []
    current = ""
    for token in tokens:
        candidate = token if not current else current + separator + token
        if current and draw.textbbox((0, 0), candidate, font=text_font)[2] > max_width:
            lines.append(current)
            current = token
        else:
            current = candidate
    if current:
        lines.append(current)
    return "\n".join(lines)


def draw_heading(draw: ImageDraw.ImageDraw, scene: dict, fonts: dict[str, Path], x: int = 90, y: int = 170, width: int = 920) -> int:
    draw.text((x, y), scene["eyebrow"], font=font(fonts["bold"], 24), fill=ACCENT)
    title_font = font(fonts["bold"], 94)
    draw.multiline_text((x, y + 52), scene["title"], font=title_font, fill=WHITE, spacing=2)
    title_box = draw.multiline_textbbox((x, y + 52), scene["title"], font=title_font, spacing=2)
    subtitle_y = title_box[3] + 34
    subtitle_font = font(fonts["regular"], 34)
    subtitle = wrap_text(draw, scene["subtitle"], subtitle_font, width)
    draw.multiline_text((x, subtitle_y), subtitle, font=subtitle_font, fill=MUTED, spacing=8)
    return subtitle_y


def rounded_panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill=(255, 255, 255, 18), outline=(255, 255, 255, 30), radius=28) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=2)


def render_slide(scene: dict, fonts: dict[str, Path], destination: Path) -> None:
    image = gradient_background()
    draw = ImageDraw.Draw(image, "RGBA")
    draw_brand(draw, fonts)
    kind = scene["kind"]

    if kind == "title":
        draw_heading(draw, scene, fonts, y=225)
        draw.rounded_rectangle((90, 850, 640, 916), radius=33, fill=ACCENT)
        draw.text((365, 883), "arXiv:2609.37402", font=font(fonts["bold"], 26), fill=INK, anchor="mm")

    elif kind == "problem":
        draw_heading(draw, scene, fonts, width=720)
        labels = [
            ("01", "Acquire supervision", "Upfront expenditure  C₀"),
            ("02", "Save per request", "Serving saving  Cᵦ − Cᵣ"),
            ("03", "Recover the bill", "Payback measured by SA-BEP"),
        ]
        if scene["eyebrow"] != "THE MISSING COST":
            labels = [
                ("01", "获取监督数据", "前期支出  C₀"),
                ("02", "逐请求节省", "服务节省  Cᵦ − Cᵣ"),
                ("03", "覆盖前期账单", "用 SA-BEP 衡量回本"),
            ]
        start_x, top = 850, 245
        for index, (number, title, note) in enumerate(labels):
            y = top + index * 220
            rounded_panel(draw, (start_x, y, 1815, y + 172), fill=(255, 255, 255, 16))
            draw.text((start_x + 38, y + 32), number, font=font(fonts["bold"], 25), fill=ACCENT)
            draw.text((start_x + 112, y + 26), title, font=font(fonts["bold"], 38), fill=WHITE)
            draw.text((start_x + 112, y + 90), note, font=font(fonts["regular"], 25), fill=MUTED)
            if index < 2:
                draw.text((1330, y + 190), "↓", font=font(fonts["regular"], 28), fill=MUTED, anchor="mm")

    elif kind == "method":
        draw.text((90, 170), scene["eyebrow"], font=font(fonts["bold"], 24), fill=ACCENT)
        draw.text((90, 220), scene["title"], font=font(fonts["bold"], 72), fill=WHITE, spacing=0)
        draw.text((90, 405), scene["subtitle"], font=font(fonts["regular"], 29), fill=MUTED)
        rounded_panel(draw, (90, 505, 1830, 958), fill=(255, 255, 255, 245), outline=(255, 255, 255, 255), radius=26)
        method = fit_image(ROOT / "assets" / "method.png", (1660, 390))
        image.paste(method, (130 + (1660 - method.width) // 2, 540 + (360 - method.height) // 2))

    elif kind == "stages":
        draw_heading(draw, scene, fonts, width=760)
        cards = [
            ("01", "Adaptive acquisition", "Choose informative outcomes"),
            ("02", "Capability sharing", "Pool evidence across groups"),
            ("03", "Local refinement", "Retain query-level variation"),
        ]
        if scene["eyebrow"] != "FROM SPARSE FEEDBACK TO ROUTING":
            cards = [
                ("01", "自适应获取", "选择高信息量结果"),
                ("02", "能力信息共享", "跨查询组汇聚证据"),
                ("03", "查询级细化", "保留局部差异"),
            ]
        top = 625
        for index, (number, title, note) in enumerate(cards):
            x = 90 + index * 590
            rounded_panel(draw, (x, top, x + 550, 935), fill=(255, 255, 255, 18))
            draw.text((x + 35, top + 32), number, font=font(fonts["bold"], 24), fill=ACCENT)
            draw.text((x + 35, top + 112), title, font=font(fonts["bold"], 36), fill=WHITE)
            draw.text((x + 35, top + 185), note, font=font(fonts["regular"], 26), fill=MUTED)

    elif kind == "metrics":
        draw_heading(draw, scene, fonts, width=790)
        chart = (920, 250, 1810, 910)
        rounded_panel(draw, chart, fill=(255, 255, 255, 15))
        left, top, right, bottom = 1010, 330, 1740, 820
        draw.line((left, top, left, bottom), fill=(255, 255, 255, 80), width=2)
        draw.line((left, bottom, right, bottom), fill=(255, 255, 255, 80), width=2)
        draw.line((left, bottom, right, top + 40), fill=VIOLET, width=7)
        draw.line((left, bottom - 160, right, top + 215), fill=ACCENT, width=7)
        cross_x, cross_y = 1355, 588
        draw.line((cross_x, cross_y, cross_x, bottom), fill=(255, 255, 255, 90), width=2)
        draw.ellipse((cross_x - 10, cross_y - 10, cross_x + 10, cross_y + 10), fill=WHITE)
        draw.text((cross_x, bottom + 45), "SA-BEP", font=font(fonts["bold"], 25), fill=WHITE, anchor="ma")
        draw.text((1060, 370), "Cumulative cost", font=font(fonts["regular"], 22), fill=MUTED)
        draw.text((right, bottom + 45), "Deployment volume", font=font(fonts["regular"], 22), fill=MUTED, anchor="ra")
        draw.text((1520, 425), "Best single model", font=font(fonts["regular"], 22), fill=VIOLET)
        draw.text((1520, 625), "Router + supervision", font=font(fonts["regular"], 22), fill=ACCENT)

    elif kind == "results":
        draw_heading(draw, scene, fonts, width=730)
        cards = [("33–41%", "training feedback"), ("1.9–9.5×", "earlier break-even")]
        if scene["eyebrow"] != "FOUR ROUTING BENCHMARKS":
            cards = [("33–41%", "训练反馈"), ("1.9–9.5×", "更早回本")]
        for index, (number, label) in enumerate(cards):
            y = 650 + index * 145
            rounded_panel(draw, (90, y, 730, y + 120), fill=(255, 255, 255, 18))
            draw.text((125, y + 50), number, font=font(fonts["bold"], 48), fill=ACCENT, anchor="lm")
            draw.text((430, y + 57), label, font=font(fonts["regular"], 26), fill=MUTED, anchor="lm")
        rounded_panel(draw, (845, 180, 1815, 960), fill=(255, 255, 255, 245), outline=(255, 255, 255, 255))
        results = fit_image(ROOT / "assets" / "main_results.png", (880, 700))
        image.paste(results, (890 + (880 - results.width) // 2, 215 + (700 - results.height) // 2))
        note = "Across four evaluated benchmarks · See paper for protocol"
        if scene["eyebrow"] != "FOUR ROUTING BENCHMARKS":
            note = "来自四个评测基准 · 完整协议请见论文"
        draw.text((1330, 925), note, font=font(fonts["regular"], 19), fill=(50, 46, 62, 180), anchor="mm")

    elif kind == "cta":
        draw_heading(draw, scene, fonts, width=1050)
        qr = qrcode.make("https://lamda-model-reuse.github.io/SaveRouter/").convert("RGB").resize((270, 270), Image.Resampling.NEAREST)
        panel = Image.new("RGB", (310, 310), WHITE)
        panel.paste(qr, (20, 20))
        image.paste(panel, (1510, 650))
        commands = "pip install saverouter\nsaverouter smoke-test"
        rounded_panel(draw, (90, 705, 1200, 900), fill=(0, 0, 0, 80))
        draw.text((135, 750), commands, font=font(fonts["regular"], 34), fill=WHITE, spacing=22)
        draw.text((90, 980), scene["subtitle"], font=font(fonts["regular"], 28), fill=MUTED)

    image.save(destination, quality=95)


async def synthesize(text: str, voice: str, destination: Path, rate: str) -> None:
    for attempt in range(3):
        try:
            await edge_tts.Communicate(text, voice=voice, rate=rate).save(str(destination))
            return
        except Exception:
            if attempt == 2:
                raise
            await asyncio.sleep(2 ** attempt)


def run(command: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(command, check=True, text=True, capture_output=True)


def duration(ffmpeg: str, path: Path) -> float:
    process = subprocess.run([ffmpeg, "-i", str(path), "-f", "null", "-"], text=True, capture_output=True)
    match = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", process.stderr)
    if not match:
        raise RuntimeError(f"Could not read audio duration for {path}")
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def srt_time(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    secs, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


async def build_locale(locale: str, fonts: dict[str, Path], rate: str) -> Path:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    locale_work = WORK / locale
    locale_work.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    scenes = SCENES[locale]
    timings = []
    cursor = 0.0
    clips = []

    for index, scene in enumerate(scenes, start=1):
        stem = f"{index:02d}-{scene['kind']}"
        slide = locale_work / f"{stem}.png"
        audio = locale_work / f"{stem}.mp3"
        clip = locale_work / f"{stem}.mp4"
        render_slide(scene, fonts, slide)
        await synthesize(scene["voice"], VOICES[locale], audio, rate)
        voice_duration = duration(ffmpeg, audio)
        clip_duration = voice_duration + 0.5
        fade_out = max(0.3, clip_duration - 0.3)
        command = [
            ffmpeg, "-y", "-loop", "1", "-framerate", "30", "-i", str(slide), "-i", str(audio),
            "-vf", f"scale={WIDTH}:{HEIGHT},fade=t=in:st=0:d=0.25,fade=t=out:st={fade_out:.3f}:d=0.35,format=yuv420p",
            "-af", f"adelay=200,apad=pad_dur=0.3,afade=t=out:st={voice_duration + 0.1:.3f}:d=0.25",
            "-t", f"{clip_duration:.3f}", "-r", "30", "-c:v", "libx264", "-preset", "medium", "-crf", "19",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart", str(clip),
        ]
        run(command)
        clips.append(clip)
        timings.append((cursor + 0.15, cursor + voice_duration + 0.3, scene["voice"]))
        cursor += clip_duration

    concat_file = locale_work / "concat.txt"
    concat_file.write_text("".join(f"file '{clip.resolve()}'\n" for clip in clips), encoding="utf-8")
    combined = locale_work / "combined.mp4"
    run([ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file), "-c", "copy", str(combined)])

    subtitle = OUTPUT / f"saverouter-explainer-{locale}.srt"
    subtitle.write_text(
        "\n".join(
            f"{index}\n{srt_time(start)} --> {srt_time(end)}\n{text}\n"
            for index, (start, end, text) in enumerate(timings, start=1)
        ),
        encoding="utf-8",
    )
    destination = OUTPUT / f"saverouter-explainer-{locale}.mp4"
    run([
        ffmpeg, "-y", "-i", str(combined), "-i", str(subtitle), "-map", "0:v", "-map", "0:a", "-map", "1:0",
        "-c:v", "copy", "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-c:a", "aac", "-b:a", "192k",
        "-c:s", "mov_text", "-metadata:s:s:0", f"language={LANGUAGE_CODES[locale]}",
        "-movflags", "+faststart", str(destination),
    ])
    print(f"Built {destination} ({cursor:.1f}s)")
    return destination


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=("en", "zh", "all"), default="all")
    parser.add_argument("--rate", default="+38%", help="Edge TTS speaking rate")
    args = parser.parse_args()
    fonts = ensure_fonts()
    locales = ("en", "zh") if args.language == "all" else (args.language,)
    for locale in locales:
        await build_locale(locale, fonts, args.rate)


if __name__ == "__main__":
    asyncio.run(main())
