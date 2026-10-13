"""Render the demo video from captured/*.json and a running `contexteval ui`.

Usage:
  python render.py UI_URL OUT.mp4               encode every frame to H.264
  python render.py UI_URL --stills DIR 3,20,41  step the timeline, save PNGs at those seconds

Frames go straight from Chrome screenshots into ffmpeg; a frame where nothing moved reuses the
previous image instead of taking a new screenshot.
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import imageio_ffmpeg
from playwright.sync_api import Route, sync_playwright

HERE = Path(__file__).resolve().parent
FPS = 30


def serve_stage(route: Route) -> None:
    rel = route.request.url.split("/__stage/", 1)[1].split("?", 1)[0]
    path = HERE / rel if rel.startswith("captured/") else HERE / "stage" / rel
    route.fulfill(path=str(path))


def encoder(out: Path) -> subprocess.Popen:
    return subprocess.Popen(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", str(FPS),
         "-c:v", "png", "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", str(out)],
        stdin=subprocess.PIPE,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ui_url")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--stills", nargs=2, metavar=("DIR", "SECONDS"))
    args = ap.parse_args()
    base = args.ui_url.rstrip("/")
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome")
        page = browser.new_page(viewport={"width": 1920, "height": 1080}, device_scale_factor=1, color_scheme="light")
        page.route(f"{base}/__stage/**", serve_stage)
        page.goto(f"{base}/__stage/stage.html")
        timeline = page.evaluate("init()")
        total = timeline["frames"]
        print(f"{total} frames ({total / FPS:.1f} s), marks {timeline['marks']}", flush=True)
        if args.stills:
            folder, seconds = Path(args.stills[0]), [float(s) for s in args.stills[1].split(",")]
            folder.mkdir(parents=True, exist_ok=True)
            wanted = {round(s * FPS): s for s in seconds}
            for n in range(min(total, max(wanted) + 1)):
                page.evaluate("step()")
                if n in wanted:
                    page.screenshot(path=str(folder / f"still-{wanted[n]:05.1f}s.png"))
        else:
            ff = encoder(Path(args.out))
            image, shots, start = b"", 0, time.monotonic()
            for n in range(total):
                if page.evaluate("step()") or not image:
                    image = page.screenshot(type="png")
                    shots += 1
                ff.stdin.write(image)
                if n % 300 == 0:
                    print(f"frame {n}/{total}, {shots} screenshots, {time.monotonic() - start:.0f} s", flush=True)
            ff.stdin.close()
            if ff.wait() != 0:
                raise SystemExit("ffmpeg failed")
        browser.close()


if __name__ == "__main__":
    main()
