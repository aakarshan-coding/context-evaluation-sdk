"""Cut the README GIF from the MP4: the run, diagnose, and sweep in the terminal.

Usage: python gif.py IN.mp4 OUT.gif [START_SECONDS DURATION_SECONDS]
"""

import subprocess
import sys

import imageio_ffmpeg

START, DURATION = 15.4, 20.0

src, out = sys.argv[1], sys.argv[2]
start, duration = (float(sys.argv[3]), float(sys.argv[4])) if len(sys.argv) > 4 else (START, DURATION)
graph = (
    "fps=10,scale=960:-1:flags=lanczos,split[a][b];"
    "[a]palettegen=max_colors=96:stats_mode=diff[p];"
    "[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle"
)
subprocess.run(
    [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-ss", str(start), "-t", str(duration),
     "-i", src, "-filter_complex", graph, "-loop", "0", out],
    check=True,
)
