# Demo video

The scripts here produce `docs/media/contexteval-demo.mp4` and `docs/media/contexteval-demo.gif`. Every line of terminal output in the video is real output from `capture.py`. The dashboard scenes drive the real `contexteval ui`.

- `capture.py` runs the commands the video shows and saves their output to `captured/demo.json` and `captured/realbot.json`.
- `stage/` is the page that Chrome renders: the background, the window, the terminal replay, the cursor, the zoom, and the timeline in `stage.js`.
- `render.py` steps the timeline at 30 fps, screenshots each frame in Chrome, and pipes the frames into ffmpeg.
- `gif.py` cuts the GIF loop from the MP4.

## Re-render

You need the repository's virtual environment with `playwright` and `imageio-ffmpeg`, plus Google Chrome. The renderer uses the installed Chrome (`channel="chrome"`).

1. Capture the demo output. It costs nothing, because the demo uses the fake provider. `WORK` is any scratch directory.

   ```bash
   python demo/video/capture.py demo WORK
   ```

2. Capture the real bot output. This step needs `OPENAI_API_KEY`. It stops when the estimates or the tallied spend pass $0.40. Our run cost about $0.15. Skip this step to reuse the committed `captured/realbot.json`.

   ```bash
   python demo/video/capture.py realbot WORK
   ```

3. Serve the dashboard for the demo project in a second terminal:

   ```bash
   cd WORK/demo-project
   PYTHONPATH=/path/to/context-evaluation-sdk python -m contexteval ui --no-open --port 8765
   ```

4. Render the video, then cut the GIF:

   ```bash
   python demo/video/render.py http://127.0.0.1:8765 docs/media/contexteval-demo.mp4
   python demo/video/gif.py docs/media/contexteval-demo.mp4 docs/media/contexteval-demo.gif
   ```

To check a few frames without encoding, pass `--stills DIR 5,30,60` to `render.py` in place of the output path. It saves PNGs at those seconds.
