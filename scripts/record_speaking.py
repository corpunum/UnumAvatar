"""
Deterministic 30fps avatar video recorder.
Renders one frame at a time via page.evaluate(renderFrameAt(t)),
captures PNG per frame, assembles with ffmpeg.
No MediaRecorder — no timing jitter, true 30fps.
"""
import subprocess, time, os, json, base64
from pathlib import Path
from playwright.sync_api import sync_playwright

BASE    = Path(__file__).parent.parent
OUTPUT  = BASE / "output"
FRAMES  = OUTPUT / "frames"
VIDEO   = OUTPUT / "avatar_speaking.mp4"
AUDIO   = OUTPUT / "speech.wav"
LIPSYNC = OUTPUT / "speech_lipsync.json"
SERVER  = "http://localhost:8899"
FPS     = 30


def main():
    lipsync_duration = 3.33
    if LIPSYNC.exists():
        with open(LIPSYNC) as f:
            lipsync_duration = json.load(f).get('duration', 3.33)

    tail = 1.0  # 1 second after speech ends
    total_time = lipsync_duration + tail
    total_frames = int(total_time * FPS)
    print(f"[record] {total_frames} frames @ {FPS}fps = {total_time:.2f}s")

    FRAMES.mkdir(exist_ok=True)
    # Clean old frames
    for f in FRAMES.glob("frame_*.png"):
        f.unlink()

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--use-gl=egl", "--enable-webgl", "--ignore-gpu-blocklist",
                  "--no-sandbox", "--disable-dev-shm-usage",
                  "--disable-web-security"]
        )
        ctx = browser.new_context(viewport={"width": 1024, "height": 768})
        page = ctx.new_page()
        page.goto(f"{SERVER}/talking.html")

        # Wait for both model and lipsync to be ready
        page.wait_for_function("window.__ready === true", timeout=30000)
        page.wait_for_function("typeof window.renderFrameAt === 'function'", timeout=10000)
        time.sleep(0.5)  # let WebGL finish initializing textures
        print("[record] Ready. Rendering frames...")

        for i in range(total_frames):
            t = i / FPS
            # Call deterministic render
            page.evaluate(f"window.renderFrameAt({t})")
            # Capture PNG via canvas.toDataURL (synchronous after render)
            png_b64 = page.evaluate("""() => {
                const canvas = document.querySelector('canvas');
                return canvas.toDataURL('image/png').split(',')[1];
            }""")
            frame_path = FRAMES / f"frame_{i:05d}.png"
            frame_path.write_bytes(base64.b64decode(png_b64))

            if i % 30 == 0:
                print(f"  frame {i}/{total_frames} (t={t:.2f}s)")

        browser.close()

    print(f"[record] {total_frames} frames captured. Assembling MP4...")
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(FPS),
        "-i", str(FRAMES / "frame_%05d.png"),
        "-i", str(AUDIO),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        str(VIDEO),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    size_mb = VIDEO.stat().st_size / 1024 / 1024
    print(f"[record] Done: {VIDEO}  {size_mb:.1f}MB  {lipsync_duration:.1f}s speech")


if __name__ == "__main__":
    main()
