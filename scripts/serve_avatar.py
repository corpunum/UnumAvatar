"""
Simple HTTP server for the Unum Avatar with TTS API endpoint.
Serves static files from output/ and handles /api/speak for on-demand TTS.
"""
import http.server
import json
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(__file__))
try:
    from generate_lipsync_rhubarb import run as rhubarb_run
    USE_RHUBARB = True
except ImportError:
    USE_RHUBARB = False
from generate_lipsync import generate

PORT = 8765
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
speech_counter = 0
speech_lock = threading.Lock()


class AvatarHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=OUTPUT_DIR, **kwargs)

    def do_POST(self):
        global speech_counter
        if self.path == '/api/speak':
            length = int(self.headers.get('Content-Length', 0))
            body = json.loads(self.rfile.read(length))
            text = body.get('text', '').strip()

            if not text:
                self.send_error(400, 'No text provided')
                return

            with speech_lock:
                speech_counter += 1
                name = f"speech_{speech_counter}"

            try:
                os.chdir(os.path.dirname(os.path.dirname(__file__)))
                if USE_RHUBARB:
                    rhubarb_run(text)
                    import json as _j
                    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "output", "speech_lipsync.json")) as f:
                        result = _j.load(f)
                else:
                    result = generate(text, output_name=name)

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({
                    "audio": f"{name}.wav",
                    "lipsync": f"{name}_lipsync.json",
                    "duration": result["duration"],
                    "frames": len(result["frames"])
                }).encode())
            except Exception as e:
                self.send_error(500, str(e))
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        if '/api/' in str(args[0]):
            super().log_message(format, *args)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(__file__)))
    server = http.server.HTTPServer(('0.0.0.0', PORT), AvatarHandler)
    print(f"Unum Avatar Server running on http://localhost:{PORT}")
    print(f"  Viewer: http://localhost:{PORT}/talking.html")
    print(f"  API:    POST http://localhost:{PORT}/api/speak")
    server.serve_forever()
