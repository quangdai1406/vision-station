"""Live view of the 4 station cameras in a browser, for aiming and lighting.

Run:   python -m tools.liveview            (needs opencv-python; imported lazily)
Open:  http://127.0.0.1:8090/   (on the Pi screen)

Stop the station first: only one program can hold a camera at a time.
Binds to 127.0.0.1 by default (no login). Use --host only on a network you trust.
Nothing is saved. The 'Save snapshot' link writes 4 PNGs to ~/Pictures/vision-liveview/.
"""
import argparse
import os
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tools import ToolDependencyError

cv2 = None  # loaded by _require_cv2() so importing this module needs no opencv


def _require_cv2():
    global cv2
    if cv2 is None:
        try:
            import cv2 as _cv2  # noqa: PLC0415 - lazy on purpose
        except ImportError as exc:
            raise ToolDependencyError("OpenCV (cv2) is needed for the live view") from exc
        cv2 = _cv2
    return cv2


PAGE = """<!doctype html><meta charset=utf-8><title>Live view</title>
<style>
body{margin:0;background:#111;color:#eee;font:14px sans-serif}
.g{display:grid;grid-template-columns:1fr 1fr;gap:4px;padding:4px}
.c{position:relative;background:#000}
.c img{width:100%;display:block}
.c:before,.c:after{content:"";position:absolute;background:#fff8;pointer-events:none}
.c:before{left:50%;top:0;bottom:0;width:1px}
.c:after{top:50%;left:0;right:0;height:1px}
.bar{padding:6px 10px}
a{color:#7cf}
</style>
<div class=bar>Live view (nothing is saved) &middot; <a href="/snap" target=_blank>Save snapshot</a>
&middot; Cross = picture centre. Brightness: aim for a mid value, no red text.</div>
<div class=g>@@TILES@@</div>"""


class Cam:
    def __init__(self, index, label, width, height):
        self.index, self.label = index, label
        self.frame = None
        self.stats = (0.0, 0.0, 0.0)
        self.error = None
        self.lock = threading.Lock()
        self.stop = False
        self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        while not self.stop:
            ok, f = self.cap.read()
            if not ok:
                self.error = "no frame"
                time.sleep(0.2)
                continue
            self.error = None
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            stats = (float(g.mean()), float((g < 30).mean() * 100), float((g > 225).mean() * 100))
            with self.lock:
                self.frame, self.stats = f, stats

    def jpeg(self, quality=70, width=640):
        with self.lock:
            f = None if self.frame is None else self.frame.copy()
            mean, dark, bright = self.stats
        if f is None:
            return None
        err = self.error
        h = int(f.shape[0] * width / f.shape[1])
        f = cv2.resize(f, (width, h))
        bad = dark > 10 or bright > 5 or mean < 60 or mean > 200
        color = (60, 60, 255) if bad else (80, 255, 80)
        text = "%s  mean %d  dark %.0f%%  bright %.0f%%" % (self.label, mean, dark, bright)
        cv2.putText(f, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(f, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if err:
            cv2.putText(f, "CAMERA ERROR: " + err, (8, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 255), 2)
        return cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()

    def snapshot(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()


def make_handler(cams):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path == "/":
                tiles = "".join('<div class=c><img src="/stream/%d"></div>' % i for i in range(len(cams)))
                body = PAGE.replace("@@TILES@@", tiles).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path.startswith("/stream/"):
                try:
                    cam = cams[int(self.path.rsplit("/", 1)[1])]
                except (ValueError, IndexError):
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        data = cam.jpeg()
                        if data:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(data))
                            self.wfile.write(data + b"\r\n")
                        time.sleep(0.1)
                except OSError:
                    pass
            elif self.path == "/snap":
                if self.headers.get("Host", "").split(":")[0] not in ("127.0.0.1", "localhost"):
                    self.send_error(403)
                    return
                d = os.path.expanduser("~/Pictures/vision-liveview")
                os.makedirs(d, exist_ok=True)
                ts = time.strftime("%y_%m%d_%H%M%S")
                saved = []
                for c in cams:
                    f = c.snapshot()
                    if f is not None:
                        p = os.path.join(d, "%s_cam%d.png" % (ts, c.index))
                        cv2.imwrite(p, f)
                        saved.append(p)
                body = ("Saved:<br>" + "<br>".join(saved)).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_error(404)

    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--devices", type=int, nargs="+", default=[0, 2, 4, 6])
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8090)
    a = ap.parse_args()
    try:
        _require_cv2()
    except ToolDependencyError as exc:
        raise SystemExit("error: %s" % exc)
    cams = [Cam(d, "cam%d (C%d)" % (d, i + 1), a.width, a.height) for i, d in enumerate(a.devices)]
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(cams))
    srv.daemon_threads = True
    def stop_now(signum, frame):
        raise KeyboardInterrupt

    # A background launch can leave SIGINT ignored; take both signals explicitly.
    signal.signal(signal.SIGINT, stop_now)
    signal.signal(signal.SIGTERM, stop_now)
    print("Live view: http://%s:%d/   Ctrl-C to stop" % (a.host, a.port), flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for c in cams:
            c.stop = True
        srv.server_close()
        for c in cams:
            c.thread.join(timeout=1)
            c.cap.release()
        print("stopped")


if __name__ == "__main__":
    main()
