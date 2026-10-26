"""tools/liveview.py: page, stream routing and the /snap host check, with fake cameras (no opencv)."""
import http.client
import threading
import unittest
from http.server import ThreadingHTTPServer

from tools import liveview


class FakeCam:
    def __init__(self, index):
        self.index = index

    def jpeg(self):
        return b"\xff\xd8fake\xff\xd9"

    def snapshot(self):
        return None


class LiveViewTests(unittest.TestCase):
    def setUp(self):
        cams = [FakeCam(i) for i in (0, 2, 4, 6)]
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), liveview.make_handler(cams))
        self.srv.daemon_threads = True
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()

    def get(self, path, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
        conn.request("GET", path, headers=headers or {})
        return conn, conn.getresponse()

    def test_root_page_has_four_streams(self):
        conn, resp = self.get("/")
        body = resp.read().decode()
        conn.close()
        self.assertEqual(resp.status, 200)
        for n in range(4):
            self.assertIn('<img src="/stream/%d">' % n, body)
        self.assertNotIn("@@TILES@@", body)

    def test_stream_serves_multipart_jpeg(self):
        conn, resp = self.get("/stream/0")
        self.assertEqual(resp.status, 200)
        self.assertIn("multipart/x-mixed-replace", resp.getheader("Content-Type"))
        chunk = resp.read(60)
        conn.close()
        self.assertIn(b"--frame", chunk)

    def test_unknown_stream_is_404(self):
        for path in ("/stream/9", "/stream/x", "/nope"):
            conn, resp = self.get(path)
            resp.read()
            conn.close()
            self.assertEqual(resp.status, 404, path)

    def test_snap_rejects_foreign_host(self):
        conn, resp = self.get("/snap", {"Host": "evil.example"})
        resp.read()
        conn.close()
        self.assertEqual(resp.status, 403)

    def test_missing_cv2_is_a_tool_dependency_error(self):
        import builtins

        real_import = builtins.__import__

        def no_cv2(name, *a, **k):
            if name == "cv2":
                raise ImportError("no cv2")
            return real_import(name, *a, **k)

        saved = liveview.cv2
        liveview.cv2 = None
        builtins.__import__ = no_cv2
        try:
            with self.assertRaises(liveview.ToolDependencyError):
                liveview._require_cv2()
        finally:
            builtins.__import__ = real_import
            liveview.cv2 = saved


if __name__ == "__main__":
    unittest.main()
