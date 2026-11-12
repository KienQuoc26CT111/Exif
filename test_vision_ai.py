import base64
import io
import json
import os
import sys
import tempfile
import unittest
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image

import vision_ai as v


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return json.dumps(self.payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class VisionTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = os.path.join(self.dir.name, "a.jpg")
        Image.new("RGB", (3000, 2000), (90, 120, 150)).save(self.path)

    def test_request_shape(self):
        req = v.build_request(self.path, "sk-test")
        self.assertEqual(req.get_method(), "POST")
        self.assertEqual(req.full_url, v.API_URL)
        self.assertEqual(req.get_header("X-api-key"), "sk-test")
        self.assertEqual(req.get_header("Anthropic-version"), v.API_VERSION)
        body = json.loads(req.data)
        self.assertEqual(body["model"], v.MODEL)
        blocks = body["messages"][0]["content"]
        self.assertEqual(blocks[0]["type"], "image")
        self.assertEqual(blocks[1]["type"], "text")

    def test_image_downscaled(self):
        raw = base64.b64decode(v.prepare_image(self.path))
        img = Image.open(io.BytesIO(raw))
        self.assertLessEqual(max(img.size), v.MAX_SIDE)
        self.assertEqual(len(img.getexif()), 0)

    def test_describe_success(self):
        payload = {"content": [{"type": "text", "text": "1. Địa điểm: lớp học"}]}
        out = v.describe_image(self.path, "k", opener=lambda req, timeout: FakeResponse(payload))
        self.assertIn("lớp học", out)

    def test_no_key(self):
        with self.assertRaises(ValueError):
            v.describe_image(self.path, "")

    def test_unauthorized_message(self):
        def opener(req, timeout):
            raise urllib.error.HTTPError(req.full_url, 401, "x", {}, io.BytesIO(b'{"error":{"message":"bad"}}'))
        with self.assertRaises(RuntimeError) as cm:
            v.describe_image(self.path, "k", opener=opener)
        self.assertIn("401", str(cm.exception))

    def test_network_error(self):
        def opener(req, timeout):
            raise urllib.error.URLError("offline")
        with self.assertRaises(RuntimeError) as cm:
            v.describe_image(self.path, "k", opener=opener)
        self.assertIn("kết nối", str(cm.exception))

    def test_empty_response(self):
        with self.assertRaises(ValueError):
            v.parse_response(json.dumps({"content": []}))

    def test_to_lines_marks_high_risk(self):
        lines = v.to_lines("1. Địa điểm: trường\n6. Mức rủi ro: Cao. Nên che mặt.")
        tags = [t for t, _ in lines]
        self.assertEqual(lines[0][0], "h")
        self.assertIn("bad", tags)


if __name__ == "__main__":
    unittest.main(verbosity=2)
