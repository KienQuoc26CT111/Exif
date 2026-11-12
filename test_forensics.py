import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image, ImageDraw, ImageFont

import forensics as f


def noisy(size=(800, 600)):
    return Image.effect_noise(size, 30).convert("RGB")


def has_text(findings, needle):
    return any(needle in x for _, x in findings)


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def p(self, name):
        return os.path.join(self.dir.name, name)

    def prov(self, path):
        with Image.open(path) as img:
            return f.analyze_provenance(path, img)

    def test_libjpeg_quality_detected(self):
        path = self.p("a.jpg")
        noisy().save(path, quality=75)
        with Image.open(path) as img:
            self.assertEqual(f.libjpeg_quality(img.quantization), 75)

    def test_custom_tables_not_libjpeg(self):
        path = self.p("a.jpg")
        luma = [max(1, (i % 7) + 2) for i in range(64)]
        chroma = [max(1, (i % 5) + 3) for i in range(64)]
        noisy().save(path, qtables=[luma, chroma])
        with Image.open(path) as img:
            self.assertIsNone(f.libjpeg_quality(img.quantization))

    def test_progressive_detected(self):
        path = self.p("a.jpg")
        noisy().save(path, progressive=True, quality=80)
        self.assertTrue(f.inspect_jpeg(path)["progressive"])
        self.assertTrue(self.prov(path)["progressive"])

    def test_baseline_not_progressive(self):
        path = self.p("a.jpg")
        noisy().save(path, quality=80)
        self.assertFalse(f.inspect_jpeg(path)["progressive"])

    def test_chat_style_file_verdict(self):
        name = "2aOboR9kCEW8dkpTblX7JkG0cOn8cuvuS4qu.jpg"
        path = self.p(name)
        noisy((960, 1280)).save(path, quality=78, progressive=True)
        prov = self.prov(path)
        self.assertGreaterEqual(prov["reenc"], 3)
        self.assertIn("CHAT", prov["verdict"])
        self.assertTrue(prov["banner"])

    def test_original_like_file_verdict(self):
        path = self.p("IMG_20261024_143210.jpg")
        luma = [max(1, (i % 7) + 2) for i in range(64)]
        chroma = [max(1, (i % 5) + 3) for i in range(64)]
        noisy((4032, 3024)).save(path, qtables=[luma, chroma])
        prov = self.prov(path)
        self.assertGreaterEqual(prov["orig"], 3)
        self.assertIn("điện thoại", prov["verdict"])
        self.assertIsNotNone(prov["name_time"])

    def test_exif_present_verdict(self):
        path = self.p("a.jpg")
        e = Image.Exif()
        e[0x010F] = "Apple"
        noisy().save(path, exif=e)
        self.assertEqual(self.prov(path)["verdict_tag"], "ok")

    def test_filename_patterns(self):
        cases = {
            "FB_IMG_1698123456789.jpg": "Facebook",
            "received_123456789012345.jpeg": "Messenger",
            "z4912345678901_0a1b2c3d4e5f60718293a4b5c6d7e8f9.jpg": "Zalo",
            "photo_2026-10-24_14-32-10.jpg": "Telegram",
            "WhatsApp Image 2026-10-24 at 14.32.10.jpeg": "WhatsApp",
            "PXL_20261024_143210123.jpg": "Pixel",
            "IMG_1234.JPG": "iPhone",
        }
        for name, needle in cases.items():
            out, _, _, _ = f.filename_hints(name)
            self.assertTrue(has_text(out, needle), name)

    def test_filename_time_parsed(self):
        _, _, _, when = f.filename_hints("20261024_143210.jpg")
        self.assertEqual((when.year, when.month, when.hour, when.second), (2026, 10, 14, 10))

    def test_plain_name_no_hint(self):
        out, r, o, when = f.filename_hints("anh cua toi.jpg")
        self.assertEqual((out, r, o, when), ([], 0.0, 0.0, None))

    def test_native_size_credited(self):
        path = self.p("a.jpg")
        Image.new("RGB", (4032, 3024), (80, 90, 100)).save(path)
        self.assertTrue(has_text(self.prov(path)["findings"], "kích thước gốc"))

    def test_resize_size_flagged(self):
        path = self.p("a.jpg")
        Image.new("RGB", (1280, 960), (80, 90, 100)).save(path)
        self.assertTrue(has_text(self.prov(path)["findings"], "mức thu nhỏ"))

    def test_server_comment_flagged(self):
        path = self.p("a.jpg")
        noisy().save(path, comment=b"CREATOR: gd-jpeg v1.0 (using IJG JPEG v62), quality = 82")
        self.assertTrue(has_text(self.prov(path)["findings"], "ghi chú của phần mềm"))

    def test_png_tall_screenshot(self):
        path = self.p("s.png")
        Image.new("RGB", (1080, 2400), (20, 20, 20)).save(path)
        self.assertTrue(has_text(self.prov(path)["findings"], "chụp màn hình"))


class MetadataScanTests(unittest.TestCase):
    def test_pixel_data_excluded_from_scan(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "a.jpg")
            noisy().save(path, comment=b"hello")
            raw = bytearray(open(path, "rb").read())
            mid = len(raw) // 2
            raw[mid:mid + 4] = b"sdxl"
            open(path, "wb").write(bytes(raw))
            self.assertTrue(len(raw) > 60000)
            self.assertFalse(b"sdxl" in f.metadata_bytes(path))

    def test_header_comment_included(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "a.jpg")
            noisy().save(path, comment=b"made with Midjourney")
            self.assertIn(b"midjourney", f.metadata_bytes(path))

    def test_png_text_included_idat_excluded(self):
        from PIL import PngImagePlugin
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "a.png")
            info = PngImagePlugin.PngInfo()
            info.add_text("parameters", "Steps: 20, Sampler: Euler")
            noisy().save(path, pnginfo=info)
            data = f.metadata_bytes(path)
            self.assertIn(b"sampler", data)
            self.assertLess(len(data), 2000)

    def test_meaningful_exif(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "a.jpg")
            e = Image.Exif()
            e[0x0112] = 1
            e.get_ifd(0x8769)[0xA002] = 640
            noisy().save(path, exif=e)
            with Image.open(path) as img:
                self.assertFalse(f.meaningful_exif(img))
            e[0x010F] = "Apple"
            noisy().save(path, exif=e)
            with Image.open(path) as img:
                self.assertTrue(f.meaningful_exif(img))


class OcrFilterTests(unittest.TestCase):
    def test_garbage_dropped(self):
        self.assertEqual(f.clean_line(["Ñ", "À", "“ay"], [90, 80, 75]), "")
        self.assertEqual(f.clean_line(["yy", "BN"], [60, 60]), "")

    def test_real_line_kept(self):
        self.assertEqual(f.clean_line(["HOP", "CHA", "HOC", "SINH"], [90, 88, 91, 85]), "HOP CHA HOC SINH")

    def test_single_long_confident_word(self):
        self.assertEqual(f.clean_line(["SCHOOL"], [92]), "SCHOOL")
        self.assertEqual(f.clean_line(["SCHOOL"], [60]), "")

    def test_digits_kept(self):
        self.assertEqual(f.clean_line(["Ngay", "2/6/2023"], [90, 85]), "Ngay 2/6/2023")


class EntityTests(unittest.TestCase):
    def kinds(self, text):
        return {label for _, label, _, _ in f.find_entities([text])}

    def test_phone(self):
        self.assertIn("số điện thoại", self.kinds("Liên hệ 0987 654 321 gặp cô Lan"))
        self.assertIn("số điện thoại", self.kinds("+84 912 345 678"))

    def test_email(self):
        self.assertIn("email", self.kinds("lienhe: abc.def@gmail.com"))

    def test_plate(self):
        self.assertIn("biển số xe", self.kinds("Xe 59A1-123.45"))
        self.assertIn("biển số xe", self.kinds("51F-123.45"))

    def test_id_number(self):
        self.assertIn("dãy 12 số (giống số căn cước)", self.kinds("CCCD 079123456789"))

    def test_address_keywords(self):
        self.assertTrue(self.kinds("Trường Đại học Lạc Hồng, phường Bửu Long"))

    def test_date(self):
        self.assertIn("ngày tháng", self.kinds("Ngày 2/6/2023 thi vào lớp 10"))

    def test_nothing(self):
        self.assertEqual(self.kinds("xin chao"), set())


@unittest.skipUnless(shutil.which("tesseract"), "chưa có Tesseract")
class OcrTests(unittest.TestCase):
    def test_reads_text(self):
        img = Image.new("RGB", (900, 300), "white")
        d = ImageDraw.Draw(img)
        try:
            font = ImageFont.load_default(size=60)
        except TypeError:
            font = ImageFont.load_default()
        d.text((30, 40), "SCHOOL NOTICE", fill="black", font=font)
        d.text((30, 150), "Call 0987654321", fill="black", font=font)
        lines, note = f.ocr_lines(img)
        self.assertIsNotNone(lines)
        joined = " ".join(lines).upper()
        self.assertIn("SCHOOL", joined)
        self.assertIn("0987654321", joined.replace(" ", ""))

    def test_blank_has_no_text(self):
        lines, _ = f.ocr_lines(Image.new("RGB", (400, 300), (120, 130, 140)))
        self.assertEqual(lines, [])


try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None


@unittest.skipIf(cv2 is None, "chưa có OpenCV")
class VisionTests(unittest.TestCase):
    def test_qr_decoded(self):
        enc = cv2.QRCodeEncoder.create()
        qr = enc.encode("https://example.com/secret")
        qr = cv2.resize(qr, None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST)
        canvas = np.full((qr.shape[0] + 80, qr.shape[1] + 80), 255, dtype=np.uint8)
        canvas[40:40 + qr.shape[0], 40:40 + qr.shape[1]] = qr
        img = Image.fromarray(canvas).convert("RGB")
        self.assertIn("https://example.com/secret", f.decode_qr(img))

    def test_no_qr_in_noise(self):
        self.assertEqual(f.decode_qr(noisy()), [])

    def test_faces_on_astronaut(self):
        try:
            from skimage import data
        except ImportError:
            self.skipTest("chưa có scikit-image")
        img = Image.fromarray(data.astronaut())
        n, note = f.count_faces(img)
        self.assertIsNotNone(n, note)
        self.assertGreaterEqual(n, 1)

    def test_no_faces_in_noise(self):
        n, note = f.count_faces(noisy())
        self.assertIsNotNone(n, note)
        self.assertEqual(n, 0)


class ContentLinesTests(unittest.TestCase):
    def test_pipeline_runs(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "a.jpg")
            noisy().save(path)
            lines = f.content_lines(path)
            self.assertEqual(lines[0][0], "h")
            self.assertGreater(len(lines), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
