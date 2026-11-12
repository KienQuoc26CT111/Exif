import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

try:
    import tkinter
    from PIL import ImageTk
except ImportError:
    for name in ("tkinter", "tkinter.ttk", "tkinter.filedialog", "tkinter.messagebox", "PIL.ImageTk"):
        sys.modules[name] = MagicMock()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image, ImageCms, PngImagePlugin

import exif_tool as t


def make_exif(make=None, model=None, ev_args=None, flash=None, gps=None, gps_err=None):
    e = Image.Exif()
    if make:
        e[0x010F] = make
    if model:
        e[0x0110] = model
    ex = e.get_ifd(0x8769)
    if ev_args:
        ex[0x829A], ex[0x829D], ex[0x8827] = ev_args
        ex[0x9003] = "2026:10:24 14:32:10"
    if flash is not None:
        ex[0x9209] = flash
    if gps:
        g = e.get_ifd(0x8825)
        g[1], g[2], g[3], g[4] = "N", (10.0, 46.0, 36.84), "E", (106.0, 42.0, 3.24)
        if gps_err:
            g[31] = gps_err
    return e


def sky_image(size=(640, 480)):
    img = Image.new("RGB", size, (60, 110, 40))
    px = img.load()
    for y in range(size[1] // 2):
        for x in range(size[0]):
            px[x, y] = (100 + y // 10, 150 + y // 10, 235)
    return img


class Tests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def p(self, name):
        return os.path.join(self.dir.name, name)

    def test_gps_decimal(self):
        lat = t.to_decimal((10.0, 46.0, 36.84), "N")
        lon = t.to_decimal((106.0, 42.0, 3.24), "W")
        self.assertAlmostEqual(lat, 10.7769, places=4)
        self.assertAlmostEqual(lon, -106.7009, places=4)

    def test_read_exif_coords(self):
        path = self.p("a.jpg")
        sky_image().save(path, exif=make_exif("Apple", "iPhone", gps=True))
        rows, coords = t.read_exif(path)
        self.assertIsNotNone(coords)
        self.assertAlmostEqual(coords[1], 106.7009, places=4)

    def test_strip_removes_everything(self):
        path = self.p("a.jpg")
        sky_image().save(path, exif=make_exif("Apple", "iPhone", gps=True))
        out = t.strip_exif(path)
        rows, coords = t.read_exif(out)
        self.assertEqual(rows, [])
        self.assertIsNone(coords)

    def test_strip_keeps_icc(self):
        path = self.p("a.jpg")
        icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        sky_image().save(path, exif=make_exif("Apple", "iPhone", gps=True), icc_profile=icc)
        out = t.strip_exif(path)
        with Image.open(out) as check:
            self.assertTrue(check.info.get("icc_profile"))

    def test_strip_no_overwrite(self):
        path = self.p("a.jpg")
        sky_image().save(path)
        first = t.strip_exif(path)
        second = t.strip_exif(path)
        self.assertNotEqual(first, second)
        self.assertTrue(os.path.exists(first) and os.path.exists(second))

    def test_strip_rejects_animation(self):
        path = self.p("a.gif")
        frames = [Image.new("RGB", (20, 20), c) for c in ((255, 0, 0), (0, 0, 255))]
        frames[0].save(path, save_all=True, append_images=frames[1:])
        with self.assertRaises(ValueError):
            t.strip_exif(path)

    def test_png_ai_params(self):
        path = self.p("ai.png")
        info = PngImagePlugin.PngInfo()
        info.add_text("parameters", "a cat\nSteps: 20, Sampler: Euler a")
        Image.new("RGB", (1024, 1024), (120, 60, 200)).save(path, pnginfo=info)
        d = t.analyze_file(path)
        self.assertGreaterEqual(d["ai_score"], 70)

    def test_c2pa_is_not_ai(self):
        path = self.p("real.jpg")
        sky_image().save(path, exif=make_exif("Google", "Pixel 10", ev_args=(1 / 500, 1.7, 50)))
        with open(path, "ab") as f:
            f.write(b"\x00jumbf c2pa manifest\x00")
        d = t.analyze_file(path)
        self.assertLess(d["ai_score"], 40)
        self.assertTrue(any("C2PA" in x for _, x in d["ai_reasons"]))

    def test_real_photo_low_score(self):
        path = self.p("real.jpg")
        sky_image().save(path, exif=make_exif("Apple", "iPhone", ev_args=(1 / 2000, 1.8, 25)))
        self.assertLess(t.analyze_file(path)["ai_score"], 40)

    def test_bare_ai_sized_image_suspicious(self):
        path = self.p("bare.png")
        Image.new("RGB", (1792, 1024), (10, 60, 200)).save(path)
        self.assertGreaterEqual(t.analyze_file(path)["ai_score"], 40)

    def test_sky_detected(self):
        path = self.p("sky.jpg")
        sky_image().save(path)
        blue, _ = t.analyze_sky(Image.open(path), 1)
        self.assertGreater(blue, 0.3)

    def test_no_sky_on_ground(self):
        img = Image.new("RGB", (320, 240), (70, 90, 40))
        blue, gray = t.analyze_sky(img, 1)
        self.assertEqual(blue, 0)

    def test_sky_respects_orientation(self):
        img = sky_image().rotate(90, expand=True)
        blue_wrong, _ = t.analyze_sky(img, 1)
        blue_ok, _ = t.analyze_sky(img, 6)
        self.assertGreater(blue_ok, blue_wrong)

    def test_indoor_flash(self):
        path = self.p("room.jpg")
        Image.new("RGB", (400, 300), (90, 90, 90)).save(
            path, exif=make_exif("Samsung", "S23", ev_args=(1 / 30, 1.8, 1600), flash=1, gps=True, gps_err=65.0))
        d = t.analyze_file(path)
        self.assertTrue(d["scene_label"].startswith("TRONG NHÀ"))

    def test_outdoor_sunny(self):
        path = self.p("sun.jpg")
        sky_image().save(path, exif=make_exif("Apple", "iPhone", ev_args=(1 / 2000, 1.8, 25)))
        self.assertEqual(t.analyze_file(path)["scene_label"], "NGOÀI TRỜI")

    def test_unknown_without_data(self):
        path = self.p("x.png")
        Image.new("RGB", (333, 211), (150, 140, 135)).save(path)
        self.assertTrue(t.analyze_file(path)["scene_label"].startswith("KHÔNG XÁC ĐỊNH"))

    def test_random_bytes_in_pixels_do_not_flag_ai(self):
        path = self.p("real.jpg")
        sky_image().save(path, exif=make_exif("Apple", "iPhone", ev_args=(1 / 2000, 1.8, 25)))
        with open(path, "ab") as f:
            f.write(b"\x00SDxl\x00stable diffusion\x00")
        self.assertLess(t.analyze_file(path)["ai_score"], 40)

    def test_comment_keyword_still_flags_ai(self):
        path = self.p("ai.jpg")
        sky_image().save(path, comment=b"Generated with Stable Diffusion XL")
        self.assertGreaterEqual(t.analyze_file(path)["ai_score"], 70)

    def test_chat_processed_missing_metadata_is_not_suspicious(self):
        path = self.p("2aOboR9kCEW8dkpTblX7JkG0cOn8cuvuS4qu.jpg")
        sky_image((1280, 960)).save(path, quality=78, progressive=True)
        d = t.analyze_file(path)
        self.assertEqual(d["ai_score"], 0)
        self.assertTrue(d["ai_label"].startswith("KHÔNG ĐỦ DỮ KIỆN"))

    def test_residual_exif_counts_as_stripped(self):
        path = self.p("a.jpg")
        e = Image.Exif()
        e[0x0112] = 1
        e.get_ifd(0x8769)[0xA002] = 640
        sky_image().save(path, exif=e)
        d = t.analyze_file(path)
        self.assertFalse(d["prov"]["exif_present"])

    def test_greenery_means_outdoor(self):
        path = self.p("leaf.jpg")
        Image.new("RGB", (400, 300), (50, 140, 40)).save(path)
        self.assertEqual(t.analyze_file(path)["scene_label"], "NGOÀI TRỜI")

    def test_csv_row(self):
        path = self.p("a.jpg")
        sky_image().save(path, exif=make_exif("Apple", "iPhone", ev_args=(1 / 2000, 1.8, 25), gps=True))
        row = t.csv_row(t.analyze_file(path))
        self.assertEqual(set(row), set(t.CSV_FIELDS))
        self.assertEqual(row["has_gps"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
