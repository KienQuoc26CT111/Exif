import datetime
import io
import os
import re
import shutil
import struct

from PIL import Image, ImageCms, ImageOps

STD_LUMA = (
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
)
STD_CHROMA = (
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99, 47, 66, 99, 99, 99, 99, 99, 99,
) + (99,) * 32

NATIVE_SIZES = {
    (4032, 3024), (3024, 4032), (4000, 3000), (3000, 4000), (3264, 2448), (2448, 3264),
    (4608, 3456), (3456, 4608), (4624, 3472), (3472, 4624), (4160, 3120), (3120, 4160),
    (5120, 3840), (3840, 5120), (8000, 6000), (6000, 8000), (4080, 3072), (3072, 4080),
    (4284, 5712), (5712, 4284), (4000, 2250), (2250, 4000), (4096, 3072), (3072, 4096),
    (3840, 2160), (2160, 3840), (2592, 1944), (1944, 2592), (2560, 1920), (1920, 2560),
}
RESIZE_LONG_SIDES = {640, 720, 800, 960, 1024, 1080, 1280, 1440, 1600, 1920, 2048, 2560}
SERVER_COMMENTS = ("gd-jpeg", "lavc", "ijg", "imagemagick", "photoshop", "created with", "optimized")

TAG_COLOR = {"bad": "bad", "warn": "warn", "ok": "ok", "dim": "dim"}


TRIVIAL_MAIN_TAGS = {0x0112, 0x8769, 0x8825, 0xA005, 0x011A, 0x011B, 0x0128, 0x0213}
TRIVIAL_SUB_TAGS = {0x9000, 0x9101, 0xA000, 0xA001, 0xA002, 0xA003, 0xA005}


def meaningful_exif(img):
    exif = img.getexif()
    if any(k not in TRIVIAL_MAIN_TAGS for k in exif.keys()):
        return True
    if any(k not in TRIVIAL_SUB_TAGS for k in exif.get_ifd(0x8769).keys()):
        return True
    return len(exif.get_ifd(0x8825)) > 0


def metadata_bytes(path):
    try:
        with open(path, "rb") as f:
            head = f.read(2_000_000)
            if head[:2] == b"\xff\xd8":
                sos = head.find(b"\xff\xda")
                tail_marker = b""
                f.seek(0, os.SEEK_END)
                f.seek(max(0, f.tell() - 16_000))
                tail_marker = f.read()
                cut = head[:sos] if sos > 0 else head[:65536]
                return (cut + b"\x00" + tail_marker).lower()
            if head[:8] == b"\x89PNG\r\n\x1a\n":
                f.seek(8)
                out = b""
                while True:
                    hdr = f.read(8)
                    if len(hdr) < 8:
                        break
                    length, ctype = struct.unpack(">I4s", hdr)
                    if ctype == b"IDAT":
                        f.seek(length + 4, os.SEEK_CUR)
                        continue
                    if length > 5_000_000:
                        break
                    out += ctype + f.read(length)
                    f.read(4)
                return out.lower()
            return head[:65536].lower()
    except OSError:
        return b""


def scaled_table(std, q):
    scale = 5000 // q if q < 50 else 200 - q * 2
    return [min(255, max(1, (v * scale + 50) // 100)) for v in std]


def libjpeg_quality(tables):
    luma = tables.get(0) if tables else None
    if not luma or len(luma) != 64:
        return None
    luma = list(luma)
    chroma = tables.get(1)
    for q in range(1, 101):
        if luma == scaled_table(STD_LUMA, q):
            if chroma is None or list(chroma) == scaled_table(STD_CHROMA, q):
                return q
    return None


def inspect_jpeg(path):
    info = {"jfif": False, "exif": False, "xmp": False, "icc": False, "mpf": False,
            "iptc": False, "progressive": False, "comment": "", "motion": False}
    try:
        with open(path, "rb") as f:
            if f.read(2) != b"\xff\xd8":
                return None
            while True:
                b = f.read(1)
                if not b:
                    break
                if b != b"\xff":
                    continue
                m = f.read(1)
                while m == b"\xff":
                    m = f.read(1)
                if not m:
                    break
                code = m[0]
                if code in (0x00, 0x01, 0xD8) or 0xD0 <= code <= 0xD7:
                    continue
                if code == 0xD9:
                    break
                raw = f.read(2)
                if len(raw) < 2:
                    break
                size = struct.unpack(">H", raw)[0]
                payload = f.read(max(0, size - 2))
                head = payload[:40]
                if code == 0xE0 and head.startswith(b"JFIF"):
                    info["jfif"] = True
                elif code == 0xE1 and head.startswith(b"Exif"):
                    info["exif"] = True
                elif code == 0xE1 and b"ns.adobe.com/xap" in head:
                    info["xmp"] = True
                elif code == 0xE2 and head.startswith(b"ICC_PROFILE"):
                    info["icc"] = True
                elif code == 0xE2 and head.startswith(b"MPF"):
                    info["mpf"] = True
                elif code == 0xED:
                    info["iptc"] = True
                elif code == 0xFE:
                    info["comment"] = payload.decode("latin-1", "ignore").strip()[:80]
                elif code in (0xC2, 0xC6, 0xCA, 0xCE):
                    info["progressive"] = True
                elif code == 0xDA:
                    break
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 3_000_000))
            tail = f.read()
        if b"ftypmp4" in tail or b"ftypisom" in tail or b"MotionPhoto" in tail:
            info["motion"] = True
    except OSError:
        return None
    return info


def icc_description(img):
    icc = img.info.get("icc_profile")
    if not icc:
        return None
    try:
        return ImageCms.getProfileDescription(ImageCms.ImageCmsProfile(io.BytesIO(icc))).strip()
    except (ImageCms.PyCMSError, OSError, TypeError):
        return "không đọc được"


def parse_time(fmt, text):
    try:
        return datetime.datetime.strptime(text, fmt)
    except ValueError:
        return None


def filename_hints(name):
    stem = os.path.splitext(os.path.basename(name))[0]
    out = []
    reenc = 0.0
    orig = 0.0
    when = None
    m = re.match(r"^FB_IMG_(\d{13})", stem)
    if m:
        out.append(("warn", "Tên file kiểu FB_IMG_...: ảnh lưu từ Facebook/Messenger"))
        reenc += 2
        try:
            when = datetime.datetime.fromtimestamp(int(m.group(1)) / 1000)
        except (OverflowError, OSError, ValueError):
            when = None
    elif re.match(r"^received_\d+", stem):
        out.append(("warn", "Tên file kiểu received_...: ảnh nhận qua Messenger"))
        reenc += 2
    elif re.match(r"^z\d{7,}_[0-9a-f]{16,}", stem, re.I):
        out.append(("warn", "Tên file dạng z<số>_<mã>: kiểu đặt tên thường gặp của Zalo bản máy tính"))
        reenc += 2
    elif re.match(r"^photo_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", stem):
        out.append(("warn", "Tên file kiểu photo_ngày_giờ: ảnh tải từ Telegram"))
        reenc += 2
        when = parse_time("%Y-%m-%d_%H-%M-%S", stem[6:25])
    elif re.match(r"^WhatsApp Image", stem):
        out.append(("warn", "Tên file kiểu WhatsApp Image: ảnh nhận qua WhatsApp"))
        reenc += 2
        m2 = re.search(r"(\d{4}-\d{2}-\d{2}) at (\d{2})\.(\d{2})\.(\d{2})", stem)
        if m2:
            when = parse_time("%Y-%m-%d %H:%M:%S", f"{m2.group(1)} {m2.group(2)}:{m2.group(3)}:{m2.group(4)}")
    elif re.match(r"^(screenshot|ảnh chụp màn hình|screen shot)", stem, re.I):
        out.append(("warn", "Tên file cho thấy đây là ảnh chụp màn hình, không phải ảnh từ camera"))
    elif re.match(r"^PXL_\d{8}_\d{6}", stem):
        out.append(("ok", "Tên file kiểu PXL_ngày_giờ: đặt tên mặc định của camera Google Pixel"))
        orig += 1.5
        when = parse_time("%Y%m%d_%H%M%S", stem[4:19])
    elif re.match(r"^IMG[_-]\d{8}[_-]\d{6}", stem):
        out.append(("ok", "Tên file kiểu IMG_ngày_giờ: đặt tên mặc định của camera nhiều máy Android"))
        orig += 1.5
        when = parse_time("%Y%m%d_%H%M%S", stem[4:19].replace("-", "_"))
    elif re.match(r"^\d{8}_\d{6}", stem):
        out.append(("ok", "Tên file kiểu ngày_giờ: đặt tên mặc định của camera điện thoại (Samsung, Xiaomi...)"))
        orig += 1.5
        when = parse_time("%Y%m%d_%H%M%S", stem[:15])
    elif re.match(r"^IMG_\d{4}(?!\d)", stem):
        out.append(("ok", "Tên file kiểu IMG_1234: đặt tên mặc định của iPhone / máy ảnh"))
        orig += 1.0
    elif re.match(r"^DSC[_-]?\d{4,}", stem):
        out.append(("ok", "Tên file kiểu DSC_xxxx: đặt tên mặc định của máy ảnh"))
        orig += 1.0
    elif len(stem) >= 20 and re.fullmatch(r"[A-Za-z0-9_\-]+", stem) and not stem.isdigit():
        classes = sum([any(c.islower() for c in stem), any(c.isupper() for c in stem),
                       any(c.isdigit() for c in stem)])
        if classes >= 2:
            out.append(("warn", "Tên file là chuỗi ngẫu nhiên dài: đặc trưng của ảnh tải về từ ứng dụng chat/mạng xã hội hay trình duyệt, không phải tên do camera đặt"))
            reenc += 2
    if when:
        out.append(("dim", f"Gợi ý thời điểm theo tên file: {when:%d/%m/%Y %H:%M:%S} (chỉ là gợi ý, tên file có thể bị đổi)"))
    return out, reenc, orig, when


def analyze_provenance(path, img):
    findings = []
    reenc = 0.0
    orig = 0.0
    fmt = (img.format or "").upper()
    w, h = img.size
    exif_present = meaningful_exif(img)
    residual_exif = (not exif_present) and len(img.getexif()) > 0
    quality = None
    progressive = False
    findings.append(("dim", f"Định dạng {fmt or '?'}, {w}x{h} px ({w * h / 1e6:.1f} MP)"))

    if fmt == "JPEG":
        seg = inspect_jpeg(path) or {}
        quality = libjpeg_quality(getattr(img, "quantization", None))
        progressive = bool(seg.get("progressive") or img.info.get("progressive") or img.info.get("progression"))
        if residual_exif:
            findings.append(("warn", "Chỉ còn EXIF tối thiểu (hướng ảnh, kích thước): dấu hiệu máy chủ đã ghi lại ảnh và xóa thông tin chụp"))
            reenc += 1.5
        if seg.get("jfif") and not seg.get("exif"):
            findings.append(("warn", "Đầu file chỉ có JFIF, không có Exif: kiểu file do phần mềm/máy chủ mã hóa lại (camera điện thoại gốc thường ghi Exif)"))
            reenc += 1
        if seg.get("exif") and exif_present:
            orig += 1
        if progressive:
            findings.append(("warn", "JPEG dạng progressive: thường do máy chủ web/mạng xã hội tối ưu để tải nhanh, camera điện thoại hầu như không tạo kiểu này"))
            reenc += 2
        if quality is not None and quality <= 92:
            findings.append(("warn", f"Bảng lượng tử hóa là bảng chuẩn libjpeg ở chất lượng {quality}: ảnh đã bị nén lại bằng phần mềm hoặc máy chủ, không phải file gốc của camera"))
            reenc += 2
        elif quality is not None:
            findings.append(("dim", f"Bảng lượng tử hóa chuẩn libjpeg ở chất lượng {quality}: có thể là file gốc của một số máy Android hoặc ảnh lưu lại chất lượng cao"))
        else:
            findings.append(("ok", "Bảng lượng tử hóa tùy biến (không phải bảng chuẩn): kiểu của camera điện thoại như iPhone, Samsung; ít khi gặp ở ảnh bị máy chủ nén lại"))
            orig += 1.5
        if seg.get("mpf"):
            findings.append(("ok", "Có Multi-Picture Format (ảnh nhúng như bản đồ độ sâu / HDR): đặc trưng của file gốc từ điện thoại"))
            orig += 2
        if seg.get("motion"):
            findings.append(("ok", "Có dữ liệu Motion Photo (video ngắn nhúng trong ảnh): file gốc từ điện thoại Samsung/Pixel"))
            orig += 2
        comment = seg.get("comment", "")
        if comment and any(k in comment.lower() for k in SERVER_COMMENTS):
            findings.append(("warn", f"Có ghi chú của phần mềm xử lý ảnh trong file: \"{comment}\""))
            reenc += 2
        elif comment:
            findings.append(("dim", f"Có ghi chú trong file: \"{comment}\""))
        if seg.get("xmp") or seg.get("iptc"):
            findings.append(("dim", "Còn khối XMP/IPTC (có thể chứa thông tin tác giả, phần mềm, mô tả)"))
    elif fmt == "PNG":
        if max(w, h) / max(1, min(w, h)) >= 1.9:
            findings.append(("warn", "PNG tỉ lệ rất dài: nhiều khả năng là ảnh chụp màn hình điện thoại, không phải ảnh từ camera"))
        else:
            findings.append(("dim", "PNG: không phải định dạng camera điện thoại thường lưu, ảnh có thể đã được chuyển đổi"))
        reenc += 1

    desc = icc_description(img)
    if desc and "p3" in desc.lower():
        findings.append(("ok", f"Hồ sơ màu \"{desc}\": thường do iPhone hoặc điện thoại cao cấp xuất ra"))
        orig += 1
    elif desc:
        findings.append(("dim", f"Hồ sơ màu: {desc}"))
    elif fmt == "JPEG":
        findings.append(("dim", "Không có hồ sơ màu ICC (ứng dụng chat/mạng xã hội thường loại bỏ)"))
        reenc += 0.5

    if (w, h) in NATIVE_SIZES:
        findings.append(("ok", f"Kích thước {w}x{h} trùng kích thước gốc của camera điện thoại: nhiều khả năng chưa bị thu nhỏ"))
        orig += 2
    elif max(w, h) in RESIZE_LONG_SIDES:
        findings.append(("warn", f"Cạnh dài {max(w, h)} px khớp mức thu nhỏ thường gặp của ứng dụng chat/mạng xã hội"))
        reenc += 1.5

    name_findings, r, o, when = filename_hints(path)
    findings += name_findings
    reenc += r
    orig += o

    try:
        mtime = datetime.datetime.fromtimestamp(os.stat(path).st_mtime)
        findings.append(("dim", f"Ngày sửa file trên máy: {mtime:%d/%m/%Y %H:%M:%S} (file chép thẳng từ điện thoại thường giữ giờ chụp, file tải từ ứng dụng thường là giờ tải về)"))
    except OSError:
        pass

    if exif_present:
        verdict = "Còn EXIF: nguồn gốc file chưa bị làm sạch"
        tag = "ok"
        banner = ""
    elif reenc >= 3:
        verdict = "ĐÃ QUA ỨNG DỤNG CHAT/MẠNG XÃ HỘI hoặc máy chủ nén lại. EXIF bị xóa là việc tự động; ảnh gốc ban đầu rất có thể có GPS, giờ chụp, model máy"
        tag = "warn"
        banner = "EXIF đã bị xóa: ảnh nhiều khả năng đã qua ứng dụng chat/MXH"
    elif orig >= 3:
        verdict = "Cấu trúc file giống ảnh xuất ra từ điện thoại (bảng nén riêng, hồ sơ màu P3...) nhưng thông tin chụp đã bị gỡ: có thể do ứng dụng gửi ảnh trên điện thoại tự nén lại, hoặc do công cụ xóa metadata"
        tag = "warn"
        banner = "EXIF đã bị gỡ; file có cấu trúc kiểu ảnh điện thoại"
    else:
        verdict = "Không có EXIF và không đủ dấu hiệu để xác định ảnh đến từ đâu"
        tag = "dim"
        banner = "Không có EXIF, chưa rõ nguồn gốc file"
    if not exif_present:
        findings.append(("dim", "Dữ liệu EXIF đã mất (GPS, giờ chụp, model máy) không thể khôi phục từ file này. Muốn có lại, nhờ người gửi chuyển file gốc qua Email, Google Drive hoặc Telegram (chế độ File)"))
    return {"findings": findings, "exif_present": exif_present, "verdict": verdict,
            "verdict_tag": tag, "banner": banner,
            "reenc": reenc, "orig": orig, "quality": quality, "progressive": progressive,
            "name_time": when}


ENTITY_RULES = (
    ("bad", "số điện thoại", r"(?<!\d)(?:\+?84|0)(?:[\s.\-]?\d){8,10}(?!\d)"),
    ("bad", "email", r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    ("bad", "dãy 12 số (giống số căn cước)", r"(?<!\d)\d{12}(?!\d)"),
    ("bad", "biển số xe", r"\b\d{2}[\s\-]?[A-ZĐ]{1,2}\d?[\s\-]?\d{3}[\s.]?\d{2}\b"),
    ("warn", "đường dẫn web", r"(?:https?://|www\.)\S+"),
    ("warn", "ngày tháng", r"\b\d{1,2}\s*[/\-.]\s*\d{1,2}\s*[/\-.]\s*20\d{2}\b"),
    ("warn", "năm", r"\b20[12]\d\b"),
    ("warn", "từ khóa địa chỉ / tổ chức",
     r"\b(đường|phường|quận|huyện|thôn|xã|thành phố|tỉnh|số nhà|tòa nhà|toà nhà|trường|đại học|bệnh viện|công ty|phòng|lớp|khoa|duong|phuong|quan|truong|cong ty|school|university|hospital|street|road|room)\b"),
)


def find_entities(lines):
    found = []
    for line in lines:
        for tag, label, pattern in ENTITY_RULES:
            m = re.search(pattern, line, re.I if label.startswith("từ khóa") else 0)
            if m:
                found.append((tag, label, m.group(0).strip(), line))
    return found


def locate_tesseract(pytesseract):
    if shutil.which("tesseract"):
        return True
    for p in (r"C:\Program Files\Tesseract-OCR\tesseract.exe",
              r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"):
        if os.path.exists(p):
            pytesseract.pytesseract.tesseract_cmd = p
            return True
    return False


def ocr_tiles(img, max_tiles=9):
    w, h = img.size
    cols = max(1, min(4, round(w / 700)))
    rows = max(1, min(4, round(h / 700)))
    while cols * rows > max_tiles:
        if cols >= rows:
            cols -= 1
        else:
            rows -= 1
    tw = w / cols
    th = h / rows
    for r in range(rows):
        for c in range(cols):
            x0 = max(0, int(c * tw - 0.15 * tw))
            y0 = max(0, int(r * th - 0.15 * th))
            x1 = min(w, int((c + 1) * tw + 0.15 * tw))
            y1 = min(h, int((r + 1) * th + 0.15 * th))
            yield img.crop((x0, y0, x1, y1))


def good_token(w, c, min_conf):
    w = w.strip(" |_~`'\"“”‘’")
    if c < min_conf or not w:
        return ""
    core = re.sub(r"[/.,:;\-]", "", w)
    if not core or sum(ch.isalnum() for ch in core) / len(core) < 0.8:
        return ""
    if len(w) >= 3 or (len(w) >= 2 and any(ch.isdigit() for ch in w)):
        return w
    return ""


def clean_line(words, confs):
    good = [g for g in (good_token(w, c, 70) for w, c in zip(words, confs)) if g]
    if len(good) >= 2:
        return " ".join(good)
    for w, c in zip(words, confs):
        g = good_token(w, c, 85)
        if len(g) >= 6:
            return g
    return ""


def ocr_lines(img):
    try:
        import pytesseract
    except ImportError:
        return None, "Chưa cài pytesseract (pip install pytesseract) nên chưa đọc được chữ trong ảnh"
    if not locate_tesseract(pytesseract):
        return None, "Chưa cài chương trình Tesseract OCR nên chưa đọc được chữ trong ảnh"
    try:
        langs = set(pytesseract.get_languages(config=""))
    except Exception as e:
        return None, f"Không khởi động được Tesseract: {e}"
    note = ""
    if "vie" in langs and "eng" in langs:
        lang = "vie+eng"
    elif "vie" in langs:
        lang = "vie"
    else:
        lang = "eng" if "eng" in langs else next(iter(langs), "eng")
        note = "Tesseract chưa có dữ liệu tiếng Việt (vie): chữ có dấu sẽ đọc kém"
    found = []
    seen = set()
    try:
        for tile in ocr_tiles(img):
            g = ImageOps.autocontrast(tile.convert("L"))
            f = max(1.0, 1400 / max(g.size))
            if f != 1.0:
                g = g.resize((int(g.width * f), int(g.height * f)), Image.LANCZOS)
            d = pytesseract.image_to_data(g, lang=lang, config="--psm 6",
                                          output_type=pytesseract.Output.DICT, timeout=60)
            grouped = {}
            for i, txt in enumerate(d["text"]):
                txt = txt.strip()
                try:
                    conf = float(d["conf"][i])
                except (TypeError, ValueError):
                    conf = -1
                if conf >= 50 and txt:
                    key = (d["block_num"][i], d["par_num"][i], d["line_num"][i])
                    grouped.setdefault(key, []).append((txt, conf))
            for items in grouped.values():
                words = [w for w, _ in items]
                confs = [c for _, c in items]
                line = clean_line(words, confs)
                norm = re.sub(r"\W+", "", line.lower())
                if line and norm not in seen:
                    seen.add(norm)
                    found.append(line)
    except Exception as e:
        return None, f"Đọc chữ thất bại: {e}"
    return found, note


def decode_qr(img):
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    det = cv2.QRCodeDetector()
    arr = np.array(img.convert("RGB"))
    try:
        ok, decoded, _, _ = det.detectAndDecodeMulti(arr)
    except cv2.error:
        return []
    return [d for d in decoded if d] if ok else []


def faces_with_skimage(gray):
    from skimage import data, transform
    from skimage.feature import Cascade
    det = Cascade(data.lbp_frontal_face_cascade_filename())
    h, w = gray.shape
    f = min(1.0, 1200 / max(h, w))
    small = transform.resize(gray, (max(1, int(h * f)), max(1, int(w * f))), anti_aliasing=True)
    found = det.detect_multi_scale(
        img=small, scale_factor=1.2, step_ratio=1, min_size=(24, 24),
        max_size=(small.shape[1] // 2, small.shape[0] // 2), min_neighbor_number=5)
    return len(found)


def faces_with_opencv(gray):
    import cv2
    path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    if not os.path.exists(path):
        return None
    casc = cv2.CascadeClassifier(path)
    if casc.empty():
        return None
    h, w = gray.shape
    if max(h, w) > 1200:
        f = 1200 / max(h, w)
        gray = cv2.resize(gray, (int(w * f), int(h * f)))
    return len(casc.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=5, minSize=(22, 22)))


def count_faces(img):
    try:
        import numpy as np
    except ImportError:
        return None, "Chưa cài numpy nên chưa dò khuôn mặt"
    gray = np.array(img.convert("L"))
    try:
        return faces_with_skimage(gray), ""
    except ImportError:
        pass
    try:
        n = faces_with_opencv(gray)
        if n is not None:
            return n, ""
    except (ImportError, AttributeError):
        pass
    return None, "Chưa có dữ liệu dò mặt. Cài: pip install scikit-image"


def content_lines(path):
    with Image.open(path) as raw:
        img = ImageOps.exif_transpose(raw)
        img.load()
    lines = [("h", "5. Nội dung nhìn thấy trong ảnh (dù không còn EXIF vẫn có thể lộ thông tin)")]
    texts, note = ocr_lines(img)
    if texts is None:
        lines.append(("dim", "   • " + note))
    else:
        if note:
            lines.append(("dim", "   • " + note))
        if texts:
            lines.append(("warn", f"   • Đọc được {len(texts)} dòng chữ trong ảnh, ví dụ:"))
            for t in texts[:6]:
                lines.append(("dim", "       \"" + t[:90] + "\""))
            for tag, label, value, line in find_entities(texts)[:8]:
                lines.append((tag, f"   • Có {label}: {value}"))
        else:
            lines.append(("ok", "   • Không đọc được chữ rõ ràng (biển hiệu, tên, số điện thoại...)"))
    qr = decode_qr(img)
    faces, face_note = count_faces(img)
    if qr is None:
        lines.append(("dim", "   • Chưa cài OpenCV (pip install opencv-python) nên chưa dò mã QR"))
    if faces is None and qr is not None:
        lines.append(("dim", "   • " + face_note))
    if qr:
        lines.append(("bad", "   • Có mã QR chứa: " + "; ".join(q[:80] for q in qr[:3])))
    elif qr is not None:
        lines.append(("ok", "   • Không thấy mã QR"))
    if faces:
        lines.append(("warn", f"   • Có ít nhất khoảng {faces} khuôn mặt nhìn thẳng (số thật thường nhiều hơn vì mặt nghiêng/che bị sót): ảnh có người, có thể nhận ra danh tính"))
    elif faces is not None:
        lines.append(("ok", "   • Không thấy khuôn mặt nhìn thẳng (bộ dò đơn giản, có thể sót mặt nghiêng/che/nhỏ)"))
    return lines


def prov_lines(prov):
    lines = [("h", "3. Dấu vết nguồn gốc file (dò ngoài EXIF)")]
    lines.append((prov["verdict_tag"], "   ► " + prov["verdict"]))
    lines += [(t, "   • " + x) for t, x in prov["findings"]]
    return lines
