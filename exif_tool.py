import os
import csv
import math
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import webbrowser
from tkinter import simpledialog
from PIL import Image, ImageOps, ImageTk, ImageFilter
from PIL.ExifTags import TAGS, GPSTAGS

from forensics import analyze_provenance, prov_lines, content_lines, metadata_bytes, meaningful_exif
import vision_ai

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

GPS_IFD = 0x8825
EXIF_IFD = 0x8769
EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".heic", ".heif")


def to_decimal(values, ref):
    d, m, s = (float(x) for x in values)
    dec = d + m / 60 + s / 3600
    if ref in ("S", "W"):
        dec = -dec
    return dec


def read_exif(path):
    with Image.open(path) as img:
        return read_exif_from(img)


def read_exif_from(img):
    exif = img.getexif()
    rows = []
    for tag_id, value in exif.items():
        if tag_id in (GPS_IFD, EXIF_IFD):
            continue
        rows.append((TAGS.get(tag_id, str(tag_id)), clean_value(value)))
    for tag_id, value in exif.get_ifd(EXIF_IFD).items():
        rows.append((TAGS.get(tag_id, str(tag_id)), clean_value(value)))
    gps_raw = exif.get_ifd(GPS_IFD)
    gps = {GPSTAGS.get(k, str(k)): v for k, v in gps_raw.items()}
    for k, v in gps.items():
        rows.append(("GPS " + k, clean_value(v)))
    coords = None
    try:
        lat = to_decimal(gps["GPSLatitude"], gps.get("GPSLatitudeRef", "N"))
        lon = to_decimal(gps["GPSLongitude"], gps.get("GPSLongitudeRef", "E"))
        coords = (lat, lon)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        pass
    return rows, coords


def clean_value(v):
    if isinstance(v, bytes):
        try:
            return v.decode("utf-8", "ignore").strip("\x00")
        except Exception:
            return repr(v)
    if isinstance(v, tuple):
        return ", ".join(str(clean_value(x)) for x in v)
    try:
        if hasattr(v, "numerator"):
            return str(round(float(v), 6))
    except (TypeError, ValueError, OverflowError):
        pass
    s = str(v)
    return s if len(s) < 200 else s[:200] + "..."


def strip_exif(src, dst_dir=None):
    with Image.open(src) as original:
        return strip_loaded(original, src, dst_dir)


def strip_loaded(original, src, dst_dir):
    if getattr(original, "n_frames", 1) > 1:
        raise ValueError("ảnh động (nhiều khung hình) chưa được hỗ trợ")
    icc = original.info.get("icc_profile")
    img = ImageOps.exif_transpose(original)
    if img.mode not in ("RGB", "RGBA", "L", "LA"):
        has_alpha = "A" in img.getbands() or "transparency" in img.info
        img = img.convert("RGBA" if has_alpha else "RGB")
    base, ext = os.path.splitext(os.path.basename(src))
    folder = dst_dir or os.path.dirname(src)
    ext = ext.lower()
    if ext in (".heic", ".heif"):
        ext = ".jpg"
    out = os.path.join(folder, base + "_clean" + ext)
    n = 2
    while os.path.exists(out):
        out = os.path.join(folder, f"{base}_clean{n}{ext}")
        n += 1
    clean = Image.frombytes(img.mode, img.size, img.tobytes())
    save_kwargs = {}
    if icc:
        save_kwargs["icc_profile"] = icc
    if ext in (".jpg", ".jpeg"):
        if clean.mode not in ("RGB", "L"):
            clean = clean.convert("RGB")
        save_kwargs["quality"] = 95
    clean.save(out, **save_kwargs)
    with Image.open(out) as check:
        leftover = len(check.getexif())
    if leftover > 0:
        os.remove(out)
        raise RuntimeError("không xóa sạch được metadata")
    return out


AI_KEYWORDS = (
    "stable diffusion", "midjourney", "dall-e", "dall\u00b7e", "firefly", "sdxl",
    "comfyui", "automatic1111", "novelai", "leonardo.ai", "made with google ai",
    "trainedalgorithmicmedia", "compositesynthetic", "ai generated",
    "ai-generated", "invokeai",
)
AI_PNG_KEYS = ("parameters", "prompt", "workflow", "sd-metadata", "dream",
               "invokeai_metadata", "negative_prompt")
AI_SIZES = {(1024, 1024), (1024, 1792), (1792, 1024), (1536, 1024), (1024, 1536),
            (832, 1216), (1216, 832), (1344, 768), (768, 1344), (768, 768), (512, 512)}


def num(v):
    if isinstance(v, (tuple, list)) and v:
        v = v[0]
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def reorient(img, orientation):
    ops = {2: Image.FLIP_LEFT_RIGHT, 3: Image.ROTATE_180, 4: Image.FLIP_TOP_BOTTOM,
           5: Image.TRANSPOSE, 6: Image.ROTATE_270, 7: Image.TRANSVERSE, 8: Image.ROTATE_90}
    if orientation in ops:
        return img.transpose(ops[orientation])
    return img


def analyze_ai(path, img, main, sub, prov=None):
    score = 0
    reasons = []
    data = metadata_bytes(path)
    hits = []
    for kw in AI_KEYWORDS:
        if (kw.encode("utf-8") in data or kw.encode("utf-16-le") in data
                or kw.encode("utf-16-be") in data):
            hits.append(kw)
    if b"steps:" in data and b"sampler" in data:
        hits.append("tham số Stable Diffusion (Steps/Sampler)")
    if hits:
        score += 70
        reasons.append(("bad", "Metadata chứa dấu vết công cụ AI: " + ", ".join(hits[:4])))
    elif b"c2pa" in data or b"jumbf" in data:
        reasons.append(("warn", "Có chứng thực nội dung C2PA: đây là chuẩn ghi nguồn gốc, cả ảnh AI lẫn một số máy ảnh/điện thoại đời mới đều dùng, nên không tính là bằng chứng AI"))
    png_keys = [k for k in img.info.keys() if str(k).lower() in AI_PNG_KEYS]
    if png_keys:
        score += 60
        reasons.append(("bad", "Khối text của PNG có khóa đặc trưng AI: " + ", ".join(map(str, png_keys))))
    software = str(main.get(0x0131, "") or "")
    if "photoshop" in software.lower():
        reasons.append(("warn", f"Ảnh đã qua phần mềm chỉnh sửa ({software}): có thể chứa vùng AI nhưng không xác định được"))
    make = main.get(0x010F)
    model = main.get(0x0110)
    has_cam = bool(make or model)
    has_time = bool(sub.get(0x9003))
    has_exposure = bool(sub.get(0x829A) and sub.get(0x829D))
    processed = bool(prov and prov["reenc"] >= 2)
    missing = (0 if has_cam else 1) + (0 if has_time else 1)
    if missing and processed:
        reasons.append(("dim", "Thiếu thông tin máy chụp/giờ chụp, nhưng ảnh đã bị ứng dụng chat/MXH xử lý nên việc thiếu này KHÔNG nói lên ảnh là thật hay AI"))
    elif missing == 2:
        score += 25
        reasons.append(("warn", "Thiếu cả thiết bị chụp lẫn thời điểm chụp"))
    elif missing == 1:
        score += 8
        reasons.append(("warn", "Thiếu một trong hai: thiết bị chụp hoặc thời điểm chụp"))
    fmt = (img.format or "").upper()
    w, h = img.size
    if not has_cam and not processed and fmt in ("PNG", "WEBP") and (w, h) in AI_SIZES:
        score += 15
        reasons.append(("warn", f"Kích thước {w}x{h} khớp mẫu đầu ra thường gặp của công cụ tạo ảnh AI"))
    if has_cam and has_time and has_exposure:
        score -= 40
        reasons.append(("ok", "Có đủ thông số máy ảnh thật (thiết bị, thời gian, khẩu độ, tốc độ)"))
    score = max(0, min(100, score))
    if score >= 70:
        label = "NHIỀU KHẢ NĂNG là ảnh AI"
        tag = "bad"
    elif score >= 40:
        label = "NGHI NGỜ ảnh AI hoặc ảnh đã bị xóa metadata"
        tag = "warn"
    elif has_cam and has_time and has_exposure:
        label = "Có vẻ là ảnh chụp thật"
        tag = "ok"
    else:
        label = "KHÔNG ĐỦ DỮ KIỆN: không thấy dấu hiệu AI, nhưng cũng không đủ thông tin để khẳng định ảnh thật"
        tag = "dim"
    return score, label, tag, reasons


def analyze_sky(img, orientation):
    small = img.convert("RGB")
    small.thumbnail((160, 160))
    small = reorient(small, orientation)
    w, h = small.size
    hsv = small.convert("HSV").load()
    edges = small.convert("L").filter(ImageFilter.FIND_EDGES).load()
    rgb = small.load()
    blue = 0
    gray = 0
    limit = int(h * 0.7)
    for x in range(w):
        for y in range(limit):
            hh, s, v = hsv[x, y]
            e = 0 if x in (0, w - 1) or y in (0, h - 1) else edges[x, y]
            r, g, b = rgb[x, y]
            if 120 <= hh <= 190 and s >= 35 and v >= 90 and e < 25 and b >= r:
                blue += 1
            elif s < 30 and v >= 185 and e < 18:
                gray += 1
            else:
                break
    total = w * h
    return blue / total, gray / total


def analyze_green(img, orientation):
    small = img.convert("RGB")
    small.thumbnail((160, 160))
    small = reorient(small, orientation)
    raw = small.convert("HSV").tobytes()
    total = small.width * small.height
    count = sum(1 for hh, ss, vv in zip(raw[0::3], raw[1::3], raw[2::3])
                if 30 <= hh <= 120 and ss >= 50 and vv >= 40)
    return count / total


def analyze_scene(sub, gps, blue, gray, green=0.0):
    score = 0
    reasons = []
    t = num(sub.get(0x829A))
    n = num(sub.get(0x829D))
    iso = num(sub.get(0x8827))
    ev = None
    if t and n and iso and t > 0 and n > 0 and iso > 0:
        ev = math.log2(n * n / t) - math.log2(iso / 100)
        if ev >= 12:
            score += 2
            reasons.append(("ok", f"Độ sáng cảnh EV100 ≈ {ev:.1f}: mức ánh sáng ban ngày ngoài trời"))
        elif ev >= 10:
            score += 1
            reasons.append(("ok", f"EV100 ≈ {ev:.1f}: sáng, nghiêng về ngoài trời (hoặc trời âm u)"))
        elif ev <= 8:
            score -= 2
            reasons.append(("warn", f"EV100 ≈ {ev:.1f}: ánh sáng yếu, đặc trưng trong nhà hoặc ban đêm"))
        else:
            score -= 1
            reasons.append(("warn", f"EV100 ≈ {ev:.1f}: mức sáng của phòng có đèn / cửa sổ"))
    flash = sub.get(0x9209)
    try:
        if flash is not None and int(flash) & 1:
            score -= 2
            reasons.append(("warn", "Đèn flash đã nháy: thường dùng ở nơi thiếu sáng, trong nhà"))
    except (TypeError, ValueError):
        pass
    dop = num(gps.get(11))
    herr = num(gps.get(31))
    if (herr is not None and herr >= 30) or (dop is not None and dop >= 5):
        score -= 1
        reasons.append(("warn", "Sai số GPS lớn: tín hiệu vệ tinh yếu, hay gặp khi ở trong nhà"))
    if blue >= 0.15:
        score += 2
        reasons.append(("ok", f"Bầu trời xanh chiếm ~{blue * 100:.0f}% khung hình"))
    elif blue >= 0.05:
        score += 1
        reasons.append(("ok", f"Thấy một phần bầu trời (~{blue * 100:.0f}%)"))
    if green >= 0.25:
        score += 2
        reasons.append(("ok", f"Cây cối xanh chiếm ~{green * 100:.0f}% khung hình: đặc trưng ngoài trời"))
    elif green >= 0.12:
        score += 1
        reasons.append(("ok", f"Có nhiều cây xanh (~{green * 100:.0f}% khung hình)"))
    if blue < 0.05 and gray >= 0.30:
        reasons.append(("warn", f"Vùng trắng/xám sáng ở phía trên (~{gray * 100:.0f}%): có thể là trời âm u HOẶC trần nhà, không phân biệt được"))
    if score >= 2:
        label, tag = "NGOÀI TRỜI", "ok"
    elif score <= -2:
        label, tag = "TRONG NHÀ / thiếu sáng", "warn"
    else:
        label, tag = "KHÔNG XÁC ĐỊNH được", "dim"
    if not reasons:
        reasons.append(("dim", "Không đủ thông số EXIF và dấu hiệu hình ảnh để kết luận"))
    return label, tag, reasons


def analyze_file(path):
    with Image.open(path) as img:
        return analyze_loaded(path, img)


def analyze_loaded(path, img):
    exif = img.getexif()
    main = dict(exif.items())
    sub = exif.get_ifd(EXIF_IFD)
    gps = exif.get_ifd(GPS_IFD)
    orientation = main.get(0x0112, 1)
    prov = analyze_provenance(path, img)
    ai_score, ai_label, ai_tag, ai_reasons = analyze_ai(path, img, main, sub, prov)
    blue, gray = analyze_sky(img, orientation)
    green = analyze_green(img, orientation)
    scene_label, scene_tag, scene_reasons = analyze_scene(sub, gps, blue, gray, green)
    _, coords = read_exif_from(img)
    return {
        "prov": prov, "path": path, "coords": coords, "make": clean_value(main.get(0x010F, "")),
        "model": clean_value(main.get(0x0110, "")), "ai_score": ai_score,
        "ai_label": ai_label, "ai_tag": ai_tag, "ai_reasons": ai_reasons,
        "blue": blue, "gray": gray, "scene_label": scene_label,
        "scene_tag": scene_tag, "scene_reasons": scene_reasons,
    }


def risk_lines(d):
    coords = d["coords"]
    scene = d["scene_label"]
    risks = []
    if coords and scene.startswith("TRONG NHÀ"):
        risks.append(("bad", "GPS + trong nhà: tọa độ nhiều khả năng là nhà ở / cơ quan, rủi ro rất cao"))
    elif coords and scene == "NGOÀI TRỜI":
        risks.append(("warn", "GPS + ngoài trời: lộ địa điểm và lộ trình di chuyển"))
    elif coords:
        risks.append(("bad", "Ảnh có GPS: lộ vị trí chính xác"))
    if coords and d["blue"] >= 0.4:
        risks.append(("warn", "Bầu trời chiếm phần lớn nền: nhìn ảnh không thấy địa điểm nào, nhưng GPS vẫn lộ chính xác (dễ gây chủ quan)"))
    if coords and d["ai_score"] >= 70:
        risks.append(("warn", "Ảnh nghi là AI nhưng có GPS: tọa độ có thể bị làm giả, đừng tin tuyệt đối"))
    if not coords and d["ai_score"] >= 70:
        risks.append(("ok", "Ảnh AI không gắn với địa điểm thật: rủi ro lộ vị trí thấp"))
    if not coords and d["ai_score"] < 70:
        risks.append(("ok", "Không có GPS trong metadata. Vẫn có thể lộ qua nội dung ảnh (biển hiệu, cảnh vật) và thông tin thiết bị"))
    return risks


def format_report(d):
    lines = [("h", f"1. Nguồn gốc ảnh: {d['ai_label']}  (điểm nghi ngờ {d['ai_score']}/100)")]
    lines += [(t, "   • " + x) for t, x in d["ai_reasons"]]
    lines.append(("h", f"2. Bối cảnh chụp: {d['scene_label']}"))
    lines += [(t, "   • " + x) for t, x in d["scene_reasons"]]
    lines += prov_lines(d["prov"])
    lines.append(("h", "4. Hệ quả quyền riêng tư"))
    lines += [(t, "   • " + x) for t, x in risk_lines(d)]
    lines.append(("dim", "Lưu ý: đây là suy đoán bằng heuristic, không phải kết luận chắc chắn."))
    return lines


def report(path, coords=None):
    d = analyze_file(path)
    if coords is not None:
        d["coords"] = coords
    return format_report(d)


CSV_FIELDS = ["file", "ai_score", "ai_label", "scene", "blue_sky_pct", "has_gps",
              "lat", "lon", "make", "model", "origin_verdict", "jpeg_quality"]


def csv_row(d):
    c = d["coords"]
    return {
        "file": os.path.basename(d["path"]), "ai_score": d["ai_score"],
        "ai_label": d["ai_label"], "scene": d["scene_label"],
        "blue_sky_pct": round(d["blue"] * 100, 1), "has_gps": int(bool(c)),
        "lat": round(c[0], 6) if c else "", "lon": round(c[1], 6) if c else "",
        "make": d["make"], "model": d["model"],
        "origin_verdict": d["prov"]["verdict"], "jpeg_quality": d["prov"]["quality"] or "",
    }


BG = "#1e1e2e"
PANEL = "#2a2a3c"
FG = "#e6e6f0"
ACCENT = "#7aa2f7"
RED = "#f7768e"
GREEN = "#9ece6a"


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("EXIF Inspector & Cleaner")
        self.geometry("1100x760")
        self.minsize(900, 620)
        self.configure(bg=BG)
        self.files = []
        self.current = None
        self.scan_id = 0
        self.api_key = ""
        self.pending = 0
        self.sections = {"content": [], "vision": []}
        self.scan_queue = queue.Queue()
        self.base_lines = []
        self.coords = None
        self.thumb = None
        self.setup_style()
        self.build()

    def setup_style(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("Treeview", background=PANEL, fieldbackground=PANEL,
                    foreground=FG, rowheight=24, borderwidth=0)
        s.configure("Treeview.Heading", background=BG, foreground=ACCENT,
                    relief="flat", font=("Segoe UI", 10, "bold"))
        s.map("Treeview", background=[("selected", ACCENT)],
              foreground=[("selected", BG)])
        s.configure("TButton", background=ACCENT, foreground=BG,
                    padding=8, font=("Segoe UI", 10, "bold"), borderwidth=0)
        s.map("TButton", background=[("active", "#9db8ff"), ("disabled", "#444")])
        s.configure("Red.TButton", background=RED)
        s.map("Red.TButton", background=[("active", "#ff9bb0"), ("disabled", "#444")])
        s.configure("Vertical.TScrollbar", background=PANEL, troughcolor=BG)

    def build(self):
        top = tk.Frame(self, bg=BG)
        top.pack(fill="x", padx=12, pady=10)
        ttk.Button(top, text="Chọn ảnh", command=self.pick_files).pack(side="left")
        ttk.Button(top, text="Chọn thư mục", command=self.pick_folder).pack(side="left", padx=8)
        self.btn_clean = ttk.Button(top, text="Xóa metadata ảnh này", style="Red.TButton",
                                    command=self.clean_one, state="disabled")
        self.btn_clean.pack(side="right")
        self.btn_all = ttk.Button(top, text="Xóa tất cả", style="Red.TButton",
                                  command=self.clean_all, state="disabled")
        self.btn_all.pack(side="right", padx=8)
        self.btn_csv = ttk.Button(top, text="Xuất CSV", command=self.export_csv, state="disabled")
        self.btn_csv.pack(side="right")
        self.btn_map = ttk.Button(top, text="Xem vị trí trên bản đồ",
                                  command=self.open_map, state="disabled")
        self.btn_map.pack(side="right")
        self.btn_vision = ttk.Button(top, text="Phân tích bằng AI", command=self.run_vision,
                                     state="disabled")
        self.btn_vision.pack(side="right", padx=8)

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=12)

        left = tk.Frame(body, bg=BG, width=260)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)
        self.listbox = tk.Listbox(left, bg=PANEL, fg=FG, selectbackground=ACCENT,
                                  selectforeground=BG, highlightthickness=0,
                                  borderwidth=0, activestyle="none", font=("Segoe UI", 10))
        self.listbox.pack(fill="both", expand=True)
        self.listbox.bind("<<ListboxSelect>>", self.on_select)
        self.preview = tk.Label(left, bg=PANEL, height=10)
        self.preview.pack(fill="x", pady=(8, 0))

        right = tk.Frame(body, bg=BG)
        right.pack(side="left", fill="both", expand=True, padx=(12, 0))
        self.banner = tk.Label(right, text="Chưa chọn ảnh", bg=PANEL, fg=FG,
                               font=("Segoe UI", 11, "bold"), anchor="w", padx=12, pady=10)
        self.banner.pack(fill="x")
        wrap = tk.Frame(right, bg=BG)
        wrap.pack(fill="both", expand=True, pady=(8, 0))
        self.tree = ttk.Treeview(wrap, columns=("tag", "value"), show="headings")
        self.tree.heading("tag", text="Thẻ")
        self.tree.heading("value", text="Giá trị")
        self.tree.column("tag", width=200, anchor="w")
        self.tree.column("value", width=420, anchor="w")
        self.tree.tag_configure("gps", foreground=RED)
        sb = ttk.Scrollbar(wrap, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        self.analysis = tk.Text(right, bg=PANEL, fg=FG, height=14, wrap="word",
                                borderwidth=0, highlightthickness=0, padx=10, pady=8,
                                font=("Segoe UI", 10), state="disabled")
        self.analysis.pack(fill="x", pady=(8, 0))
        self.analysis.tag_configure("h", foreground=ACCENT, font=("Segoe UI", 10, "bold"),
                                    spacing1=6)
        self.analysis.tag_configure("bad", foreground=RED)
        self.analysis.tag_configure("warn", foreground="#e0af68")
        self.analysis.tag_configure("ok", foreground=GREEN)
        self.analysis.tag_configure("dim", foreground="#8a8aa0")

        self.status = tk.Label(self, text="", bg=BG, fg=GREEN, anchor="w")
        self.status.pack(fill="x", padx=12, pady=8)

    def add_files(self, paths):
        for p in paths:
            if p not in self.files:
                self.files.append(p)
                self.listbox.insert("end", os.path.basename(p))
        if self.files:
            self.btn_all.config(state="normal")
            self.btn_csv.config(state="normal")
            if self.current is None:
                self.listbox.selection_set(0)
                self.on_select()

    def pick_files(self):
        paths = filedialog.askopenfilenames(
            title="Chọn ảnh",
            filetypes=[("Ảnh", " ".join("*" + e for e in EXT)), ("Tất cả", "*.*")])
        self.add_files(paths)

    def pick_folder(self):
        folder = filedialog.askdirectory(title="Chọn thư mục ảnh")
        if folder:
            paths = [os.path.join(folder, f) for f in sorted(os.listdir(folder))
                     if f.lower().endswith(EXT) and "_clean" not in f]
            self.add_files(paths)

    def on_select(self, _event=None):
        sel = self.listbox.curselection()
        if not sel:
            return
        path = self.files[sel[0]]
        self.current = path
        self.tree.delete(*self.tree.get_children())
        self.coords = None
        self.show_report([("dim", "Đang phân tích...")])
        try:
            rows, coords = read_exif(path)
            img = Image.open(path)
            img.thumbnail((240, 160))
            self.thumb = ImageTk.PhotoImage(img)
            self.preview.config(image=self.thumb, height=0)
        except Exception as e:
            self.banner.config(text=f"Không đọc được ảnh: {e}", fg=RED)
            self.btn_clean.config(state="disabled")
            self.btn_map.config(state="disabled")
            return
        for tag, val in rows:
            self.tree.insert("", "end", values=(tag, val),
                             tags=("gps",) if tag.startswith("GPS") else ())
        self.coords = coords
        banner_hint = ""
        try:
            data = analyze_file(path)
            data["coords"] = coords
            banner_hint = data["prov"]["banner"]
            self.base_lines = format_report(data)
            self.sections = {"content": [("h", "5. Nội dung nhìn thấy trong ảnh"),
                                         ("dim", "   Đang quét chữ, mã QR, khuôn mặt...")],
                             "vision": []}
            self.show_report(self.base_lines + self.sections["content"])
            self.start_content_scan(path)
        except Exception as e:
            self.base_lines = []
            self.show_report([("bad", f"Không phân tích được: {e}")])
        self.btn_clean.config(state="normal")
        self.btn_vision.config(state="normal")
        self.btn_map.config(state="normal" if coords else "disabled")
        if coords:
            self.banner.config(
                text=f"CẢNH BÁO: ảnh chứa GPS  ({coords[0]:.6f}, {coords[1]:.6f})", fg=RED)
        elif rows:
            self.banner.config(text=f"Có {len(rows)} thẻ metadata, không có GPS", fg="#e0af68")
        else:
            self.banner.config(text=banner_hint or "Không có EXIF, chưa rõ nguồn gốc file", fg="#e0af68")

    def start_content_scan(self, path):
        self.scan_id += 1
        self.spawn(self.scan_id, "content", lambda: content_lines(path))

    def spawn(self, sid, kind, func):
        q = self.scan_queue
        self.pending += 1

        def work():
            try:
                q.put((sid, kind, func()))
            except Exception as e:
                q.put((sid, kind, [("bad", f"   Không thực hiện được: {e}")]))

        threading.Thread(target=work, daemon=True).start()
        if self.pending == 1:
            self.after(200, self.poll_scan)

    def render(self):
        self.show_report(self.base_lines + self.sections["content"] + self.sections["vision"])

    def poll_scan(self):
        while True:
            try:
                sid, kind, lines = self.scan_queue.get_nowait()
            except queue.Empty:
                break
            self.pending -= 1
            if sid == self.scan_id:
                self.sections[kind] = lines
                self.render()
        if self.pending > 0:
            self.after(200, self.poll_scan)

    def run_vision(self):
        path = self.current
        if not path:
            return
        ok = messagebox.askyesno(
            "Gửi ảnh tới dịch vụ AI?",
            "Chức năng này GỬI ẢNH (đã thu nhỏ) lên máy chủ của Anthropic để mô hình AI mô tả nội dung.\n\n"
            "Chỉ dùng với ảnh bạn đồng ý chia sẻ. Có thể phát sinh chi phí API nhỏ.\n\nTiếp tục?")
        if not ok:
            return
        key = os.environ.get("ANTHROPIC_API_KEY") or self.api_key
        if not key:
            key = simpledialog.askstring(
                "API key", "Dán Anthropic API key (chỉ giữ trong bộ nhớ, không lưu file):",
                show="*", parent=self)
            if not key:
                return
            self.api_key = key.strip()
            key = self.api_key
        self.sections["vision"] = [("h", "6. Phân tích nội dung bằng mô hình AI"),
                                   ("dim", "   Đang gửi ảnh và chờ phản hồi...")]
        self.render()

        def job():
            try:
                return vision_ai.to_lines(vision_ai.describe_image(path, key))
            except Exception as e:
                if "401" in str(e):
                    self.api_key = ""
                return [("h", "6. Phân tích nội dung bằng mô hình AI"),
                        ("bad", f"   Không phân tích được: {e}")]

        self.spawn(self.scan_id, "vision", job)

    def open_map(self):
        if self.coords:
            webbrowser.open(f"https://www.google.com/maps?q={self.coords[0]},{self.coords[1]}")

    def clean_one(self):
        if not self.current:
            return
        try:
            out = strip_exif(self.current)
        except Exception as e:
            messagebox.showerror("Lỗi", str(e))
            return
        self.status.config(text=f"Đã lưu: {out}")
        self.add_files([out])

    def set_busy(self, busy):
        state = "disabled" if busy else "normal"
        self.btn_all.config(state=state)
        self.btn_csv.config(state=state)
        self.btn_clean.config(state="disabled" if busy or not self.current else "normal")

    def run_jobs(self, func, targets, on_ok, on_finish):
        self.set_busy(True)
        self.jobs = queue.Queue()
        self.errors = []
        self.count = 0
        self.total = len(targets)

        def worker():
            for f in targets:
                try:
                    self.jobs.put(("ok", f, func(f)))
                except Exception as e:
                    self.jobs.put(("err", f, str(e)))
            self.jobs.put(("done", None, None))

        threading.Thread(target=worker, daemon=True).start()
        self.poll(on_ok, on_finish)

    def poll(self, on_ok, on_finish):
        finished = False
        while True:
            try:
                kind, src, val = self.jobs.get_nowait()
            except queue.Empty:
                break
            if kind == "ok":
                self.count += 1
                on_ok(src, val)
            elif kind == "err":
                self.errors.append(f"{os.path.basename(src)}: {val}")
            else:
                finished = True
        if finished:
            self.set_busy(False)
            on_finish()
            if self.errors:
                messagebox.showwarning("Một số ảnh bị lỗi", "\n".join(self.errors[:15]))
        else:
            done = self.count + len(self.errors)
            self.status.config(text=f"Đang xử lý {done}/{self.total}...", fg=ACCENT)
            self.after(100, lambda: self.poll(on_ok, on_finish))

    def clean_all(self):
        targets = [f for f in self.files if "_clean" not in os.path.basename(f)]
        if not targets:
            return

        def on_ok(src, out):
            self.add_files([out])

        def on_finish():
            self.status.config(
                text=f"Đã xóa metadata {self.count}/{self.total} ảnh, lỗi {len(self.errors)} (lưu dạng *_clean)",
                fg=GREEN if not self.errors else RED)

        self.run_jobs(strip_exif, targets, on_ok, on_finish)

    def export_csv(self):
        if not self.files:
            return
        dest = filedialog.asksaveasfilename(
            title="Lưu kết quả", defaultextension=".csv",
            filetypes=[("CSV", "*.csv")], initialfile="ket_qua_exif.csv")
        if not dest:
            return
        rows = []

        def on_ok(src, d):
            rows.append(csv_row(d))

        def on_finish():
            with open(dest, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                w.writeheader()
                w.writerows(rows)
            self.status.config(text=f"Đã xuất {len(rows)} dòng: {dest}", fg=GREEN)

        self.run_jobs(analyze_file, list(self.files), on_ok, on_finish)


if __name__ == "__main__":
    App().mainloop()
