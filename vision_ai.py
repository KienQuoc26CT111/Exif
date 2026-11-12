import base64
import io
import json
import os
import urllib.error
import urllib.request

from PIL import Image, ImageOps

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"
MODEL = os.environ.get("EXIF_TOOL_MODEL", "claude-sonnet-5-5")
MAX_SIDE = 1568

PROMPT = (
    "Bạn là chuyên gia phân tích quyền riêng tư của ảnh. Ảnh này không còn dữ liệu EXIF. "
    "Hãy chỉ ra những gì NHÌN THẤY trong ảnh có thể làm lộ thông tin cá nhân. "
    "Trả lời bằng tiếng Việt, ngắn gọn, đúng 6 mục sau, mỗi mục bắt đầu bằng số thứ tự:\n"
    "1. Địa điểm: loại nơi chốn, dấu hiệu nhận biết (biển hiệu, kiến trúc, cây cối, vật dụng, khu vực/quốc gia) "
    "và mức chắc chắn (thấp/vừa/cao). Không đoán bừa.\n"
    "2. Thời điểm: mùa, buổi trong ngày, ngày tháng hay năm đọc được trong ảnh.\n"
    "3. Con người: số lượng, nhóm tuổi ước lượng, đồng phục/logo trường hay công ty. "
    "TUYỆT ĐỐI không nhận dạng danh tính ai.\n"
    "4. Chữ và số đọc được: chép nguyên văn những dòng quan trọng.\n"
    "5. Vật nhạy cảm: biển số xe, giấy tờ, màn hình, mã QR, địa chỉ, số điện thoại, tên riêng.\n"
    "6. Mức rủi ro quyền riêng tư: Thấp, Trung bình hoặc Cao, kèm 1-2 câu khuyến nghị cụ thể "
    "(nên che/cắt phần nào trước khi đăng)."
)


def prepare_image(path):
    with Image.open(path) as raw:
        img = ImageOps.exif_transpose(raw)
        img = img.convert("RGB")
        img.thumbnail((MAX_SIDE, MAX_SIDE))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def build_request(path, api_key, model=MODEL):
    body = {
        "model": model,
        "max_tokens": 1200,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                             "data": prepare_image(path)}},
                {"type": "text", "text": PROMPT},
            ],
        }],
    }
    headers = {
        "content-type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": API_VERSION,
    }
    return urllib.request.Request(API_URL, data=json.dumps(body).encode("utf-8"),
                                  headers=headers, method="POST")


def parse_response(raw):
    data = json.loads(raw)
    parts = [b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"]
    text = "\n".join(p for p in parts if p).strip()
    if not text:
        raise ValueError("phản hồi không có nội dung văn bản")
    return text


def describe_image(path, api_key, opener=urllib.request.urlopen, timeout=90):
    if not api_key:
        raise ValueError("chưa có API key")
    req = build_request(path, api_key)
    try:
        with opener(req, timeout=timeout) as resp:
            return parse_response(resp.read())
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = json.loads(e.read()).get("error", {}).get("message", "")
        except (ValueError, OSError):
            pass
        if e.code == 401:
            raise RuntimeError("API key không hợp lệ (401)") from e
        if e.code == 429:
            raise RuntimeError("vượt giới hạn gọi API hoặc hết hạn mức (429)") from e
        raise RuntimeError(f"lỗi API {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"không kết nối được tới API: {e.reason}") from e


def to_lines(text):
    lines = [("h", "6. Phân tích nội dung bằng mô hình AI (ảnh đã được gửi tới Anthropic API)")]
    for raw in text.splitlines():
        raw = raw.rstrip()
        if not raw:
            continue
        head = raw.lstrip()[:2]
        tag = "ok"
        low = raw.lower()
        if head[:1].isdigit():
            tag = "warn" if head[:1] != "6" else ("bad" if "cao" in low else "warn")
        lines.append((tag if raw.lstrip()[:1].isdigit() else "dim", "   " + raw))
    return lines
