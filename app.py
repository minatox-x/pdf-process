"""
Watermark remover + transparent proxy service
(PyMuPDF + OpenCV + numpy + RapidOCR + httpx + Starlette).

POST /process
  Body: the "data" string, OR the full JSON {"success":true,"data":"..."}
        (raw text, a JSON string, or JSON object all work)
  Query params: ignored. Page 1 is always processed.
  Returns: PNG of page 1 with the watermarks "StudyRays.cc", "AllApps.cfd" and
           "DevJisu" removed and the area behind them reconstructed.

  Response headers
    X-Total-Pages        page count of the PDF
    X-Watermarks-Removed number of watermark instances removed (vector + raster)
    X-Removal-Method     comma list of: vector, raster-signature, raster-ocr, none
    X-Render-Scale       scale the page was rendered at

How the removal works (best method first)
  1. VECTOR  - The three strings are located in the PDF's own text layer and
               those glyphs (and only those) are deleted from the page content.
               Nothing is painted or guessed: whatever lies behind the
               watermark (gradient, photo, pie chart ...) renders natively and
               pixel-perfectly. Real text overlapping the watermark is kept.
  2. RASTER  - For watermarks burned into an image, the watermark layer is
               separated by its colour signature, grouped into text blobs and
               each blob is VERIFIED by OCR to really read as one of the target
               strings before anything is touched. Only those pixels are then
               reconstructed (inpainting + grain matching); every other pixel
               of the page is byte-identical to the source render.
  3. OCR     - Fallback: full-page OCR locates the strings directly when the
               colour model does not match.

GET|POST|* /proxy?url=<base64url-encoded-url>   (unchanged)
"""
import asyncio
import base64
import json
import logging
import math
import os
import re
import threading
import urllib.parse
from difflib import SequenceMatcher

import cv2
import httpx
import numpy as np

try:  # new import name first, old one as fallback
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz  # type: ignore

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import ClientDisconnect
from starlette.responses import JSONResponse, PlainTextResponse, Response
from starlette.routing import Route

log = logging.getLogger("wm")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
KEY = (90, 165, 195, 60, 15, 240, 150, 105)
XOR_MASK = np.array([KEY[i % 8] ^ (i % 256) for i in range(256)], dtype=np.uint8)

MAX_BODY = int(float(os.getenv("MAX_BODY_MB", "1")) * 1024 * 1024)
MAX_PIXELS = int(os.getenv("MAX_PIXELS", str(25_000_000)))
MAX_JOBS = int(os.getenv("MAX_JOBS", "2"))
MAX_QUEUE = int(os.getenv("MAX_QUEUE", "50"))
QUEUE_TIMEOUT = float(os.getenv("QUEUE_TIMEOUT", "30"))

RENDER_SCALE = float(os.getenv("RENDER_SCALE", "2.0"))      # used when the page has no big image
MAX_SCALE = float(os.getenv("MAX_SCALE", "4.0"))
MIN_SCALE = float(os.getenv("MIN_SCALE", "1.0"))

# watermark colour prior(s) for the raster stage, e.g. "220,220,220;200,200,200"
WM_COLORS = [tuple(int(v) for v in c.split(",")) for c in
             os.getenv("WM_COLORS", "220,220,220").split(";") if c.strip()]
OCR_FALLBACK = os.getenv("OCR_FALLBACK", "1") != "0"

# The strings to remove. Matching is case-insensitive and ignores dots/spaces.
VECTOR_RE = re.compile(r"studyrays(?:cc)?|allapps(?:cfd)?|devjisu")
# OCR-side targets (confusable characters folded: 0->o, 1/i/|->l, 5->s)
_CONF = str.maketrans({"0": "o", "1": "l", "i": "l", "|": "l", "5": "s"})


def _fold(s: str) -> str:
    return re.sub(r"[^a-z0-9|]", "", s.lower()).translate(_CONF)


OCR_TARGETS = [_fold(t) for t in ("StudyRays.cc", "AllApps.cfd", "DevJisu", "StudyRays", "AllApps")]

_sem = asyncio.Semaphore(MAX_JOBS)
_pending = 0


def busy():
    return JSONResponse({"error": "Server busy, please try again later"},
                        status_code=503, headers={"Retry-After": "3"})


class BadRequest(Exception):
    pass


class Cancelled(Exception):
    """Client went away; abort work as soon as possible."""


class QueueTimeout(Exception):
    pass


# ---------------------------------------------------------------------------
# Input decoding (unchanged)
# ---------------------------------------------------------------------------
def extract_b64(body: bytes) -> str:
    s = body.strip()
    if not s:
        raise BadRequest("Empty body")
    if s[:1] in (b"{", b'"'):
        try:
            obj = json.loads(s)
        except ValueError:
            raise BadRequest("Invalid JSON")
        if isinstance(obj, dict):
            if obj.get("success") is False:
                raise BadRequest("Upstream response reported failure")
            obj = obj.get("data")
        if not isinstance(obj, str):
            raise BadRequest('No "data" string found')
        s = obj.encode()
    s = s.decode("ascii", "ignore")
    if s.startswith("data:"):
        s = s.split(",", 1)[-1]
    s = "".join(s.split())
    s += "=" * (-len(s) % 4)
    return s


def decode_pdf(b64: str) -> bytes:
    try:
        raw = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
    except Exception:
        raise BadRequest("Invalid base64")
    return (raw ^ np.resize(XOR_MASK, raw.size)).tobytes()


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def pick_scale(page) -> float:
    """Render at the native pixel density of the page's dominant image so no
    detail is lost or resampled; vector-only pages use RENDER_SCALE."""
    best = 0.0
    try:
        area = page.rect.get_area()
        for info in page.get_image_info():
            r = fitz.Rect(info["bbox"])
            if r.is_empty or r.get_area() < 0.2 * area:
                continue
            best = max(best, info["width"] / max(r.width, 1e-3), info["height"] / max(r.height, 1e-3))
    except Exception:
        best = 0.0
    scale = best if best > 0 else RENDER_SCALE
    scale = max(MIN_SCALE, min(MAX_SCALE, scale))
    r = page.rect
    px = r.width * r.height * scale * scale
    if px > MAX_PIXELS:
        scale = math.sqrt(MAX_PIXELS / (r.width * r.height))
    return scale


def render_rgb(page, scale: float) -> np.ndarray:
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=fitz.csRGB, alpha=False)
    h, w = pix.height, pix.width
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(h, pix.stride)[:, :w * 3]
    return arr.reshape(h, w, 3).copy()


# ---------------------------------------------------------------------------
# STAGE 1 - vector removal (lossless)
# ---------------------------------------------------------------------------
def _trace_glyphs(page):
    """All glyphs in drawing order. Consecutive spans with identical style are
    one group, so a watermark drawn char-by-char still reads as one string."""
    glyphs, gid, prev = [], -1, None
    for span in page.get_texttrace():
        key = (span["font"], round(span["size"], 2),
               tuple(round(c, 3) for c in span["color"]),
               tuple(round(x, 3) for x in span["dir"]))
        if key != prev:
            gid += 1
            prev = key
        chars = span["chars"]
        dx, dy = span["dir"]
        advs = []
        for i, ch in enumerate(chars):
            if i + 1 < len(chars):
                nx, ny = chars[i + 1][2]
                ox, oy = ch[2]
                advs.append(max((nx - ox) * dx + (ny - oy) * dy, 0.0))
            else:
                advs.append(advs[-1] if advs else 0.6 * span["size"])
        for ch, adv in zip(chars, advs):
            uni, _gl, origin, bbox = ch
            glyphs.append({
                "c": chr(uni) if uni and uni > 0 else "",
                "origin": (float(origin[0]), float(origin[1])),
                "bbox": tuple(float(v) for v in bbox),
                "dir": (float(dx), float(dy)),
                "size": float(span["size"]),
                "adv": adv if adv > 0 else 0.5 * span["size"],
                "gid": gid,
                "color": tuple(int(round(c * 255)) for c in (span["color"] if len(span["color"]) == 3
                                                             else (span["color"][0],) * 3)),
            })
    return glyphs


def _find_wm_glyphs(glyphs):
    """Indices of glyphs forming one of the target strings."""
    hit = set()
    n_matches = 0
    by_group = {}
    for i, g in enumerate(glyphs):
        by_group.setdefault(g["gid"], []).append(i)
    for idxs in by_group.values():
        norm, back = [], []
        for i in idxs:
            ch = glyphs[i]["c"].lower()
            if ch.isalnum():
                norm.append(ch)
                back.append(i)
        text = "".join(norm)
        for m in VECTOR_RE.finditer(text):
            hit.update(back[m.start():m.end()])
            n_matches += 1
    # include the dots belonging to a matched run, e.g. the '.' in "StudyRays.cc"
    extra = set()
    for gidx in {glyphs[i]["gid"] for i in hit}:
        idxs = by_group[gidx]
        pos = {i: k for k, i in enumerate(idxs)}
        for i in list(hit):
            if glyphs[i]["gid"] != gidx:
                continue
            k = pos[i]
            for j in (k - 1, k + 1):
                if 0 <= j < len(idxs) and glyphs[idxs[j]]["c"] in (".", " ", "\u00a0"):
                    # a '.' between two matched alnum glyphs
                    other = idxs[k + (k - j)] if 0 <= k + (k - j) < len(idxs) else None
                    if other in hit:
                        extra.add(idxs[j])
    return hit | extra, n_matches


def _probe_grid(g):
    """Candidate probe points inside a watermark glyph, best (most central) first."""
    ox, oy = g["origin"]
    dx, dy = g["dir"]
    ux, uy = dy, -dx  # "up" in y-down page space
    adv, size = g["adv"], g["size"]
    pts = []
    for v in np.linspace(0.05, 0.75, 8):
        for u in np.linspace(0.12, 0.88, 7):
            score = (u - 0.5) ** 2 + (v - 0.35) ** 2
            pts.append((score, (ox + dx * adv * u + ux * size * v, oy + dy * adv * u + uy * size * v)))
    pts.sort(key=lambda t: t[0])
    return [p for _, p in pts]


def _apply_redactions(page):
    try:
        page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE,
                              graphics=fitz.PDF_REDACT_LINE_ART_NONE,
                              text=fitz.PDF_REDACT_TEXT_REMOVE)
    except (TypeError, AttributeError):
        page.apply_redactions(images=0)


def _remove_annotations(page) -> int:
    n = 0
    try:
        for a in list(page.annots() or []):
            blob = " ".join(str(v) for v in a.info.values())
            if VECTOR_RE.search(re.sub(r"[^a-z0-9]", "", blob.lower())):
                page.delete_annot(a)
                n += 1
    except Exception:
        pass
    return n


def _blocked(px, py, others, learned):
    """True if (px, py) is in/near a real glyph or in an area learned to be unsafe."""
    for (a, b, c, d) in learned:
        if a <= px <= c and b <= py <= d:
            return True
    if others.size == 0:
        return False
    sz = others[:, 4]
    mx = 0.05 * sz + 0.3
    my = 0.12 * sz + 0.3
    return bool(np.any((others[:, 0] - mx <= px) & (px <= others[:, 2] + mx) &
                       (others[:, 1] - my <= py) & (py <= others[:, 3] + my)))


def _glyph_key(g):
    return (g["c"], round(g["origin"][0], 1), round(g["origin"][1], 1))


def _trial(pdf_bytes, probes):
    """Apply probe redactions on a scratch copy; return its glyph list."""
    d = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        p = d[0]
        for (px, py) in probes:
            p.add_redact_annot(fitz.Rect(px - 0.25, py - 0.25, px + 0.25, py + 0.25),
                               fill=None, cross_out=False)
        _apply_redactions(p)
        return _trace_glyphs(p)
    finally:
        d.close()


def vector_stage(pdf_bytes, page, check):
    """Delete watermark glyphs from the page content, and nothing else.

    A probe point is placed inside every watermark glyph where no real glyph
    is. The plan is rehearsed on a scratch copy of the PDF and the text layer
    is read back; if a real glyph was lost, the area is learned as unsafe and
    the plan is revised. Only a verified plan is applied to the real page.

    Returns (instances_removed, collateral_boxes, watermark_colours)."""
    removed = _remove_annotations(page)
    before = _trace_glyphs(page)
    wm, n_inst = _find_wm_glyphs(before)
    if not wm:
        return removed, [], []
    wm_colors = sorted({before[i]["color"] for i in wm})
    wm_keys = {i: _glyph_key(before[i]) for i in wm}
    real = [(i, g) for i, g in enumerate(before) if i not in wm]
    real_keys = {_glyph_key(g): g for _, g in real}
    others = np.array([g["bbox"] + (g["size"],) for _, g in real], np.float32).reshape(-1, 5)
    grids = {i: _probe_grid(before[i]) for i in wm}
    ptr = {i: 0 for i in wm}          # next candidate to try, per watermark glyph
    learned = []
    plan = {}

    for it in range(10):
        check()
        plan = {}
        for i in wm:
            g = grids[i]
            while ptr[i] < len(g) and _blocked(g[ptr[i]][0], g[ptr[i]][1], others, learned):
                ptr[i] += 1
            if ptr[i] < len(g):
                plan[i] = g[ptr[i]]
        if not plan:
            break
        after = _trial(pdf_bytes, list(plan.values()))
        akeys = {_glyph_key(g) for g in after}
        lost = [g for k, g in real_keys.items() if k not in akeys]
        survivors = [i for i in wm if wm_keys[i] in akeys]
        if lost:
            for g in lost:
                e = 0.6 * g["size"] + 1.0
                x0, y0, x1, y1 = g["bbox"]
                near = [p for p in plan.values()
                        if x0 - e <= p[0] <= x1 + e and y0 - e <= p[1] <= y1 + e]
                xs = [x0, x1] + [p[0] for p in near]
                ys = [y0, y1] + [p[1] for p in near]
                learned.append((min(xs) - 0.3, min(ys) - 0.3, max(xs) + 0.3, max(ys) + 0.3))
            # re-plan the glyphs whose probes sat near a lost real glyph
            for i, p in list(plan.items()):
                if _blocked(p[0], p[1], others, learned):
                    ptr[i] += 1
            continue
        if not survivors:
            break
        for i in survivors:  # probe missed the glyph: try the next point
            if i in plan:
                ptr[i] += 1
        if all(ptr[i] >= len(grids[i]) for i in survivors):
            break

    for p in plan.values():
        page.add_redact_annot(fitz.Rect(p[0] - 0.25, p[1] - 0.25, p[0] + 0.25, p[1] + 0.25),
                              fill=None, cross_out=False)
    if plan:
        _apply_redactions(page)

    after = _trace_glyphs(page)
    akeys = {_glyph_key(g) for g in after}
    left, _ = _find_wm_glyphs(after)
    collateral = [g["bbox"] for k, g in real_keys.items() if k not in akeys]
    if len(left) < len(wm):
        removed += max(1, n_inst - (1 if left else 0))
    log.debug("vector: %d wm glyphs, %d left, %d collateral", len(wm), len(left), len(collateral))
    return removed, collateral, wm_colors


# ---------------------------------------------------------------------------
# STAGE 2 - raster: signature layer -> blobs -> OCR verification -> inpaint
# ---------------------------------------------------------------------------
_ocr_obj = None
_ocr_lock = threading.Lock()


def _get_ocr():
    global _ocr_obj
    if _ocr_obj is None:
        from rapidocr_onnxruntime import RapidOCR
        _ocr_obj = RapidOCR()
    return _ocr_obj


def _ell(k: int):
    k = max(1, int(k)) | 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _bg_median(rgb: np.ndarray, ksize_px: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    f = 2 if min(h, w) > 600 else 1
    small = rgb if f == 1 else cv2.resize(rgb, (w // f, h // f), interpolation=cv2.INTER_AREA)
    bg = cv2.medianBlur(small, max(3, int(ksize_px / f)) | 1)
    return bg if f == 1 else cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)


def _signature(rgb, bg, c, check, thr=5.0, roi=None):
    """Pixels that look like a blend of the background towards colour `c`
    (colour-model of an opaque or semi-transparent overlay) -> (sig, real_ink).
    `real_ink` = strongly different from the background but NOT explainable by
    the overlay colour (i.e. genuine foreground such as black text)."""
    H, W = rgb.shape[:2]
    sig = np.zeros((H, W), bool)
    real = np.zeros((H, W), bool)
    cvec = np.array(c, np.float32)
    for y in range(0, H, 192):
        check()
        if roi is not None and not roi[y:y + 192].any():
            continue
        f = rgb[y:y + 192].astype(np.float32)
        b = bg[y:y + 192].astype(np.float32)
        d = f - b
        u = cvec - b
        un = np.sqrt((u * u).sum(2))
        dn = np.sqrt((d * d).sum(2))
        dot = (d * u).sum(2)
        cos = dot / np.maximum(un * dn, 1e-3)
        k = dot / np.maximum(un * un, 1e-3)
        strict = (dn > thr) & (un > 12) & (cos > 0.88) & (k > 0.08) & (k < 1.3)
        loose = (dn > thr) & (un > 12) & (cos > 0.75) & (k > 0.03) & (k < 1.7)
        rl = (dn > 28) & ~loose
        if roi is not None:
            r = roi[y:y + 192]
            strict &= r
            rl &= r
        sig[y:y + 192] = strict
        real[y:y + 192] = rl
    return sig, real


def _thick(cand: np.ndarray):
    """Drop thin structures (anti-aliasing fringes of real text) and keep
    stroke-thick shapes. Returns (ink, stroke_halfwidth) or (None, 0)."""
    u8 = cand.astype(np.uint8)
    if u8.sum() < 200:
        return None, 0.0
    dist = cv2.distanceTransform(u8, cv2.DIST_L2, 3)
    nz = dist[dist > 0]
    s = float(np.percentile(nz, 98))
    rho = max(1, int(round(0.4 * s)))
    ker = _ell(2 * rho + 1)
    core = cv2.morphologyEx(u8, cv2.MORPH_OPEN, ker)
    ink = (cv2.dilate(core, ker) > 0) & cand
    return ink, s


def _blobs(ink: np.ndarray):
    """Group ink pixels into text-line blobs -> list of (x, y, mask)."""
    u8 = ink.astype(np.uint8)
    n, _lab, st, _c = cv2.connectedComponentsWithStats(u8, connectivity=8)
    if n <= 1:
        return []
    areas = st[1:, cv2.CC_STAT_AREA]
    big = st[1:][areas >= max(40, 0.15 * np.percentile(areas, 90))]
    if len(big) == 0:
        return []
    g = float(np.median(np.maximum(big[:, cv2.CC_STAT_WIDTH], big[:, cv2.CC_STAT_HEIGHT])))
    r = int(np.clip(0.5 * g, 4, 90))
    merged = cv2.dilate(u8, _ell(2 * r + 1))
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
    out = []
    for i in range(1, n2):
        x, y, w, h, _a = st2[i]
        sub = ink[y:y + h, x:x + w] & (lab2[y:y + h, x:x + w] == i)
        if sub.sum() < 80:
            continue
        out.append((int(x), int(y), sub))
    return out


def _rectified_crop(sub: np.ndarray, flip: bool):
    """Render the blob's ink as black-on-white, rotated so its main axis is horizontal."""
    ys, xs = np.nonzero(sub)
    if len(xs) < 50:
        return None
    pts = np.column_stack([xs, ys]).astype(np.float32)
    mu = pts.mean(0)
    cov = np.cov((pts - mu).T)
    ev, vec = np.linalg.eigh(cov)
    ax = vec[:, 1]
    ang = math.atan2(ax[1], ax[0]) + (math.pi if flip else 0.0)
    ca, sa = math.cos(-ang), math.sin(-ang)
    R = np.array([[ca, -sa], [sa, ca]], np.float32)
    q = (pts - mu) @ R.T
    x0, y0 = q.min(0)
    x1, y1 = q.max(0)
    pad = 10
    W = int(x1 - x0) + 2 * pad + 1
    H = int(y1 - y0) + 2 * pad + 1
    if W > 6000 or H > 3000 or H < 6:
        return None
    M = np.zeros((2, 3), np.float32)
    M[:, :2] = R
    M[:, 2] = -(mu @ R.T) - np.array([x0, y0]) + pad
    layer = cv2.warpAffine(sub.astype(np.uint8) * 255, M, (W, H), flags=cv2.INTER_LINEAR)
    img = 255 - layer
    # normalise height for the recogniser
    th = 64
    img = cv2.resize(img, (max(8, int(W * th / H)), th), interpolation=cv2.INTER_AREA)
    img = cv2.copyMakeBorder(img, 4, 4, 8, 8, cv2.BORDER_CONSTANT, value=255)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def _fuzzy_target(text: str):
    """(score, coverage) of the best fuzzy match of `text` against the targets."""
    t = _fold(text)
    if len(t) < 5:
        return 0.0, 0.0
    best, cov = 0.0, 0.0
    for tgt in OCR_TARGETS:
        L = len(tgt)
        for wl in (L - 1, L, L + 1):
            if wl < 5 or wl > len(t):
                continue
            for s in range(0, len(t) - wl + 1):
                r = SequenceMatcher(None, t[s:s + wl], tgt).ratio()
                if r > best:
                    best, cov = r, wl / len(t)
    return best, cov


def _is_target_blob(sub: np.ndarray) -> bool:
    ys, xs = np.nonzero(sub)
    if len(xs) == 0:
        return False
    # a text line is elongated; reject round/square junk early
    h, w = sub.shape
    if max(h, w) < 2.0 * min(h, w) and sub.sum() < 0.5 * h * w * 0.3:
        pass
    ocr = _get_ocr()
    for flip in (False, True):
        crop = _rectified_crop(sub, flip)
        if crop is None:
            continue
        with _ocr_lock:
            res, _ = ocr(crop, use_det=False, use_cls=False, use_rec=True)
        if not res:
            continue
        txt = res[0][0]
        score, cov = _fuzzy_target(txt)
        log.debug("blob ocr %r score=%.2f cov=%.2f", txt, score, cov)
        if score >= 0.82 and cov >= 0.55:
            return True
    return False


def _color_modes(rgb, bg, check, max_modes=2):
    """Extra colour hypotheses: dominant light/mid colours of thin structures."""
    dev = np.abs(rgb.astype(np.int16) - bg.astype(np.int16)).max(2)
    sel = (dev > 10)
    px = rgb[sel]
    if len(px) < 500:
        return []
    mx = px.max(1)
    mn = px.min(1)
    px = px[(mn > 55) & (mx < 247)]
    if len(px) < 300:
        return []
    q = (px // 12).astype(np.int32)
    key = q[:, 0] * 4096 + q[:, 1] * 64 + q[:, 2]
    vals, counts = np.unique(key, return_counts=True)
    order = np.argsort(-counts)[:max_modes * 3]
    modes = []
    for o in order:
        sel_px = px[key == vals[o]]
        modes.append(tuple(int(v) for v in sel_px.mean(0)))
    return modes[:max_modes]


def raster_signature_stage(rgb, check):
    """-> (accepted_ink | None, colour, stroke_halfwidth, n_blobs)"""
    H, W = rgb.shape[:2]
    bg = _bg_median(rgb, max(41, int(0.05 * max(H, W))))
    hyps = list(WM_COLORS)
    for m in _color_modes(rgb, bg, check):
        if all(max(abs(m[i] - h[i]) for i in range(3)) > 14 for h in hyps):
            hyps.append(m)
    accepted = np.zeros((H, W), bool)
    s_used, n_ok, c_used = 0.0, 0, None
    for c in hyps:
        check()
        sig, _real = _signature(rgb, bg, c, check)
        ink, s = _thick(sig)
        if ink is None:
            continue
        for (x, y, sub) in _blobs(ink):
            check()
            if _is_target_blob(sub):
                accepted[y:y + sub.shape[0], x:x + sub.shape[1]] |= sub
                n_ok += 1
                s_used = max(s_used, s)
        if n_ok:
            c_used = c
            break  # first hypothesis that verifies is the watermark colour
    if not n_ok:
        return None, None, 0.0, 0
    return accepted, c_used, s_used, n_ok


# ---- OCR-anchored fallback --------------------------------------------------
def _boost(rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    bg = cv2.medianBlur(g, 31)
    dev = np.maximum(bg.astype(np.int16) - g, g.astype(np.int16) - bg).clip(0, 255).astype(np.uint8)
    k = 255.0 / max(25.0, float(np.percentile(dev, 99.7)))
    out = 255 - np.clip(dev * k, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)


def raster_ocr_stage(rgb, covered, check):
    """Locate the strings with full-page OCR; segment glyphs inside each box."""
    H, W = rgb.shape[:2]
    ocr = _get_ocr()
    boxes = []
    TILE, OV = 1600, 160
    ys = list(range(0, max(1, H - OV), TILE - OV)) or [0]
    xs = list(range(0, max(1, W - OV), TILE - OV)) or [0]
    for variant in (rgb, _boost(rgb)):
        for y0 in ys:
            for x0 in xs:
                check()
                tile = variant[y0:y0 + TILE, x0:x0 + TILE]
                if tile.shape[0] < 24 or tile.shape[1] < 24:
                    continue
                with _ocr_lock:
                    res, _ = ocr(cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))
                for box, txt, sc in (res or []):
                    s, cov = _fuzzy_target(txt)
                    if s >= 0.82 and cov >= 0.5:
                        poly = np.array(box, np.float32) + np.array([x0, y0], np.float32)
                        boxes.append(poly)
    if not boxes:
        return None, 0
    mask = np.zeros((H, W), bool)
    n = 0
    bg = _bg_median(rgb, max(41, int(0.05 * max(H, W))))
    for poly in boxes:
        roi = np.zeros((H, W), np.uint8)
        cv2.fillPoly(roi, [poly.astype(np.int32)], 1)
        hh = float(np.linalg.norm(poly[1] - poly[2]))
        roi = cv2.dilate(roi, _ell(int(0.25 * hh) * 2 + 1))
        if (covered & (roi > 0)).sum() > 0.5 * (roi > 0).sum() * 0.15:
            continue
        x, y, w, h = cv2.boundingRect(roi)
        f = rgb[y:y + h, x:x + w].astype(np.float32)
        b = bg[y:y + h, x:x + w].astype(np.float32)
        dev = np.abs(f - b).max(2)
        r = roi[y:y + h, x:x + w] > 0
        sigma = 1.4826 * float(np.median(np.abs(dev[r] - np.median(dev[r]))))
        thr = max(5.0, np.median(dev[r]) + 3.0 * sigma)
        cand = (dev > thr) & r
        ink, _s = _thick(cand)
        if ink is None:
            continue
        mask[y:y + h, x:x + w] |= ink
        n += 1
    return (mask if n else None), n


# ---- reconstruction ----------------------------------------------------------
def build_mask(accepted, info, s):
    """Mask for the OCR-located fallback (no colour model): grow + nothing protected."""
    rho = max(1, int(round(0.4 * s)))
    gk = max(3, int(round(0.35 * s)))
    core = cv2.morphologyEx(accepted.astype(np.uint8), cv2.MORPH_CLOSE, _ell(2 * rho + 1)) > 0
    grow = cv2.dilate(core.astype(np.uint8), _ell(2 * gk + 1)) > 0
    return grow, np.zeros_like(grow)


def _grow_by_color(rgb, core, radius, tol, rho):
    """Add pixels within `radius` of the core whose colour matches the colour of
    the nearest SOLID interior of the core (catches strokes the background model
    missed). Interior only: the core's anti-aliased rim is too close to the
    background colour and would let the region leak."""
    interior = cv2.erode(core.astype(np.uint8), _ell(2 * rho + 1)) > 0
    ys, xs = np.nonzero(interior)
    if len(ys) == 0:
        return core
    src = np.where(interior, 0, 255).astype(np.uint8)
    dist, lab = cv2.distanceTransformWithLabels(src, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
    cols = np.zeros((len(ys) + 1, 3), np.int16)
    cols[1:] = rgb[ys, xs]            # labels are 1..N in raster order of the zero pixels
    near = cols[np.clip(lab, 0, len(ys))]
    diff = np.abs(rgb.astype(np.int16) - near).max(2)
    grown = (dist <= radius) & (diff <= tol)
    n, cc = cv2.connectedComponents((grown | core).astype(np.uint8), connectivity=8)
    keep = np.unique(cc[core])
    keep = keep[keep > 0]
    return (np.isin(cc, keep) | core)


def refine_mask(rgb, accepted, c, s, check):
    """Refinement with a much better background model.

    The verified glyph core is grown generously and inpainted: the result is a
    watermark-free background estimate that is accurate even at slice/edge
    boundaries (it is built from outside the hole). The colour-signature test
    is then repeated against it, which recovers the anti-aliased fringe and
    separates genuine foreground (black/white text, chart lines) from the mark.
    Runs twice so that the estimate improves where the first one was poor."""
    rho = max(1, int(round(0.4 * s)))
    gk = max(3, int(round(0.35 * s)))
    acc = accepted.copy()
    thin2 = real2 = None
    for _ in range(2):
        check()
        core = cv2.morphologyEx(acc.astype(np.uint8), cv2.MORPH_CLOSE, _ell(2 * rho + 1)) > 0
        core = _grow_by_color(rgb, core, 4 * gk, 10, rho)
        hole1 = cv2.dilate(core.astype(np.uint8), _ell(2 * max(4, int(round(0.7 * s))) + 1)) > 0
        bg2 = inpaint_regions(rgb, hole1, np.zeros_like(hole1), check, grain=False)
        roi = cv2.dilate(hole1.astype(np.uint8), _ell(9)) > 0
        sig2, real2 = _signature(rgb, bg2, c, check, thr=4.0, roi=roi)
        ink2, _ = _thick(sig2)
        cand = core | (ink2 if ink2 is not None else False)
        n, lab = cv2.connectedComponents(cand.astype(np.uint8), connectivity=8)
        keep = np.unique(lab[core])
        keep = keep[keep > 0]
        acc = (np.isin(lab, keep) if len(keep) else core) | core
        thin2 = sig2 & ~acc
    core2 = cv2.morphologyEx(acc.astype(np.uint8), cv2.MORPH_CLOSE, _ell(2 * rho + 1)) > 0
    grow = cv2.dilate(core2.astype(np.uint8), _ell(2 * gk + 1)) > 0
    near = cv2.dilate(acc.astype(np.uint8), _ell(2 * (gk + 1) + 1)) > 0
    protect = (real2 & ~acc) | (thin2 & ~near)
    return grow & ~protect, protect


def inpaint_regions(rgb, mask, protect, check, grain=True):
    """Rebuild masked pixels from the surrounding *background* (real ink is not
    used as a source) and match the local grain. Everything outside `mask` is
    returned byte-identical."""
    out = rgb.copy()
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    H, W = mask.shape
    rng = np.random.default_rng(1234)
    for i in range(1, n):
        check()
        x, y, w, h, a = st[i]
        mg = 24 + int(0.3 * max(w, h) ** 0.5)
        x0, y0, x1, y1 = max(0, x - mg), max(0, y - mg), min(W, x + w + mg), min(H, y + h + mg)
        crop = rgb[y0:y1, x0:x1]
        mm = mask[y0:y1, x0:x1]
        pr = protect[y0:y1, x0:x1]
        hole = (mm | (cv2.dilate(pr.astype(np.uint8), _ell(3)) > 0)).astype(np.uint8) * 255
        fixed = cv2.inpaint(crop, hole, 3, cv2.INPAINT_TELEA)
        # grain: copy the local high-frequency noise level of the surrounding background
        ring = (cv2.dilate(hole, _ell(25)) > 0) & ~(cv2.dilate(hole, _ell(7)) > 0)
        if grain and ring.sum() > 300:
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32)
            hp = gray - cv2.GaussianBlur(gray, (0, 0), 1.5)
            v = hp[ring]
            sig = 1.4826 * float(np.median(np.abs(v - np.median(v))))
            if 0.7 < sig < 4.0:
                noise = rng.normal(0.0, sig * 0.8, size=crop.shape[:2]).astype(np.float32)
                fixed = np.clip(fixed.astype(np.float32) + noise[..., None], 0, 255).astype(np.uint8)
        region = out[y0:y1, x0:x1]
        sel = mm & ~pr
        region[sel] = fixed[sel]
    return out


def _restore_collateral(rgb, orig, boxes, scale, tl, wm_colors, check):
    H, W = rgb.shape[:2]
    for (x0, y0, x1, y1) in boxes:
        check()
        X0 = max(0, int((x0 - tl.x) * scale) - 3)
        Y0 = max(0, int((y0 - tl.y) * scale) - 3)
        X1 = min(W, int((x1 - tl.x) * scale) + 4)
        Y1 = min(H, int((y1 - tl.y) * scale) + 4)
        if X1 <= X0 or Y1 <= Y0:
            continue
        a = orig[Y0:Y1, X0:X1]
        b = rgb[Y0:Y1, X0:X1]
        wmpix = np.zeros(a.shape[:2], bool)
        for c in wm_colors:
            sg, _ = _signature(a, b, c, lambda: None, thr=3.0)
            wmpix |= sg
        wmpix = cv2.dilate(wmpix.astype(np.uint8), _ell(3)) > 0
        restored = a.copy()
        restored[wmpix] = b[wmpix]
        rgb[Y0:Y1, X0:X1] = restored


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def process(body: bytes, cancel: threading.Event):
    def check():
        if cancel.is_set():
            raise Cancelled

    check()
    pdf_bytes = decode_pdf(extract_b64(body))
    check()
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        raise BadRequest("Decoded data is not a valid PDF (wrong key/data?)")
    methods, removed = [], 0
    try:
        total = doc.page_count
        if total < 1:
            raise BadRequest("PDF has no pages")
        page = doc[0]  # page 1 only, always
        scale = pick_scale(page)
        check()

        # ---- stage 1: lossless vector removal
        collateral, wm_colors = [], []
        try:
            v_removed, collateral, wm_colors = vector_stage(pdf_bytes, page, check)
        except Cancelled:
            raise
        except Exception as e:  # never let the vector stage break the request
            log.warning("vector stage failed: %s", e, exc_info=True)
            v_removed = 0
        if v_removed:
            methods.append("vector")
            removed += v_removed

        rgb = render_rgb(page, scale)
        check()

        # A real glyph that MuPDF removed together with the watermark is put back
        # from an untouched render; only pixels matching the watermark colour are
        # then replaced by the (watermark-free, glyph-free) background.
        if collateral:
            orig = fitz.open(stream=pdf_bytes, filetype="pdf")
            try:
                a = render_rgb(orig[0], scale)
            finally:
                orig.close()
            if a.shape == rgb.shape:
                _restore_collateral(rgb, a, collateral, scale, page.rect.tl, wm_colors, check)
            del a
    finally:
        doc.close()
    del pdf_bytes

    # ---- stage 2: baked-in (raster) watermarks
    h, w = rgb.shape[:2]
    try:
        final_mask, protect = None, None
        accepted, c_used, s, n_ok = raster_signature_stage(rgb, check)
        if accepted is not None:
            final_mask, protect = refine_mask(rgb, accepted, c_used, s, check)
            methods.append("raster-signature")
            removed += n_ok
        if OCR_FALLBACK and not methods:
            covered = np.zeros((h, w), bool)
            m2, n2 = raster_ocr_stage(rgb, covered, check)
            if m2 is not None:
                fm, pr = build_mask(m2, None, 3.0)
                final_mask, protect = fm, pr
                methods.append("raster-ocr")
                removed += n2
        if final_mask is not None and final_mask.any():
            check()
            rgb = inpaint_regions(rgb, final_mask, protect, check)
    except (Cancelled, BadRequest):
        raise
    except Exception as e:
        log.warning("raster stage failed: %s", e, exc_info=True)

    check()
    ok, png = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise RuntimeError("PNG encode failed")
    return png.tobytes(), total, removed, methods or ["none"], scale


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------
async def run_job(body, cancel):
    """Wait for a slot, then run the job in a thread. If the client leaves, the
    worker is told to stop at its next checkpoint and the slot is released only
    once the thread has really stopped (keeps memory bounded)."""
    try:
        await asyncio.wait_for(_sem.acquire(), QUEUE_TIMEOUT)
    except asyncio.TimeoutError:
        raise QueueTimeout
    fut = asyncio.get_running_loop().run_in_executor(None, lambda: process(body, cancel))
    handed_off = False
    try:
        return await asyncio.shield(fut)
    except asyncio.CancelledError:
        cancel.set()
        handed_off = True

        def _done(f):
            if not f.cancelled():
                f.exception()
            _sem.release()

        fut.add_done_callback(_done)
        raise
    finally:
        if not handed_off:
            _sem.release()


async def watch_disconnect(request):
    while not await request.is_disconnected():
        await asyncio.sleep(0.25)


async def process_endpoint(request):
    global _pending
    if _pending >= MAX_JOBS + MAX_QUEUE:
        return busy()

    cl = request.headers.get("content-length")
    if cl and cl.isdigit() and int(cl) > MAX_BODY:
        return JSONResponse({"error": "Body too large"}, status_code=413)

    body = bytearray()
    try:
        async for chunk in request.stream():
            body += chunk
            if len(body) > MAX_BODY:
                return JSONResponse({"error": "Body too large"}, status_code=413)
    except ClientDisconnect:
        return Response(status_code=499)

    cancel = threading.Event()
    _pending += 1
    job = asyncio.ensure_future(run_job(bytes(body), cancel))
    watcher = asyncio.ensure_future(watch_disconnect(request))
    del body
    try:
        done, _ = await asyncio.wait({job, watcher}, return_when=asyncio.FIRST_COMPLETED)
        if job not in done:
            job.cancel()
            return Response(status_code=499)
        data, total, removed, methods, scale = job.result()
    except QueueTimeout:
        return busy()
    except Cancelled:
        return Response(status_code=499)
    except BadRequest as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except asyncio.CancelledError:
        job.cancel()
        raise
    except Exception as e:  # noqa
        log.exception("processing failed")
        return JSONResponse({"error": f"Processing failed: {e}"}, status_code=500)
    finally:
        watcher.cancel()
        _pending -= 1

    return Response(
        data,
        media_type="image/png",
        headers={
            "X-Total-Pages": str(total),
            "X-Watermarks-Removed": str(removed),
            "X-Removal-Method": ",".join(methods),
            "X-Render-Scale": f"{scale:.3f}",
            "Cache-Control": "no-store",
        },
    )


# ---------------------------------------------------------------------------
# Transparent proxy (unchanged)
# ---------------------------------------------------------------------------
_HOP_BY_HOP = frozenset([
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "proxy-connection",
])
_STRIP_FROM_RESPONSE = _HOP_BY_HOP | frozenset(["content-encoding", "content-length"])

_proxy_client = httpx.AsyncClient(
    follow_redirects=False,
    timeout=httpx.Timeout(
        connect=float(os.getenv("PROXY_CONNECT_TIMEOUT", "10")),
        read=float(os.getenv("PROXY_READ_TIMEOUT", "60")),
        write=float(os.getenv("PROXY_WRITE_TIMEOUT", "30")),
        pool=float(os.getenv("PROXY_POOL_TIMEOUT", "10")),
    ),
    limits=httpx.Limits(
        max_connections=int(os.getenv("PROXY_MAX_CONN", "100")),
        max_keepalive_connections=int(os.getenv("PROXY_MAX_KEEPALIVE", "20")),
    ),
    verify=os.getenv("PROXY_VERIFY_SSL", "true").lower() != "false",
)


def _decode_proxy_url(raw: str) -> str:
    unquoted = urllib.parse.unquote(raw)
    padded = unquoted + "=" * (-len(unquoted) % 4)
    try:
        return base64.b64decode(padded).decode("utf-8")
    except Exception:
        raise BadRequest("url param is not valid base64")


def _forward_headers(incoming_headers, host: str) -> dict:
    _skip = _HOP_BY_HOP | frozenset(["host", "accept-encoding"])
    return {k: v for k, v in incoming_headers.items() if k.lower() not in _skip}


def _response_headers(upstream_headers) -> dict:
    return {k: v for k, v in upstream_headers.items() if k.lower() not in _STRIP_FROM_RESPONSE}


async def proxy_endpoint(request):
    raw_param = request.query_params.get("url", "").strip()
    if not raw_param:
        return JSONResponse({"error": "Missing 'url' query parameter"}, status_code=400)
    try:
        target_url = _decode_proxy_url(raw_param)
    except BadRequest as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    if not target_url.lower().startswith(("http://", "https://")):
        return JSONResponse({"error": "Decoded URL must start with http:// or https://"}, status_code=400)
    try:
        body = await request.body()
    except ClientDisconnect:
        return Response(status_code=499)
    out_headers = _forward_headers(request.headers, target_url)
    try:
        upstream = await _proxy_client.request(
            method=request.method, url=target_url, headers=out_headers, content=body or None)
    except httpx.TimeoutException:
        return JSONResponse({"error": "Upstream request timed out"}, status_code=504)
    except httpx.RequestError as exc:
        return JSONResponse({"error": f"Upstream connection error: {exc}"}, status_code=502)
    return Response(content=upstream.content, status_code=upstream.status_code,
                    headers=_response_headers(upstream.headers))


async def test_page(request):
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "test.html"), "rb") as f:
            html = f.read()
    except FileNotFoundError:
        return PlainTextResponse("test.html not found", status_code=404)
    return Response(html, media_type="text/html; charset=utf-8")


async def health(request):
    return PlainTextResponse("ok")


app = Starlette(routes=[
    Route("/process", process_endpoint, methods=["POST"]),
    Route("/proxy", proxy_endpoint, methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]),
    Route("/", health, methods=["GET"]),
    Route("/test", test_page, methods=["GET"]),
], middleware=[
    Middleware(
        CORSMiddleware,
        allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Total-Pages", "X-Watermarks-Removed", "X-Removal-Method",
                        "X-Render-Scale", "Content-Disposition", "Content-Type", "Content-Length"],
    ),
])
