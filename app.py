"""
Watermark remover + transparent proxy service (PyMuPDF + numpy + httpx + Starlette).

POST /process
  Body: the "data" string, OR the full JSON {"success":true,"data":"..."}
        (raw text, a JSON string, or JSON object all work)
  Query parameters are IGNORED. Everything is fixed:
    page       always 1
    scale      chosen automatically from the page size (target 3.0, 2.5 floor
               unless the memory cap forces it lower)
    watermark  opacity 20.60 %, colour RGB(158, 159, 157)
    tolerance  4 RGB levels
    format     PNG (lossless)
  Returns: the processed PNG bytes.
  Headers: X-Total-Pages = page count, X-Scale = scale actually used.

How the watermark is removed
  The watermark was alpha-composited over the page:
        W = (1 - a) * C + a * K          a = 0.206, K = (158, 159, 157)
  so the clean pixel is   C = (W - a * K) / (1 - a).
  1. Pixels within +-4 levels of the "watermark over white" colour are found.
  2. Only pixels that sit in a flat patch of such pixels are kept as seeds
     (this stops anti-aliased text edges from being mistaken for watermark).
  3. Seeds are grown a few pixels so that the watermark where it overlaps
     text / anti-aliased edges is covered too.
  4. Inside that mask the exact inverse of the compositing formula is applied
     (per-channel lookup table), and near-white leftovers snap to pure white.
  Everything outside the mask is left byte-for-byte untouched.

GET|POST|* /proxy?url=<base64url-encoded-url>
  Decodes the `url` param (URL-decoded first, then base64-decoded) and
  forwards the request verbatim — method, headers, body, cookies,
  Authorization — to the target. The target's status, headers, and body
  are streamed back unchanged. Hop-by-hop headers are stripped per RFC 7230.
"""
import base64
import json
import math
import os
import asyncio
import threading
import urllib.parse

import httpx

import fitz  # PyMuPDF
import numpy as np
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import ClientDisconnect
from starlette.responses import JSONResponse, PlainTextResponse, Response
from starlette.routing import Route

KEY = (90, 165, 195, 60, 15, 240, 150, 105)
# key[i % 8] ^ (i % 256) repeats every 256 bytes -> precompute one period
XOR_MASK = np.array([KEY[i % 8] ^ (i % 256) for i in range(256)], dtype=np.uint8)

MAX_BODY = int(float(os.getenv("MAX_BODY_MB", "1")) * 1024 * 1024)  # ~300 KB typical
MAX_PIXELS = int(os.getenv("MAX_PIXELS", str(25_000_000)))  # hard cap per render
MAX_JOBS = int(os.getenv("MAX_JOBS", "2"))  # concurrent renders per worker (bounds RAM)
BAND_ROWS = 256  # process image in horizontal bands to keep temp memory tiny

MAX_QUEUE = int(os.getenv("MAX_QUEUE", "50"))  # waiting requests per worker
QUEUE_TIMEOUT = float(os.getenv("QUEUE_TIMEOUT", "30"))  # max seconds spent waiting

# ---------------------------------------------------------------------------
# Fixed watermark model (measured) — request parameters can NOT change these
# ---------------------------------------------------------------------------
PAGE_INDEX = 0                      # always page 1
WM_ALPHA = 0.2060                   # estimated watermark opacity (20.60 %)
WM_COLOR = (158, 159, 157)          # estimated watermark colour K
WM_ON_WHITE = (238, 238, 238)       # measured watermark-over-white (sample 1)
TOL = 4                             # +-4 RGB levels around WM_ON_WHITE

SCALE_TARGET = 3.0                  # preferred render scale
SCALE_PREF_MIN = 2.5                # don't go below this unless forced by MAX_PIXELS
SOFT_PIXELS = int(os.getenv("SOFT_PIXELS", str(12_000_000)))  # above this, ease toward 2.5

MIN_NEIGHBORS = 4                   # of 8 neighbours must also match for a seed pixel
SNAP_WHITE = 250                    # restored channel values >= this become 255

_sem = asyncio.Semaphore(MAX_JOBS)
_pending = 0  # running + waiting (single event loop => no lock needed)


def busy():
    return JSONResponse(
        {"error": "Server busy, please try again later"},
        status_code=503,
        headers={"Retry-After": "3"},
    )


class BadRequest(Exception):
    pass


class Cancelled(Exception):
    """Client went away; abort work as soon as possible."""


class QueueTimeout(Exception):
    pass


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
    out = raw ^ np.resize(XOR_MASK, raw.size)
    return out.tobytes()


# ---------------------------------------------------------------------------
# Scale selection
# ---------------------------------------------------------------------------

def pick_scale(width_pt: float, height_pt: float) -> float:
    """Pick the render scale from the page size.

    Normal pages (A4, Letter, A3 ...) are rendered at 3.0. Very large pages ease
    down toward 2.5 so the image stays a sane size. Only if even that would
    exceed MAX_PIXELS is the scale reduced further (hard memory cap).
    """
    area = width_pt * height_pt
    if area <= 0:
        raise BadRequest("Page has no area")
    s = SCALE_TARGET
    if area * s * s > SOFT_PIXELS:
        s = max(SCALE_PREF_MIN, math.sqrt(SOFT_PIXELS / area))
    if area * s * s > MAX_PIXELS:
        s = math.sqrt(MAX_PIXELS / area)
    if s < 0.5:
        raise BadRequest("Page too large to render")
    return s


# ---------------------------------------------------------------------------
# Watermark removal
# ---------------------------------------------------------------------------

def _build_luts():
    """Per-channel lookup tables: exact inverse of  W = (1-a)*C + a*K."""
    inv = 1.0 / (1.0 - WM_ALPHA)
    v = np.arange(256, dtype=np.float32)
    luts = []
    for k in WM_COLOR:
        c = np.clip(np.rint((v - WM_ALPHA * k) * inv), 0, 255)
        c[c >= SNAP_WHITE] = 255
        luts.append(c.astype(np.uint8))
    return luts


_LUTS = _build_luts()


def _neighbor_count(m: np.ndarray) -> np.ndarray:
    """Number of True pixels among the 8 neighbours of each pixel."""
    hh, ww = m.shape
    p = np.pad(m, 1)
    cnt = np.zeros((hh, ww), dtype=np.uint8)
    for dy in range(3):
        for dx in range(3):
            if dy == 1 and dx == 1:
                continue
            cnt += p[dy:dy + hh, dx:dx + ww]
    return cnt


def _dilate(m: np.ndarray, r: int) -> np.ndarray:
    """Square dilation with radius r (separable, boolean)."""
    out = m.copy()
    for d in range(1, r + 1):
        out[d:] |= m[:-d]
        out[:-d] |= m[d:]
    out2 = out.copy()
    for d in range(1, r + 1):
        out2[:, d:] |= out[:, :-d]
        out2[:, :-d] |= out[:, d:]
    return out2


def remove_watermark(buf: bytearray, w: int, h: int, scale: float, cancel):
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
    center = np.array(WM_ON_WHITE, dtype=np.int16)

    # Pass 0: pixels that look like "watermark over white" (within +-TOL levels)
    match = np.empty((h, w), dtype=bool)
    for y in range(0, h, BAND_ROWS):
        if cancel.is_set():
            raise Cancelled
        band = arr[y:y + BAND_ROWS]
        diff = np.abs(band.astype(np.int16) - center)
        match[y:y + BAND_ROWS] = (diff <= TOL).all(axis=2)
    if not match.any():
        return

    # Pass 1: keep only pixels inside a flat patch (rejects anti-aliased text edges)
    seeds = np.zeros((h, w), dtype=bool)
    for y in range(0, h, BAND_ROWS):
        if cancel.is_set():
            raise Cancelled
        y0, y1 = max(0, y - 1), min(h, y + BAND_ROWS + 1)
        blk = match[y0:y1]
        cnt = _neighbor_count(blk)
        ya = y - y0
        yb = ya + min(BAND_ROWS, h - y)
        seeds[y:y + BAND_ROWS] = blk[ya:yb] & (cnt[ya:yb] >= MIN_NEIGHBORS)
    del match
    if not seeds.any():
        return

    # Pass 2: grow seeds (covers watermark over text / glyph edges), then invert
    r = max(2, int(math.ceil(1.5 * scale)))
    for y in range(0, h, BAND_ROWS):
        if cancel.is_set():
            raise Cancelled
        y0, y1 = max(0, y - r), min(h, y + BAND_ROWS + r)
        grown = _dilate(seeds[y0:y1], r)
        ya = y - y0
        yb = ya + min(BAND_ROWS, h - y)
        m = grown[ya:yb]
        if not m.any():
            continue
        band = arr[y:y + BAND_ROWS]
        for c in range(3):
            ch = band[:, :, c]
            ch[m] = _LUTS[c][ch[m]]


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
    try:
        total = doc.page_count
        if total < 1:
            raise BadRequest("PDF has no pages")
        pg = doc[PAGE_INDEX]  # always the first page
        check()
        r = pg.rect
        scale = pick_scale(r.width, r.height)
        pix = pg.get_pixmap(matrix=fitz.Matrix(scale, scale),
                            colorspace=fitz.csRGB, alpha=False)
    finally:
        doc.close()
    del pdf_bytes
    check()

    w, h = pix.width, pix.height
    buf = bytearray(pix.samples)
    del pix
    check()
    remove_watermark(buf, w, h, scale, cancel)
    check()
    out_pix = fitz.Pixmap(fitz.csRGB, w, h, bytes(buf), False)
    del buf
    check()
    data = out_pix.tobytes("png")
    return data, total, scale


async def run_job(body, cancel):
    """Wait for a slot, then run the render in a thread.

    If this task is cancelled (client left), the waiting is abandoned at once
    or the running thread is told to stop at its next checkpoint. The slot is
    only released when the thread has really stopped, so memory stays bounded.
    """
    try:
        await asyncio.wait_for(_sem.acquire(), QUEUE_TIMEOUT)
    except asyncio.TimeoutError:
        raise QueueTimeout
    fut = asyncio.get_running_loop().run_in_executor(
        None, lambda: process(body, cancel))
    handed_off = False
    try:
        return await asyncio.shield(fut)
    except asyncio.CancelledError:
        cancel.set()
        handed_off = True

        def _done(f):
            if not f.cancelled():
                f.exception()  # mark retrieved, ignore Cancelled
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
    # reject early, before reading the body, so a full queue costs ~no memory
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

    # NOTE: query parameters (page, scale, r, g, b, tol, format, quality) are
    # deliberately ignored — all settings are fixed above.

    cancel = threading.Event()
    _pending += 1
    job = asyncio.ensure_future(run_job(bytes(body), cancel))
    watcher = asyncio.ensure_future(watch_disconnect(request))
    del body
    try:
        done, _ = await asyncio.wait({job, watcher}, return_when=asyncio.FIRST_COMPLETED)
        if job not in done:
            # client disconnected while queued or mid-process: drop immediately
            job.cancel()
            return Response(status_code=499)
        data, total, scale = job.result()
    except QueueTimeout:
        return busy()
    except Cancelled:
        return Response(status_code=499)
    except BadRequest as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except asyncio.CancelledError:
        # the request handler itself was cancelled by the server
        job.cancel()
        raise
    except Exception as e:  # noqa
        return JSONResponse({"error": f"Processing failed: {e}"}, status_code=500)
    finally:
        watcher.cancel()
        _pending -= 1

    return Response(
        data,
        media_type="image/png",
        headers={
            "X-Total-Pages": str(total),
            "X-Scale": f"{scale:.3f}",
            "Cache-Control": "no-store",
        },
    )


# ---------------------------------------------------------------------------
# Transparent proxy
# ---------------------------------------------------------------------------

# Hop-by-hop headers that must never be forwarded (RFC 7230 §6.1 + common extras)
_HOP_BY_HOP = frozenset([
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade",
    "proxy-connection",
])

# Headers that become invalid after httpx transparently decompresses the body
_STRIP_FROM_RESPONSE = _HOP_BY_HOP | frozenset(["content-encoding", "content-length"])

# Single shared async client — keeps a connection pool, reuses TLS sessions
# decompress=True (default): httpx decodes gzip/br/deflate for us so we always
# forward plain bytes — we just must not re-advertise Content-Encoding downstream.
_proxy_client = httpx.AsyncClient(
    follow_redirects=False,          # let the caller handle redirects
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
    """URL-decode then base64-decode the `url` query param."""
    # Step 1: undo any %XX percent-encoding the caller applied
    unquoted = urllib.parse.unquote(raw)
    # Step 2: fix padding and base64-decode
    padded = unquoted + "=" * (-len(unquoted) % 4)
    try:
        return base64.b64decode(padded).decode("utf-8")
    except Exception:
        raise BadRequest("url param is not valid base64")


def _forward_headers(incoming_headers, host: str) -> dict:
    """Build the outbound header dict: strip hop-by-hop, rewrite Host.

    We also drop Accept-Encoding so the upstream always replies with plain
    (uncompressed) bytes — httpx would decompress anyway, but this avoids
    any edge-case where a server ignores decompress=True behaviour.
    """
    _skip = _HOP_BY_HOP | frozenset(["host", "accept-encoding"])
    out = {}
    for k, v in incoming_headers.items():
        if k.lower() in _skip:
            continue
        out[k] = v
    return out


def _response_headers(upstream_headers) -> dict:
    """Strip hop-by-hop + encoding/length headers from the upstream response.

    httpx transparently decompresses gzip/br/deflate, so the body we forward
    is already plain bytes.  We must drop Content-Encoding (no longer true)
    and Content-Length (now wrong) so the client doesn't choke on the mismatch.
    """
    out = {}
    for k, v in upstream_headers.items():
        if k.lower() in _STRIP_FROM_RESPONSE:
            continue
        out[k] = v
    return out


async def proxy_endpoint(request):
    raw_param = request.query_params.get("url", "").strip()
    if not raw_param:
        return JSONResponse({"error": "Missing 'url' query parameter"}, status_code=400)

    try:
        target_url = _decode_proxy_url(raw_param)
    except BadRequest as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    # Basic sanity: must look like an http(s) URL
    if not target_url.lower().startswith(("http://", "https://")):
        return JSONResponse({"error": "Decoded URL must start with http:// or https://"}, status_code=400)

    # Read the body (stream it to avoid buffering huge payloads needlessly)
    try:
        body = await request.body()
    except ClientDisconnect:
        return Response(status_code=499)

    out_headers = _forward_headers(request.headers, target_url)

    try:
        upstream = await _proxy_client.request(
            method=request.method,
            url=target_url,
            headers=out_headers,
            content=body or None,
        )
    except httpx.TimeoutException:
        return JSONResponse({"error": "Upstream request timed out"}, status_code=504)
    except httpx.RequestError as exc:
        return JSONResponse({"error": f"Upstream connection error: {exc}"}, status_code=502)

    resp_headers = _response_headers(upstream.headers)

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=resp_headers,
    )


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
        expose_headers=["X-Total-Pages", "X-Scale", "Content-Disposition", "Content-Type", "Content-Length"],
    ),
])
