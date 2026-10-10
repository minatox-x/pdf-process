"""
Watermark remover + transparent proxy service (PyMuPDF + numpy + httpx + Starlette).

POST /process
  Body: the "data" string, OR the full JSON {"success":true,"data":"..."}
        (raw text, a JSON string, or JSON object all work)
  Query params (optional):
    page    1-based page number (default 1)
    format  png | jpeg          (default png)
  Returns: the processed image bytes. Header X-Total-Pages = page count.
"""
import base64
import json
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
XOR_MASK = np.array([KEY[i % 8] ^ (i % 256) for i in range(256)], dtype=np.uint8)

MAX_BODY = int(float(os.getenv("MAX_BODY_MB", "1")) * 1024 * 1024)
MAX_PIXELS = int(os.getenv("MAX_PIXELS", str(35_000_000)))
MAX_JOBS = int(os.getenv("MAX_JOBS", "2"))
BAND_ROWS = 256

MAX_QUEUE = int(os.getenv("MAX_QUEUE", "50"))
QUEUE_TIMEOUT = float(os.getenv("QUEUE_TIMEOUT", "30"))

# --- LOCKED OPTIMAL DEFAULTS FOR PERFECT WATERMARK REMOVAL ---
OPTIMAL_RGB = (240, 240, 240)  # Universal light-gray watermark target
OPTIMAL_TOL = 40             # Balanced tolerance to catch faded traces
OPTIMAL_SCALE = 2.0          # High resolution scale for razor-sharp text

_sem = asyncio.Semaphore(MAX_JOBS)
_pending = 0


def busy():
    return JSONResponse(
        {"error": "Server busy, please try again later"},
        status_code=503,
        headers={"Retry-After": "3"},
    )


class BadRequest(Exception):
    pass


class Cancelled(Exception):
    pass


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


def remove_color(buf: bytearray, w: int, h: int, cancel):
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
    t = np.array(OPTIMAL_RGB, dtype=np.float32)
    tol_f = float(OPTIMAL_TOL)
    tol2 = tol_f * tol_f

    for y in range(0, h, BAND_ROWS):
        if cancel.is_set():
            raise Cancelled
        band = arr[y:y + BAND_ROWS]
        band_f = band.astype(np.float32)
        
        # 1. Vectorized Euclidean distance squared from target watermark RGB
        diff = band_f - t
        dist2 = np.sum(diff * diff, axis=2)
        
        mask = dist2 <= tol2
        if not np.any(mask):
            continue
            
        # 2. Cosine smoothstep feathering for natural, anti-aliased edge transition
        dist = np.sqrt(dist2)
        weight = np.zeros_like(dist)
        valid = mask & (dist <= tol_f)
        
        x = np.clip(dist[valid] / tol_f, 0.0, 1.0)
        weight[valid] = 0.5 * (1.0 + np.cos(x * np.pi))
        
        # 3. Smart Foreground & Text Stroke Protection (luminance ramp)
        luminance = band_f.mean(axis=2)
        text_protection = np.clip((luminance - 50.0) / 70.0, 0.0, 1.0)
        
        final_weight = weight * text_protection
        
        # 4. Precision in-place blending toward immaculate white (255.0)
        if np.any(final_weight > 0):
            w_exp = final_weight[..., np.newaxis]
            blended = band_f * (1.0 - w_exp) + 255.0 * w_exp
            band[...] = np.clip(blended, 0, 255).astype(np.uint8)


def process(body: bytes, page: int, fmt: str, quality: int, cancel: threading.Event):
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
        if not 1 <= page <= total:
            raise BadRequest(f"page must be 1..{total}")
        pg = doc[page - 1]
        check()
        r = pg.rect
        if r.width * OPTIMAL_SCALE * r.height * OPTIMAL_SCALE > MAX_PIXELS:
            raise BadRequest("Render too large")
        pix = pg.get_pixmap(matrix=fitz.Matrix(OPTIMAL_SCALE, OPTIMAL_SCALE),
                            colorspace=fitz.csRGB, alpha=False)
    finally:
        doc.close()
    del pdf_bytes
    check()

    w, h = pix.width, pix.height
    buf = bytearray(pix.samples)
    del pix
    
    # Run the optimized watermark remover with locked-in defaults
    remove_color(buf, w, h, cancel)
    
    check()
    out_pix = fitz.Pixmap(fitz.csRGB, w, h, bytes(buf), False)
    del buf
    check()
    if fmt == "jpeg":
        data = out_pix.tobytes("jpeg", jpg_quality=quality)
    else:
        data = out_pix.tobytes("png")
    return data, total


def _int(q, name, default, lo, hi):
    try:
        v = int(q.get(name, default))
    except ValueError:
        raise BadRequest(f"{name} must be an integer")
    return max(lo, min(hi, v))


async def run_job(body, args, cancel):
    try:
        await asyncio.wait_for(_sem.acquire(), QUEUE_TIMEOUT)
    except asyncio.TimeoutError:
        raise QueueTimeout
    fut = asyncio.get_running_loop().run_in_executor(
        None, lambda: process(body, *args, cancel))
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

    q = request.query_params
    try:
        page = _int(q, "page", 1, 1, 100000)
        quality = _int(q, "quality", 90, 1, 100)
        fmt = "jpeg" if q.get("format", "png").lower() in ("jpeg", "jpg") else "png"
    except BadRequest as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    cancel = threading.Event()
    args = (page, fmt, quality)
    _pending += 1
    job = asyncio.ensure_future(run_job(bytes(body), args, cancel))
    watcher = asyncio.ensure_future(watch_disconnect(request))
    del body
    try:
        done, _ = await asyncio.wait({job, watcher}, return_when=asyncio.FIRST_COMPLETED)
        if job not in done:
            job.cancel()
            return Response(status_code=499)
        data, total = job.result()
    except QueueTimeout:
        return busy()
    except Cancelled:
        return Response(status_code=499)
    except BadRequest as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except asyncio.CancelledError:
        job.cancel()
        raise
    except Exception as e:
        return JSONResponse({"error": f"Processing failed: {e}"}, status_code=500)
    finally:
        watcher.cancel()
        _pending -= 1

    return Response(
        data,
        media_type="image/jpeg" if fmt == "jpeg" else "image/png",
        headers={"X-Total-Pages": str(total), "Cache-Control": "no-store"},
    )


# ---------------------------------------------------------------------------
# Transparent proxy
# ---------------------------------------------------------------------------

_HOP_BY_HOP = frozenset([
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade",
    "proxy-connection",
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
    out = {}
    for k, v in incoming_headers.items():
        if k.lower() in _skip:
            continue
        out[k] = v
    return out


def _response_headers(upstream_headers) -> dict:
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

    if not target_url.lower().startswith(("http://", "https://")):
        return JSONResponse({"error": "Decoded URL must start with http:// or https://"}, status_code=400)

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
        expose_headers=["X-Total-Pages", "Content-Disposition", "Content-Type", "Content-Length"],
    ),
])
