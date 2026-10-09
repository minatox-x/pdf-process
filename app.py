"""
Watermark remover service (PyMuPDF + numpy + Starlette).

POST /process
  Body: the "data" string, OR the full JSON {"success":true,"data":"..."}
        (raw text, a JSON string, or JSON object all work)
  Query params (all optional):
    page    1-based page number           (default 1)
    scale   render scale, 0.5-4           (default 1.5)
    r,g,b   watermark color to remove     (default 220,220,220)
    tol     RGB distance tolerance 0-150  (default 35)
    format  png | jpeg                    (default png)
    quality jpeg quality 1-100            (default 90)
  Returns: the processed image bytes. Header X-Total-Pages = page count.
"""
import base64
import json
import os
import asyncio
import threading

import fitz  # PyMuPDF
import numpy as np
from starlette.applications import Starlette
from starlette.requests import ClientDisconnect
from starlette.responses import JSONResponse, PlainTextResponse, Response
from starlette.routing import Route

KEY = (90, 165, 195, 60, 15, 240, 150, 105)
# key[i % 8] ^ (i % 256) repeats every 256 bytes -> precompute one period
XOR_MASK = np.array([KEY[i % 8] ^ (i % 256) for i in range(256)], dtype=np.uint8)

MAX_BODY = int(float(os.getenv("MAX_BODY_MB", "1")) * 1024 * 1024)  # ~300 KB typical
MAX_PIXELS = int(os.getenv("MAX_PIXELS", str(25_000_000)))
MAX_JOBS = int(os.getenv("MAX_JOBS", "2"))  # concurrent renders per worker (bounds RAM)
BAND_ROWS = 256  # process image in horizontal bands to keep temp memory tiny

MAX_QUEUE = int(os.getenv("MAX_QUEUE", "50"))  # waiting requests per worker
QUEUE_TIMEOUT = float(os.getenv("QUEUE_TIMEOUT", "30"))  # max seconds spent waiting

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


def remove_color(buf: bytearray, w: int, h: int, rgb, tol: int, cancel):
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
    t = np.array(rgb, dtype=np.int32)
    tol2 = tol * tol
    for y in range(0, h, BAND_ROWS):
        if cancel.is_set():
            raise Cancelled
        band = arr[y:y + BAND_ROWS]
        d = band.astype(np.int32) - t
        d *= d
        mask = d.sum(axis=2) <= tol2
        band[mask] = 255


def process(body: bytes, page: int, scale: float, rgb, tol: int, fmt: str,
            quality: int, cancel: threading.Event):
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
        if r.width * scale * r.height * scale > MAX_PIXELS:
            raise BadRequest("Render too large; lower scale")
        pix = pg.get_pixmap(matrix=fitz.Matrix(scale, scale),
                            colorspace=fitz.csRGB, alpha=False)
    finally:
        doc.close()
    del pdf_bytes
    check()

    w, h = pix.width, pix.height
    buf = bytearray(pix.samples)
    del pix
    if tol > 0:
        remove_color(buf, w, h, rgb, tol, cancel)
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
        None, lambda: process(body, *args, cancel))
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

    q = request.query_params
    try:
        page = _int(q, "page", 1, 1, 100000)
        tol = _int(q, "tol", 35, 0, 150)
        rgb = (_int(q, "r", 220, 0, 255), _int(q, "g", 220, 0, 255), _int(q, "b", 220, 0, 255))
        quality = _int(q, "quality", 90, 1, 100)
        try:
            scale = max(0.5, min(4.0, float(q.get("scale", 1.5))))
        except ValueError:
            raise BadRequest("scale must be a number")
        fmt = "jpeg" if q.get("format", "png").lower() in ("jpeg", "jpg") else "png"
    except BadRequest as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    cancel = threading.Event()
    args = (page, scale, rgb, tol, fmt, quality)
    _pending += 1
    job = asyncio.ensure_future(run_job(bytes(body), args, cancel))
    watcher = asyncio.ensure_future(watch_disconnect(request))
    del body
    try:
        done, _ = await asyncio.wait({job, watcher}, return_when=asyncio.FIRST_COMPLETED)
        if job not in done:
            # client disconnected while queued or mid-process: drop immediately
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
        media_type="image/jpeg" if fmt == "jpeg" else "image/png",
        headers={"X-Total-Pages": str(total), "Cache-Control": "no-store"},
    )


async def health(request):
    return PlainTextResponse("ok")


app = Starlette(routes=[
    Route("/process", process_endpoint, methods=["POST"]),
    Route("/", health, methods=["GET"]),
])
