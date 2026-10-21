import os
import time
import logging
from typing import Dict, Any, Tuple, Optional, List
from urllib.parse import urlparse

import httpx
import numpy as np

logger = logging.getLogger("satquery.acquisition.windowed_reader")

class WindowedRetrievalError(Exception):
    """Raised when safe windowed retrieval fails or violates security controls."""
    pass

class SafeWindowedCOGReader:
    """
    Scientific Cloud-Optimized GeoTIFF (COG) Windowed Reader (Stage 6B).
    
    Strict Security & Scientific Safeguards:
    1. Host & Protocol Safety: Enforces HTTPS and strictly validates host against trusted allowlist.
    2. HTTP Range Verification: Validates that remote server returns HTTP 206 Partial Content.
       If the server returns HTTP 200 (attempting full object transfer), it aborts immediately.
    3. Strict Cumulative Transfer Budget: Bounds total bytes transferred to prevent runaway egress/memory.
    4. Bounded Output Dimensions: Clamps output window to maximum allowed pixel dimensions (<= 1024x1024).
    5. Georeferencing Integrity: Requires genuine projected/geographic CRS and affine transform.
       Never assumes or invents georeferencing for unprojected/identity swath rasters.
    6. Scientific Provenance: Retains actual raster dtype (e.g. uint16 surface reflectance),
       calibrated window transform, nodata values, and source metadata.
    """

    ALLOWED_DOWNLOAD_HOSTS = {
        "sentinel-cogs.s3.us-west-2.amazonaws.com",
        "earth-search.aws.element84.com",
        "sentinel-s1-l1c.s3.amazonaws.com",
        "sentinel-s2-l2a.s3.amazonaws.com",
        "landsat-pds.s3.amazonaws.com",
        "s3.us-west-2.amazonaws.com",
        "s3.amazonaws.com",
        "sentinel1euwestrtc.blob.core.windows.net",
        "planetarycomputer.microsoft.com"
    }

    DEFAULT_CUMULATIVE_BYTE_LIMIT = int(os.getenv("MAX_WINDOW_TRANSFER_BYTES", str(20 * 1024 * 1024)))  # 20 MB default
    MAX_WINDOW_DIMENSION_PIXELS = 1024
    MAX_PROBE_BYTES = 64 * 1024  # 64 KiB maximum allowed for probe response payload
    REQUEST_TIMEOUT_SECONDS = 15.0

    @staticmethod
    def _sanitize_url(u: Optional[str]) -> str:
        """Strips query parameters (e.g. SAS tokens) to prevent leaking credentials in logs/errors."""
        if not u:
            return ""
        try:
            parsed = urlparse(u)
            return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        except Exception:
            return "url_sanitized"

    @classmethod
    def validate_url_security(cls, target_url: str) -> None:
        """
        Enforces HTTPS protocol and strict allowlist membership at request boundary,
        preventing SSRF and redirection to untrusted destinations.
        """
        if not target_url.startswith("https://"):
            sanitized = cls._sanitize_url(target_url)
            raise WindowedRetrievalError(f"Insecure protocol: only HTTPS URLs are allowed ({sanitized}).")

        host = urlparse(target_url).hostname or ""
        if not any(host == allowed or host.endswith("." + allowed) for allowed in cls.ALLOWED_DOWNLOAD_HOSTS):
            raise WindowedRetrievalError(
                f"Security violation: host '{host}' is not in the trusted satellite host allowlist."
            )

    @classmethod
    def read_cog_window(
        cls,
        url: str,
        bbox_wgs84: List[float],
        max_bytes: Optional[int] = None,
        max_dimension: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reads a spatial bounding box window [min_lon, min_lat, max_lon, max_lat] from a remote COG
        using HTTP range requests.
        Returns:
            (numpy array [bands, height, width] or [height, width], metadata dict)
        """
        budget = min(max_bytes or cls.DEFAULT_CUMULATIVE_BYTE_LIMIT, cls.DEFAULT_CUMULATIVE_BYTE_LIMIT)
        max_dim = max_dimension or cls.MAX_WINDOW_DIMENSION_PIXELS
        cumulative_bytes = 0

        # 1. Protocol and Host Validation on initial URL
        cls.validate_url_security(url)
        current_url = url

        # 2. Probe Remote Server for HTTP Range Support, Following Redirects Safely
        with httpx.Client(timeout=cls.REQUEST_TIMEOUT_SECONDS, follow_redirects=False) as client:
            head_resp = None
            redirect_count = 0
            while redirect_count < 5:
                try:
                    resp = client.head(current_url)
                except Exception as e:
                    raise WindowedRetrievalError(f"Network error connecting to remote asset: {e}")

                # Handle redirects explicitly to validate target host before navigating
                if resp.status_code in (301, 302, 303, 307, 308):
                    loc = resp.headers.get("Location")
                    if not loc:
                        raise WindowedRetrievalError("Redirect response missing Location header.")
                    import urllib.parse
                    next_url = urllib.parse.urljoin(current_url, loc)
                    cls.validate_url_security(next_url)
                    current_url = next_url
                    redirect_count += 1
                else:
                    head_resp = resp
                    break

            if head_resp is None:
                raise WindowedRetrievalError("Too many HTTP redirects encountered during probe.")

            if head_resp.status_code != 200:
                raise WindowedRetrievalError(f"Remote asset returned HTTP {head_resp.status_code} on HEAD probe.")

            total_file_size = int(head_resp.headers.get("content-length", 0))
            accept_ranges = head_resp.headers.get("accept-ranges", "").lower()
            if "bytes" not in accept_ranges:
                raise WindowedRetrievalError(
                    f"Remote server does not advertise 'Accept-Ranges: bytes'. Windowed retrieval unsupported."
                )

            # Test an explicit byte range probe using a streaming request so status can be checked before reading body
            stream_cm = None
            try:
                stream_cm = client.stream("GET", current_url, headers={"Range": "bytes=0-1023"})
                test_range = stream_cm.__enter__()
            except Exception as e:
                raise WindowedRetrievalError(f"Failed range probe request: {e}")

            try:
                # Check if range request triggered another redirect
                if test_range.status_code in (301, 302, 303, 307, 308):
                    loc = test_range.headers.get("Location")
                    if loc:
                        import urllib.parse
                        next_url = urllib.parse.urljoin(current_url, loc)
                        cls.validate_url_security(next_url)
                        current_url = next_url
                        if stream_cm is not None:
                            try:
                                stream_cm.__exit__(None, None, None)
                            except Exception:
                                pass
                            stream_cm = None
                        stream_cm = client.stream("GET", current_url, headers={"Range": "bytes=0-1023"})
                        test_range = stream_cm.__enter__()

                if test_range.status_code == 200:
                    raise WindowedRetrievalError(
                        f"Server returned HTTP 200 instead of 206 Partial Content during range probe. "
                        "Aborting to prevent unintentional full-file download."
                    )
                elif test_range.status_code == 416:
                    raise WindowedRetrievalError(
                        "Server returned HTTP 416 (Range Not Satisfiable). Range request is unsupported or byte range is invalid."
                    )
                elif test_range.status_code in (500, 501, 502, 503):
                    raise WindowedRetrievalError(
                        f"Server failed to fulfill range probe request (HTTP {test_range.status_code}). Range requests may be unsupported."
                    )
                elif test_range.status_code != 206:
                    raise WindowedRetrievalError(
                        f"Server returned HTTP {test_range.status_code} instead of 206 Partial Content during range probe. "
                        "Aborting to prevent unintentional full-file download."
                    )

                # Verify Content-Range header when headers are provided (RFC 7233 compliance)
                cr_header = None
                if hasattr(test_range, "headers") and test_range.headers is not None:
                    cr_header = test_range.headers.get("content-range")

                if isinstance(test_range.headers, (dict, httpx.Headers)) or isinstance(cr_header, str):
                    if not cr_header or not isinstance(cr_header, str):
                        raise WindowedRetrievalError(
                            "Server returned HTTP 206 Partial Content but omitted mandatory 'Content-Range' response header."
                        )
                    import re
                    cr_clean = cr_header.strip()
                    cr_match = re.match(r"^bytes\s+(\d+)-(\d+)/(\d+|\*)$", cr_clean, re.IGNORECASE)
                    if not cr_match:
                        raise WindowedRetrievalError(
                            f"Server returned invalid 'Content-Range' header format ('{cr_clean}'). Expected 'bytes START-END/TOTAL'."
                        )
                    range_start = int(cr_match.group(1))
                    range_end = int(cr_match.group(2))
                    total_from_cr = cr_match.group(3)

                    if range_start != 0:
                        raise WindowedRetrievalError(
                            f"Content-Range start offset mismatch: requested 0, received {range_start}."
                        )
                    if range_end < range_start:
                        raise WindowedRetrievalError(
                            f"Content-Range end offset ({range_end}) is less than start offset ({range_start})."
                        )
                    if total_from_cr != "*" and total_file_size == 0:
                        total_file_size = int(total_from_cr)

                # Incremental, strictly bounded consumption of the probe response payload
                # Enforces hard 64 KiB maximum while reading chunks; aborts immediately without buffering full body
                max_probe_allowed = cls.MAX_PROBE_BYTES
                probe_bytes_read = 0
                if hasattr(test_range, "iter_bytes") and callable(test_range.iter_bytes):
                    for chunk in test_range.iter_bytes(chunk_size=1024):
                        if not chunk:
                            continue
                        probe_bytes_read += len(chunk)
                        if probe_bytes_read > max_probe_allowed:
                            raise WindowedRetrievalError(
                                f"Server returned excessively large payload (>{probe_bytes_read} bytes) for range probe (bytes=0-1023)."
                            )
                elif hasattr(test_range, "read") and callable(test_range.read):
                    raw = test_range.read()
                    if isinstance(raw, (bytes, bytearray)):
                        probe_bytes_read = len(raw)
                        if probe_bytes_read > max_probe_allowed:
                            raise WindowedRetrievalError(
                                f"Server returned excessively large payload ({probe_bytes_read} bytes) for range probe (bytes=0-1023)."
                            )
                elif hasattr(test_range, "content"):
                    raw = getattr(test_range, "content", None)
                    if isinstance(raw, (bytes, bytearray)):
                        probe_bytes_read = len(raw)
                        if probe_bytes_read > max_probe_allowed:
                            raise WindowedRetrievalError(
                                f"Server returned excessively large payload ({probe_bytes_read} bytes) for range probe (bytes=0-1023)."
                            )

                probe_bytes = probe_bytes_read if probe_bytes_read > 0 else 1024
                cumulative_bytes += probe_bytes
                if cumulative_bytes > budget:
                    raise WindowedRetrievalError(
                        f"Cumulative transfer limit of {budget} bytes exceeded during initial range probe."
                    )
            finally:
                if stream_cm is not None:
                    try:
                        stream_cm.__exit__(None, None, None)
                    except Exception:
                        pass

        # 3. Rasterio / GDAL Windowed Read with Tight vsicurl Environment
        import rasterio
        from rasterio.windows import from_bounds
        from rasterio.warp import transform_bounds

        gdal_env = {
            "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
            "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif, .tiff",
            "CPL_VSIL_CURL_CHUNK_SIZE": "65536"
        }

        start_time = time.time()
        with rasterio.Env(**gdal_env):
            try:
                with rasterio.open(current_url) as src:
                    # Georeferencing Verification: Genuine CRS and calibrated transform required
                    if not src.crs:
                        raise WindowedRetrievalError(
                            "Asset lacks a valid Coordinate Reference System (CRS is None). "
                            "Unprojected swath rasters cannot be mapped to a spatial AOI without orthorectification."
                        )

                    # Reject identity or uncalibrated transform
                    if hasattr(src.transform, "is_identity") and src.transform.is_identity:
                        raise WindowedRetrievalError("Asset has identity affine transform without geospatial calibration.")
                    if (src.transform[0] == 1.0 and src.transform[4] == 1.0 and src.transform[2] == 0.0 and src.transform[5] == 0.0):
                        raise WindowedRetrievalError("Asset georeferencing is uncalibrated identity matrix.")

                    # Transform AOI bbox from WGS84 to the raster's native CRS
                    if str(src.crs) != "EPSG:4326":
                        target_bounds = transform_bounds("EPSG:4326", src.crs, *bbox_wgs84)
                    else:
                        target_bounds = bbox_wgs84

                    # Detect empty spatial intersection between target bounds and raster extent
                    inter_minx = max(src.bounds.left, target_bounds[0])
                    inter_maxx = min(src.bounds.right, target_bounds[2])
                    inter_miny = max(src.bounds.bottom, target_bounds[1])
                    inter_maxy = min(src.bounds.top, target_bounds[3])

                    if inter_minx >= inter_maxx or inter_miny >= inter_maxy:
                        raise WindowedRetrievalError(
                            "Requested AOI bounding box does not intersect the asset extent (empty spatial intersection)."
                        )

                    # Compute raster window covering the intersecting bounds
                    raw_win = from_bounds(inter_minx, inter_miny, inter_maxx, inter_maxy, transform=src.transform)
                    col_off = max(0, min(src.width - 1, int(np.floor(raw_win.col_off))))
                    row_off = max(0, min(src.height - 1, int(np.floor(raw_win.row_off))))
                    
                    req_w = max(1, int(np.ceil(raw_win.width)))
                    req_h = max(1, int(np.ceil(raw_win.height)))

                    # Clamp window within raster extent and dimension cap
                    w_px = min(max_dim, min(src.width - col_off, req_w))
                    h_px = min(max_dim, min(src.height - row_off, req_h))

                    if w_px <= 0 or h_px <= 0:
                        raise WindowedRetrievalError("Requested AOI bounding box produced empty spatial window.")

                    # Cumulative transfer budget check on expected read buffer
                    item_bytes = np.dtype(src.dtypes[0]).itemsize if src.dtypes else 2
                    expected_read_bytes = int(w_px * h_px * src.count * item_bytes)
                    if (cumulative_bytes + expected_read_bytes) > budget:
                        raise WindowedRetrievalError(
                            f"Cumulative transfer limit of {budget} bytes exceeded: window read requires {expected_read_bytes} bytes "
                            f"(total would be {cumulative_bytes + expected_read_bytes} bytes)."
                        )

                    clamped_win = rasterio.windows.Window(col_off, row_off, w_px, h_px)
                    
                    # Read window data
                    if src.count == 1:
                        data = src.read(1, window=clamped_win)
                    else:
                        data = src.read(window=clamped_win)
                        # Transpose from (bands, H, W) to (H, W, bands)
                        data = np.transpose(data, (1, 2, 0))

                    cumulative_bytes += expected_read_bytes
                    win_transform = src.window_transform(clamped_win)
                    elapsed = round(time.time() - start_time, 2)

                    meta = {
                        "driver": src.driver,
                        "dtype": str(data.dtype),
                        "layout": "HWC" if data.ndim == 3 else "HW",
                        "nodata": src.nodata,
                        "crs": str(src.crs),
                        "crs_display": str(src.crs),
                        "transform": list(win_transform)[:6],
                        "bounds": [
                            float(win_transform.c),
                            float(win_transform.f + win_transform.e * h_px),
                            float(win_transform.c + win_transform.a * w_px),
                            float(win_transform.f)
                        ],
                        "width": int(w_px),
                        "height": int(h_px),
                        "bands": int(src.count),
                        "full_source_shape": [int(src.width), int(src.height)],
                        "full_source_bytes": int(total_file_size),
                        "window": {
                            "col_off": int(col_off),
                            "row_off": int(row_off),
                            "width": int(w_px),
                            "height": int(h_px)
                        },
                        "is_geotiff": True,
                        "georeferenced": True,
                        "retrieval_method": "http_range_windowed_cog",
                        "elapsed_seconds": elapsed,
                        "transfer_telemetry": {
                            "cumulative_bytes": cumulative_bytes,
                            "budget_bytes": budget,
                            "probe_bytes": probe_bytes,
                            "read_bytes": expected_read_bytes,
                            "elapsed_seconds": elapsed
                        },
                        "resolution": [float(src.res[0]), float(src.res[1])] if hasattr(src, "res") and src.res else None,
                        "tags": dict(src.tags()) if hasattr(src, "tags") else {}
                    }

                    return data, meta

            except WindowedRetrievalError:
                raise
            except Exception as e:
                raise WindowedRetrievalError(f"Error during windowed COG read: {e}")
