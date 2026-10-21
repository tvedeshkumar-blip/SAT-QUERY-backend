"""
Targeted test suite for Step 1: Satellite Asset Retrieval Safeguards.

Tests all Step 1 requirements:
1. Preserves 35 MiB full-download safety limit.
2. Uses existing AOI windowed retrieval workflow for large COG assets when AOI is provided.
3. Reports precise blocker when large COG asset cannot be retrieved without an AOI.
4. HTTP Range and Content-Range behavior:
   - Rejects HTTP 416 (Range Not Satisfiable) safely.
   - Rejects HTTP 501 / 500 range failures safely.
   - Rejects HTTP 206 missing Content-Range header.
   - Rejects HTTP 206 with malformed Content-Range header.
   - Accepts valid HTTP 206 with RFC 7233 compliant Content-Range.
5. Truncation and empty download payload detection.
6. Payload validation failure distinction: corrupt or invalid rasters return status="error".
"""

import os
import sys
import io
import base64

backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import httpx

from app.schemas.acquisition import (
    STACRetrieveRequest,
    STACWindowedRetrieveRequest
)
from app.acquisition.stac_service import STACAcquisitionService, stac_service
from app.acquisition.windowed_reader import SafeWindowedCOGReader, WindowedRetrievalError
from app.acquisition.asset_validator import AssetValidator


# ==============================================================================
# Requirement 1 & 2: Large COG & 35 MiB Full-Download Safety Limit
# ==============================================================================

def test_large_cog_with_aoi_transitions_to_windowed_retrieval():
    """
    Verifies that a COG asset exceeding 35 MiB automatically delegates to the
    existing AOI windowed retrieval workflow when an AOI bounding box is provided.
    """
    service = STACAcquisitionService()
    service.MAX_DOWNLOAD_BYTES = 35 * 1024 * 1024

    service._scene_cache["test_s2_scene"] = {
        "collection": "sentinel-2-l2a",
        "datetime": "2024-03-20T00:00:00Z",
        "raw_assets": {
            "visual": {
                "href": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/test_visual.tif",
                "type": "image/tiff; application=geotiff; profile=cloud-optimized",
                "file:size": 52 * 1024 * 1024  # 52 MB (> 35 MB)
            }
        },
        "search_bbox": [77.58, 12.96, 77.60, 12.98]
    }

    req = STACRetrieveRequest(
        scene_id="test_s2_scene",
        asset_key="visual",
        aoi_bbox=[77.58, 12.96, 77.60, 12.98],
        collection="sentinel-2-l2a"
    )

    with patch.object(service, "retrieve_window") as mock_window:
        mock_window.return_value = MagicMock(status="success", scene_id="test_s2_scene", asset_key="visual")
        resp = service.retrieve(req)

        assert mock_window.called
        call_arg = mock_window.call_args[0][0]
        assert isinstance(call_arg, STACWindowedRetrieveRequest)
        assert call_arg.scene_id == "test_s2_scene"
        assert call_arg.asset_key == "visual"
        assert call_arg.aoi_bbox == [77.58, 12.96, 77.60, 12.98]
        assert resp.status == "success"


def test_large_cog_without_aoi_preserves_safety_limit_and_reports_blocker():
    """
    Verifies that a COG asset exceeding 35 MiB without any AOI is safely blocked
    and returns an error explaining that bounded sub-windowing requires an AOI.
    """
    service = STACAcquisitionService()
    service.MAX_DOWNLOAD_BYTES = 35 * 1024 * 1024

    service._scene_cache["test_s2_scene_no_aoi"] = {
        "collection": "sentinel-2-l2a",
        "datetime": "2024-03-20T00:00:00Z",
        "raw_assets": {
            "visual": {
                "href": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/test_visual.tif",
                "type": "image/tiff",
                "file:size": 75 * 1024 * 1024  # 75 MB
            }
        }
        # Note: No search_bbox in cache
    }

    req = STACRetrieveRequest(
        scene_id="test_s2_scene_no_aoi",
        asset_key="visual",
        aoi_bbox=None,  # No AOI provided
        collection="sentinel-2-l2a"
    )

    resp = service.retrieve(req)
    assert resp.status == "error"
    assert "exceeds the configured retrieval safety limit of 35MB" in resp.error
    assert "Full-tile Sentinel COG rasters require bounded sub-windowing" in resp.error


# ==============================================================================
# Requirement 3: HTTP Range and Content-Range Verification
# ==============================================================================

def test_http_range_416_not_satisfiable_rejected_safely():
    """Ensures server returning HTTP 416 (Range Not Satisfiable) raises WindowedRetrievalError."""
    test_url = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/test.tif"

    mock_head = MagicMock()
    mock_head.status_code = 200
    mock_head.headers = {"content-length": "1000000", "accept-ranges": "bytes"}

    mock_range = MagicMock()
    mock_range.status_code = 416  # Range Not Satisfiable

    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.head.return_value = mock_head
    mock_client.stream.return_value.__enter__.return_value = mock_range

    with patch("httpx.Client", return_value=mock_client):
        with pytest.raises(WindowedRetrievalError, match="HTTP 416"):
            SafeWindowedCOGReader.read_cog_window(
                url=test_url,
                bbox_wgs84=[77.58, 12.96, 77.60, 12.98]
            )


def test_http_range_server_error_rejected_safely():
    """Ensures server returning HTTP 501 / 500 on range request raises WindowedRetrievalError."""
    test_url = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/test.tif"

    mock_head = MagicMock()
    mock_head.status_code = 200
    mock_head.headers = {"content-length": "1000000", "accept-ranges": "bytes"}

    mock_range = MagicMock()
    mock_range.status_code = 501  # Not Implemented

    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.head.return_value = mock_head
    mock_client.stream.return_value.__enter__.return_value = mock_range

    with patch("httpx.Client", return_value=mock_client):
        with pytest.raises(WindowedRetrievalError, match="HTTP 501"):
            SafeWindowedCOGReader.read_cog_window(
                url=test_url,
                bbox_wgs84=[77.58, 12.96, 77.60, 12.98]
            )


def test_http_range_missing_content_range_header_rejected():
    """Ensures HTTP 206 response omitting Content-Range header is rejected."""
    test_url = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/test.tif"

    mock_head = MagicMock()
    mock_head.status_code = 200
    mock_head.headers = {"content-length": "1000000", "accept-ranges": "bytes"}

    mock_range = MagicMock()
    mock_range.status_code = 206
    # Dictionary without Content-Range header
    mock_range.headers = {"content-length": "1024"}

    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.head.return_value = mock_head
    mock_client.stream.return_value.__enter__.return_value = mock_range

    with patch("httpx.Client", return_value=mock_client):
        with pytest.raises(WindowedRetrievalError, match="omitted mandatory 'Content-Range' response header"):
            SafeWindowedCOGReader.read_cog_window(
                url=test_url,
                bbox_wgs84=[77.58, 12.96, 77.60, 12.98]
            )


def test_http_range_malformed_content_range_header_rejected():
    """Ensures HTTP 206 response with malformed Content-Range header is rejected."""
    test_url = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/test.tif"

    mock_head = MagicMock()
    mock_head.status_code = 200
    mock_head.headers = {"content-length": "1000000", "accept-ranges": "bytes"}

    mock_range = MagicMock()
    mock_range.status_code = 206
    mock_range.headers = {"content-range": "items 0-100/500"}  # Invalid range unit (items instead of bytes)

    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.head.return_value = mock_head
    mock_client.stream.return_value.__enter__.return_value = mock_range

    with patch("httpx.Client", return_value=mock_client):
        with pytest.raises(WindowedRetrievalError, match="invalid 'Content-Range' header format"):
            SafeWindowedCOGReader.read_cog_window(
                url=test_url,
                bbox_wgs84=[77.58, 12.96, 77.60, 12.98]
            )


# ==============================================================================
# Requirement 4: Truncation, Empty Payloads & Validation Status Distinction
# ==============================================================================

def test_download_truncated_payload_detected_and_rejected():
    """Ensures stream ending with total_bytes < Content-Length is flagged as error."""
    service = STACAcquisitionService()

    service._scene_cache["test_scene_trunc"] = {
        "collection": "sentinel-2-l2a",
        "datetime": "2024-03-20T00:00:00Z",
        "raw_assets": {
            "thumbnail": {
                "href": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/thumb.jpg",
                "type": "image/jpeg"
            }
        }
    }

    req = STACRetrieveRequest(
        scene_id="test_scene_trunc",
        asset_key="thumbnail",
        collection="sentinel-2-l2a"
    )

    mock_stream = MagicMock()
    mock_stream.status_code = 200
    mock_stream.headers = {"content-length": "50000"}  # Server claims 50,000 bytes
    mock_stream.iter_bytes.return_value = [b"truncated_data_100_bytes" * 5]  # Only ~125 bytes delivered

    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.stream.return_value.__enter__.return_value = mock_stream

    with patch("httpx.Client", return_value=mock_client):
        resp = service.retrieve(req)
        assert resp.status == "error"
        assert "Asset download truncated" in resp.error


def test_download_empty_payload_detected_and_rejected():
    """Ensures stream delivering 0 bytes is flagged as error."""
    service = STACAcquisitionService()

    service._scene_cache["test_scene_empty"] = {
        "collection": "sentinel-2-l2a",
        "datetime": "2024-03-20T00:00:00Z",
        "raw_assets": {
            "thumbnail": {
                "href": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/thumb.jpg",
                "type": "image/jpeg"
            }
        }
    }

    req = STACRetrieveRequest(
        scene_id="test_scene_empty",
        asset_key="thumbnail",
        collection="sentinel-2-l2a"
    )

    mock_stream = MagicMock()
    mock_stream.status_code = 200
    mock_stream.headers = {}
    mock_stream.iter_bytes.return_value = []  # 0 bytes

    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.stream.return_value.__enter__.return_value = mock_stream

    with patch("httpx.Client", return_value=mock_client):
        resp = service.retrieve(req)
        assert resp.status == "error"
        assert "empty payload (0 bytes received)" in resp.error


def test_download_corrupt_payload_validation_fails_and_returns_error_status():
    """
    Ensures that when a downloaded payload is corrupted or invalid (e.g. random garbage bytes),
    physical asset validation marks it invalid and retrieve returns status="error" rather than success.
    """
    service = STACAcquisitionService()

    service._scene_cache["test_scene_corrupt"] = {
        "collection": "sentinel-2-l2a",
        "datetime": "2024-03-20T00:00:00Z",
        "raw_assets": {
            "visual": {
                "href": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/corrupt.tif",
                "type": "image/tiff"
            }
        }
    }

    req = STACRetrieveRequest(
        scene_id="test_scene_corrupt",
        asset_key="visual",
        collection="sentinel-2-l2a"
    )

    # 512 bytes of completely corrupted garbage (not a GeoTIFF, not a JPEG)
    corrupted_bytes = b"CORRUPTED_NON_RASTER_DATA_" * 20

    mock_stream = MagicMock()
    mock_stream.status_code = 200
    mock_stream.headers = {"content-length": str(len(corrupted_bytes))}
    mock_stream.iter_bytes.return_value = [corrupted_bytes]

    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.stream.return_value.__enter__.return_value = mock_stream

    with patch("httpx.Client", return_value=mock_client):
        resp = service.retrieve(req)
        assert resp.status == "error"
        assert resp.data_base64 == ""
        assert "Retrieved asset payload validation failed" in resp.error
        assert resp.validation is not None
        assert resp.validation.status == "invalid"


def test_http_range_probe_streaming_200_rejected_without_consuming_body():
    """
    Step 1A Regression Test:
    Proves that an HTTP 200 response to a range probe is rejected immediately
    via streaming probe without consuming or buffering the response body.
    """
    test_url = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/large_cog.tif"

    mock_head = MagicMock()
    mock_head.status_code = 200
    mock_head.headers = {"content-length": "250000000", "accept-ranges": "bytes"}

    mock_stream = MagicMock()
    mock_stream.status_code = 200  # Server ignored Range header and attempted full transfer
    mock_stream.read = MagicMock()
    mock_stream.iter_bytes = MagicMock()

    mock_stream_cm = MagicMock()
    mock_stream_cm.__enter__.return_value = mock_stream

    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.head.return_value = mock_head
    mock_client.stream.return_value = mock_stream_cm

    with patch("httpx.Client", return_value=mock_client):
        with pytest.raises(WindowedRetrievalError, match="returned HTTP 200 instead of 206"):
            SafeWindowedCOGReader.read_cog_window(
                url=test_url,
                bbox_wgs84=[77.58, 12.96, 77.60, 12.98]
            )

        # Assert stream was opened with GET and Range header
        assert mock_client.stream.called
        call_args = mock_client.stream.call_args
        assert call_args[0][0] == "GET"
        assert call_args[1]["headers"] == {"Range": "bytes=0-1023"}

        # PROOF: Response body was NEVER consumed or read
        assert not mock_stream.read.called
        assert not mock_stream.iter_bytes.called

        # PROOF: Stream was closed immediately
        assert mock_stream_cm.__exit__.called


def test_oversized_http_206_response_rejected_during_streaming_before_full_consumption():
    """
    Step 1B Regression Test:
    Verifies that when a remote server returns HTTP 206 with an oversized payload
    (> 64 KiB), the streaming reader aborts immediately during chunk iteration
    BEFORE consuming or buffering the full response body.
    """
    test_url = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/large_cog.tif"

    mock_head = MagicMock()
    mock_head.status_code = 200
    mock_head.headers = {"content-length": "250000000", "accept-ranges": "bytes"}

    mock_stream = MagicMock()
    mock_stream.status_code = 206
    mock_stream.headers = {"content-range": "bytes 0-1023/250000000"}

    # Track how many chunks are actually generated/yielded by the stream
    chunks_yielded = []

    def chunk_generator(chunk_size=1024):
        # Server attempts to send 200 chunks of 1024 bytes (total 200 KiB > 64 KiB cap)
        for i in range(200):
            chunks_yielded.append(i)
            yield b"X" * 1024

    mock_stream.iter_bytes = chunk_generator

    mock_stream_cm = MagicMock()
    mock_stream_cm.__enter__.return_value = mock_stream

    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.head.return_value = mock_head
    mock_client.stream.return_value = mock_stream_cm

    with patch("httpx.Client", return_value=mock_client):
        with pytest.raises(WindowedRetrievalError, match="excessively large payload"):
            SafeWindowedCOGReader.read_cog_window(
                url=test_url,
                bbox_wgs84=[77.58, 12.96, 77.60, 12.98]
            )

        # PROOF OF EARLY TERMINATION:
        # The generator must have been stopped after exceeding 64 KiB (65 chunks * 1024 = 66560 bytes).
        # It must NOT have consumed all 200 chunks!
        assert len(chunks_yielded) < 200, "Stream was NOT terminated early; full oversized body was consumed!"
        assert len(chunks_yielded) == 65, f"Expected abort on chunk 65 (>64 KiB), got {len(chunks_yielded)}"

        # PROOF: Stream context manager was closed immediately
        assert mock_stream_cm.__exit__.called


def test_http_range_probe_valid_206_with_content_range_succeeds():
    """
    Step 1B Regression Test:
    Verifies that a valid HTTP 206 response with an RFC 7233 compliant Content-Range
    header is accepted and the bounded probe payload is read safely.
    """
    test_url = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/test_cog.tif"

    mock_head = MagicMock()
    mock_head.status_code = 200
    mock_head.headers = {"content-length": "10000000", "accept-ranges": "bytes"}

    mock_stream = MagicMock()
    mock_stream.status_code = 206
    mock_stream.headers = {"content-range": "bytes 0-1023/10000000"}
    mock_stream.iter_bytes = MagicMock(return_value=[b"R" * 1024])

    mock_stream_cm = MagicMock()
    mock_stream_cm.__enter__.return_value = mock_stream

    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.head.return_value = mock_head
    mock_client.stream.return_value = mock_stream_cm

    # Mock rasterio DatasetReader to avoid actual vsicurl network access in this unit test
    import rasterio
    from rasterio.transform import Affine

    mock_ds = MagicMock()
    mock_ds.__enter__.return_value = mock_ds
    mock_ds.crs = "EPSG:4326"
    mock_ds.width = 512
    mock_ds.height = 512
    mock_ds.count = 1
    mock_ds.dtypes = ["uint16"]
    mock_ds.nodata = 0
    mock_ds.bounds = MagicMock(left=77.5, bottom=12.9, right=77.7, top=13.1)
    mock_ds.transform = Affine(0.0002, 0.0, 77.5, 0.0, -0.0002, 13.1)
    mock_ds.read.return_value = np.zeros((1, 100, 100), dtype=np.uint16)

    with patch("httpx.Client", return_value=mock_client), \
         patch("rasterio.open", return_value=mock_ds):
        data, meta = SafeWindowedCOGReader.read_cog_window(
            url=test_url,
            bbox_wgs84=[77.58, 12.96, 77.60, 12.98]
        )

        assert mock_client.stream.called
        assert mock_stream.iter_bytes.called
        assert mock_stream_cm.__exit__.called
        assert meta["crs"] == "EPSG:4326"
        assert meta["retrieval_method"] == "http_range_windowed_cog"


# ==============================================================================
# Step 2: Explicit AOI Propagation and Stateless Retrieval Tests
# ==============================================================================

def test_step2_explicit_aoi_dispatches_to_windowed_retrieval_with_exact_aoi():
    """
    Step 2 Requirement 1:
    Verifies that an oversized eligible COG asset with an explicitly supplied request AOI
    delegates to retrieve_window() using that exact AOI, ignoring any different search_bbox in cache.
    """
    service = STACAcquisitionService()
    service.MAX_DOWNLOAD_BYTES = 35 * 1024 * 1024

    cached_aoi = [10.0, 20.0, 11.0, 21.0]
    request_aoi = [77.58, 12.96, 77.60, 12.98]

    service._scene_cache["scene_explicit_aoi"] = {
        "collection": "sentinel-2-l2a",
        "datetime": "2024-03-20T00:00:00Z",
        "raw_assets": {
            "visual": {
                "href": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/visual.tif",
                "type": "image/tiff; application=geotiff; profile=cloud-optimized",
                "file:size": 65 * 1024 * 1024  # 65 MB (> 35 MB)
            }
        },
        "search_bbox": cached_aoi  # Stale/earlier search bounding box
    }

    req = STACRetrieveRequest(
        scene_id="scene_explicit_aoi",
        asset_key="visual",
        aoi_bbox=request_aoi,
        collection="sentinel-2-l2a"
    )

    mock_resp = MagicMock()
    mock_resp.status = "success"
    mock_resp.scene_id = "scene_explicit_aoi"
    mock_resp.asset_key = "visual"

    with patch.object(service, "retrieve_window", return_value=mock_resp) as mock_win:
        resp = service.retrieve(req)
        assert resp == mock_resp
        mock_win.assert_called_once()
        win_arg: STACWindowedRetrieveRequest = mock_win.call_args[0][0]
        # Crucial check: Must match the explicit request AOI, NOT the cached search_bbox
        assert win_arg.aoi_bbox == request_aoi
        assert win_arg.aoi_bbox != cached_aoi


def test_step2_no_aoi_does_not_inherit_cached_search_bbox():
    """
    Step 2 Requirement 2:
    Verifies that if an earlier search populated search_bbox in the shared cache,
    an oversized retrieval request WITHOUT aoi_bbox fails clearly rather than
    silently inheriting the cached search_bbox.
    """
    service = STACAcquisitionService()
    service.MAX_DOWNLOAD_BYTES = 35 * 1024 * 1024

    cached_aoi = [10.0, 20.0, 11.0, 21.0]

    service._scene_cache["scene_no_aoi_cached"] = {
        "collection": "sentinel-2-l2a",
        "datetime": "2024-03-20T00:00:00Z",
        "raw_assets": {
            "visual": {
                "href": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/visual.tif",
                "type": "image/tiff; application=geotiff; profile=cloud-optimized",
                "file:size": 80 * 1024 * 1024  # 80 MB (> 35 MB)
            }
        },
        "search_bbox": cached_aoi  # Cached from prior query
    }

    req = STACRetrieveRequest(
        scene_id="scene_no_aoi_cached",
        asset_key="visual",
        aoi_bbox=None,  # No AOI supplied on this request
        collection="sentinel-2-l2a"
    )

    with patch.object(service, "retrieve_window") as mock_win:
        resp = service.retrieve(req)
        # Windowed retrieval must NOT be triggered
        mock_win.assert_not_called()
        assert resp.status == "error"
        assert "exceeds the configured retrieval safety limit of 35MB" in resp.error
        assert "Full-tile Sentinel COG rasters require bounded sub-windowing" in resp.error


def test_step2_request_aoi_isolation_between_distinct_retrievals():
    """
    Step 2 Requirement 3:
    Verifies that two retrieval requests for the same scene with different AOIs
    each strictly receive their own request-specific AOI without cross-talk or mutation.
    """
    service = STACAcquisitionService()
    service.MAX_DOWNLOAD_BYTES = 35 * 1024 * 1024

    service._scene_cache["scene_shared"] = {
        "collection": "sentinel-2-l2a",
        "datetime": "2024-03-20T00:00:00Z",
        "raw_assets": {
            "b04": {
                "href": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/b04.tif",
                "type": "image/tiff; application=geotiff; profile=cloud-optimized",
                "file:size": 50 * 1024 * 1024
            }
        },
        "search_bbox": [0.0, 0.0, 1.0, 1.0]  # Initial search bbox
    }

    aoi_user_a = [77.58, 12.96, 77.60, 12.98]
    aoi_user_b = [78.10, 13.20, 78.15, 13.25]

    req_a = STACRetrieveRequest(
        scene_id="scene_shared",
        asset_key="b04",
        aoi_bbox=aoi_user_a,
        collection="sentinel-2-l2a"
    )

    req_b = STACRetrieveRequest(
        scene_id="scene_shared",
        asset_key="b04",
        aoi_bbox=aoi_user_b,
        collection="sentinel-2-l2a"
    )

    captured_aois = []
    def fake_retrieve_window(win_req: STACWindowedRetrieveRequest):
        captured_aois.append(list(win_req.aoi_bbox))
        m = MagicMock()
        m.status = "success"
        return m

    with patch.object(service, "retrieve_window", side_effect=fake_retrieve_window):
        service.retrieve(req_a)
        service.retrieve(req_b)

        assert len(captured_aois) == 2
        assert captured_aois[0] == aoi_user_a
        assert captured_aois[1] == aoi_user_b
        # Ensure neither request fell back to or altered the cache's search_bbox
        assert service._scene_cache["scene_shared"]["search_bbox"] == [0.0, 0.0, 1.0, 1.0]


def test_step2_invalid_or_malformed_aoi_rejected_by_schema():
    """
    Step 2 Requirement 4:
    Verifies that malformed or non-compliant AOI bounds are strictly rejected
    by the Pydantic schema validation contract.
    """
    from pydantic import ValidationError

    # Case 1: Wrong coordinate length (needs 4)
    with pytest.raises(ValidationError, match="aoi_bbox must have exactly 4 coordinates"):
        STACRetrieveRequest(
            scene_id="scene_val",
            asset_key="visual",
            aoi_bbox=[77.5, 12.9, 77.6]
        )

    # Case 2: Inverted latitude (min_lat >= max_lat)
    with pytest.raises(ValidationError, match="min_lat cannot be greater than or equal to max_lat"):
        STACRetrieveRequest(
            scene_id="scene_val",
            asset_key="visual",
            aoi_bbox=[77.5, 13.0, 77.6, 12.9]
        )

    # Case 3: Out of range latitude (> 90 degrees)
    with pytest.raises(ValidationError, match="Latitude values must be between -90 and 90"):
        STACRetrieveRequest(
            scene_id="scene_val",
            asset_key="visual",
            aoi_bbox=[77.5, 12.0, 77.6, 95.0]
        )

    # Case 4: Out of range longitude (> 180 degrees)
    with pytest.raises(ValidationError, match="Longitude values must be between -180 and 180"):
        STACRetrieveRequest(
            scene_id="scene_val",
            asset_key="visual",
            aoi_bbox=[77.5, 12.0, 195.0, 13.0]
        )


def test_step2_content_length_oversized_no_aoi_fails_without_cache_inheritance():
    """
    Step 2 Requirement 5:
    Verifies that when file:size is missing from metadata but HTTP Content-Length
    exceeds 35 MiB during the streaming probe, a request without an AOI halts and
    fails cleanly, rather than inheriting a cached search_bbox.
    """
    service = STACAcquisitionService()
    service.MAX_DOWNLOAD_BYTES = 35 * 1024 * 1024

    cached_aoi = [10.0, 20.0, 11.0, 21.0]
    service._scene_cache["scene_cl_check"] = {
        "collection": "sentinel-2-l2a",
        "datetime": "2024-03-20T00:00:00Z",
        "raw_assets": {
            "visual": {
                "href": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/visual.tif",
                "type": "image/tiff; application=geotiff; profile=cloud-optimized"
                # Note: No 'file:size' metadata
            }
        },
        "search_bbox": cached_aoi
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    # Server reports 55 MB Content-Length
    mock_resp.headers = {"content-length": str(55 * 1024 * 1024)}

    mock_stream_cm = MagicMock()
    mock_stream_cm.__enter__.return_value = mock_resp

    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.stream.return_value = mock_stream_cm

    req = STACRetrieveRequest(
        scene_id="scene_cl_check",
        asset_key="visual",
        aoi_bbox=None,  # No AOI
        collection="sentinel-2-l2a"
    )

    with patch("httpx.Client", return_value=mock_client), \
         patch.object(service, "retrieve_window") as mock_win:
        resp = service.retrieve(req)
        # Windowed retrieval must not be invoked
        mock_win.assert_not_called()
        assert resp.status == "error"
        assert "exceeds the configured retrieval safety limit of 35MB" in resp.error


# ==============================================================================
# Step 3: End-to-End AOI-Windowed Retrieval & Raster Validation Tests
# ==============================================================================

def test_step3_ineligible_asset_format_with_aoi_rejected_safely():
    """
    Step 3 Requirement: Test 3 - Ineligible asset format.
    Verifies that an oversized asset with an explicit AOI that is NOT eligible
    for supported COG windowed retrieval (e.g. tar.gz or zip archive) is rejected
    truthfully and not sent to the COG window reader or downloaded.
    """
    service = STACAcquisitionService()
    service.MAX_DOWNLOAD_BYTES = 35 * 1024 * 1024

    service._scene_cache["scene_ineligible_format"] = {
        "collection": "sentinel-2-l2a",
        "datetime": "2024-03-20T00:00:00Z",
        "raw_assets": {
            "full_archive": {
                "href": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/bundle.tar.gz",
                "type": "application/gzip",
                "file:size": 750 * 1024 * 1024  # 750 MB archive
            }
        }
    }

    req = STACRetrieveRequest(
        scene_id="scene_ineligible_format",
        asset_key="full_archive",
        aoi_bbox=[77.58, 12.96, 77.60, 12.98],
        collection="sentinel-2-l2a"
    )

    with patch.object(service, "retrieve_window") as mock_win, \
         patch("httpx.Client") as mock_http:
        resp = service.retrieve(req)
        # Windowed retrieval must NOT be invoked because format is not a COG candidate
        mock_win.assert_not_called()
        # Full download must NOT be attempted
        mock_http.assert_not_called()
        assert resp.status == "error"
        assert "exceeds the configured retrieval safety limit of 35MB" in resp.error


def test_step3_windowed_retrieval_produces_real_validated_raster():
    """
    Step 3 Requirement: Test 4 - Windowed retrieval produces a real raster.
    Exercises the full end-to-end windowed reader, in-memory GeoTIFF serialization,
    and physical AssetValidator inspection using a real local GeoTIFF sample.
    Verifies real raster bytes, nonzero dimensions, CRS, affine transform, and bands.
    """
    service = STACAcquisitionService()
    service.MAX_DOWNLOAD_BYTES = 35 * 1024 * 1024

    test_cog_url = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/cartosat_bengaluru.tif"
    req_aoi = [80.375, 12.848, 80.378, 12.852]  # Intersects Cartosat sample extent

    service._scene_cache["scene_real_raster_test"] = {
        "collection": "sentinel-2-l2a",
        "datetime": "2024-03-20T00:00:00Z",
        "raw_assets": {
            "visual": {
                "href": test_cog_url,
                "type": "image/tiff; application=geotiff; profile=cloud-optimized",
                "file:size": 85 * 1024 * 1024  # 85 MB (> 35 MB)
            }
        }
    }

    # Mock the HTTP range probe to succeed
    mock_head = MagicMock()
    mock_head.status_code = 200
    mock_head.headers = {"content-length": "89128960", "accept-ranges": "bytes"}

    mock_stream = MagicMock()
    mock_stream.status_code = 206
    mock_stream.headers = {"content-range": "bytes 0-1023/89128960"}
    mock_stream.iter_bytes = MagicMock(return_value=[b"G" * 1024])

    mock_stream_cm = MagicMock()
    mock_stream_cm.__enter__.return_value = mock_stream

    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.head.return_value = mock_head
    mock_client.stream.return_value = mock_stream_cm

    # Point rasterio.open directly to the real repository test raster sample
    import rasterio
    from rasterio.io import MemoryFile
    real_sample_path = os.path.abspath(os.path.join(backend_dir, "..", "data", "samples", "cartosat_optical_bengaluru.tif"))
    assert os.path.exists(real_sample_path), f"Sample raster missing at {real_sample_path}"

    real_open = rasterio.open
    def redirect_open(path, *args, **kwargs):
        if str(path).startswith("https://") or "cartosat_bengaluru.tif" in str(path):
            return real_open(real_sample_path, *args, **kwargs)
        return real_open(path, *args, **kwargs)

    req = STACRetrieveRequest(
        scene_id="scene_real_raster_test",
        asset_key="visual",
        aoi_bbox=req_aoi,
        collection="sentinel-2-l2a"
    )

    with patch("httpx.Client", return_value=mock_client), \
         patch("rasterio.open", side_effect=redirect_open):
        resp = service.retrieve(req)

        # 1. API Response & Truthfulness
        assert resp.status == "success"
        assert resp.category in ("scientific_raster", "rgb_visual_raster")
        assert resp.data_base64.startswith("data:image/tiff;base64,")
        assert resp.validation is not None
        assert resp.validation.status == "valid"

        # 2. Decode returned binary payload and verify with Rasterio
        raw_b64 = resp.data_base64.split(",", 1)[1]
        decoded_bytes = base64.b64decode(raw_b64)
        assert len(decoded_bytes) > 0

        with MemoryFile(decoded_bytes) as mem:
            with mem.open() as ds:
                assert ds.driver == "GTiff"
                assert ds.width > 0
                assert ds.height > 0
                assert ds.count == 3  # 3-band RGB Cartosat
                assert str(ds.crs) == "EPSG:32644"
                assert ds.dtypes[0] == "uint8"
                assert ds.transform is not None
                assert len(ds.transform[:6]) == 6
                # Verify bounds match intersecting subset
                assert ds.bounds.left < ds.bounds.right
                assert ds.bounds.bottom < ds.bounds.top


def test_step3_invalid_corrupt_window_data_rejected_without_fabrication():
    """
    Step 3 Requirement: Test 5 - Invalid, empty, or truncated window response.
    Verifies that when raster reading or validation fails on corrupted binary data,
    the service returns status="error" and never fabricates metadata or pixel arrays.
    """
    service = STACAcquisitionService()
    service.MAX_DOWNLOAD_BYTES = 35 * 1024 * 1024

    test_cog_url = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/corrupt_test.tif"
    service._scene_cache["scene_corrupt_test"] = {
        "collection": "sentinel-2-l2a",
        "datetime": "2024-03-20T00:00:00Z",
        "raw_assets": {
            "b04": {
                "href": test_cog_url,
                "type": "image/tiff; application=geotiff; profile=cloud-optimized",
                "file:size": 60 * 1024 * 1024
            }
        }
    }

    req = STACRetrieveRequest(
        scene_id="scene_corrupt_test",
        asset_key="b04",
        aoi_bbox=[77.58, 12.96, 77.60, 12.98],
        collection="sentinel-2-l2a"
    )

    with patch.object(SafeWindowedCOGReader, "read_cog_window", side_effect=WindowedRetrievalError("Corrupted TIFF header")):
        resp = service.retrieve(req)
        assert resp.status == "error"
        assert resp.data_base64 == ""
        assert "Windowed retrieval failed: Corrupted TIFF header" in resp.error


def test_step3_transfer_budget_exhaustion_in_windowed_retrieval_rejected():
    """
    Step 3 Requirement: Test 6 - Transfer and size protections.
    Verifies that when cumulative transfer budget is exceeded during window retrieval,
    SafeWindowedCOGReader raises WindowedRetrievalError and retrieve() returns a clean error.
    """
    service = STACAcquisitionService()
    service.MAX_DOWNLOAD_BYTES = 35 * 1024 * 1024

    test_cog_url = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/budget_test.tif"
    service._scene_cache["scene_budget_test"] = {
        "collection": "sentinel-2-l2a",
        "datetime": "2024-03-20T00:00:00Z",
        "raw_assets": {
            "b04": {
                "href": test_cog_url,
                "type": "image/tiff; application=geotiff; profile=cloud-optimized",
                "file:size": 60 * 1024 * 1024
            }
        }
    }

    req = STACRetrieveRequest(
        scene_id="scene_budget_test",
        asset_key="b04",
        aoi_bbox=[77.58, 12.96, 77.60, 12.98],
        collection="sentinel-2-l2a"
    )

    with patch.object(SafeWindowedCOGReader, "read_cog_window", side_effect=WindowedRetrievalError("Cumulative transfer limit of 20971520 bytes exceeded")):
        resp = service.retrieve(req)
        assert resp.status == "error"
        assert "Cumulative transfer limit" in resp.error
