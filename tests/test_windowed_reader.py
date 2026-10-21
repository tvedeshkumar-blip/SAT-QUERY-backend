import os
import sys

backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import pytest
import numpy as np
import httpx
from unittest.mock import patch, MagicMock

from app.acquisition.windowed_reader import (
    SafeWindowedCOGReader,
    WindowedRetrievalError
)

def test_insecure_http_protocol_rejected():
    with pytest.raises(WindowedRetrievalError, match="Insecure protocol"):
        SafeWindowedCOGReader.read_cog_window(
            url="http://sentinel-cogs.s3.us-west-2.amazonaws.com/test.tif",
            bbox_wgs84=[77.5, 12.9, 77.6, 13.0]
        )

def test_untrusted_host_rejected():
    with pytest.raises(WindowedRetrievalError, match="Security violation: host"):
        SafeWindowedCOGReader.read_cog_window(
            url="https://evil-untrusted-host.com/fake_cog.tif",
            bbox_wgs84=[77.5, 12.9, 77.6, 13.0]
        )

def test_server_without_accept_ranges_rejected():
    with patch("httpx.Client.head") as mock_head:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-length": "1000000"}  # No Accept-Ranges header
        mock_head.return_value = mock_resp

        with pytest.raises(WindowedRetrievalError, match="Accept-Ranges: bytes"):
            SafeWindowedCOGReader.read_cog_window(
                url="https://sentinel-cogs.s3.us-west-2.amazonaws.com/test.tif",
                bbox_wgs84=[77.5, 12.9, 77.6, 13.0]
            )

def test_server_returning_200_instead_of_206_aborts_immediately():
    """
    Verifies that when a remote server ignores the Range request and returns HTTP 200,
    the streaming probe rejects the response and closes the stream without reading/consuming the body.
    """
    mock_head_resp = MagicMock()
    mock_head_resp.status_code = 200
    mock_head_resp.headers = {"accept-ranges": "bytes", "content-length": "100000000"}  # 100 MB file

    mock_stream_resp = MagicMock()
    mock_stream_resp.status_code = 200  # Server ignores Range: bytes=0-1023
    mock_stream_resp.headers = {"content-length": "100000000"}
    mock_stream_resp.read = MagicMock()
    mock_stream_resp.iter_bytes = MagicMock()

    mock_stream_cm = MagicMock()
    mock_stream_cm.__enter__.return_value = mock_stream_resp

    with patch("httpx.Client.head", return_value=mock_head_resp), \
         patch("httpx.Client.stream", return_value=mock_stream_cm) as mock_stream_call:

        with pytest.raises(WindowedRetrievalError, match="returned HTTP 200 instead of 206"):
            SafeWindowedCOGReader.read_cog_window(
                url="https://sentinel-cogs.s3.us-west-2.amazonaws.com/test.tif",
                bbox_wgs84=[77.5, 12.9, 77.6, 13.0]
            )

        # PROOF: Verify streaming probe was invoked
        assert mock_stream_call.called

        # PROOF: Verify response body was NOT consumed or read
        assert not mock_stream_resp.read.called
        assert not mock_stream_resp.iter_bytes.called

        # PROOF: Verify stream context manager was properly exited/closed
        assert mock_stream_cm.__exit__.called

def test_live_s2_scientific_b04_window_retrieval():
    """
    Demonstrates physical windowed reading of live Sentinel-2 B04 (red surface reflectance)
    without downloading the 235 MB full file.
    """
    s2_b04_url = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/43/P/GQ/2024/3/S2A_43PGQ_20240328_0_L2A/B04.tif"
    aoi_bengaluru = [77.58, 12.96, 77.60, 12.98]  # ~2km x 2km window

    data, meta = SafeWindowedCOGReader.read_cog_window(
        url=s2_b04_url,
        bbox_wgs84=aoi_bengaluru,
        max_bytes=25 * 1024 * 1024
    )

    # 1. Scientific Dtype & Values: uint16 BOA reflectance
    assert data.dtype == np.uint16
    assert data.ndim == 2
    assert meta["dtype"] == "uint16"
    assert meta["bands"] == 1

    # 2. Georeferencing preserved
    assert meta["crs"] == "EPSG:32643"
    assert meta["georeferenced"] is True
    assert len(meta["transform"]) == 6
    assert meta["transform"][0] == 10.0  # 10m pixel width
    assert meta["transform"][4] == -10.0  # 10m pixel height

    # 3. Bounded dimensions
    assert meta["width"] <= 1024
    assert meta["height"] <= 1024
    assert meta["width"] > 0
    assert meta["height"] > 0

    # 4. Provenance
    assert meta["retrieval_method"] == "http_range_windowed_cog"
    assert meta["full_source_bytes"] > 200 * 1024 * 1024  # Source is ~235 MB
    assert meta["elapsed_seconds"] < 15.0  # Finished within timeout budget


def test_live_s1_rtc_scientific_vv_window_retrieval():
    """
    Demonstrates physical windowed reading of live Sentinel-1 RTC (terrain-flattened gamma-naught backscatter)
    without downloading the 1.86 GB full scene.
    """
    token_url = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-1-rtc"
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(token_url)
            if resp.status_code != 200:
                pytest.skip("Planetary Computer SAS token endpoint unavailable")
            token = resp.json().get("token")
    except Exception as e:
        pytest.skip(f"Network error accessing Planetary Computer token API: {e}")

    raw_href = "https://sentinel1euwestrtc.blob.core.windows.net/sentinel1-grd-rtc/GRD/2024/3/22/IW/DV/S1A_IW_GRDH_1SDV_20240322T004029_20240322T004054_053087_066E0A_9C2C/measurement/iw-vv.rtc.tiff"
    signed_url = f"{raw_href}?{token}"
    aoi_bengaluru = [77.58, 12.96, 77.60, 12.98]  # ~2km x 2km window

    data, meta = SafeWindowedCOGReader.read_cog_window(
        url=signed_url,
        bbox_wgs84=aoi_bengaluru,
        max_bytes=20 * 1024 * 1024
    )

    # 1. Scientific Dtype & Values: float32 linear power backscatter
    assert data.dtype == np.float32
    assert data.ndim == 2
    assert meta["dtype"] == "float32"
    assert meta["bands"] == 1
    assert meta["nodata"] == -32768.0

    # 2. Georeferencing preserved
    assert meta["crs"] == "EPSG:32643"
    assert meta["georeferenced"] is True
    assert len(meta["transform"]) == 6
    assert meta["transform"][0] == 10.0   # 10m pixel width
    assert meta["transform"][4] == -10.0  # 10m pixel height

    # 3. Valid pixels check: backscatter values are genuine positive power
    valid_mask = data != meta["nodata"]
    assert np.count_nonzero(valid_mask) > 0
    assert float(np.min(data[valid_mask])) > 0.0

    # 4. Bounded dimensions
    assert meta["width"] <= 1024
    assert meta["height"] <= 1024
    assert meta["width"] > 0
    assert meta["height"] > 0

    # 5. Provenance & Budget
    assert meta["retrieval_method"] == "http_range_windowed_cog"
    assert meta["full_source_bytes"] > 1_500_000_000  # Source is ~1.86 GB
    assert meta["elapsed_seconds"] < 15.0


def test_s1_rtc_and_s2_b04_spatial_pair_alignment():
    """
    Verifies that real Sentinel-2 B04 and Sentinel-1 RTC windows acquired over the same AOI
    share identical CRS and transform grid, passing pair validation co-registration gates
    without synthetic resizing or missing-CRS inference.
    """
    from app.remote_sensing.registration import align_image_pair

    # 1. Acquire S2 B04 window
    s2_url = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/43/P/GQ/2024/3/S2A_43PGQ_20240328_0_L2A/B04.tif"
    aoi_bengaluru = [77.58, 12.96, 77.60, 12.98]
    s2_data, s2_meta = SafeWindowedCOGReader.read_cog_window(url=s2_url, bbox_wgs84=aoi_bengaluru)

    # 2. Acquire S1 RTC VV window
    token_url = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-1-rtc"
    try:
        with httpx.Client(timeout=10.0) as client:
            token = client.get(token_url).json().get("token")
    except Exception as e:
        pytest.skip(f"Token service unreachable: {e}")

    s1_raw = "https://sentinel1euwestrtc.blob.core.windows.net/sentinel1-grd-rtc/GRD/2024/3/22/IW/DV/S1A_IW_GRDH_1SDV_20240322T004029_20240322T004054_053087_066E0A_9C2C/measurement/iw-vv.rtc.tiff"
    s1_url = f"{s1_raw}?{token}"
    s1_data, s1_meta = SafeWindowedCOGReader.read_cog_window(url=s1_url, bbox_wgs84=aoi_bengaluru)

    # 3. Align the pair using standard registration engine
    aligned_s2, aligned_s1, reg_info = align_image_pair(s2_data, s1_data, s2_meta, s1_meta)

    # 4. Strict physical pair validation gates
    assert reg_info["registration_method"] == "identical_grid_co_registered"
    assert reg_info["co_registered"] is True
    assert reg_info["crs_matched"] is True
    assert reg_info["primary_crs"] == "EPSG:32643"
    assert reg_info["secondary_crs"] == "EPSG:32643"
    assert reg_info["spatial_overlap_percent"] == 100.0
    assert reg_info["is_display_only"] is False
    assert reg_info["valid_pixel_count"] > 0
    assert reg_info["valid_pixel_count"] == aligned_s2.shape[0] * aligned_s2.shape[1]
    assert aligned_s2.shape == aligned_s1.shape

