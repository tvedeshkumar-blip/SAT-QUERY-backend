"""
Sample GeoTIFF Generator for SatQuery AI
Generates genuine multi-band GeoTIFF test rasters using Rasterio with authentic CRS and transforms.
Validates with rasterio.open() to guarantee dataset.crs != None.

CRITICAL LABELING: These files are strictly SYNTHETIC DEMONSTRATION DATA.
They simulate sensor characteristics for local pipeline testing and must never be portrayed
as actual raw Cartosat-2S or RISAT-1A telemetry.
"""

import os
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS

def create_geotiff_rasterio(
    filepath: str,
    width: int = 256,
    height: int = 256,
    channels: int = 3,
    generator_func = None,
    pixel_size: float = 2.5,
    origin_x: float = 432000.0,
    origin_y: float = 1420000.0,
    epsg_code: int = 32644,
    scene_label: str = "Synthetic Demonstration Scene"
):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if generator_func:
        data = generator_func(height, width, channels)
    else:
        data = np.random.randint(40, 220, (height, width, channels), dtype=np.uint8)

    # rasterio expects (channels, height, width)
    if data.ndim == 2:
        raster_data = np.expand_dims(data, axis=0)
        channels = 1
    elif data.ndim == 3 and data.shape[2] <= 4:
        raster_data = np.transpose(data, (2, 0, 1))
    else:
        raster_data = data

    transform = from_origin(origin_x, origin_y, pixel_size, pixel_size)
    crs = CRS.from_epsg(epsg_code)

    with rasterio.open(
        filepath,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=channels,
        dtype=raster_data.dtype,
        crs=crs,
        transform=transform,
        nodata=0
    ) as dst:
        dst.write(raster_data)
        dst.update_tags(
            DATASET_TYPE="SYNTHETIC_DEMONSTRATION_DATA",
            SCENE_LABEL=scene_label,
            IS_AUTHENTIC_RAW_TELEMETRY="FALSE"
        )

    # Validate output using rasterio.open()
    with rasterio.open(filepath) as validated_ds:
        assert validated_ds.crs is not None, f"CRS generation failed for {filepath}"
        assert validated_ds.transform is not None, f"Transform generation failed for {filepath}"
        print(f"Verified GeoTIFF: {os.path.basename(filepath)} | CRS: {validated_ds.crs} | Shape: {validated_ds.shape} | [{scene_label}]")

def generate_optical_urban(h, w, c):
    # Procedural synthetic urban grid with vegetation and river
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = [70, 110, 60]  # Base scrub vegetation
    # Road network
    img[h//4:h//4+8, :] = [110, 115, 120]
    img[3*h//4:3*h//4+8, :] = [110, 115, 120]
    img[:, w//2:w//2+8] = [110, 115, 120]
    # Building cluster markers
    for y in range(20, h - 30, 45):
        for x in range(20, w - 30, 45):
            img[y:y+25, x:x+25] = [190, 180, 170]
    # Water curve
    for y in range(h):
        for x in range(w):
            if (x - 60)**2 + (y - 180)**2 < 35**2:
                img[y, x] = [20, 65, 120]
    return img

def generate_sar_flood(h, w, c):
    # Simulated C-band SAR backscatter with speckle simulation
    base = np.random.gamma(shape=2.0, scale=35.0, size=(h, w)).clip(0, 255).astype(np.uint8)
    # Specular smooth water (low backscatter)
    for y in range(h):
        for x in range(w):
            if (x - 60)**2 + (y - 180)**2 < 38**2:
                base[y, x] = max(1, min(255, int(np.random.normal(8, 3))))
    # Urban corner reflectors
    for y in range(20, h - 30, 45):
        for x in range(20, w - 30, 45):
            base[y:y+8, x:x+8] = 250
    return np.stack([base, base, base], axis=-1)

def generate_temporal_t1(h, w, c):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = [140, 120, 80]
    for y in range(h):
        x = int(w/2 + 20 * np.sin(y / 30.0))
        img[y, max(0, x-4):min(w, x+4)] = [90, 80, 60]
    return img

def generate_temporal_t2(h, w, c):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = [80, 130, 70]
    for y in range(h):
        x = int(w/2 + 20 * np.sin(y / 30.0))
        img[y, max(0, x-28):min(w, x+28)] = [25, 75, 135]
    return img

def generate_all_samples():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "samples"))
    
    # 1. Cartosat-style synthetic optical scene
    create_geotiff_rasterio(
        os.path.join(base_dir, "cartosat_optical_bengaluru.tif"),
        generator_func=generate_optical_urban,
        pixel_size=2.5,
        origin_x=432100.0,
        origin_y=1421000.0,
        epsg_code=32644,
        scene_label="Cartosat-style synthetic optical scene"
    )

    # 2. RISAT-style synthetic SAR scene
    create_geotiff_rasterio(
        os.path.join(base_dir, "risat_sar_mumbai.tif"),
        generator_func=generate_sar_flood,
        pixel_size=3.0,
        origin_x=280500.0,
        origin_y=2101000.0,
        epsg_code=32643,
        scene_label="RISAT-style synthetic SAR scene"
    )

    # 3. Bi-temporal T1 synthetic scene
    create_geotiff_rasterio(
        os.path.join(base_dir, "temporal_t1_pre_monsoon.tif"),
        generator_func=generate_temporal_t1,
        pixel_size=10.0,
        origin_x=720000.0,
        origin_y=1850000.0,
        epsg_code=32643,
        scene_label="Bi-temporal T1 synthetic demonstration scene"
    )

    # 4. Bi-temporal T2 synthetic scene
    create_geotiff_rasterio(
        os.path.join(base_dir, "temporal_t2_post_monsoon.tif"),
        generator_func=generate_temporal_t2,
        pixel_size=10.0,
        origin_x=720000.0,
        origin_y=1850000.0,
        epsg_code=32643,
        scene_label="Bi-temporal T2 synthetic demonstration scene"
    )

    print("All synthetic demonstration GeoTIFFs generated and verified with Rasterio in data/samples/")

if __name__ == "__main__":
    generate_all_samples()
