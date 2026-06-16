#!/usr/bin/env python3
"""
swdi_era5land.py
================
Reproduce the Soil Water Deficit Index (SWDI) flash-drought metric using the
ERA5-Land member of the three-model ensemble described in Mohammadi and Wang, 2025
(ERA5-Land / GLEAM / GLDAS-Noah).

    SWDI = ( (theta - theta_FC) / (theta_FC - theta_WP) ) * 10

where
    theta     = root-zone volumetric soil water content   [m^3 m^-3]
    theta_FC  = field capacity                            [m^3 m^-3]
    theta_WP  = permanent wilting point                   [m^3 m^-3]

Drought categories (Martinez-Fernandez et al. 2015):
    mild      -2 <= SWDI <  0
    moderate  -5 <= SWDI < -2
    severe   -10 <= SWDI < -5
    extreme         SWDI < -10
SWDI == -10 corresponds to soil moisture at the wilting point.

This script implements ONE ensemble member (ERA5-Land) end-to-end. The paper
averages the SWDI signal from all three products; to build the full ensemble you
would run an analogous routine for GLEAM and GLDAS-Noah (see GLEAM/GLDAS notes
at the bottom of this file) and average the three resulting 0.5-degree / 5-day
SWDI stacks.

------------------------------------------------------------------------------
WHY ERA5-Land for the worked example
------------------------------------------------------------------------------
* It is retrieved fully programmatically through the Copernicus Climate Data
  Store (CDS) API -- no manual web download.
* Its soil moisture is published directly in volumetric units (m^3 m^-3), so no
  mass-to-volume conversion is needed (unlike GLDAS-Noah, which is in kg m^-2).
* Its field capacity and wilting point are *defined by the land-surface model*
  (HTESSEL). The paper is explicit that the FC/WP "employed in the development
  of each soil moisture product" must be used so the SWDI is internally
  consistent. For HTESSEL these are fixed van Genuchten-derived values per soil
  texture class -- see HTESSEL_SOIL_PARAMS below.

------------------------------------------------------------------------------
SOIL HYDRAULIC PARAMETERS (the critical, product-specific ingredient)
------------------------------------------------------------------------------
HTESSEL assigns every land grid cell a single dominant soil texture class taken
from the FAO dataset, then derives hydraulic properties with the van Genuchten
(1980) formulation. Field capacity is the water content at a matric potential of
-0.10 bar; the permanent wilting point is the water content at -15 bar
(Balsamo et al. 2009, J. Hydrometeorol., doi:10.1175/2008JHM1068.1, their
Table 1). Those FC/WP values -- NOT a generic soil database -- are what makes the
ERA5-Land SWDI self-consistent.

The dominant soil texture class is itself an ERA5 field ("Soil type", short name
`slt`), which this script downloads and maps to FC/WP per cell.

------------------------------------------------------------------------------
SETUP
------------------------------------------------------------------------------
  pip install "cdsapi>=0.7.4" xarray rioxarray rasterio netCDF4 numpy pandas

  1. Create a free CDS account: https://cds.climate.copernicus.eu/
  2. Accept the ERA5-Land and ERA5 single-levels licences (once, in the browser).
  3. Put your credentials in ~/.cdsapirc :

         url: https://cds.climate.copernicus.eu/api
         key: <YOUR-PERSONAL-ACCESS-TOKEN>

Run:
  python swdi_era5land.py
Outputs land in ./swdi_output/ as one GeoTIFF per 5-day window:
  SWDI_ERA5Land_YYYYMMDD.tif   (0.5-degree, EPSG:4326, single band, float32)

Author: training-course example. Tested logic; CDS calls require credentials.
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray  # noqa: F401  (registers the .rio accessor)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("swdi")


# =============================================================================
# 1. CONFIGURATION  -- edit this block to change region / period / resolution
# =============================================================================
@dataclass
class Config:
    # Time window (inclusive). Keep modest for a first run: CDS jobs queue and
    # ERA5-Land is high-volume. A single season is plenty for a demo.
    start_date: str = "2021-06-01"
    end_date: str = "2021-08-31"

    # Hour of day sampled for the daily soil-moisture snapshot (UTC). Soil
    # moisture is a slowly varying state variable, so one snapshot/day is a
    # reasonable, low-volume choice. Set sample_all_hours=True to instead
    # download every hour and use the daily mean (much larger download).
    snapshot_hour: int = 12
    sample_all_hours: bool = False

    # Geographic subset [North, West, South, East] in degrees. Default is the
    # contiguous US. Use a small box first to keep the CDS job fast.
    area: tuple[float, float, float, float] = (50.0, -125.0, 24.0, -66.0)

    # Target grid / aggregation. ERA5-Land native = 0.1 deg, so factor 5 -> 0.5.
    native_res: float = 0.1
    target_res: float = 0.5
    temporal_step_days: int = 5

    # I/O
    work_dir: str = "./swdi_work"
    out_dir: str = "./swdi_output"
    sm_file: str = "era5land_rootzone_sm.nc"
    slt_file: str = "era5_soil_type.nc"

    # Skip re-downloading if the cached NetCDF already exists.
    use_cache: bool = True

    @property
    def coarsen_factor(self) -> int:
        f = round(self.target_res / self.native_res)
        if not np.isclose(f * self.native_res, self.target_res):
            raise ValueError("target_res must be an integer multiple of native_res")
        return f


# =============================================================================
# 2. PRODUCT-SPECIFIC SOIL HYDRAULIC PARAMETERS (HTESSEL / ERA5-Land)
# =============================================================================
# Mapping: ERA5 soil-type code -> (theta_FC, theta_WP) in m^3 m^-3.
# Values: Balsamo et al. (2009), Table 1 (van Genuchten; FC @ -0.10 bar,
# WP @ -15 bar). HTESSEL/ERA5 'slt' codes:
#   1 Coarse, 2 Medium, 3 Medium-fine, 4 Fine, 5 Very-fine, 6 Organic,
#   7 Tropical-organic. The published table lists six mineral/organic classes;
#   code 7 is mapped to the organic class here as a documented approximation --
#   verify against your IFS cycle if your domain contains tropical organic soils.
HTESSEL_SOIL_PARAMS: dict[int, tuple[float, float]] = {
    1: (0.244, 0.059),   # Coarse
    2: (0.347, 0.151),   # Medium
    3: (0.383, 0.133),   # Medium fine
    4: (0.448, 0.279),   # Fine
    5: (0.541, 0.335),   # Very fine
    6: (0.663, 0.267),   # Organic
    7: (0.663, 0.267),   # Tropical organic (approx -> organic; see note above)
}

# ERA5-Land soil layer geometry (cm). The paper uses layer 2 + layer 3 as the
# root zone (7-100 cm). We weight by layer thickness.
#   L1 0-7, L2 7-28, L3 28-100, L4 100-289
ERA5LAND_LAYER2_THICKNESS_CM = 28 - 7    # = 21
ERA5LAND_LAYER3_THICKNESS_CM = 100 - 28  # = 72


# =============================================================================
# 3. DATA RETRIEVAL (Copernicus CDS API)
# =============================================================================
def _date_lists(cfg: Config):
    days = pd.date_range(cfg.start_date, cfg.end_date, freq="D")
    years = sorted({f"{d.year:04d}" for d in days})
    months = sorted({f"{d.month:02d}" for d in days})
    dom = sorted({f"{d.day:02d}" for d in days})
    return years, months, dom


def download_era5land_soil_moisture(cfg: Config) -> str:
    """Download ERA5-Land root-zone soil moisture (volumetric layers 2 & 3)."""
    target = os.path.join(cfg.work_dir, cfg.sm_file)
    if cfg.use_cache and os.path.exists(target):
        log.info("Using cached soil moisture: %s", target)
        return target

    import cdsapi
    years, months, dom = _date_lists(cfg)
    hours = [f"{h:02d}:00" for h in range(24)] if cfg.sample_all_hours \
        else [f"{cfg.snapshot_hour:02d}:00"]

    log.info("Requesting ERA5-Land soil moisture from CDS (this may queue)...")
    cdsapi.Client().retrieve(
        "reanalysis-era5-land",
        {
            "variable": [
                "volumetric_soil_water_layer_2",
                "volumetric_soil_water_layer_3",
            ],
            "year": years,
            "month": months,
            "day": dom,
            "time": hours,
            "area": list(cfg.area),       # N, W, S, E
            "data_format": "netcdf",
            "download_format": "unarchived",
        },
        target,
    )
    log.info("Saved %s", target)
    return target


def download_era5_soil_type(cfg: Config) -> str:
    """Download the static ERA5 'Soil type' field (used to assign FC/WP).

    Soil type is a constant field, so we grab a single timestep. It is served by
    the ERA5 single-levels dataset (0.25 deg) and is regridded to the ERA5-Land
    grid later with nearest-neighbour (categorical-safe) interpolation.
    """
    target = os.path.join(cfg.work_dir, cfg.slt_file)
    if cfg.use_cache and os.path.exists(target):
        log.info("Using cached soil type: %s", target)
        return target

    import cdsapi
    log.info("Requesting ERA5 soil type from CDS...")
    cdsapi.Client().retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": "soil_type",
            "year": "2021", "month": "01", "day": "01", "time": "00:00",
            "area": list(cfg.area),
            "data_format": "netcdf",
            "download_format": "unarchived",
        },
        target,
    )
    log.info("Saved %s", target)
    return target


# =============================================================================
# 4. PROCESSING
# =============================================================================
def _pick(ds: xr.Dataset, *candidates: str) -> str:
    """Return the first variable/coord name present (CDS naming varies)."""
    for name in candidates:
        if name in ds.variables:
            return name
    raise KeyError(f"None of {candidates} found in {list(ds.variables)}")


def open_root_zone_soil_moisture(sm_path: str, cfg: Config) -> xr.DataArray:
    """Open ERA5-Land, build thickness-weighted root-zone theta, daily series."""
    ds = xr.open_dataset(sm_path)

    # CDS may name the time dimension 'time' or 'valid_time'.
    tname = _pick(ds, "time", "valid_time")
    if tname != "time":
        ds = ds.rename({tname: "time"})

    l2 = ds[_pick(ds, "swvl2", "volumetric_soil_water_layer_2")]
    l3 = ds[_pick(ds, "swvl3", "volumetric_soil_water_layer_3")]

    # If multiple hours/day were downloaded, reduce to a daily mean first.
    if cfg.sample_all_hours:
        l2 = l2.resample(time="1D").mean()
        l3 = l3.resample(time="1D").mean()

    t2, t3 = ERA5LAND_LAYER2_THICKNESS_CM, ERA5LAND_LAYER3_THICKNESS_CM
    theta = (t2 * l2 + t3 * l3) / (t2 + t3)
    theta.name = "theta_rootzone"
    log.info("Root-zone theta: %s  (%s..%s)", dict(theta.sizes),
             str(theta.time.values[0])[:10], str(theta.time.values[-1])[:10])
    return theta


def build_fc_wp_maps(slt_path: str, like: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """Regrid ERA5 soil type to the soil-moisture grid and map to FC / WP."""
    ds = xr.open_dataset(slt_path)
    slt = ds[_pick(ds, "slt", "soil_type")]
    slt = slt.squeeze(drop=True)  # drop singleton time/level

    # Nearest-neighbour onto the ERA5-Land grid (preserves integer classes).
    slt_on_grid = slt.interp(
        latitude=like.latitude, longitude=like.longitude, method="nearest"
    )
    slt_round = np.rint(slt_on_grid)

    fc = xr.full_like(slt_on_grid, np.nan, dtype="float32")
    wp = xr.full_like(slt_on_grid, np.nan, dtype="float32")
    for code, (f, w) in HTESSEL_SOIL_PARAMS.items():
        fc = fc.where(slt_round != code, f)
        wp = wp.where(slt_round != code, w)

    n_assigned = int(np.isfinite(fc).sum())
    log.info("Assigned FC/WP to %d land cells (soil types present: %s)",
             n_assigned, sorted(np.unique(slt_round.values[np.isfinite(slt_round.values)]).astype(int).tolist()))
    return fc.rename("theta_FC"), wp.rename("theta_WP")


def compute_swdi(theta: xr.DataArray, fc: xr.DataArray, wp: xr.DataArray) -> xr.DataArray:
    """SWDI = ((theta - FC) / (FC - WP)) * 10, broadcast over time."""
    swdi = ((theta - fc) / (fc - wp)) * 10.0
    swdi.name = "SWDI"
    swdi.attrs.update(
        long_name="Soil Water Deficit Index",
        units="dimensionless",
        description="((theta - FC)/(FC - WP))*10; SWDI=-10 at wilting point",
        source_product="ERA5-Land (HTESSEL FC/WP)",
    )
    return swdi


def aggregate_spatial(da: xr.DataArray, factor: int) -> xr.DataArray:
    """Block-average to the coarser target grid (0.1 deg -> 0.5 deg)."""
    out = da.coarsen(latitude=factor, longitude=factor, boundary="trim").mean()
    log.info("Spatial aggregation -> %s", dict(out.sizes))
    return out


def aggregate_temporal(da: xr.DataArray, step_days: int) -> xr.DataArray:
    """Non-overlapping N-day means, stepping every N days from the start."""
    out = da.resample(time=f"{step_days}D").mean()
    log.info("Temporal aggregation -> %d windows of %d days",
             out.sizes["time"], step_days)
    return out


def write_geotiffs(da: xr.DataArray, cfg: Config) -> list[str]:
    """Write one georeferenced GeoTIFF per 5-day window."""
    os.makedirs(cfg.out_dir, exist_ok=True)
    da = da.rio.write_crs("EPSG:4326")
    da = da.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")
    da = da.rio.write_nodata(np.nan, encoded=True)

    paths = []
    for i in range(da.sizes["time"]):
        sl = da.isel(time=i).astype("float32")
        stamp = pd.Timestamp(sl.time.values).strftime("%Y%m%d")
        path = os.path.join(cfg.out_dir, f"SWDI_ERA5Land_{stamp}.tif")
        sl.rio.to_raster(path, driver="GTiff", compress="deflate")
        paths.append(path)
    log.info("Wrote %d GeoTIFFs to %s", len(paths), cfg.out_dir)
    return paths


# =============================================================================
# 5. ORCHESTRATION
# =============================================================================
def main(cfg: Config | None = None) -> list[str]:
    cfg = cfg or Config()
    os.makedirs(cfg.work_dir, exist_ok=True)

    sm_path = download_era5land_soil_moisture(cfg)
    slt_path = download_era5_soil_type(cfg)

    theta = open_root_zone_soil_moisture(sm_path, cfg)
    fc, wp = build_fc_wp_maps(slt_path, like=theta.isel(time=0))

    swdi_native = compute_swdi(theta, fc, wp)              # 0.1 deg, daily
    swdi_05 = aggregate_spatial(swdi_native, cfg.coarsen_factor)   # 0.5 deg
    swdi_5d = aggregate_temporal(swdi_05, cfg.temporal_step_days)  # 5-day

    return write_geotiffs(swdi_5d, cfg)


if __name__ == "__main__":
    written = main()
    print(f"\nDone. {len(written)} SWDI GeoTIFF(s):")
    for p in written:
        print("  ", p)


# =============================================================================
# EXTENDING TO THE FULL THREE-MODEL ENSEMBLE
# =============================================================================
# The paper averages the SWDI signal across ERA5-Land, GLEAM and GLDAS-Noah on a
# common 0.5-degree / 5-day grid. To add the other two members:
#
# GLDAS-Noah (NASA GES DISC, Earthdata login; OPeNDAP/Hyrax or direct HTTP):
#   * Product GLDAS_NOAH025_3H 2.1. Root zone = layer 2 (10-40 cm) + layer 3
#     (40-100 cm). Soil moisture is mass content (kg m^-2): convert to
#     volumetric by dividing by (layer_thickness_mm) -> theta = SM_kgm2 / mm.
#   * FC/WP: the Noah LSM uses soil-class parameters SMCREF (~field capacity)
#     and SMCWLT (wilting point) from its SOILPARM.TBL, keyed by the model's
#     soil-texture map. Use those, not a generic database, for consistency.
#
# GLEAM (gleam.eu; SFTP after free registration):
#   * Provides root-zone soil moisture directly (root depth varies by land
#     cover). GLEAM's water-balance uses its own critical soil moisture and
#     residual/saturation terms; use the wilting point / field capacity adopted
#     in the GLEAM version you download.
#
# Then, for each member, produce a 0.5-degree / 5-day SWDI stack with the same
# time windows, align them (xr.align), and average:
#   ensemble = xr.concat([swdi_era5, swdi_gleam, swdi_gldas], "model").mean("model")
# before writing the final GeoTIFFs.