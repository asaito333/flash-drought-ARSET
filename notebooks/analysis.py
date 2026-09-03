import csv
import glob
import logging
import os
import warnings
from datetime import datetime
from typing import TypeVar

import earthaccess
import fsspec
import numpy as np
import rasterio
import xarray as xr
import zarr
from dask.distributed import Client, LocalCluster
from download import download_model_constants
from pyproj import Transformer
from rasterio.transform import from_origin
from rasterio.windows import Window, from_bounds

XarrayObj = TypeVar("XarrayObj", xr.Dataset, xr.DataArray)

# The threshold and scale factor parameters come from the documentation: https://data.globalecology.unh.edu/data/GOSIF_v2/Fair_Data_Use_Policy_and_Readme_GOSIF_v2.pdf
# 32767 = water bodies, 32766 = ice/snow
GOSIF_DATA_THRESH = 32765
# This value tells our code the conversion between pixel values in the GeoTIFF images to units of W/m^2/sr/μm
GOSIF_SCALE_FACTOR = 0.0001

# EASE-Grid 2.0 Global (9 km) projection used by SMAP L4 x/y coordinates (meters).
EASE2_GLOBAL_EPSG = "EPSG:6933"

# SPL4SMLM land-model constants used to turn soil moisture into an SWDI.  These
# live on the same 9 km EASE-Grid 2.0 cells as the SPL4SMGP soil moisture, so
# their (y, x) arrays align 1:1 with sm_rootzone.
LMC_GROUP = "Land-Model-Constants_Data"
FIELD_CAPACITY_VAR = "clsm_cdcr2"  # column water capacity (kg m-2)
PROFILE_DEPTH_VAR = "clsm_dzpr"  # soil profile thickness (m)
WILTING_POINT_VAR = "clsm_wp"  # wilting point (m3 m-3)

# Density of liquid water (kg m-3), used to convert clsm_cdcr2's column water
# capacity (kg m-2) into a volumetric field capacity (m3 m-3).
WATER_DENSITY = 1000.0


def get_read_window(
        geotiff_path: str,
        west: float,
        south: float,
        east: float,
        north: float
) -> Window:
    with rasterio.open(geotiff_path) as src:
        read_window = from_bounds(west, south, east, north, src.transform)
        read_window = read_window.round_offsets().round_lengths()
    return read_window


def read_roi(path: str, read_window: Window) -> np.ndarray:
    """Read the ROI, mask non-data (water/ice/fill), scale to physical units."""
    with rasterio.open(path) as src:
        arr = src.read(1, window=read_window).astype("float64")
    arr[arr > GOSIF_DATA_THRESH] = np.nan
    return arr * GOSIF_SCALE_FACTOR


def write_step_geotiff(
        ras_grid: np.ndarray,
        out_path: str,
        crs,
        transform,
        dtype="float64",
) -> None:
    """Write a single-band raster grid to a georeferenced GeoTIFF."""
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=ras_grid.shape[0],
        width=ras_grid.shape[1],
        count=1,
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=np.nan,
    ) as dst:
        dst.write(ras_grid, 1)


def compute_rci(
        z_jy_grid: np.ndarray,
        z_prev_grid: np.ndarray,
        rci_prev_grid: np.ndarray
) -> np.ndarray:
    """Compute the per grid cell SIF-RCI using the formula from notebook 1."""
    neg_anom = z_jy_grid < -0.75                  # Z(j,y) < -0.75       (case 1)
    pos_anom = z_jy_grid > 0.75                   # Z(j,y) >  0.75       (case 2)
    sign_change = (z_prev_grid * z_jy_grid) < 0   # Z(j-1,y)·Z(j,y) < 0  (case 3)

    neg_term = np.sqrt(np.where(neg_anom, np.abs(z_jy_grid) - 0.75, 0.0))
    pos_term = np.sqrt(np.where(pos_anom, z_jy_grid + 0.75, 0.0))

    rci_jy_grid = rci_prev_grid.copy()
    rci_jy_grid = np.where(neg_anom, rci_prev_grid - neg_term, rci_jy_grid)
    rci_jy_grid = np.where(pos_anom, rci_prev_grid + pos_term, rci_jy_grid)
    rci_jy_grid = np.where(sign_change, 0.0, rci_jy_grid)
    return rci_jy_grid


def compute_sif_time_series(
        gosif_geotiffs: list[str],
        clim_dir: str,
        time_series_fname: str,
        west: float,
        south: float,
        east: float,
        north: float,
        raster_dir: str | None = None,
) -> tuple[str, int]:
    prev_grid: np.ndarray | None = None
    # Initialize the arrays for the rows of our CSV
    # The variable names correspond to what is mentioned in the description above
    dates: list[datetime] = []
    sif_jy: list[float] = []
    mean_sif_j: list[float] = []
    z_jy: list[float] = []
    rci_jy: list[float] = []

    read_window = get_read_window(gosif_geotiffs[0], west, south, east, north)

    # Capture the CRS and windowed transform once so each per-step RCI grid can
    # be written out as a georeferenced GeoTIFF aligned to the read window.
    ras_crs = None
    ras_transform = None
    if raster_dir is not None:
        os.makedirs(raster_dir, exist_ok=True)
        with rasterio.open(gosif_geotiffs[0]) as src:
            ras_crs = src.crs
            ras_transform = src.window_transform(read_window)

    # Recursive state carried between time windows for RCI.
    # RCI(j0, y) = 0 and Z(j0, y) = 0 everywhere
    # For simplicity, our incon j0 = DOY 1
    rci_prev_grid: np.ndarray = np.zeros((read_window.height, read_window.width))
    z_prev_grid: np.ndarray = np.zeros((read_window.height, read_window.width))

    for j, geotiff in enumerate(gosif_geotiffs[1:], start=1):
        # Parse the date from the filename, e.g. GOSIF_2017073.tif = DOY 73
        yr = os.path.splitext(os.path.basename(geotiff))[0][-7:-3]
        doy = os.path.splitext(os.path.basename(geotiff))[0][-3:]
        dates.append(datetime.strptime(f"{yr}{doy}", "%Y%j")) # noqa: DTZ007

        sif_grid = read_roi(geotiff, read_window)
        if j == 1:
            prev_grid = read_roi(gosif_geotiffs[j-1], read_window)
        # Get the SIF increment at j
        dsif_grid = sif_grid - prev_grid
        # Set the current raster to the previous for the next iteration
        prev_grid = sif_grid

        # Get the climatology input file produced by the appendix notebook
        clim_input = f"{clim_dir}/GOSIF_dSIF_clim_{doy}.tif"
        with rasterio.open(clim_input) as clim_src:
            mean_dsif_band = clim_src.read(1).astype(float)
            std_dsif_band = clim_src.read(2).astype(float)
            mean_sif_band = clim_src.read(3).astype(float)

        z_jy_grid = (dsif_grid - mean_dsif_band) / std_dsif_band

        rci_jy_grid = compute_rci(z_jy_grid, z_prev_grid, rci_prev_grid)
        z_prev_grid = z_jy_grid
        rci_prev_grid = rci_jy_grid

        rci_masked = np.where(np.isnan(z_jy_grid), np.nan, rci_jy_grid)

        # Optionally save the non-spatially-averaged RCI grid for this step.
        # TO DO: I may make the write_step_geotiff function more modular to save
        # the other metrics in other bands of the geotiff.
        if raster_dir is not None:
            date = dates[-1]
            out_path = os.path.join(
                raster_dir,
                f"sif_rci_{date.year}_{date.month:02d}_{date.day:02d}.tif",
            )
            write_step_geotiff(rci_masked, out_path, ras_crs, ras_transform)

        # Compute the spatial average at the end
        sif_jy.append(float(np.nanmean(sif_grid)))
        mean_sif_j.append(float(np.nanmean(mean_sif_band)))
        z_jy.append(float(np.nanmean(z_jy_grid)))
        rci_jy.append(float(np.nanmean(rci_masked)))

    # Save the output as a CSV so it can be used in the next notebook
    csv_path = os.path.join("data", time_series_fname)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "sif", "mean_sif", "zscore", "sif_rci"])
        # We have 4 sig figs from the source data
        writer.writerows(
            zip(
                [d.strftime("%Y-%m-%d") for d in dates],
                [f"{sjy:.4f}" for sjy in sif_jy],
                [f"{msj:.4f}" for msj in mean_sif_j],
                [f"{zjy:.4f}" for zjy in z_jy],
                [f"{rjy:.4f}" for rjy in rci_jy],
            )
        )

    return csv_path, len(dates)


def detect_flash_drought_sif(
        raster_dir: str,
        output_dir: str,
        time_series_csv: str,
        threshold: float = -0.5,
        n_steps: int = 3,
) -> str:
    """Flag per-cell flash drought from the SIF-RCI raster series.

    Reads the "sif_rci_{year}_{month:02d}_{day:02d}.tif" rasters written by
    :func:`compute_sif_time_series` and applies a rule to each grid cell: a
    flash drought is detected at a time step when that step and the `n_steps`
    - 1 immediately preceding steps all have a SIF-RCI value below `threshold`.
    The chronological filename convention means a lexical sort of the rasters
    is also a temporal sort.

    One GeoTIFF is written per time step to `output_dir` (same filename with a
    "fd_sifrci_" prefix), carrying the source raster's CRS and transform, where
    1 marks a detection and 0 marks no detection. The earliest `n_steps` - 1
    steps lack enough history to satisfy the rule and are therefore all 0.

    The fraction of valid (non-NaN) grid cells flagged at each step is written
    back into the existing `time_series_csv` as a new "fd_percent" column,
    matched to each row by its date so it stays aligned with the other columns.

    Arguments:
        raster_dir (str): Directory holding the SIF-RCI GeoTIFFs.
        output_dir (str): Directory to write the detection GeoTIFFs to.
        time_series_csv (str): Path to the existing time series CSV (with a
            leading "date" column) to add the "fd_percent" column to.
        threshold (float): SIF-RCI value a cell must fall below to count toward
            a detection.
        n_steps (int): Number of consecutive steps (including the current one)
            that must be below `threshold` to flag a detection.

    Returns:
        str: The output directory path.
    """
    paths = sorted(glob.glob(os.path.join(raster_dir, "sif_rci_*.tif")))
    os.makedirs(output_dir, exist_ok=True)

    # Rolling buffer of the last `n_steps` "below threshold" masks so detection
    # only needs each raster in memory once, not the whole stack.
    recent_below: list[np.ndarray] = []
    # Percent of valid cells flagged at each step, keyed by "%Y-%m-%d" date so
    # it can be merged into the CSV by row rather than relying on row order.
    fd_percent_by_date: dict[str, float] = {}

    for path in paths:
        with rasterio.open(path) as src:
            rci_grid = src.read(1).astype("float64")
            crs = src.crs
            transform = src.transform

        # NaN (water/ice/fill) compares False, so it never counts as a detection.
        recent_below.append(rci_grid < threshold)
        recent_below = recent_below[-n_steps:]

        if len(recent_below) == n_steps:
            detection = np.logical_and.reduce(recent_below)
        else:
            detection = np.zeros(rci_grid.shape, dtype=bool)

        # Percent over valid land cells only; water/ice/fill (NaN) can never be
        # flagged, so counting them would dilute the detection fraction.
        n_valid = int(np.count_nonzero(~np.isnan(rci_grid)))
        fd_percent = 100.0 * int(np.count_nonzero(detection)) / n_valid if n_valid else 0.0

        # Reconstruct the "%Y-%m-%d" date from "sif_rci_{year}_{month}_{day}".
        year, month, day = os.path.splitext(os.path.basename(path))[0].split("_")[-3:]
        fd_percent_by_date[f"{year}-{month}-{day}"] = fd_percent

        out_name = os.path.basename(path).replace("sif_rci_", "fd_sifrci_", 1)
        out_path = os.path.join(output_dir, out_name)
        write_step_geotiff(detection.astype("float32"), out_path, crs, transform, dtype="float32")

    _add_fd_percent_column(time_series_csv, fd_percent_by_date)

    return output_dir


def _add_fd_percent_column(
        time_series_csv: str,
        fd_percent_by_date: dict[str, float],
) -> None:
    """Add an "fd_percent" column to an existing time series CSV.

    Rows are matched to their detection percent by the leading "date" column so
    the new values stay aligned with the existing rows regardless of order.
    """
    with open(time_series_csv, newline="") as f:
        rows = list(csv.reader(f))

    header, *data_rows = rows
    header.append("fd_percent")
    for row in data_rows:
        fd_percent = fd_percent_by_date.get(row[0])
        row.append(f"{fd_percent:.2f}" if fd_percent is not None else "")

    with open(time_series_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data_rows)


# Change the number of workers to meet the capabilities of your own computer if needed
def create_dask_cluster(
        n_workers: int = 8,
) -> tuple[Client, LocalCluster, bool]:

    # Reuse an existing local cluster if one is already running so repeated
    # calls don't spin up (and leak) a new LocalCluster each time. Dask
    # registers every Client as the global/default client, so querying for the
    # current one tells us whether a cluster is already up. The returned flag
    # reports whether we created the cluster, so callers know whether it is
    # theirs to shut down.
    try:
        client = Client.current()
    except ValueError:
        print("Creating new local Dask client")
        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=1,
            silence_logs=logging.ERROR)

        client = Client(cluster)
        return (client, cluster, True)
    else:
        print("Reusing existing local Dask client")
        return (client, client.cluster, False) # type: ignore


def silence_worker_warnings() -> None:
    warnings.filterwarnings("ignore")
    for name in ["distributed", "xarray", "py.warnings", "fsspec", "h5netcdf", "h5py"]:
        logging.getLogger(name).setLevel(logging.ERROR)


def open_virtual_dataset(
        ref_url: str,
) -> tuple[xr.Dataset, Client, LocalCluster, bool]:
    client, cluster, created = create_dask_cluster()
    client.run(silence_worker_warnings)

    earthaccess.login()
    daac_fs = earthaccess.get_fsspec_https_session()

    fs = fsspec.filesystem(
        "reference",
        fo=ref_url,
        remote_protocol="https",
        asynchronous=True,
        remote_options={"asynchronous": True, **daac_fs.storage_options},
    )

    store = zarr.storage.FsspecStore(fs, read_only=True) # type: ignore
    ds = xr.open_zarr(store, consolidated=False)
    return ds, client, cluster, created


def latlon_bbox_to_ease(
    bbox: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Convert a lat/lon bounding box to EASE-Grid 2.0 Global x/y bounds.

    The SMAP L4 x/y coordinates are in meters in the EASE-Grid 2.0
    Global projection (EPSG:6933), so a geographic bounding box must be
    reprojected before it can be used to index the grid.  EPSG:6933 is a
    cylindrical equal-area projection, so x depends only on longitude and
    y only on latitude; transforming the four corners and taking the
    min/max therefore yields exact axis-aligned bounds.

    Arguments:
        bbox: (west, south, east, north) in degrees (lon/lat, EPSG:4326).

    Returns:
        (x_min, y_min, x_max, y_max) in meters (EPSG:6933).
    """
    west, south, east, north = bbox
    transformer = Transformer.from_crs("EPSG:4326", EASE2_GLOBAL_EPSG, always_xy=True)
    xs, ys = transformer.transform([west, east, west, east], [south, south, north, north])
    return min(xs), min(ys), max(xs), max(ys)


def _bounds_slice(coord: xr.DataArray, lo: float, hi: float) -> slice:
    """Build a slice from lo to hi that respects a coordinate's order.

    xarray label slicing follows the coordinate's stored direction, and the
    SMAP L4 y coordinate is descending (north to south), so the slice bounds
    must be reversed for descending coordinates.
    """
    if float(coord[0]) > float(coord[-1]):
        return slice(hi, lo)
    return slice(lo, hi)


def _select_bbox(
    obj: XarrayObj,
    bbox: tuple[float, float, float, float],
) -> XarrayObj:
    """Select the x/y cells of a SMAP L4 grid falling inside a lat/lon box.

    Both the soil moisture and the land-model constants ride on the same
    EASE-Grid 2.0 cells, so selecting each with this shared helper guarantees
    their subsets carry identical x/y coordinates and align cell-for-cell.

    Raises:
        ValueError: If the bounding box does not overlap the dataset grid.
    """
    x_min, y_min, x_max, y_max = latlon_bbox_to_ease(bbox)
    subset = obj.sel(
        x=_bounds_slice(obj.x, x_min, x_max),
        y=_bounds_slice(obj.y, y_min, y_max),
    )
    if subset.sizes["x"] == 0 or subset.sizes["y"] == 0:
        msg = f"Bounding box {bbox} does not overlap the dataset grid."
        raise ValueError(msg)
    return subset


def load_field_capacity_wilting_point(
    constants_path: str,
    bbox: tuple[float, float, float, float],
) -> tuple[xr.DataArray, xr.DataArray]:
    """Extract volumetric field capacity and wilting point over a lat/lon box.

    The SPL4SMLM file stores the geophysical constants in the
    Land-Model-Constants_Data group while the x/y coordinates live in the root
    group, so the two are opened separately and the coordinates attached to the
    constants before subsetting.

    The wilting point (clsm_wp) is already a volumetric water content
    (m3 m-3), matching sm_rootzone.  The field capacity, however, is taken from
    clsm_cdcr2 -- the column water-holding capacity in kg m-2 integrated over
    the soil profile depth clsm_dzpr (m).  Dividing it by the mass of a full
    water column (depth x water density) converts it to a volumetric field
    capacity (m3 m-3) so the SWDI's numerator and denominator are dimensionally
    consistent.

    Arguments:
        constants_path (str): Path to the SPL4SMLM HDF-5 granule (see
            :func:`download_model_constants`).
        bbox: (west, south, east, north) in degrees (lon/lat).

    Returns:
        tuple[xr.DataArray, xr.DataArray]: The (field_capacity, wilting_point)
        DataArrays over the box, both in m3 m-3 and carrying the SMAP L4 x/y
        coordinates so they align with a soil moisture subset over the same box.
    """
    open_kwargs = {"engine": "h5netcdf", "phony_dims": "sort"}
    root = xr.open_dataset(constants_path, **open_kwargs)  # type: ignore[arg-type]
    lmc = xr.open_dataset(constants_path, group=LMC_GROUP, **open_kwargs)  # type: ignore[arg-type]
    lmc = lmc.assign_coords(x=root["x"], y=root["y"])

    field_capacity = (
        lmc[FIELD_CAPACITY_VAR] / (lmc[PROFILE_DEPTH_VAR] * WATER_DENSITY)
    ).rename("field_capacity")
    wilting_point = lmc[WILTING_POINT_VAR].rename("wilting_point")

    return _select_bbox(field_capacity, bbox), _select_bbox(wilting_point, bbox)


def ease2_grid_transform(x: np.ndarray, y: np.ndarray):
    """Affine transform for a north-up EASE-Grid 2.0 raster from cell centers.

    The SMAP L4 x/y coordinates are the cell centers (meters) of a regular
    grid, so the pixel size is the coordinate spacing and the raster origin is
    the outer corner of the north-west cell (half a pixel beyond the extreme
    centers).  min/max are used so the transform is correct regardless of
    whether x/y are stored ascending or descending; callers must orient the
    array itself north-up (row 0 = northernmost row) to match.
    """
    xres = abs(float(x[1] - x[0]))
    yres = abs(float(y[1] - y[0]))
    west = float(x.min()) - xres / 2.0
    north = float(y.max()) + yres / 2.0
    return from_origin(west, north, xres, yres)


def swdi_timeseries(
    ds: xr.Dataset,
    constants_path: str,
    bbox: tuple[float, float, float, float],
    freq: str = "3D",
    variable: str = "sm_rootzone",
    start: str | None = None,
    stop: str | None = None,
    raster_dir: str | None = None,
) -> xr.DataArray:
    """Compute a box-averaged Soil Water Deficit Index (SWDI) time series.
    The SWDI is computed per cell and then spatially averaged over the bbox.

    Field capacity and wilting point are constant in time, so aggregating the
    soil moisture to `freq` windows before forming the (linear) SWDI is
    equivalent to forming it first and then aggregating; the soil moisture is
    resampled first so each window's SWDI is built from that window's moisture.

    Arguments:
        ds (xr.Dataset): The SMAP L4 virtual dataset.
        constants_path (str): Path to the SPL4SMLM HDF-5 granule.
        bbox: (west, south, east, north) in degrees (lon/lat).
        freq (str): Pandas offset alias for the temporal aggregation window
            ("3D" = 3-day means).
        variable (str): The root-zone soil moisture variable to use.
        start (str | None): Optional start date (e.g. "2019" or "2019-01-01").
            If None, begins at the start of the dataset.
        stop (str | None): Optional end date (e.g. "2019" or "2019-12-31"),
            inclusive. If None, runs to the end of the dataset.
        raster_dir (str | None): Optional directory in which to save the
            non-spatially-averaged per-cell SWDI grid for each time step as a
            georeferenced GeoTIFF (EPSG:6933), named
            "swdi_{year}_{month}_{day}.tif" (ordered so the files sort
            chronologically).  If None, no rasters are written.

    Returns:
        xr.DataArray: A 1-D DataArray of the box-averaged SWDI indexed by time.

    Raises:
        ValueError: If the bounding box does not overlap the dataset grid.
    """
    subset = _select_bbox(ds, bbox)

    if start is not None or stop is not None:
        subset = subset.sel(time=slice(start, stop))

    sm = subset[variable].resample(time=freq).mean()

    # Per-cell field capacity and wilting point over the same box; the shared
    # _select_bbox selection guarantees identical x/y coordinates, so xarray
    # broadcasts them against the (time, y, x) soil moisture cell-for-cell.
    field_capacity, wilting_point = load_field_capacity_wilting_point(
        constants_path, bbox
    )

    swdi = (sm - field_capacity) / (field_capacity - wilting_point) * 10.0

    if raster_dir is not None:
        # Writing the rasters already materialises the full (time, y, x) grid
        # in memory, so reuse it for the spatial average instead of forcing a
        # second (slow, network-bound) read of the virtualised dataset.
        swdi = _save_swdi_rasters(swdi, raster_dir)

    # Average the SWDI (not the soil moisture) across the box.
    return swdi.mean(dim=("x", "y")).rename("swdi")


def _save_swdi_rasters(swdi: xr.DataArray, raster_dir: str) -> xr.DataArray:
    """Write each SWDI time step to a georeferenced GeoTIFF in `raster_dir`.

    The grid is oriented north-up (y descending) so its rows match the affine
    transform, then materialised once. The computed in-memory grid is returned
    so the caller can spatially average it without re-reading the (virtualised,
    network-bound) source data.
    """
    os.makedirs(raster_dir, exist_ok=True)

    # Orient north-up (row 0 = northernmost) and materialise the lazy grid so
    # every time step is computed a single time.
    swdi = swdi.sortby("y", ascending=False).compute()
    transform = ease2_grid_transform(swdi.x.values, swdi.y.values)

    for step in swdi.transpose("time", "y", "x"):
        date = step.time.values.astype("datetime64[s]").item()
        out_path = os.path.join(
            raster_dir,
            f"swdi_{date.year}_{date.month:02d}_{date.day:02d}.tif",
        )
        write_step_geotiff(step.values, out_path, EASE2_GLOBAL_EPSG, transform)

    return swdi


def compute_swdi_timeseries(
        start_date: str,
        stop_date: str,
        time_series_fname: str,
        bbox: tuple[float, float, float, float],
        ref_url: str = "https://its-live-data.s3-us-west-2.amazonaws.com/test-space/vds/SPL4SMGP.parquet",
        raster_dir: str | None = None,
) -> str:
    constants_path = download_model_constants()
    ds, client, cluster, created = open_virtual_dataset(ref_url)
    try:
        swdi_ts = swdi_timeseries(
            ds, constants_path, bbox,
            start=start_date, stop=stop_date, raster_dir=raster_dir,
        )
        csv_path = os.path.join("data", time_series_fname)

        swdi_ts.to_dataframe().to_csv(csv_path)
        return csv_path
    finally:
        # Only tear down the cluster if this workflow created it, so a client
        # the user already had running isn't shut down out from under them.
        if created:
            client.close()
            cluster.close()