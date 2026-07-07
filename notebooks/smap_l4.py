import os
import warnings
from collections.abc import Sequence
from typing import TypeVar

import earthaccess
import fsspec
import virtualizarr  # noqa: F401  (registers the .vz dataset accessor)
import xarray as xr
from pyproj import Transformer

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

# Preserves whether _select_bbox was handed a Dataset or a DataArray.
XarrayObj = TypeVar("XarrayObj", xr.Dataset, xr.DataArray)


def collapse_time_to_scalar(ds: xr.Dataset) -> xr.Dataset:
    """Collapse a SMAP L4 granule's single timestamp to a scalar coordinate.

    Each SMAP L4 granule carries one timestamp stored as a length-1 time
    variable along its own (non-indexed) dimension rather than as a scalar.
    When open_virtual_mfdataset concatenates with concat_dim="time",
    xarray tries to expand_dims("time") to create the stacking dimension
    and raises "time already exists as coordinate or variable name" because
    a non-scalar time variable is already present.

    Squeezing the length-1 dimension turns time into a scalar coordinate,
    which expand_dims is able to promote into a time dimension.

    Arguments:
        ds (xr.Dataset): A single (virtual) granule dataset.

    Returns:
        xr.Dataset: The dataset with time collapsed to a scalar coordinate.
    """
    if "time" in ds.variables:
        squeeze_dims = [d for d in ds["time"].dims if ds.sizes[d] == 1]
        if squeeze_dims:
            ds = ds.squeeze(squeeze_dims)
        ds = ds.set_coords("time")
    return ds


def virtualize_smap_l4(
    temporal: tuple[str, str],
    group: str = "Geophysical_Data",
    variables: Sequence[str] | None = ("sm_rootzone",),
    load: bool = False,
) -> xr.Dataset:
    """Build a virtual dataset of SMAP L4 geophysical variables over a time range.

    SMAP L4 (SPL4SMGP) stores the coordinate variables (time, x, y)
    in the file's root group, while the geophysical variables such as
    sm_rootzone live in the Geophysical_Data subgroup as 2-D (y, x)
    arrays with no time coordinate of their own.  Because
    open_virtual_mfdataset parses one group per call, the root group and the
    geophysical group are opened separately and the root coordinates are then
    attached to the stacked geophysical variables.

    Arguments:
        temporal (tuple[str, str]): (start, end) date strings passed to earthaccess.
        group (str): The subgroup holding the geophysical variables.
        variables (Sequence[str], optional): Variables to keep from group.  None keeps the whole
            group; the default keeps only sm_rootzone.
        load (bool, optional): When False (default) the geophysical variables stay as
            VirtualiZarr ManifestArrays (byte-range references), which is
            what :func:`save_virtual_zarr` needs to write a compact kerchunk
            file.  When True they are materialised into a concrete,
            lazily-loaded dataset that can be computed on directly (e.g. by
            :func:`sm_rootzone_timeseries`) without a kerchunk round-trip.

    Returns:
        A virtual xarray.Dataset (or a concrete one when load=True) with
        the requested variables stacked along a time dimension and carrying
        the time/x/y coordinates.
    """
    auth = earthaccess.login()
    if not auth.authenticated:
        auth.login(strategy="interactive", persist=True)

    warnings.filterwarnings("ignore", "As of version 1.0*", FutureWarning)
    results = earthaccess.search_data(
        short_name="SPL4SMGP",
        temporal=temporal,
    )

    open_options = {
        "access": "indirect",
        "concat_dim": "time",
        "coords": "minimal",
        "compat": "override",
        "combine_attrs": "override",
    }

    warnings.filterwarnings(
        "ignore",
        message="This DMRpp contains the variable EASE2_global_projection*",
        category=UserWarning
    )
    # The root group is always loaded so that time/x/y come back as concrete
    # coordinates; these are tiny and get inlined into the kerchunk file.
    result_root = earthaccess.virtualize(
        granules=results,
        load=True,
        data_vars="minimal",
        preprocess=collapse_time_to_scalar,
        loadable_variables=["time", "x", "y"],
        **open_options, # type: ignore
    )

    result_gph = earthaccess.virtualize(
        granules=results,
        load=load,
        group=group,
        data_vars="all",
        **open_options, # type: ignore
    )

    result = result_gph.assign_coords(
        time=result_root["time"],
        x=result_root["x"],
        y=result_root["y"],
    )

    if variables is not None:
        result = result[list(variables)]

    return result


def save_virtual_zarr(ds: xr.Dataset, filepath: str) -> str:
    """Persist a virtual dataset as a kerchunk virtual Zarr reference file.

    The dataset returned by :func:`virtualize_smap_l4` does not hold any
    array data itself; it holds references (byte-range offsets into the
    remote SMAP L4 granules) produced by VirtualiZarr.  Serializing those
    references to a single kerchunk JSON file lets the whole time stack be
    reopened later as one Zarr store without re-running the search and
    virtualization step, while the actual chunks are still streamed from the
    original granules on demand.

    Arguments:
        ds (xr.Dataset): A virtual dataset from :func:`virtualize_smap_l4`.
        filepath (str): Destination path for the kerchunk references.  Use a
            .json suffix for the JSON format.

    Returns:
        str: The path the references were written to.
    """
    ds.vz.to_kerchunk(filepath, format="json")
    return filepath


def open_virtual_zarr(filepath: str) -> xr.Dataset:
    """Open a kerchunk virtual Zarr file written by :func:`save_virtual_zarr`.

    The references point at the original SMAP L4 granules on NASA Earthdata,
    which are served over authenticated HTTPS, so an Earthdata bearer token
    is attached to every remote request.  fsspec's "reference" filesystem
    maps the kerchunk references onto that remote store, and xarray reads it
    back through the Zarr engine.  Only the chunks that are actually accessed
    are fetched, so opening the store is cheap.

    Arguments:
        filepath: Path to a kerchunk JSON reference file.

    Returns:
        The lazily-backed xarray.Dataset described by the references.
    """
    auth = earthaccess.login()
    if not auth.authenticated:
        auth.login(strategy="interactive", persist=True)
    token = auth.token["access_token"]  # type: ignore[index]

    # zarr v3's fsspec store requires the reference filesystem and the remote
    # HTTPS filesystem it wraps to share the same asynchronous setting.
    fs = fsspec.filesystem(
        "reference",
        fo=filepath,
        remote_protocol="https",
        remote_options={
            "headers": {"Authorization": f"Bearer {token}"},
            "asynchronous": True,
        },
        asynchronous=True,
    )
    return xr.open_dataset(
        fs.get_mapper(""),
        engine="zarr",
        consolidated=False,
    )


def download_model_constants(
    output_dir: str = "data/smap_model_constants",
) -> str:
    """Download the single SPL4SMLM land-model-constants granule.

    The SWDI needs the field capacity and wilting point, which are not carried
    in the SPL4SMGP soil moisture granules; they are static constants of the
    Catchment land surface model shared by the whole SMAP L4 record.  The
    SPL4SMLM collection therefore contains exactly one granule, which this
    downloads (skipping the transfer if it is already present locally).

    Arguments:
        output_dir (str): Directory to store the granule in.  Created if absent.

    Returns:
        str: Path to the downloaded HDF-5 constants file.

    Raises:
        FileNotFoundError: If no HDF-5 granule was returned by the download.
    """
    expected_path = "data/smap_model_constants/SMAP_L4_SM_lmc_00000000T000000_Vv8011_001.h5"
    if os.path.exists(expected_path):
        return expected_path

    auth = earthaccess.login()
    if not auth.authenticated:
        auth.login(strategy="interactive", persist=True)

    os.makedirs(output_dir, exist_ok=True)

    warnings.filterwarnings("ignore", "As of version 1.0*", FutureWarning)
    results = earthaccess.search_data(short_name="SPL4SMLM")
    downloaded = earthaccess.download(results, local_path=output_dir)

    h5_files = [str(p) for p in downloaded if str(p).endswith(".h5")]
    if not h5_files:
        msg = f"No SPL4SMLM HDF-5 granule was downloaded to {output_dir}."
        raise FileNotFoundError(msg)
    return h5_files[0]


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


def sm_rootzone_timeseries(
    ds: xr.Dataset,
    bbox: tuple[float, float, float, float],
    freq: str = "3D",
    variable: str = "sm_rootzone",
) -> xr.DataArray:
    """Spatially average a variable over a lat/lon box and aggregate in time.

    Arguments:
        ds (xr.Dataset): The dataset returned by :func:`virtualize_smap_l4`.
        bbox: (west, south, east, north) in degrees (lon/lat).
        freq (str): Pandas offset alias for the temporal aggregation window
            ("3D" = 3-day means).
        variable (str): The variable to aggregate.

    Returns:
        xr.DataArray: A 1-D DataArray of the box-averaged, freq-aggregated variable
        indexed by time.

    Raises:
        ValueError: If the bounding box does not overlap the dataset grid.
    """
    subset = _select_bbox(ds, bbox)

    # Mean over the box (NaN fill values over water/ice are skipped), then
    # aggregate the native 3-hourly steps into `freq` windows.
    spatial_mean = subset[variable].mean(dim=("x", "y"))
    return spatial_mean.resample(time=freq).mean()


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


def swdi_timeseries(
    ds: xr.Dataset,
    constants_path: str,
    bbox: tuple[float, float, float, float],
    freq: str = "3D",
    variable: str = "sm_rootzone",
) -> xr.DataArray:
    """Compute a box-averaged Soil Water Deficit Index (SWDI) time series.

    The SWDI of a grid cell is::

        SWDI = ((sm - sm_fc) / (sm_fc - sm_wp)) * 10

    where ``sm`` is the root-zone soil moisture, ``sm_fc`` the field capacity
    and ``sm_wp`` the wilting point (all m3 m-3). The SWDI is computed
    per cell and then spatially averaged over the bbox.

    Field capacity and wilting point are constant in time, so aggregating the
    soil moisture to `freq` windows before forming the (linear) SWDI is
    equivalent to forming it first and then aggregating; the soil moisture is
    resampled first so each window's SWDI is built from that window's moisture.

    Arguments:
        ds (xr.Dataset): Soil moisture dataset from :func:`virtualize_smap_l4`
            or :func:`open_virtual_zarr`.
        constants_path (str): Path to the SPL4SMLM HDF-5 granule.
        bbox: (west, south, east, north) in degrees (lon/lat).
        freq (str): Pandas offset alias for the temporal aggregation window
            ("3D" = 3-day means).
        variable (str): The root-zone soil moisture variable to use.

    Returns:
        xr.DataArray: A 1-D DataArray of the box-averaged SWDI indexed by time.

    Raises:
        ValueError: If the bounding box does not overlap the dataset grid.
    """
    # Root-zone soil moisture over the box, aggregated to `freq` windows per
    # cell (NaN fill values over water/ice are skipped).
    sm = _select_bbox(ds, bbox)[variable].resample(time=freq).mean()

    # Per-cell field capacity and wilting point over the same box; the shared
    # _select_bbox selection guarantees identical x/y coordinates, so xarray
    # broadcasts them against the (time, y, x) soil moisture cell-for-cell.
    field_capacity, wilting_point = load_field_capacity_wilting_point(
        constants_path, bbox
    )

    swdi = (sm - field_capacity) / (field_capacity - wilting_point) * 10.0

    # Average the SWDI (not the soil moisture) across the box.
    return swdi.mean(dim=("x", "y")).rename("swdi")


if __name__ == "__main__":
    #ds = virtualize_smap_l4(("2026-06-01", "2026-06-15"))

    # Persist the virtual dataset as a kerchunk virtual Zarr reference file.
    #zarr_path = save_virtual_zarr(ds, "smap_l4_virtual.json")
    #print(f"Saved virtual Zarr references to {zarr_path}")

    # Reopen straight from the references -- no re-search or re-virtualize.
    zarr_path = "inputs/SPL4SMGP_virtual_https.json"
    ds = open_virtual_zarr(zarr_path)

    # The field capacity and wilting point come from the single, static
    # SPL4SMLM land-model-constants granule.
    constants_path = download_model_constants()
    print(f"Downloaded land-model constants to {constants_path}")

    # Same bounding box as notebook (west, south, east, north):
    bbox = (-111.0, 45.0, -106.0, 50.0)
    timeseries = swdi_timeseries(ds, constants_path, bbox, freq="3D")

    out_path = "data/swdi_timeseries.csv"
    timeseries.to_dataframe().to_csv(out_path)
    print(timeseries)
    print(f"Saved time series to {out_path}")