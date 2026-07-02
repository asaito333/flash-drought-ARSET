import warnings
from collections.abc import Sequence

import earthaccess
import xarray as xr
from pyproj import Transformer

# EASE-Grid 2.0 Global (9 km) projection used by SMAP L4 x/y coordinates (meters).
EASE2_GLOBAL_EPSG = "EPSG:6933"


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

    Parameters:
        ds: A single (virtual) granule dataset.

    Returns:
        The dataset with time collapsed to a scalar coordinate.
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
) -> xr.Dataset:
    """Build a virtual dataset of SMAP L4 geophysical variables over a time range.

    SMAP L4 (SPL4SMGP) stores the coordinate variables (time, x, y)
    in the file's root group, while the geophysical variables such as
    sm_rootzone live in the Geophysical_Data subgroup as 2-D (y, x)
    arrays with no time coordinate of their own.  Because
    open_virtual_mfdataset parses one group per call, the root group and the
    geophysical group are opened separately and the root coordinates are then
    attached to the stacked geophysical variables.

    Parameters:
        temporal: (start, end) date strings passed to earthaccess.
        group: The subgroup holding the geophysical variables.
        variables: Variables to keep from group.  None keeps the whole
            group; the default keeps only sm_rootzone.

    Returns:
        A virtual xarray.Dataset with the requested variables stacked along
        a time dimension and carrying the time/x/y coordinates.
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
        "load": True,
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
    result_root = earthaccess.virtualize(
        granules=results,
        data_vars="minimal",
        preprocess=collapse_time_to_scalar,
        loadable_variables=["time", "x", "y"],
        **open_options, # type: ignore
    )

    result_gph = earthaccess.virtualize(
        granules=results,
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

    Parameters:
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


def sm_rootzone_timeseries(
    ds: xr.Dataset,
    bbox: tuple[float, float, float, float],
    freq: str = "3D",
    variable: str = "sm_rootzone",
) -> xr.DataArray:
    """Spatially average a variable over a lat/lon box and aggregate in time.

    Parameters:
        ds: The dataset returned by :func:`virtualize_smap_l4`.
        bbox: (west, south, east, north) in degrees (lon/lat).
        freq: Pandas offset alias for the temporal aggregation window
            ("3D" = 3-day means).
        variable: The variable to aggregate.

    Returns:
        A 1-D DataArray of the box-averaged, freq-aggregated variable
        indexed by time.

    Raises:
        ValueError: If the bounding box does not overlap the dataset grid.
    """
    x_min, y_min, x_max, y_max = latlon_bbox_to_ease(bbox)
    subset = ds.sel(
        x=_bounds_slice(ds.x, x_min, x_max),
        y=_bounds_slice(ds.y, y_min, y_max),
    )
    if subset.sizes["x"] == 0 or subset.sizes["y"] == 0:
        msg = f"Bounding box {bbox} does not overlap the dataset grid."
        raise ValueError(msg)

    # Mean over the box (NaN fill values over water/ice are skipped), then
    # aggregate the native 3-hourly steps into `freq` windows.
    spatial_mean = subset[variable].mean(dim=("x", "y"))
    return spatial_mean.resample(time=freq).mean()


if __name__ == "__main__":
    ds = virtualize_smap_l4(("2026-06-01", "2026-06-15"))

    # Same bounding box as notebook (west, south, east, north):
    bbox = (-111.0, 45.0, -106.0, 50.0)
    timeseries = sm_rootzone_timeseries(ds, bbox, freq="3D")

    out_path = "sm_rootzone_timeseries.csv"
    timeseries.to_dataframe().to_csv(out_path)
    print(timeseries)
    print(f"Saved time series to {out_path}")