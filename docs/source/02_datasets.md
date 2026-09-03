# astrokit.datasets
This module provides several python classes to deal with large datasets including: Subaru HSC survey, DESI Legacy Imaging Surveys, NED database. More datasets will be added in the future when I use them :)

## HSCRetriever

## LegacySurvey

`LegacySurvey` provides utilities for working with local DESI Legacy Imaging Surveys
DR11 North and South tractor catalogs. It helps you locate bricks, read local
tractor files, collect catalogs around a source list, and download coadd images.

### Data Preparation

Download the Legacy Survey DR11 data from https://www.legacysurvey.org and organize
the files in a local directory. The expected directory tree is:

```text
<DIR_LEGACYSURVEY>/
        dr11_north/
            survey-bricks-dr11-north.fits.gz
            tractor/<AAA>/tractor-<brickname>.fits
        dr11_south/
            survey-bricks-dr11-south.fits.gz
            tractor/<AAA>/tractor-<brickname>.fits
```

Here `<AAA>` is the first three characters of a brick name. For example,
`tractor-3375m395.fits` should be placed under `tractor/337/`.

### Initialization

```python
from pathlib import Path
from astrokit.datasets import LegacySurvey

dir_legacysurvey = Path("/path/to/legacysurvey")
legacysurvey = LegacySurvey(dir_legacysurvey)
```

During initialization, `LegacySurvey` loads a combined brick table from DR11 North
and South:

```python
bricksinfo = legacysurvey.bricksinfo
bricksinfo.head()
```

The `bricksinfo` table contains one row per brick, with columns such as:

| Column | Description |
| --- | --- |
| `region` | `north` or `south` |
| `AAA` | First three characters of the brick name |
| `brickname` | Legacy Survey brick name |
| `ra`, `dec` | Brick center coordinate in degrees |
| `ra1`, `ra2`, `dec1`, `dec2` | Brick boundary in degrees |
| `survey_primary` | Whether this brick is the survey-primary coverage |

The combined table is cached as `legacysurvey_bricksinfo.parquet` in
`dir_legacysurvey` after it is created for the first time.

### Find Bricks Around One Position

Use `find_brickname()` when you only need the brick containing a coordinate:

```python
brickname_dict = legacysurvey.find_brickname(ra=337.55, dec=-39.45)
print(brickname_dict)
```

Use `find_bricks()` when you need all bricks overlapping a circular search region
around one sky position:

```python
bricks = legacysurvey.find_bricks(
    ra=337.55,
    dec=-39.45,
    search_radius=60,  # arcsec
    show=True,
)
bricks
```

`search_radius` is in arcseconds. If `show=True`, the method plots the target,
the search circle, and the overlapping bricks.

### Find Bricks Around Many Positions

For a large source list, use `find_bricks_from_list()`. This method first uses a
KD-tree on the sphere to select candidate bricks and then performs a fast local
plane circle-brick overlap check. It is much faster than looping over
`find_bricks()` source by source and is intended for arcmin-scale search radii.

```python
bricks = legacysurvey.find_bricks_from_list(
    ra=source_table["ra"],
    dec=source_table["dec"],
    search_radius="5 arcmin",
    max_workers=8,
)
bricks.head()
```

The returned object is a `pandas.DataFrame` with the same style as
`legacysurvey.bricksinfo`, but it only keeps the bricks overlapping at least one
input source search region. Duplicated bricks are removed.

`search_radius` accepts:

- a number, interpreted as arcseconds, for consistency with `find_bricks()`;
- a string such as `"5 arcmin"` or `"300 arcsec"`;
- an `astropy.units.Quantity`, such as `5*u.arcmin`.

Useful options:

| Option | Description |
| --- | --- |
| `max_workers` | Number of parallel workers used during exact filtering. Use `1` to disable parallel execution. Filtering is automatically chunked from the source count and worker count. |
| `check_file_exists` | If `True`, add `file_is_ready`, indicating whether the local tractor file exists. |
| `show_progress` | If `True`, show a progress bar. |

### Locate and Read Tractor Catalogs

Use `find_tractor_file()` to locate the local tractor file for one brick:

```python
path = legacysurvey.find_tractor_file(
    region="south",
    brickname="3375m395",
)
print(path)
```

Use `load_tractor_catalog()` to read a tractor catalog into a
`pandas.DataFrame`:

```python
cat = legacysurvey.load_tractor_catalog(
    brickname="3375m395",
    region="south",
    columns=["ra", "dec", "type", "flux_g", "flux_r", "flux_z"],
)
cat.head()
```

Multi-dimensional FITS columns are expanded into separate columns with
1-based suffixes, for example `flux_ivar_1`, `flux_ivar_2`, etc.

### Collect Matches Around a Source List

`collect_matches()` finds overlapping bricks, loads each local tractor catalog,
and returns every Legacy Survey source within the search radius. DR11 North and
South are matched separately with `fast_match(mode="all")`. Nothing is written to disk.

```python
matches = legacysurvey.collect_matches(
    ra=source_table["ra"],
    dec=source_table["dec"],
    search_radius="1 arcmin",
    id=source_table["id"],
)
matches["south"].head()
```

`search_radius` accepts a number (arcseconds), a string such as `"1 arcmin"`,
or an astropy quantity. If `id` is omitted, positional indices `0 .. N-1` are
stored in the output `id` column. By default all CPU cores are used.

The returned object is a dict of DataFrames:

| Key | Description |
| --- | --- |
| `north` | Matched DR11 North tractor rows, with input `id`, `sep` (arcsec), and `ls_id_dr11` |
| `south` | Matched DR11 South tractor rows, with the same extra columns |

### Download Coadd Images

Use `download_image()` to download one coadd image file from the Legacy Survey
portal:

```python
legacysurvey.download_image(
    region="south",
    brickname="3375m395",
    band="r",
    dir_output=Path("images"),
)
```

## query_NED
