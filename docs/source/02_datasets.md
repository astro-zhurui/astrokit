# astrokit.datasets
This module provides several python classes to deal with large datasets including: Subaru HSC survey, DESI Legacy Imaging Surveys, NED database. More datasets will be added in the future when I use them :)

## HSCRetriever

## LegacySurvey

`LegacySurvey` provides utilities for working with local DESI Legacy Imaging Surveys
data, especially DR9 North and DR10 South tractor catalogs. It helps you locate
bricks, read local tractor files, collect catalogs around a source list, and
download coadd images.

### Data Preparation

Download the Legacy Survey data from https://www.legacysurvey.org and organize
the files in a local directory. The expected directory tree is:

```text
<DIR_LEGACYSURVEY>/
        dr9_north/
            survey-bricks-dr9-north.fits.gz
            tractor/<AAA>/tractor-<brickname>.fits
        dr10_south/
            survey-bricks-dr10-south.fits.gz
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

During initialization, `LegacySurvey` loads a combined brick table from DR9 North
and DR10 South:

```python
bricksinfo = legacysurvey.bricksinfo
bricksinfo.head()
```

The `bricksinfo` table contains one row per brick, with columns such as:

| Column | Description |
| --- | --- |
| `release` | Data release, either `dr9` or `dr10` |
| `AAA` | First three characters of the brick name |
| `brickname` | Legacy Survey brick name |
| `ra`, `dec` | Brick center coordinate in degrees |
| `ra1`, `ra2`, `dec1`, `dec2` | Brick boundary in degrees |

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
KD-tree on the sphere to select candidate bricks and then performs an exact
circle-brick overlap check. It is much faster than looping over
`find_bricks()` source by source.

```python
bricks = legacysurvey.find_bricks_from_list(
    ra=source_table["ra"],
    dec=source_table["dec"],
    search_radius="5 arcmin",
    max_workers=8,
    chunksize=5000,
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

If you also need to know which bricks belong to each input source, use
`return_per_source=True`:

```python
bricks, bricks_per_source = legacysurvey.find_bricks_from_list(
    ra=source_table["ra"],
    dec=source_table["dec"],
    search_radius="5 arcmin",
    max_workers=8,
    return_per_source=True,
)

bricksinfo_indexed = legacysurvey.bricksinfo.reset_index(drop=True)
first_source_bricks = bricksinfo_indexed.iloc[bricks_per_source[0]]
first_source_bricks[["release", "brickname"]]
```

Useful options:

| Option | Description |
| --- | --- |
| `max_workers` | Number of parallel workers used during exact filtering. Use `1` to disable parallel execution. |
| `chunksize` | Number of input sources in each parallel task. A value like `5000` or `10000` is usually reasonable for large catalogs. |
| `return_per_source` | If `True`, return `(bricks, bricks_per_source)`. |
| `check_files` | If `True`, add `file_is_ready`, indicating whether the local tractor file exists. |
| `show_progress` | If `True`, show a progress bar. |

### Locate and Read Tractor Catalogs

Use `find_tractor_file()` to locate the local tractor file for one brick:

```python
path = legacysurvey.find_tractor_file(
    release="dr10",
    brickname="3375m395",
)
print(path)
```

Use `load_tractor_catalog()` to read a tractor catalog into a
`pandas.DataFrame`:

```python
cat = legacysurvey.load_tractor_catalog(
    brickname="3375m395",
    release="dr10",
    columns=["ra", "dec", "type", "flux_g", "flux_r", "flux_z"],
)
cat.head()
```

Multi-dimensional FITS columns are expanded into separate columns with
1-based suffixes, for example `flux_ivar_1`, `flux_ivar_2`, etc.

### Query Catalogs Around a Source List

`query_catalogs_around_source()` is a higher-level workflow for source-list
matching. It:

1. writes the input source table to FITS;
2. finds all relevant bricks;
3. combines the local tractor files for DR9 and DR10 separately;
4. cross-matches the input sources with the combined tractor catalogs using
   STILTS.

```python
from pathlib import Path

legacysurvey.query_catalogs_around_source(
    input_cat=source_table,
    output_dir=Path("output/legacy_survey_match"),
    task_name="my_sources",
    search_radius=5,       # arcsec, passed to STILTS matching
    col_ra="ra",
    col_dec="dec",
    max_workers=16,
)
```

Expected outputs include:

| Output | Description |
| --- | --- |
| `<task_name>_input_catalog.fits` | Input source catalog |
| `<task_name>_bricksinfo.pkl` | Bricks found for each input source |
| `<task_name>_dr9_all.fits`, `<task_name>_dr10_all.fits` | Combined tractor catalogs |
| `<task_name>_dr9.fits`, `<task_name>_dr10.fits` | Final matched catalogs |

This method requires STILTS to be available through `astrokit.wrapper.stilts`.

### Make Legacy Survey IDs

`make_ls_id()` creates string identifiers from release, brickid, and objid:

```python
ls_id = legacysurvey.make_ls_id(
    release=cat["release"],
    brickid=cat["brickid"],
    objid=cat["objid"],
)
```

The output format is:

```text
<release>_<brickid>_<objid>
```

### Download Coadd Images

Use `download_image()` to download one coadd image file from the Legacy Survey
portal:

```python
legacysurvey.download_image(
    release="dr10",
    brickname="3375m395",
    band="r",
    dir_output=Path("images"),
)
```

## query_NED
