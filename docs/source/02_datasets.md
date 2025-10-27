# astrokit.datasets
This module provides several python classes to deal with large datasets including: Subaru HSC survey, DESI Legacy Imaging Surveys, NED database. More datasets will be added in the future when I use them :)

## HSCRetriever

## LegacySurvey

- **Data Preparation**

Before using this code. You need to download all the data from 
https://www.legacysurvey.org into your local data root directory (e.g. <DIR_DATA> = /home/rui/Data). The <DIR_DATA> should figure in the astrokit config file (e.g. ~/.astrokit_config.yaml). And the directory tree should look like this:
```
<DIR_DATA>/
    legacysurvey/
        dr9_north/
            tractor/<AAA>/tractor-<brickname>.fits  # tractor catalog files
            survey-bricks-dr9-north.fits.gz  # brick info file
        dr10_south/
            tractor/<AAA>/tractor-<brickname>.fits  # tractor catalog files
            survey-bricks-dr10-south.fits.gz  # brick info file
```

- LegacySurvey Initialization
```python
from astrokit.datasets import LegacySurvey
legacysurvey = LegacySurvey()  # If <DIR_DATA>/legacysurvey is not existed, it will raise an error
```

- check the position for each brick of DR9 and DR10
```python
legacysurvey.bricksinfo
```

- find bricks covering a specific position
```python
"""
Parameters
----------
ra: float
    Right ascension in degrees
dec: float
    Declination in degrees
search_radius: float
    Search radius in arcseconds
show: bool
    Whether to show the bricks and the target on a plot
"""
legacysurvey.find_bricks(ra=obj['ra'], dec=obj['dec'], search_radius=60, show=True)
```

## query_NED