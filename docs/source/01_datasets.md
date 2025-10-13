## astrokit.datasets

### HSCRetriever

### LegacySurvey
`LegacySurvey` is a module of astrokit.datasets that provides one main class `LegacySurvey` to deal with the DESI Legacy Imaging Surveys Tractor Catalogs (DR9 & DR10). 

Before using this code. You need to download all the datasets from 
https://www.legacysurvey.org into your local data root directory (e.g. DIR_DATA = /home/rui/Data). The DIR_DATA should figure in the astrokit config file (e.g. ~/.astrokit_config.yaml). And the directory tree should look like this:
```
DIR_DATA/
    legacysurvey/
        dr9_north/
            tractor/
            ...
        dr10_south/
            tractor/
            ...
```

### query_NED