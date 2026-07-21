# astrokit.simqso

`astrokit.simqso` is a bundled and patched copy of
[simqso](https://github.com/imcgreer/simqso), adapted for recent Python,
SciPy, and Astropy versions. It is designed to generate mock quasar spectra,
apply Lyman-series and Lyman-limit IGM absorption, compute synthetic
photometry, and produce survey-like quasar catalogs.

The public entry points are available from:

```python
from astrokit import simqso
from astrokit.simqso import sqbase, sqgrids, sqmodels, sqphoto, sqrun
```

## What it can do

The module can be used to:

- create quasar grids or point samples in luminosity-redshift or
  flux-redshift space;
- generate quasar continua, broad emission lines, Fe emission templates, and
  optional dust extinction;
- add HI IGM absorption using built-in absorber population models;
- compute synthetic photometry through bundled filter curves;
- add survey-like photometric errors for supported surveys;
- write simulated catalogs and, optionally, simulated spectra to FITS files.

The bundled data files include SDSS-style filter curves, emission-line trend
tables, Fe templates, and the Boss DR9 K-correction grid. They are loaded from
`astrokit/simqso/data` automatically.

## Recommended workflows

There are two common ways to use the module.

### High-level simulation dictionary

Use `qsoSimulation()` when your simulation can be described by the original
simqso parameter dictionary:

```python
from astropy.cosmology import FlatLambdaCDM
from astrokit import simqso

sim_params = {
    "FileName": "my_qso_mock",
    "RandomSeed": 12345,
    "Cosmology": FlatLambdaCDM(H0=70, Om0=0.3),
    "waveRange": (3000.0, 11000.0),
    "SpecDispersion": 600,
    "GridParams": {
        "GridType": "LuminosityRedshiftGrid",
        "mRange": (-29.0, -22.0, 5),
        "zRange": (0.05, 4.0, 8),
        "nPerBin": 10,
        "LumUnits": 1450.0,
        "ObsBand": "SDSS-i",
        "RestBand": 1450.0,
    },
    "ForestParams": {
        "ForestModel": "Worseck&Prochaska2011",
        "NumLinesOfSight": 200,
        "seed": 12346,
    },
    "QuasarModelParams": {
        "ContinuumParams": {
            "ContinuumModel": "BrokenPowerLaw",
            "PowerLawSlopes": [
                (-1.50, 0.30), 1100.0,
                (-0.50, 0.30), 5700.0,
                (-0.37, 0.30), 9730.0,
                (-1.70, 0.30), 22300.0,
                (-1.03, 0.30),
            ],
        },
        "EmissionLineParams": {
            "EmissionLineModel": "VariedEmissionLineGrid",
            "EmissionLineTrendFilename": "emlinetrends_v6",
            "minEW": 0.5,
        },
    },
    "PhotoMapParams": {
        "PhotoSystems": [("SDSS", "Legacy")],
    },
}

qso_grid, spectra = simqso.qsoSimulation(
    sim_params,
    saveSpectra=True,
    outputDir="simqso_output",
    nproc=1,
)
```

`qso_grid.data` is an `astropy.table.Table` containing the simulated object
parameters and photometry. When `saveSpectra=True`, `spectra` is a 2D array
with shape `(Nqso, Nwave)`.

### Explicit point-sample workflow

For notebooks and custom experiments, it is often clearer to build the quasar
points explicitly and then add model variables. This also gives full control
over the redshift and luminosity distribution.

```python
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astrokit.simqso import sqbase, sqgrids, sqmodels, sqphoto, sqrun

rng = np.random.default_rng(12345)
n_qso = 500
z = rng.uniform(0.05, 4.0, n_qso)
m1450 = rng.uniform(-29.0, -22.0, n_qso)

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
wave = sqbase.fixed_R_dispersion(3000.0, 11000.0, 600)

qso_grid = sqgrids.QsoSimPoints(
    [
        sqgrids.AbsMagVar(sqgrids.FixedSampler(m1450), restWave=1450.0),
        sqgrids.RedshiftVar(sqgrids.FixedSampler(z)),
    ],
    cosmo=cosmo,
    units="luminosity",
)

qso_grid.loadPhotoMap([("SDSS", "Legacy")])
qso_grid.addVars(
    sqmodels.get_BossDr9_model_vars(
        qso_grid,
        wave,
        nSightLines=200,
        forestseed=12346,
    )
)

# Use the sightline-aware builder when an IGMTransmissionGrid is present.
qso_grid, spectra = sqrun.buildSpectraBySightLine(
    wave,
    qso_grid,
    saveSpectra=True,
)

obs_photo = sqphoto.calcObsPhot(qso_grid.synFlux, qso_grid.photoMap, seed=12347)
qso_grid.addData(obs_photo)
```

`buildSpectraBySightLine()` is the preferred low-level builder when HI IGM
absorption is enabled. It processes each sightline in increasing redshift order,
which is required by the IGM transmission grid.

## Photometry systems

Photometry is handled by `sqphoto.load_photo_map()`, usually through
`qso_grid.loadPhotoMap()`. The supported built-in systems include:

| System | Survey | Bands | Notes |
| --- | --- | --- | --- |
| `SDSS` | `Legacy` | `ugriz` | asinh magnitudes with SDSS-like errors |
| `SDSS` | `Stripe82` | `ugriz` | AB magnitudes with empirical errors |
| `CFHT` | `CFHTLS_Wide` | `ugriz` | AB magnitudes |
| `UKIRT` | `UKIDSS_LAS` | `YJHK` | AB magnitudes |
| `UKIRT` | `UKIDSS_DXS` | `JHK` | AB magnitudes |
| `WISE` | `AllWISE` | `W1`, `W2` | AB magnitudes |
| `TMASS` | `Allsky` | `JHK` | AB magnitudes |
| `DECam` | `DECaLS` | `grz` | AB magnitudes |
| `DECam` | `DES` | `grizy` | AB magnitudes |
| `BASS-MzLS` | `BASS-MzLS` | `grz` | AB magnitudes |
| `HSC` | `Wide` | `grizy` | AB magnitudes |
| `LSST` | `Wide` | `ugrizy` | AB magnitudes |

Synthetic magnitudes are stored in `synMag`, synthetic fluxes in `synFlux`
(nanomaggies), and perturbed survey-like measurements in `obsMag`,
`obsMagErr`, `obsFlux`, and `obsFluxErr`.

## Built-in quasar and IGM models

`sqmodels` provides reusable model components:

- `get_BossDr9_model_vars()` builds a Boss DR9-like quasar model with broken
  power-law continuum, Baldwin-effect emission-line templates, Fe templates,
  and optional IGM absorption.
- `BOSS_DR9_PLEpivot()` returns a double power-law luminosity function model.
- `QLF_McGreer_2013` provides a high-redshift luminosity-function model.
- `forestModels` contains `Fan1999`, `Worseck&Prochaska2011`, and
  `McGreer+2013` absorber-population presets.

For most SDSS-like mock catalogs, `get_BossDr9_model_vars()` with
`ForestModel="Worseck&Prochaska2011"` is a practical starting point.

## Demo notebook

A complete worked example is included in:

```text
demo/notebooks/astrokit_demo_simqso.ipynb
```

The notebook simulates `0 < z < 4` quasar spectra, adds IGM absorption,
generates an SDSS-like mock quasar catalog, and compares the simulated colors
against a local SDSS DR16Q catalog.

## Caveats

`astrokit.simqso` is useful for mock catalog construction and selection
function experiments, but the default examples are not fully calibrated survey
selection models. For quantitative work, you should validate the simulated
redshift, luminosity, color, and photometric-error distributions against the
survey sample used in your science analysis.

The IGM absorber grid is stateful: when using low-level APIs, process spectra
with `buildSpectraBySightLine()` if `HIAbsorptionVar` is present.

## API Reference

### Top-level entry points

```{eval-rst}
.. automodule:: astrokit.simqso
   :members:
   :undoc-members:
```

### sqrun

```{eval-rst}
.. automodule:: astrokit.simqso.sqrun
   :members: qsoSimulation, buildWaveGrid, buildQsoGrid, buildForest, buildFeatures, buildSpectraBySightLine, buildSpectraBulk, buildQsoSpectrum, load_sim_output, save_spectra, load_spectra
```

### sqgrids

```{eval-rst}
.. automodule:: astrokit.simqso.sqgrids
   :members: FixedSampler, UniformSampler, GaussianSampler, AppMagVar, AbsMagVar, RedshiftVar, QsoSimObjects, QsoSimPoints, QsoSimGrid, generateQlfPoints, generateBEffEmissionLines
```

### sqmodels

```{eval-rst}
.. automodule:: astrokit.simqso.sqmodels
   :members: forestModels, get_BossDr9_model_vars, BOSS_DR9_PLEpivot, QLF_McGreer_2013
```

### sqphoto

```{eval-rst}
.. automodule:: astrokit.simqso.sqphoto
   :members: supported_photo_systems, load_photo_map, getPhotoCache, calcSynPhot, calcObsPhot, nmgy2abmag, abmag2nmgy, nmgy2asinhmag, asinhmag2nmgy
```
