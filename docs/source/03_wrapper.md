## astrokit.wrapper
This module provides wrappers for various classic astronomical software tools which are written in other programming languages (mostly in Java or C/C++). The wrappers are designed to facilitate the usage of these tools within Python scripts, allowing for seamless integration into data analysis workflows.

Currently, the following wrappers are available:
1. GALFIT: https://users.obs.carnegiescience.edu/peng/work/galfit/galfit.html
2. GALFITM: https://www.nottingham.ac.uk/astronomy/megamorph/
3. EAZY: http://www.astro.yale.edu/eazy/
4. SExtractor: https://sextractor.readthedocs.io/en/latest/index.html
5. STILTS: https://www.star.bris.ac.uk/~mbt/stilts/

### astrokit.wrapper.galfitm

### astrokit.wrapper.galfit

### astrokit.wrapper.eazy

### astrokit.wrapper.stilts
The STILTS software provides a fast and comprehensive suite of command-line tools for processing tabular data. The wrapper functions in this module allow users to call STILTS commands directly from Python, the parameters are very similar to the command-line usage.

The official STILTS website: https://www.star.bris.ac.uk/~mbt/stilts/

First, import the wrapper module:
```python
from astrokit.wrapper import stilts
```

Then, you can call any STILTS command via the corresponding wrapper functions.

1. [tcat](https://www.star.bristol.ac.uk/mbt/stilts/sun256/tcat.html): Concatenate multiple similar tables into one.
```python
stilts.tcat(
    path_in=["file1.fits", "file2.fits"],  # input table path list
    path_out="output.fits",
    stilts_flags='-verbose', 
    ifmt='fits',  # input format
    multi=True, 
    omode='out', 
    seqcol='from',  # or loccol, uloccol
    countrows=True,
    silent=False
    )
```

### astrokit.wrapper.sextractor