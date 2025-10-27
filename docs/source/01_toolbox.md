# astrokit.toolbox

This module provides various convenience functions for various tasks commonly used in my research work. Such as magnitude conversions and some functions related to calculations, cross-matching, and useful functions for plotting.

Any functions in this module can be called via the unified ways:
```python
from astrokit.toolbox import <function_name>
```
For example:
```python
from astrokit.toolbox import fnu_to_ABmag
```

And you can also use the quick access (some commonly used fuctions only show in <_top_level_map.py> file):
```python
import astrokit as ak

tbl = ak.read(<fits_file_path>)
ak.imshow(<image_data>)
```