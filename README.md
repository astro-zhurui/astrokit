# AstroKit

[![PyPI version](https://badge.fury.io/py/astrokit.svg?icon=si%3Apython)](https://badge.fury.io/py/astrokit)
[![PyPI Downloads](https://static.pepy.tech/badge/astrokit)](https://pepy.tech/projects/astrokit)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15321580.svg)](https://doi.org/10.5281/zenodo.15321580)


![logo](./docs/images/AstroKit.webp)

🌟 Current Version: **beta-0.0**

⚠️ **AstroKit is currently under development.**<br>
Please use it with caution.
Feedback and suggestions are very welcome — feel free to contact me at zhurui675@gmail.com.
The official release (v0.0) will be available soon (expected in late 2025).


## 🏁 Quick Start
**AstroKit** is a Python package for astronomical data analysis (hodgepodge 😁), developed by [Rui Zhu](https://github.com/astro-zhurui). It provides:

1. **Photometric data preprocessing** — including photometric redshift estimation, extinction correction, and more.  
2. **Wrappers for classical astronomical software** — such as **GALFIT(M)**, **STILTS**, **EAZY**, and **SExtractor**.  
3. **A collection of useful functions** developed for and inspired by my own research.
4. More features and documents are on the way!


✨ To install:<br>
- step 1: **pip install**
```bash
pip install astrokit
```
- step 2: **write your own config file**<br>
create a new file named `.astrokit_config.yaml` in your home directory. Here is an example (also in the `.astrokit_config_template.yaml` file):
```yaml
# software directory
PATH_SEX:  # SExtractor program file directory
  /opt/homebrew/Cellar/sextractor/2.28.0/share/sextractor
PATH_EAZY:  # EAZY program file directory
  /Users/rui/Applications/eazy-photoz

# Qucik Directory
PATH_DOWNLOAD:  # Default save file path
  /Users/rui/Downloads

# Project Tools Directory
PATH_PROJECT_TOOL:
  - <PATH_PROJECT1>
  - <PATH_PROJECT2>

# My Account and Password
Account_HSC: 
  username: <xxx>
  password: <xxx>
```
- Tip: Because a lot of dependencies are required, it is recommended to use [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to create a new environment and install AstroKit in it. All required packages are listed in the `requirement.txt` file.

```bash
# Create a new conda environment
conda create -n astro python=3.13
# Activate the environment
conda activate astro
# Install Required packages
pip install -r requirements.txt
# Install AstroKit
pip install astrokit
```


## 📖 Documentation
The documentation is on the way.


## 📜 License
This project is licensed under the terms of the MIT license.
See the [LICENSE](LICENSE) file for details.


## 📚 Citation
(Zenodo Software DOI)
```bibtex
@software{zhu_2025_15321580,
  author       = {Zhu, Rui},
  title        = {AstroKit: A Python Package for Astronomical Data
                   Analysis
                  },
  month        = may,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {beta-0.0},
  doi          = {10.5281/zenodo.15321580},
  url          = {https://doi.org/10.5281/zenodo.15321580},
  swhid        = {swh:1:dir:db15360761629c93ac723f4566e63570dfbe1d75
                   ;origin=https://doi.org/10.5281/zenodo.15321579;vi
                   sit=swh:1:snp:d83768d3d6f61e42e4028e488e6212d39d77
                   0eea;anchor=swh:1:rel:54bf5502e35b15d3f73552ebb3bc
                   00e154356c18;path=/
                  },
}
```

## 🤝 Acknowledgements
This project makes use of code and Data from the following projects:

### Data Cite
(astrokit.datasets.data)
- eazy_filters: Copied from the [EAZY Code](https://github.com/gbrammer/eazy-photoz/tree/master/filters)
- templates: 
    1. [lephare](http://www.cfht.hawaii.edu/~arnouts/LEPHARE/DOWNLOAD/lephare_dev_v2.2.tar.gz)
    2. [sdss](https://classic.sdss.org/dr5/algorithms/spectemplates/)


### Code Cite 
(astrokit.externals)
- HSC_pdr3/: Copied from [hsc-gitlab](https://hsc-gitlab.mtk.nao.ac.jp/ssp-software/data-access-tools/-/tree/master/pdr3)
- sfdmap.py
    Copied from [sfdmap2](https://github.com/AmpelAstro/sfdmap2/tree/main/sfdmap2)

(astrokit.ML)
- d2l.py: Modified from https://github.com/d2l-ai