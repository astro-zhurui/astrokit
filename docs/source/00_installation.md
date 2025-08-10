## Installation

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
```

- step 3: **Enjoy it!**<br>
```python
import astrokit as ak
```

🔔 Tip: Because a lot of dependencies are required, it is recommended to use [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to create a new environment and install AstroKit in it. All required packages are listed in the `requirement.txt` file.

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
