# Installation

## Install AstroKit

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

# Quick Directory
DIR_DOWNLOAD:  # Default save file path
  /Users/rui/Downloads
DIR_DATA:  # local data root directory. Astrokit will search datasets in this directory
  /Users/rui/Data
```

- step 3: **Enjoy it!**<br>
```python
import astrokit as ak
```

🔔 Tip: Because a lot of dependencies are required, it is recommended to use [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to create a new environment and install AstroKit in it. All required packages are listed in the `requirements.txt` file.

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

##  Install Auxiliary Softwares
🔔 Note that not all auxiliary softwares are required. It is depends on which modules you want to use. First of all, create a directory for auxiliary software installation, e.g. `~/opt/`
```bash
cd ~
mkdir opt
```
Then, add this in your shell configuration file (e.g. `~/.zshrc` or `~/.bashrc`):
```bash
# Applications
export PATH="/home/rui/opt:$PATH"
```
Finally, restart your terminal or run `source ~/.zshrc` to make the changes take effect.

### STILTS Installation
- Introduction: [STILTS](https://www.star.bristol.ac.uk/mbt/stilts/) is the command-line counterpart of [TOPCAT](https://www.star.bristol.ac.uk/mbt/topcat/) which provides a powerful set of tools for very large tabluar data. It is widely used in astronomy for manipulating and visualizing tabular data.

- For MacOS, you can use [Homebrew](https://brew.sh/) to install STILTS:
```bash
brew install --cask topcat --no-quarantine
```

- For Linux, following steps are recommended to install TOPCAT and STILTS together:
```bash
mkdir ~/opt/topcat
cd ~/opt/topcat

# download the latest version of TOPCAT and STILTS
wget https://www.star.bristol.ac.uk/mbt/topcat/topcat-extra.jar  # bloated version including Parquet file format support
wget https://www.star.bristol.ac.uk/mbt/topcat/topcat
wget https://www.star.bristol.ac.uk/mbt/stilts/stilts
chmod +x topcat
chmod +x stilts
```

Then, adding following content to environment variable
```bash
export PATH="/home/rui/opt/topcat:$PATH"
```

- Test TOPCAT and STILTS installation:
```bash
topcat -version
stilts -version
```