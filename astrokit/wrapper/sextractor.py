"""
A Python wrapper for Source Extractor

- Author: Rui Zhu
- Creation time: 2023-09-01
"""
import os
import time
import subprocess
import re
from pathlib import Path
from loguru import logger

from astropy.table import Table

from astrokit.toolbox import run_cmd_in_terminal
from astrokit.toolbox import find_process_by_name
from astrokit.toolbox import value_to_KVD_string
from astrokit import CONFIG


__all__ = ['SExtractor']


class SExtractor:
    """
    This class provides functionality to run Source Extractor in a Pythonic way.
    
    Parameter
    ----------
    path_image: path to the science image

    """

    def __init__(self, path_image: str | Path):
        """
        initialize the SExtractor class with a single parameter, 'path_image'.
        """
        self.PATH_IMAGE = Path(path_image)  # path to the science image 
        self.PATH_WORK = self.PATH_IMAGE.parent  # path to the working directory
        self.PATH_SEX = Path(CONFIG['PATH_SEX']) # path to the directory containing SExtractor's built-in files, defined previously

        self.config_params = None  # a dict to store configuration parameters

        self.fname_config = None  # input configuration file name
        self.fname_params = None  # input catalog parameters file name
        self.fname_output_catalog = None  # output catalog file name


    def default_config_file(self) -> str:
        """generate the default configuration file for SE"""

        res = subprocess.run(args=f"sex -dd", shell=True, cwd=self.PATH_WORK, 
                             timeout=1, stdout=subprocess.PIPE, 
                             stderr=subprocess.PIPE, universal_newlines=True)
        return res.stdout

    def default_catalog_param_file(self) -> str:
        """generate the catalog parameter file of SE"""

        res = subprocess.run(args=f"sex -dp", shell=True, cwd=self.PATH_WORK, 
                             timeout=1, stdout=subprocess.PIPE, 
                             stderr=subprocess.PIPE, universal_newlines=True)
        return res.stdout

    def all_catalog_params(self) -> dict:
        """Create a Python dictionary from the default catalog parameter file."""

        content = re.split(pattern="\n", string=self.default_catalog_param_file())
        del content[-1]

        params_dict = dict()
        for row in content:
            param = row[0:26].strip("# ")
            unit = re.search(pattern=r"\[.+\]", string=row[26::])
            if isinstance(unit, type(None)):
                unit = ''
                comment = row[26::].strip()
            else:
                unit = unit.group()
                comment = row[26::].replace(unit, '').strip()
            params_dict[param] = dict()
            params_dict[param]['comment'] = comment
            params_dict[param]['unit'] = unit

        return params_dict
    
    def make_catalog_param_file(self, params:list, fname=None|str) -> None:
        """
        Generate a catalog parameters file in the working directory with a list 
        of required output parameters.

        Note
        ----
        1. The created catalog parameter file will be automatically saved in 
        the working directory.
        2. If the 'fname' argument is not defined, the default value is 
        '<image name>.param'.

        Parameter
        ---------
        params: list, required
            List of parameters required for calculation by SExtractor and 
            displayed in the output catalog.
        fname: str, optional
            The name of the catalog parameter file. If not defined, the default
            value is '<image name>.param'.
        """
        params_dict = self.all_catalog_params()

        content = "# Catalog Parameter File for SExtractor\n"
        content += "# Build by AstroKit\n\n"
        content += f"# {'Parameter':<25} {'Comment':<60} {'unit'}\n\n"

        for param in params:
            if param in params_dict.keys():
                content += f"{param:<25} # {params_dict[param]['comment']:<60} {params_dict[param]['unit']}\n"

        # write the catalog parameter file
        if fname is None:
            fname = f"{self.PATH_IMAGE.stem}.param"
        with open(self.PATH_WORK / fname, 'w') as f:
            f.write(content)
        
        self.fname_params = fname
        
        return None


    def config(self, **kwargs):
        """
        Configuring parameters for this SExtractor task.

        Note
        ----
        1. The parameter `PARAMETERS_NAME` cannot be defined here since it is 
        defined in the 'make_catalog_param_file' function and will be automatically 
        filled in the configuration file.
        2. The parameter `FILTER_NAME` is only the `file name` of the convolution 
        kernel, not the full path. The full path will be automatically filled in
        the configuration file. The convolution kernel file is located in the 
        directory where the SExtractor is installed, and the default value is
        `/opt/homebrew/Cellar/sextractor/2.28.0/share/sextractor`

        Parameter
        ---------
        kwargs: dict, optional
            The key-value pairs of the configuration parameters. If no parameters
            are defined, the default configuration file will be used.
        
        Parameters can be defined
        -------------------------
        CATALOG_NAME : str
            Name of the output catalog. The default value is '<image name>.cat'.
        CATALOG_TYPE : str
            Type of output catalog: ASCII_HEAD (default), ASCII, ASCII_SKYCAT, ASCII_VOTABLE,
            FITS_1.0, FITS_LDAC.
        FILTER_NAME : str
            Name of the file containing the convolution filter. Check this path 
            `/opt/homebrew/Cellar/sextractor/2.28.0/share/sextractor` to choice 
            the best filer.

        ... (see the default configuration file for more details)
        """

        # step1: Create a dictionary of configuration parameters from the 'default.sex' file.
        content = re.sub(pattern=r"\n\s+#\s", repl=" ", string=self.default_config_file())
        content = re.sub(pattern="Filename for XSL style-sheet", 
                        repl="# Filename for XSL style-sheet", 
                        string=content)
        content = content.split("\n")

        config = dict()
        for row in content:
            if not isinstance(re.search(pattern=r"\w+\s+.+?#.+", string=row), type(None)):
                keyword = row[0:17].strip()
                value, comment = row[17::].split('#', maxsplit=1)
                value = value.strip()
                comment = comment.strip()

                config[keyword] = dict()
                config[keyword]['value'] = value
                config[keyword]['comment'] = comment
        
        # step2: Modify three parameters to align with the default configuration required to run SExtractor.
        config['CATALOG_NAME']['value'] = f"{self.PATH_IMAGE.stem}.cat"
        if self.fname_params is None:
            logger.error("You need to define a catalog parameter file first!")
        else:
            config['PARAMETERS_NAME']['value'] = f"{self.fname_params}"

        config['FILTER_NAME']['value'] = str(self.PATH_SEX / 'default.conv')
        config['STARNNW_NAME']['value'] = str(self.PATH_SEX / 'default.nnw')
        
        # step3: Update the configuration dictionary based on the input argument.
        modified_params = dict()
        for key, value in kwargs.items():
            if key in ['PARAMETERS_NAME']:
                raise KeyError(f"No need for repetitive input of the {key} parameter!")
            
            # Convert input arguments from any format to a keyword-value-description (KVD) string.
            value = value_to_KVD_string(value)
            if key not in config.keys():
                raise KeyError(f"The keyword '{key}' is not valid for SExtractor!")
            else:
                if key in ['FILTER_NAME', 'STARNNW_NAME']:
                    old_value = Path(config[key]['value']).name
                    config[key]['value'] = str(self.PATH_SEX / value)
                else:
                    old_value = config[key]['value']
                    config[key]['value'] = value
                modified_params[key] = (value, old_value)
        
        # step4: Print the modified parameters
        logger.info("The following parameters have been modified:")
        len_k, len_v, len_old_v = 0, 0, 0
        for k in modified_params.keys():
            new_v = modified_params[k][0]
            old_v = modified_params[k][1]
            if len(k) > len_k:
                len_k = len(k)
            if len(new_v) > len_v:
                len_v = len(new_v)
            if len(old_v) > len_old_v:
                len_old_v = len(old_v)
        for k in modified_params.keys():
            new_v = modified_params[k][0]
            old_v = modified_params[k][1]
            logger.info(f"    {k:^{len_k}}: {old_v:>{len_old_v}} --> {new_v:<{len_v}}")

        self.config_params = config
        self.fname_output_catalog = config['CATALOG_NAME']['value']

        return config
    
    def config_param_class(self) -> dict:
        """
        Create a dictionary to store configuration parameters along with their 
        corresponding types from the default configuration file.
        """
        content = re.sub(pattern=r"\n\s+#\s", repl=" ", string=self.default_config_file())
        content = re.sub(pattern="Filename for XSL style-sheet", 
                        repl="# Filename for XSL style-sheet", 
                        string=content)
        content = content.split("\n")

        param_class = dict()
        for row in content:
            if row.startswith("#--"):
                param_class_name = row.strip('#- ')
                param_class[param_class_name] = list()
            if not isinstance(re.search(pattern=r"\w+\s+.+?#.+", string=row), type(None)):
                param_class[param_class_name].append(row[0:17].strip())

        return param_class
    
    def make_config_file(self, fname=None):
        """
        Generate the configuration file for SExtractor.

        Note
        ----
        Configuration parameters should be defined using the 'config' function 
        first. Subsequently, the content in the 'config' dictionary will be 
        loaded into the configuration file.

        Parameter
        ---------
        fname: str, optional
            The name of the configuration file. If not defined, the default value
            is `<image file name>.sex`
        """

        if self.config_params is None:
            config = self.config()
            logger.warning("No config parameters input, use the default configuration file")
        else:
            config = self.config_params
        config_param_class = self.config_param_class()

        # Determine the optimal length for keyword and value strings.
        lenth_keyword = 0
        lenth_value = 0
        for keyword in config.keys():
            if len(keyword) > lenth_keyword:
                lenth_keyword = len(keyword)
            if len(config[keyword]['value']) > 60:
                pass
            else:
                lenth_value = len(config[keyword]['value'])

        # edit the configuration file
        content = "# Configuration File for SExtractor\n"
        content += "# Build by AstroKit\n"

        for param_class in config_param_class.keys():
            content += f"\n#{param_class:^80}\n".replace(' ', '-')
            for param in config_param_class[param_class]:
                content += f"{param:<{lenth_keyword}}  {config[param]['value']:<{lenth_value}}  # {config[param]['comment']}\n"
        
        # write the configuration file
        if fname is None:
            fname = f"{self.PATH_IMAGE.stem}.sex"
        with open(self.PATH_WORK / fname, 'w') as f:
            f.write(content)

        self.fname_config = fname

        return None
    

    def run(self, show_in_terminal=True, fname_detection_image=None):
        """
        Run SExtractor in the terminal or in the background.

        Parameter
        ---------
        show_in_terminal: bool, optional
            If True, run SExtractor in the terminal. If False, run SExtractor
            in the background.
            
        """

        # Perform a pre-run check
        path_output_catalog = self.PATH_WORK / self.fname_output_catalog
        if path_output_catalog.exists():
            ctime0 = os.path.getctime(self.PATH_WORK / self.fname_output_catalog)
        else:
            ctime0 = 0

        # Execute this command
        cmd = f"cd {self.PATH_WORK}\n"
        if fname_detection_image is None:
            cmd += f"sex {self.PATH_IMAGE.name} -c {self.fname_config}"
        else:
            cmd += f"sex {fname_detection_image} {self.PATH_IMAGE.name} -c {self.fname_config}"
            logger.info(f"SExtractor will run in double image mode. The detection image is '{fname_detection_image}'.")
        
        if show_in_terminal:
            logger.info("running SExtractor in the terminal...")
            st = time.time()
            
            run_cmd_in_terminal(cmd)  # run SExtractor in the terminal

            # Check if SExtractor has finished execution
            time.sleep(1)
            if find_process_by_name('sex') == False:
                if path_output_catalog.exists():
                    ctime = os.path.getctime(self.PATH_WORK / self.fname_output_catalog)
                    if ctime > ctime0:
                        logger.warning(f"SExtractor has finished! (Was it too fast?")
                    else:
                        logger.error("SExtractor failed!")
                else:
                    logger.error("SExtractor failed!")
            else:
                while True:
                    ctime = os.path.getctime(self.PATH_WORK / self.fname_output_catalog)
                    if (ctime > ctime0) & (find_process_by_name('sex') == False):
                        logger.success(f"SExtractor finished! cost time: {time.time()-st:.2f} s")
                        break

        else:  # run SExtractor in the background
            logger.info("running SExtractor...")
            st = time.time()
            res = subprocess.run(args=cmd, shell=True, 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE, 
                                 universal_newlines=True)
            if res.returncode == 0:
                logger.success(f"SExtractor finished! cost time: {time.time()-st:.2f} s")
            else:
                logger.error("SExtractor failed!")

        return None
    
    def get_output_catalog(self):
        """Read the output catalog of SExtractor."""
        return Table.read(self.PATH_WORK / self.fname_output_catalog, format='ascii.sextractor')


