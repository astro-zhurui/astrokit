"""
A Python interface for GALFITM

- Author: Rui Zhu
- Date: 2025-07-08
"""
from pathlib import Path
from loguru import logger

from astrokit.toolbox.utils import run_command

class GalfitMModel:
    """
    定义GalfitM模型
    """

    def __init__(self):
        self._model_idx = 0
        self.feedme = ""

    def _list2CSL(self, input, add_cheb=False):
        """将参数列表转换成逗号分隔行字符串"""
        if isinstance(input, list):
            input = str(input).strip('[]').replace(' ', '')
        if add_cheb:
            input = f"{input} cheb"

        return input

    def _feedme_row(self, c1, c2, c3, c4):
        """
        将模型结构参数写入feedme格式的字符串

        Parameter
        ---------

        c1 : Parameter number
        c2 : parameter name OR value
        c3 : the order of the Chebyshev series
            - 0 = fixed to input value(s)
            - 1 = fit a constant offset from the input value(s)
            - 2 = fit a linear function of wavelength
            - 3 = fit a quadratic function of wavelength, etc.
        c4 : comment
        """
        return f"{c1:>2}) {c2:<10} {c3:<10} {c4:<20}\n"

    def add_psf(
            self, 
            x=None, fit_x=1, 
            y=None, fit_y=1, 
            mag=None, fit_mag=1, 
            skip_in_output=False
            ):
        """
        PSF model

        Parameter
        ---------
        x : position x [pixel]
        y : position y [pixel]
        mag : total magnitude
        skip_in_output : Skip this model in output image?  (yes=1, no=0)
        """
        model_name = 'psf'
        content = f"\n# Component number: {self._model_idx}\n"

        content += self._feedme_row(
            c1='0', 
            c2=model_name, c3='', 
            c4="# object type"
        )
        content += self._feedme_row(
            c1='1', c2=x, c3=fit_x, 
            c4="# position x [pixel]"
        )
        content += self._feedme_row(
            c1='2', c2=y, c3=fit_y, 
            c4="# position y [pixel]"
        )
        content += self._feedme_row(
            c1='3', c2=mag, c3=fit_mag, 
            c4="# total magnitude"
        )
        content += self._feedme_row(
            c1='z', c2=skip_in_output, c3='', 
            c4="#  Skip this model in output image?  (yes=1, no=0)"
        )
        self.feedme += content
        self._model_idx += 1
        return None
    
    def add_sersic(
            self, 
            x='', fit_x=1, 
            y='', fit_y=1, 
            mag='', fit_mag=1, cheb_mag=False, 
            re='', fit_re=1, cheb_re=False,
            sersic_index='', fit_sersic_index=1, cheb_sersic_index=False,  
            axis_ratio='', fit_axis_ratio=1, cheb_axis_ratio=False, 
            PA='', fit_PA=1, cheb_PA=False, 
            skip_in_output=False
    ):
        model_name = 'sersic'
        content = f"\n# Component number: {self._model_idx}\n"

        content += self._feedme_row(
            c1=0, c2=model_name, c3='', 
            c4="# Object type" 
        )
        content += self._feedme_row(
            c1=1, 
            c2=self._list2CSL(x), 
            c3=self._list2CSL(fit_x), 
            c4="# position x [pixel]"
        )
        content += self._feedme_row(
            c1=2, 
            c2=self._list2CSL(y), 
            c3=self._list2CSL(fit_y), 
            c4="# position y [pixel]"
        )
        content += self._feedme_row(
            c1=3, 
            c2=self._list2CSL(mag), 
            c3=self._list2CSL(fit_mag, cheb_mag), 
            c4="# total magnitude in each band"
        )
        content += self._feedme_row(
            c1=4, 
            c2=self._list2CSL(re), 
            c3=self._list2CSL(fit_re, cheb_re), 
            c4="# R_e in each band"
        )
        content += self._feedme_row(
            c1=5, 
            c2=self._list2CSL(sersic_index), 
            c3=self._list2CSL(fit_sersic_index, cheb_sersic_index), 
            c4="# Sersic exponent in each band", 
        )
        content += self._feedme_row(
            c1=9, 
            c2=self._list2CSL(axis_ratio), 
            c3=self._list2CSL(fit_axis_ratio, cheb_axis_ratio), 
            c4="# axis ratio (b/a) in each band", 
        )
        content += self._feedme_row(
            c1=10, 
            c2=self._list2CSL(PA), 
            c3=self._list2CSL(fit_PA, cheb_PA), 
            c4="# position angle (PA), same value in each band", 
        )
        content += self._feedme_row(
            c1='z', c2=skip_in_output, c3='', 
            c4="#  Skip this model in output image?  (yes=1, no=0)"
        )
        self.feedme += content
        self._model_idx += 1
        return None

class GalfitM:
    def __init__(
            self, 
            dir_output, 
            task_name, 
            path_list_input_img=[], 
            path_list_input_psf=[], 
            path_list_input_sigma=[], 
            path_list_input_mask=[],
            output_type='optimize', 
            output_items=['input', 'model', 'residual', 'component'],
            ):
        
        self.dir_output = Path(dir_output)

        if not self.dir_output.exists():
            self.dir_output.mkdir(parents=True, exist_ok=True)

        self.path_output_img = self.dir_output / f"{task_name}_galfitm.fits"
        self.path_feedme = self.dir_output / f"{task_name}_galfitm.feedme"
        self.path_constraints = self.dir_output / f"{task_name}_galfitm.constraints"

        self.path_list_input_img = path_list_input_img
        self.path_list_input_psf = path_list_input_psf
        self.path_list_input_sigma = path_list_input_sigma
        self.path_list_input_mask = path_list_input_mask  # Bad pixel mask

        self.output_type = output_type
        self.output_items = output_items

        self.feedme = None
        self.constrains = None

        self.n_img = len(self.path_list_input_img)

        if self.path_list_input_sigma is None:
            self.path_list_input_sigma = ['none'] * self.n_img
        
        if self.path_list_input_mask is None:
            self.path_list_input_mask = ['none'] * self.n_img

    def show_feedme_example(self):
        url = "https://www.nottingham.ac.uk/astronomy/megamorph/exec/EXAMPLE.GALFITM.INPUT"
        print(url)
    
    def show_constrains_example(self):
        url = "https://www.nottingham.ac.uk/astronomy/megamorph/exec/EXAMPLE.GALFITM.CONSTRAINTS"
        print(url)

    def config(
            self, 
            models=None, 
            band_labels=[], 
            band_wavelengths=[],
            psf_fine_sampling=2, 
            fit_region=[], 
            convolution_box_size=151, 
            mag_zeropoint=[], 
            pixel_scale=None, 
            min_sigma_factor=None, 
            non_parameter_components=0,
            multinest_options=0
            ):
        """
        collect all the configuration parameters 

        Parameters
        ----------

        non_parameter_components : int | list
            - 0  # Standard parametric fitting
            - 1  # Turn on non-parametric component with SED homogenisation
            - -1 # Turn on non-parametric component without SED homogenisation
            - 3 0.75 25 4 40 0.0 1.0 # Customise the non-parametric schedule (n,a,b,c,d,r,t)
            - Every n iterations, the nonparametric image is updated
            - using a fraction of the filtered residuals, npf:
            - npf = a * b^c / (b^c + |i - d|^c),
            - where i is the iteration number.
            - If n is negative do not apply SED homogenisation.
            - r is a radius around each object centre to exclude.
            - t is a factor to modify the filtering threshold.
   
        """
        self.models = models
        self.band_labels = band_labels
        self.band_wavelengths = band_wavelengths  # 单位须一致, 选择是任意的
        self.psf_fine_sampling = psf_fine_sampling
        self.fit_region = fit_region  # Image region to fit (xmin xmax ymin ymax)
        self.convolution_box_size = convolution_box_size
        self.mag_zeropoint = mag_zeropoint
        self.pixel_scale = pixel_scale
        self.min_sigma_factor = min_sigma_factor
        self.non_parameter_components = non_parameter_components
        self.multinest_options = multinest_options

        """generate the feedme content"""
        feedme_content = "="*80 + "\n"
        feedme_content += "# GALFITM Configuration (made by astrokit)\n"

        feedme_content += '\n'
        feedme_content += '# ' + '-'*78 + '\n'
        feedme_content += "# IMAGE and GALFIT CONTROL PARAMETER\n"
        feedme_content += '# ' + '-'*78 + '\n'
        feedme_content += '\n'

        lst = [str(p) for p in self.path_list_input_img]
        feedme_content += f'A) {",".join(lst)}\n'

        feedme_content += f'A1) {",".join(self.band_labels)}  # Band labels\n'

        lst = [str(p) for p in self.band_wavelengths]
        feedme_content += f'A2) {",".join(lst)}  # Band wavelengths\n'

        feedme_content += f'B) {str(self.path_output_img)}\n'

        lst = [str(p) for p in self.path_list_input_sigma]
        feedme_content += f'C) {",".join(lst)}\n'

        lst = [str(p) for p in self.path_list_input_psf]
        feedme_content += f'D) {",".join(lst)}\n'

        feedme_content += f'E) {self.psf_fine_sampling}  # PSF fine sampling factor relative to data\n'

        lst = [str(p) for p in self.path_list_input_mask]
        feedme_content += f'F) {",".join(lst)}  # Bad pixel mask fits\n'

        if self.path_constraints is None:
            feedme_content += 'G) none  # constraints file\n'
        else:
            feedme_content += f'G) {str(self.path_constraints)}  # constraints file\n'

        if self.fit_region == []:
            raise ValueError("fit_region must be set, even if it is the whole image.")
        else:
            xmin, xmax, ymin, ymax = self.fit_region
        feedme_content += f'H) {xmin}    {xmax}   {ymin}    {ymax}  # Image region to fit (xmin xmax ymin ymax)\n'

        if isinstance(self.convolution_box_size, int):
            x = y = self.convolution_box_size
        else: 
            x, y = self.convolution_box_size
        feedme_content += f'I) {x}    {y}  # Size of the convolution box (x y)\n'

        if self.mag_zeropoint == []:
            raise ValueError("mag_zeropoint must be set!")
        else:
            lst = [str(p) for p in self.mag_zeropoint]
            feedme_content += f'J) {",".join(lst)}  # Magnitude photometric zeropoint\n'

        if self.pixel_scale is None:
            raise ValueError("pixel_scale must be set.")
        elif isinstance(self.pixel_scale, (int, float)):
            dx = dy = self.pixel_scale
        else:
            dx, dy = self.pixel_scale
        feedme_content += f'K) {dx}    {dy}  # Plate scale (dx dy) [arcsec per pixel]\n'

        feedme_content += 'O) regular  # Display type (regular, curses, both)\n'

        if self.output_type == 'optimize':
            v = 0
        elif self.output_type == 'model':
            v = 1
        elif self.output_type == 'imgblock':
            v = 2
        elif self.output_type == 'subcomps':
            v = 3
        else:
            raise ValueError(f"Unknown output_type: {self.output_type}")
        feedme_content += f'P) {v}  # Choose: 0=optimize, 1=model, 2=imgblock, 3=subcomps\n'

        if isinstance(self.non_parameter_components, int):
            feedme_content += f'U) {self.non_parameter_components}  # Non-parametric component\n'
        else:
            n, a, b, c, d, r, t = self.non_parameter_components
            feedme_content += f'U) {n} {a} {b} {c} {d} {r} {t}  # Non-parametric component\n'

        if isinstance(self.multinest_options, int):
            feedme_content += f'V) {self.multinest_options}  # MultiNest\n'
        else:
            c1, c2, c3, c4, c5, c6 = self.multinest_options
            feedme_content += f'V) {c1} {c2} {c3} {c4} {c5} {c6}  # MultiNest options\n'
        
        feedme_content += f'W) {",".join(self.output_items)}  # Output options\n'

        feedme_content += '\n'
        feedme_content += '# ' + '-'*78 + '\n'
        feedme_content += '# MODEL COMPONENTS\n'
        feedme_content += '# ' + '-'*78 + '\n'

        if models is None:
            logger.warning("Must define GalfitM models!")
        else:
            feedme_content += models.feedme

        self.feedme = feedme_content

        """write the feedme content to the cache directory"""
        with open(self.path_feedme, 'w') as f:
            f.write(self.feedme)
        return None
    
    def add_constrains(self, component, parameter, constraint):
        """
        Add constraints to the constrains file
        """

        def _constraints_row(c1, c2, c3):
            return f"{c1:^20} {c2:^20} {c3:<20}\n"
        
        if self.constrains is None:
            self.constrains = _constraints_row('# component', 'parameter', 'constraint')
        
        self.constrains += _constraints_row(component, parameter, constraint)
        
        with open(self.path_constraints, 'w') as f:
            f.write(self.constrains)
        return None

    def run(self, silent=False, timeout=None):
        returncode = run_command(
            f"galfitm {self.path_feedme}",
            dir_work=self.dir_output,
            print_output=not silent,
            timeout=timeout
        )
        return None