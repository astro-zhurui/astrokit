"""
A Python interface for GALFITM

- Author: Rui Zhu
- Date: 2025-07-08
"""
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import cmasher
from loguru import logger
import time
import os

from astropy.io import fits
from astropy.visualization import ImageNormalize, LogStretch

from astrokit.toolbox.utils import run_command, run_command_in_terminal
from astrokit.toolbox.utils import sec_to_hms

__all__ = ['GalfitM', 'GalfitMModel']

class GalfitMModel:
    """
    定义GalfitM模型
    """

    def __init__(self):
        self._model_idx = 0
        self.feedme = ""

    def _feedme_row(self, param_idx, param_value, fit_options, cheb, comment):
        if isinstance(param_value, list):
            n_params = len(param_value)
            param_value = str(param_value).strip('[]').replace(' ', '')
        else:
            param_value = f"{param_value}"

        if fit_options is None:
            fit_options = [1]*n_params
        if isinstance(fit_options, list):
            fit_options = str(fit_options).strip('[]').replace(' ', '')
        else:
            fit_options = f"{fit_options}"
        if cheb:
            fit_options = f"{fit_options} cheb"

        return f"{param_idx:>2}) {param_value:<50}  {fit_options:<20}  {comment:<20}\n"

    def add_sky(
            self, 
            background='', fit_background=1, cheb_background=False,
            sky_grad_x='', fit_sky_grad_x=1, cheb_sky_grad_x=False,
            sky_grad_y='', fit_sky_grad_y=1, cheb_sky_grad_y=False, 
            skip_in_output=False
    ):
        model_name = 'sky'
        self._model_idx += 1
        content = f"\n# Component number: {self._model_idx}\n"

        data = [
            ['0', model_name, '', False, "# object type"], 
            ['1', background, fit_background, cheb_background, "# sky background [ADU counts]"],
            ['2', sky_grad_x, fit_sky_grad_x, cheb_sky_grad_x, "# dsky/dx (sky gradient in x)"],
            ['3', sky_grad_y, fit_sky_grad_y, cheb_sky_grad_y, "# dsky/dy (sky gradient in y)"],
            ['z', skip_in_output, '', False, "#  Skip this model in output image?  (yes=1, no=0)"]
        ]
        for row in data:
            content += self._feedme_row(
                param_idx=row[0],
                param_value=row[1],
                fit_options=row[2],
                cheb=row[3],
                comment=row[4]
            )
        self.feedme += content
        return None

    def add_psf(
            self, 
            x='', fit_x=None, cheb_x=False,
            y='', fit_y=None, cheb_y=False,
            mag='', fit_mag=None, cheb_mag=False,
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
        self._model_idx += 1
        content = f"\n# Component number: {self._model_idx}\n"

        data = [
            ['0', model_name, '', False, "# object type"],
            ['1', x, fit_x, cheb_x, "# position x [pixel]"],
            ['2', y, fit_y, cheb_y, "# position y [pixel]"],
            ['3', mag, fit_mag, cheb_mag, "# total magnitude"],
            ['z', skip_in_output, '', False, "#  Skip this model in output image?  (yes=1, no=0)"]
        ]
        for row in data:
            content += self._feedme_row(
                param_idx=row[0],
                param_value=row[1],
                fit_options=row[2],
                cheb=row[3],
                comment=row[4]
            )
        self.feedme += content
        return None
    
    def add_sersic(
            self, 
            x='', fit_x=None, cheb_x=False,
            y='', fit_y=None, cheb_y=False,
            mag='', fit_mag=None, cheb_mag=False, 
            re='', fit_re=None, cheb_re=False,
            sersic_index='', fit_sersic_index=None, cheb_sersic_index=False,
            axis_ratio='', fit_axis_ratio=None, cheb_axis_ratio=False,
            PA='', fit_PA=None, cheb_PA=False,
            skip_in_output=False
    ):
        model_name = 'sersic'
        self._model_idx += 1
        content = f"\n# Component number: {self._model_idx}\n"

        data = [
            ['0', model_name, '', False, "# Object type"],
            ['1', x, fit_x, cheb_x, "# position x [pixel]"],
            ['2', y, fit_y, cheb_y, "# position y [pixel]"],
            ['3', mag, fit_mag, cheb_mag, "# total magnitude"],
            ['4', re, fit_re, cheb_re, "# R_e [pixel]"],
            ['5', sersic_index, fit_sersic_index, cheb_sersic_index, "# Sersic index"],
            ['9', axis_ratio, fit_axis_ratio, cheb_axis_ratio, "# axis ratio (b/a)"],
            ['10', PA, fit_PA, cheb_PA, "# position angle (PA)"],
            ['z', skip_in_output, '', False, "#  Skip this model in output image?  (yes=1, no=0)"]
        ]
        for row in data:
            content += self._feedme_row(
                param_idx=row[0],
                param_value=row[1],
                fit_options=row[2],
                cheb=row[3],
                comment=row[4]
            )
        self.feedme += content
        return None

class GalfitM:
    def __init__(
            self, 
            dir_output="", 
            task_name=None, 
            path_list_input_img=[], 
            path_list_input_psf=[], 
            path_list_input_sigma=[], 
            path_list_input_mask=[],
            output_type='optimize', 
            output_items=['input', 'model', 'residual', 'component'],
            min_sigma=None
            ):
        
        self.dir_output = Path(dir_output)
        self.task_name = task_name

        if not self.dir_output.exists():
            self.dir_output.mkdir(parents=True, exist_ok=True)

        self.path_output_img = self.dir_output / f"{self.task_name}_galfitm.fits"
        self.path_feedme = self.dir_output / f"{self.task_name}_galfitm.feedme"
        self.path_constraints = self.dir_output / f"{self.task_name}_galfitm.constraints"

        self.path_list_input_img = path_list_input_img
        self.path_list_input_psf = path_list_input_psf
        self.path_list_input_sigma = path_list_input_sigma
        self.path_list_input_mask = path_list_input_mask  # Bad pixel mask

        self.output_type = output_type
        self.output_items = output_items

        self.feedme = None
        self.constrains = None

        self.n_img = len(self.path_list_input_img)

    def show_feedme_example(self):
        url = "https://www.nottingham.ac.uk/astronomy/megamorph/exec/EXAMPLE.GALFITM.INPUT"
        print(url)
    
    def show_constrains_example(self):
        url = "https://www.nottingham.ac.uk/astronomy/megamorph/exec/EXAMPLE.GALFITM.CONSTRAINTS"
        print(url)

    def config(
            self, 
            models=None, 
            use_constraints=False,
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
        self.use_constraints = use_constraints
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
        feedme_content += "# IMAGE and GALFITM CONTROL PARAMETER\n"
        feedme_content += '# ' + '-'*78 + '\n'
        feedme_content += '\n'

        lst = [str(p) for p in self.path_list_input_img]
        feedme_content += f'A) {",".join(lst)}\n'

        feedme_content += f'A1) {",".join(self.band_labels)}  # Band labels\n'

        lst = [str(p) for p in self.band_wavelengths]
        feedme_content += f'A2) {",".join(lst)}  # Band wavelengths\n'

        feedme_content += f'B) {str(self.path_output_img)}\n'

        if len(self.path_list_input_sigma) == 0:
            if self.min_sigma_factor is None:
                lst = ['none'] * self.n_img
                feedme_content += f'C) {",".join(lst)}\n'
            else:
                feedme_content += f'C) none    {self.min_sigma_factor}\n'
        else:
            lst = [str(p) for p in self.path_list_input_sigma]
            feedme_content += f'C) {",".join(lst)}\n'

        lst = [str(p) for p in self.path_list_input_psf]
        feedme_content += f'D) {",".join(lst)}\n'

        feedme_content += f'E) {self.psf_fine_sampling}  # PSF fine sampling factor relative to data\n'

        if len(self.path_list_input_mask) == 0:
            lst = ['none'] * self.n_img
        else:
            lst = [str(p) for p in self.path_list_input_mask]
        feedme_content += f'F) {",".join(lst)}  # Bad pixel mask fits\n'

        if self.use_constraints:
            if self.path_constraints.exists():
                feedme_content += f'G) {str(self.path_constraints)}  # constraints file\n'
            else:
                raise ValueError(f"Constraints file {self.path_constraints} does not exist!")
        else:
            feedme_content += 'G) none  # constraints file\n'

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
        logger.info(f"GalfitM feedme written to {self.path_feedme}")
        return None
    
    def _constraints_row(self, c1, c2, c3):
        return f"{c1:^20} {c2:^20} {c3:<20}\n"
    
    def new_constrains(self):
        self.constrains = self._constraints_row('# component', 'parameter', 'constraint')
    
    def add_constrains(self, component, parameter, constraint):
        """
        Add constraints to the constrains file
        """
        if self.constrains is None:
            raise ValueError("Constraints not initialized. Call new_constrains() first.")

        self.constrains += self._constraints_row(component, parameter, constraint)

        with open(self.path_constraints, 'w') as f:
            f.write(self.constrains)
        return None

    def run(self, run_in_terminal=False, silent=False, timeout=None):
        """
        Run GalfitM via two ways. One is running in the terminal, and the other 
        is running in the jupyter cell. 

        Parameter
        ---------
        run_in_terminal: bool
            Whether to run GalfitM in the terminal (Mac) or in the Jupyter cell.

        silent: bool
            Whether to suppress output or not. Only applies when running in the 
            Jupyter cell.

        timeout: float
            Timeout for the command in seconds. If None, no timeout is applied. 
            Only applies when running in the Jupyter cell.
        """
        
        fname_log = 'fit.log'
        fname_cheb_orig = f'{self.task_name}_galfitm.galfit.01'
        fname_band_orig = f'{self.task_name}_galfitm.galfit.01.band'

        fname_cheb_new = f'{self.task_name}_galfitm.galfit.cheb'
        fname_band_new = f'{self.task_name}_galfitm.galfit.band'
        
        delete_file_ls = [
            fname_log, self.path_output_img.name, 
            fname_cheb_orig, fname_band_orig, 
            fname_cheb_new, fname_band_new
        ]
        for fname in delete_file_ls:
            path = self.dir_output / fname
            if path.exists():
                path.unlink()
                logger.warning(f"Deleted previous output file: {fname}")
        
        cmd = f"galfitm {self.path_feedme}"

        if run_in_terminal:
            if self.path_output_img.exists():
                raise ValueError("Please delete privious results!")
            run_command_in_terminal(cmd=f"cd {self.dir_output}; {cmd}")
            logger.info(f"Running GALFITM in terminal...")
            st = time.time()
            while True:
                if self.path_output_img.exists():
                    break
            logger.success(f"GalfitM Done! Cost Time: {sec_to_hms(time.time()-st)}")
        else:
            if not silent:
                logger.info("GalfitM Running...")
            st = time.time()
            returncode = run_command(
                cmd=cmd, 
                dir_work=self.dir_output,
                print_output=not silent,
                timeout=timeout
            )
            if not silent:
                logger.success(f"GalfitM Done! Cost Time: {sec_to_hms(time.time()-st)}")

        def _rename_file(fname_old, fname_new):
            path = self.dir_output / fname_old
            while True:
                if path.exists():
                    os.rename(path, self.dir_output / fname_new)
                    break
        _rename_file(fname_cheb_orig, fname_cheb_new)
        _rename_file(fname_band_orig, fname_band_new)

        return None
    
    def get_res(self):
        hdul = fits.open(self.path_output_img)
        # the final results in model extension header are same
        return hdul['MODEL_' + self.band_labels[0]].header
    
    def plot(self, filter, 
             cmap=cmasher.chroma, vmin=None, vmax=None):
        """
        Plot the results for a specific filter.
        """
        if filter not in self.band_labels:
            raise ValueError(f"Filter {filter} not in band_labels: {self.band_labels}")
        hdul = fits.open(self.path_output_img)
        data_img = hdul[f'INPUT_{filter}'].data
        data_model = hdul[f'MODEL_{filter}'].data
        data_residual = hdul[f'RESIDUAL_{filter}'].data

        fig, axes = plt.subplots(1, 3, figsize=(9, 3), constrained_layout=True)
        vmin = np.min(data_img) if (vmin is None) else vmin
        vmax = np.max(data_img) if (vmax is None) else vmax
        norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LogStretch())

        axes[0].imshow(data_img, origin='lower', cmap=cmap, norm=norm)
        axes[1].imshow(data_model, origin='lower', cmap=cmap, norm=norm)
        axes[2].imshow(data_residual, origin='lower', cmap=cmap, norm=norm)
            
        def _add_anchored_text(ax, string):
            at = AnchoredText(string, prop=dict(size=15, color='white'), pad=0.1,
                            frameon=False, loc='upper left')
            ax.add_artist(at)
        _add_anchored_text(axes[0], f"{filter}")
        _add_anchored_text(axes[1], "Model")
        _add_anchored_text(axes[2], "Residual")

        def _set_ax_style(ax):
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
            ax.tick_params(axis='x', which='major', width=0.5, length=5, direction='out')
            ax.tick_params(axis='y', which='major', width=0.5, length=5, direction='out')
        _set_ax_style(axes[0])
        _set_ax_style(axes[1])
        _set_ax_style(axes[2])

        return fig
