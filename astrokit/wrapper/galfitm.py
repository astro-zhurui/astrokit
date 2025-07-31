"""
A Python interface for GALFITM

- Author: Rui Zhu
- Date: 2025-07-08
"""

class GalfitMModel:

    def __init__(self):
        self.model_list = []
        self._model_idx = 0
        self.feedme = ""

    def _feedme_row(self, c1, c2, c3, c4):
        return f"{c1:>2}) {c2:<10} {c3:<10} {c4:<20}\n"

    def add_psf(self, x=None, y=None, mag=None, skip_in_output=False):
        model_name = 'psf'
        content = f"# Component number: {self._model_idx}\n"
        params = {
            'x': x, 
            'y': y, 
            'mag': mag, 
            'skip': 1 if skip_in_output else 0
        }
        content += self._feedme_row(
            c1='0', c2=model_name, c3='', 
            c4="# object type"
        )
        content += self._feedme_row(
            c1='1', c2=x
        )
        self.model_list.append({model_name: params})
        self.feedme += content
        self._model_idx += 1
        return None

class GalfitM:
    def __init__(
            self, 
            path_result_img=None, 
            path_list_input_img=[], 
            path_list_input_psf=[], 
            path_list_input_sigma=None, 
            path_list_input_mask=[],
            path_constraint_file=None,
            dir_cache=None,
            output_type='optimize', 
            output_items=['input', 'model', 'residual', 'component'],
            ):
        self.path_result_img = path_result_img
        self.path_list_input_img = path_list_input_img
        self.path_list_input_psf = path_list_input_psf
        self.path_list_input_sigma = path_list_input_sigma
        self.path_list_input_mask = path_list_input_mask  # Bad pixel mask
        self.path_constraint_file = path_constraint_file
        self.output_type = output_type
        self.output_items = output_items

        self.n_img = len(self.path_list_input_img)

        if dir_cache is None:
            self.dir_cache = path_result_img.parent / 'galfitm_cache'
            self.dir_cache.mkdir(parents=True, exist_ok=True)

        if self.path_list_input_sigma is None:
            self.path_list_input_sigma = ['none'] * self.n_img
        
        if self.path_list_input_mask is None:
            self.path_list_input_mask = ['none'] * self.n_img

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

        feedme_content += f'B) {str(self.path_result_img)}\n'

        lst = [str(p) for p in self.path_list_input_sigma]
        feedme_content += f'C) {",".join(lst)}\n'

        lst = [str(p) for p in self.path_list_input_psf]
        feedme_content += f'D) {",".join(lst)}\n'

        feedme_content += f'E) {self.psf_fine_sampling}  # PSF fine sampling factor relative to data\n'

        lst = [str(p) for p in self.path_list_input_mask]
        feedme_content += f'F) {",".join(lst)}  # Bad pixel mask fits\n'

        if self.path_constraint_file is None:
            feedme_content += 'G) none  # constraints file\n'
        else:
            feedme_content += f'G) {str(self.path_constraint_file)}  # constraints file\n'

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
        feedme_content += '\n'

        
        self.feedme = feedme_content

        """write the feedme content to the cache directory"""
        with open(self.dir_cache / 'galfitm.feedme', 'w') as f:
            f.write(self.feedme)
        return None