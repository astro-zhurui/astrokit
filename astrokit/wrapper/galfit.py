"""
A python warpper for Galfit.

@author: Rui Zhu
@creation time: 2023-01-09
"""
import os
import time
import numpy as np
from pathlib import Path
import subprocess
import fnmatch
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib.cm import ScalarMappable
import cmasher
plt.rcParams["image.origin"] = "lower"
plt.rc('font',family='Times New Roman')

from astropy.io import fits
from astropy.visualization import ImageNormalize
from astropy.visualization import LogStretch

import astrokit as ak

__all__ = [
    'calculate_psf_fraction',
    'append_galfit_header',
    'get_result',
    'plot',
    'Model',
    'Galfit'
]

def calculate_psf_fraction(galfit_result: dict, zero_point: float):
    '''
    calculate the psf fraction of the galaxy
    
    Parameters
    ----------
    galfit_result: the galfit result dict by astrokit.galfit
    zero_point: the zero point of the image

    Return
    ------
    psf_fraction: the psf to total flux ratio of the galaxy
    '''
    f_ps = ak.mag2flux(galfit_result['1_MAG']['value'], zero_point)
    f_sersic = ak.mag2flux(galfit_result['2_MAG']['value'], zero_point)
    return f_ps / (f_ps + f_sersic)


def append_galfit_header(header, exptime, gain=2.5, 
                         ncombine=1, bunit='ELECTRONS'):
    """
    append 4 keyworlds that GALFIT needed to existed FITS header.

    Parameters
    ----------
    header: astropy.io.fits.header.Header
        FITS header need to be appended.
    exptime: float
        exposure time of the image.
    gain: float
        commanded gain of CCD. Default: 2.5
    ncombine: int
        number of image sets combined during CR rejection. Default: 1
    bunit: str
        brightness units. Default: 'ELECTRONS'

    Returns
    -------
    header: astropy.io.fits.header.Header
        FITS header appended.

    """
    header.add_blank()
    header.add_blank(value="/ keywords for GALFIT")
    header.add_blank()
    header.append(('BUNIT', bunit, 'brightness units'), end=True)
    header.append(('EXPTIME', exptime, 'exposure depth'), end=True)
    header.append(('GAIN', gain, 'commanded gain of CCD'), end=True)
    header.append(('NCOMBINE', ncombine, 'number of image sets combined during CR rejecti'), end=True)

    return header

def get_result(path_output_img_fname, path_save=None, if_show=False) -> dict:
    """
    从galfit拟合结果fits cube中读取结果

    Parameters
    ----------
    path_output_img_fname: galfit result fits cube的路径
    path_save: 保存输出txt文件的文件夹路径, 默认为None, 不输出文件
    if_show: 是否直接打印txt文件内容
    """
    
    path_output_img_fname = Path(path_output_img_fname)
    header_res = fits.getheader(path_output_img_fname, ext=2)

    res = dict()
    params = list(header_res.keys())

    exclusion = ['XTENSION', 'COMMENT', 'LOGFILE']  # header里不需要放进字典的关键字
    for name in exclusion:
        while name in params:
            params.remove(name)

    # 遍历需要的参数名称, 结构化数据, 存进字典
    unreliable = []  # 收集拟合结果中的不可靠参数名称
    for param in params:
        if isinstance(header_res[param], str) and ('+/-' in header_res[param]):
            if '*' not in header_res[param]:
                res[param] = {
                    'value': float(header_res[param].split()[0].strip()), 
                    'err': float(header_res[param].split()[2].strip()), 
                    'comment': header_res.comments[param]
                }
            else:
                unreliable.append(param)
                res[param] = {
                    'value': float(header_res[param].split()[0].strip().replace('*', '')), 
                    'err': float(header_res[param].split()[2].strip().replace('*', '')), 
                    'comment': header_res.comments[param]
                }
        else:
            res[param] = {
                'value': header_res[param], 
                'comment': header_res.comments[param]
            }
    res['unreliable'] = unreliable

    def show_str() -> str:
        """嵌套函数: 将字典res中的内容整理成字符串, 方便预览和保存"""
        string = f"{'='*10} GALFIT result (from astrokit) {'='*10}\n"
        string += f"{'[name]':<10} {'[value]':<20} {'[err]':<10} {'[comment]'}\n"
        for key, content in res.items():
            if key == 'unreliable':
                continue
            if ('INITFILE' == key) or ('COMP' in key) or ('FLAGS' == key):
                string += '\n'
            if key in res['unreliable']:
                string += f"{f'{key}*':<10} {content['value']:<20} {content.get('err', ''):<10} {content['comment']}\n"
            else:
                string += f"{key:<10} {content['value']:<20} {content.get('err', ''):<10} {content['comment']}\n"
        if len(res['unreliable']) != 0:
            string += '\n\nNote:\n'
            string += '*: This parameter maybe get numerical problem. The solution is likely to be not reliable.\n'

        return string
    
    if path_save is not None:
        path_save = Path(path_save)
        with open(path_save / f"{path_output_img_fname.stem}.txt", 'w') as f:
            f.write(show_str())
    
    if if_show == True:
        print(show_str())

    return res


def plot(
        path_output_img_fname, 
        path_save=None, 
        vmin=None, 
        vmax=None,
        img1_name="Input Image", 
        img2_name="Model (GALFIT)", 
        img3_name="Residual Image"
        ) -> None:
    """
    plot galfit result

    Parameters
    ----------
    path_output_img_fname: galfit result fits cube的路径
    path_save: 保存路径或不保存(默认)
    """
    path_output_img_fname = Path(path_output_img_fname)
    res = get_result(path_output_img_fname)

    fig, ax = plt.subplots(1, 3, figsize=(8, 3))
    cmap = cmasher.chroma  # 精选cmap: twilight_shifted, seismic

    hdul = fits.open(path_output_img_fname)
    img = hdul[1].data
    model = hdul[2].data
    residual = hdul[3].data

    vmin = np.min(img) if (vmin is None) else vmin
    vmax = np.max(img) if (vmax is None) else vmax
    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LogStretch())

    fig_img = ax[0].imshow(img, origin='lower', cmap=cmap, norm=norm)
    fig_model = ax[1].imshow(model, origin='lower', cmap=cmap, norm=norm)
    fig_redisual = ax[2].imshow(residual, origin='lower', cmap=cmap, norm=norm)

    title_size = 15
    ax[0].set_title(img1_name, fontsize=title_size, fontweight='bold')
    ax[1].set_title(img2_name, fontsize=title_size, fontweight='bold')
    ax[2].set_title(img3_name, fontsize=title_size, fontweight='bold')

    ax_cbar = ak.add_colorbar_ax(fig, ax, pad=0.1, width=0.1)
    fig.colorbar(mappable=ScalarMappable(norm=norm, cmap=cmap), 
                 cax=ax_cbar, orientation='vertical', ticks=[norm.vmin, 0, norm.vmax])

    # 添加注释
    at = AnchoredText(
        s=r"$\chi^2_{\nu}$ = " + f"{res['CHI2NU']['value']:.3f}", 
        loc='lower right', 
        frameon=False, 
        prop=dict(size=12, color='white', fontweight='bold'), 
        borderpad=0.01, 
        pad=0.01
    )
    ax[2].add_artist(at)
    
    if len(res['unreliable']) > 0:
        at = AnchoredText(
            s="Unreliable", 
            loc='lower left', 
            frameon=True, 
            prop=dict(size=10)
        )
        ax[2].add_artist(at)

    for i in range(0, 3):
        ax[i].tick_params(top='on', right='on', which='both', width=1, direction='in')
        ax[i].minorticks_on()

    # save fig.
    if path_save is not None:
        path_save = Path(path_save)
        plt.savefig(path_save / f"{path_output_img_fname.stem}.png", dpi=300, bbox_inches='tight', facecolor='w')

    return None


class Model:
    """拟合使用的模型生成器"""

    def __init__(
            self, 
            position: tuple = (np.nan, np.nan), 
            mag: float = np.nan, 
            r_e: float = np.nan,
            n_sersic: float = np.nan, 
            axis_ratio: float = np.nan, 
            position_angle: float = np.nan, 
        ) -> None:
        """
        提供中心位置和基本形态参数的猜测值, 用于创建需要的拟合模型
        
        Parameters
        ----------
        position : 该模型中心坐标
        mag: 该成分的星等
        r_e: 有效半径(单位: pixels)
        n_sersic: Sérsic index
        axis_ratio: b/a
        position_angle: PA(Degrees: Up=0(星系半长轴和图像Y轴平行), Left=90)

        Note
        ----
            * 默认使用初始化输入的参数猜测值
            * 若要针对改模型修改参数, 可使用关键字传入指定值
            * free参数: 1-拟合时参数自由变化(默认), 0-固定参数拟合(参数名称在fix列表中, 修改为0)
        """
        self.position = (round(position[0], 2), round(position[1], 2))
        self.mag = round(mag, 2)
        self.r_e = round(r_e, 2)
        self.n_sersic = round(n_sersic, 2)
        self.axis_ratio = round(axis_ratio, 2)
        self.position_angle = round(position_angle, 2)
        self.skip = 0   # 是否在输出的图像中跳过该模型(0:否, 模型包含该成分; 1:跳过, 模型不含该成分)

        self.space_length = (3, 15, 5)  # 创建feedme字符串需要的格式控制空间大小; 写进feedme的字符串(分成4列: {参数号} {value} {fix} {comment})
    
    def show_properties(self) -> None:
        """显示模型的属性"""
        print(f'position: {self.position}')
        print(f'mag: {self.mag}')
        print(f'r_e: {self.r_e}')
        print(f'n_sersic: {self.n_sersic}')
        print(f'axis_ratio: {self.axis_ratio}')
        print(f'position_angle: {self.position_angle}')
        print(f'skip: {self.skip}')


    def sky(self, center_bkg_mean=0., sky_grad_x=0., sky_grad_y=0.,
            fix=[], **kwargs) -> dict:
        """
        sky model

        Parameters
        ----------
        center_bkg_mean: 图像中心的背景值
        sky_grad_x: x方向的背景梯度
        sky_grad_y: y方向的背景梯度
        fix: 需要固定不变的参数名列表
        """
        sky_params = {
            'model':
                {'value': 'sky', 'comment': '# object type'}, 
            'center_bkg_mean':
                {'value': center_bkg_mean, 'comment': '# sky background at center of fitting region [ADUs]', 'free': 1}, 
            'sky_grad_x':
                {'value': sky_grad_x, 'comment': '#  dsky/dx (sky gradient in x)', 'free': 1}, 
            'sky_grad_y':
                {'value': sky_grad_y, 'comment': '#  dsky/dy (sky gradient in y)', 'free': 1}, 
            'skip':
                {'value': self.skip, 'comment': "#  output option (0 = resid., 1 = Don't subtract)"}
        }
        if len(fix) != 0:
            for param in fix:
                sky_params[param]['free'] = 0

        num_space, value_space, free_space = self.space_length
        feedme = str()
        feedme += f"{'0)':>{num_space}} {sky_params['model']['value']:<{value_space}} {'':<{free_space}} {sky_params['model']['comment']}\n"
        feedme += f"{'1)':>{num_space}} {sky_params['center_bkg_mean']['value']:<{value_space}} {sky_params['center_bkg_mean']['free']:<{free_space}} {sky_params['center_bkg_mean']['comment']}\n"
        feedme += f"{'2)':>{num_space}} {sky_params['sky_grad_x']['value']:<{value_space}} {sky_params['sky_grad_x']['free']:<{free_space}} {sky_params['sky_grad_x']['comment']}\n"
        feedme += f"{'3)':>{num_space}} {sky_params['sky_grad_y']['value']:<{value_space}} {sky_params['sky_grad_y']['free']:<{free_space}} {sky_params['sky_grad_y']['comment']}\n"
        feedme += f"{'Z)':>{num_space}} {sky_params['skip']['value']:<{value_space}} {'':<{free_space}} {sky_params['skip']['comment']}\n"

        sky_params['feedme'] = feedme
        return sky_params
    
    def psf(self, fix=[], **kwargs) -> dict:
        """PSF model"""
        psf_params = {
            'model':
                {'value': 'psf', 'comment': '# object type'}, 
            'position': 
                {'value': self.position, 'comment': '# position x, y [pixel]', 'free': 1}, 
            'mag':
                {'value': self.mag, 'comment': '# total magnitude', 'free': 1}, 
            'skip':
                {'value': self.skip, 'comment': '# Skip this model in output image?  (yes=1, no=0)'}
        }
        # 修改单独指定的参数值
        if len(kwargs) != 0:
            for param in kwargs.keys():
                psf_params[param]['value'] = kwargs[param]
        if len(fix) != 0:
            for param in fix:
                psf_params[param]['free'] = 0

        num_space, value_space, free_space = self.space_length
        feedme = str()
        feedme += f"{'0)':>{num_space}} {psf_params['model']['value']:<{value_space}} {'':<{free_space}} {psf_params['model']['comment']}\n"

        position_str = f"{psf_params['position']['value'][0]}  {psf_params['position']['value'][1]}"
        str_free = f"{psf_params['position']['free']}  {psf_params['position']['free']}"
        feedme += f"{'1)':>{num_space}} {position_str:<{value_space}} {str_free:<{free_space}} {psf_params['position']['comment']}\n"

        feedme += f"{'3)':>{num_space}} {psf_params['mag']['value']:<{value_space}} {psf_params['mag']['free']:<{free_space}} {psf_params['mag']['comment']}\n"
        feedme += f"{'Z)':>{num_space}} {psf_params['skip']['value']:<{value_space}} {'':<{free_space}} {psf_params['skip']['comment']}\n"

        psf_params['feedme'] = feedme
        return psf_params

    def sersic(self, fix=[], **kwargs) -> dict:
        """Sersic model"""
        # sersic model参数值列表, 初始值设置为默认值
        sersic_params = {
            'model': 
                {'value': 'sersic', 'comment': '# Object type'}, 
            'position': 
                {'value': self.position, 'comment': '# position x, y [pixel]', 'free': 1}, 
            'mag': 
                {'value': self.mag, 'comment': '# total magnitude', 'free': 1}, 
            'r_e': 
                {'value': self.r_e, 'comment': '# R_e [Pixels]', 'free': 1}, 
            'n_sersic': 
                {'value': self.n_sersic, 'comment': '# Sersic exponent (deVauc=4, expdisk=1)', 'free': 1}, 
            'axis_ratio': 
                {'value': self.axis_ratio, 'comment': '# axis ratio (b/a)', 'free': 1}, 
            'position_angle': 
                {'value': self.position_angle, 'comment': '# position angle (PA)  [Degrees: Up=0, Left=90]', 'free': 1}, 
            'skip': 
                {'value': self.skip, 'comment': 'Skip this model in output image?  (yes=1, no=0)'}, 
        }

        # 修改单独指定的参数值
        if len(kwargs) != 0:
            for param in kwargs.keys():
                sersic_params[param]['value'] = kwargs[param]
        if len(fix) != 0:
            for param in fix:
                sersic_params[param]['free'] = 0
        
        # 写进feedme的字符串(分成4列: {参数号} {value} {free} {comment})
        num_space, value_space, free_space = self.space_length
        feedme = str()
        feedme += f"{'0)':>{num_space}} {sersic_params['model']['value']:<{value_space}} {'':<{free_space}} {sersic_params['model']['comment']}\n"

        position_str = f"{sersic_params['position']['value'][0]}  {sersic_params['position']['value'][1]}"
        str_free = f"{sersic_params['position']['free']}  {sersic_params['position']['free']}"
        feedme += f"{'1)':>{num_space}} {position_str:<{value_space}} {str_free:<{free_space}} {sersic_params['position']['comment']}\n"

        feedme += f"{'3)':>{num_space}} {sersic_params['mag']['value']:<{value_space}} {sersic_params['mag']['free']:<{free_space}} {sersic_params['mag']['comment']}\n"
        feedme += f"{'4)':>{num_space}} {sersic_params['r_e']['value']:<{value_space}} {sersic_params['r_e']['free']:<{free_space}} {sersic_params['r_e']['comment']}\n"
        feedme += f"{'5)':>{num_space}} {sersic_params['n_sersic']['value']:<{value_space}} {sersic_params['n_sersic']['free']:<{free_space}} {sersic_params['n_sersic']['comment']}\n"
        feedme += f"{'9)':>{num_space}} {sersic_params['axis_ratio']['value']:<{value_space}} {sersic_params['axis_ratio']['free']:<{free_space}} {sersic_params['axis_ratio']['comment']}\n"
        feedme += f"{'10)':>{num_space}} {sersic_params['position_angle']['value']:<{value_space}} {sersic_params['position_angle']['free']:<{free_space}} {sersic_params['position_angle']['comment']}\n"
        feedme += f"{'Z)':>{num_space}} {sersic_params['skip']['value']:<{value_space}} {'':<{free_space}} {sersic_params['skip']['comment']}\n"

        sersic_params['feedme'] = feedme
        return sersic_params


class Galfit:
    """A python wrapper for galfit"""
    
    def __init__(
            self, 
            dir_work=None,

            input_img_fname: str = None, 
            output_img_fname: str = None, 

            sigma_img_fname: str = None, 
            mask_img_fname: str = None, 
            psf_img_fname: str = None, 

            constraints_fname: str = None, 

            zeropoint=None, 
            pix_scale: float = None, 

            sampling_factor=1, 
            fit_region: tuple = None, 
            convbox_size: tuple = (50, 50), 

            display_type: str = 'regular', 
            out_option=0, 

            silent=False,
            remove_existing_output=True
        ) -> None:
        """
        填入基本信息, 用于实例化一个galfit任务

        Parameters
        ----------
        input_img_fname : sci图像的路径
        output_img_fname: 输出图像的文件名(默认galfit_xxx.fits)

        sigma_img_fname: sigma image路径
        mask_img_fname: mask image路径
        psf_img_fname: PSF image路径
        
        constraints_fname: constraints file路径

        zeropoint : 测光零点
        pix_scale : pixel scale [aresec/pixel]

        sampling_factor : PSF精细采样因子
        fit_region : 拟合区域``(x_min, x_max, y_min, y_max)``
        convbox_size : 卷积箱大小``(x, y)``

        display_type : regular, curses, both
        out_option : 输出图像内容选择(0=optimize, 1=model, 2=imgblock, 3=subcomps)

        silent : 是否静默运行, 默认False
        remove_existing_output : 是否删除已存在的输出文件, 默认True
        """
        # ========== 必要参数检查 ==========
        # check dir_work
        if isinstance(dir_work, str):
            self.dir_work = Path(dir_work)
        elif isinstance(dir_work, Path):
            self.dir_work = dir_work
        else:
            raise ValueError("[astrokit.galfit] dir_work must be a string or a Path object")
        
        # check input_img_fname
        if input_img_fname is None:
            raise ValueError("[astrokit.galfit] input_img_fname must be specified")
        elif isinstance(input_img_fname, str):
            self.input_img_fname = input_img_fname
        else:
            raise ValueError("[astrokit.galfit] input_img_fname must be a string")
        
        self.target_name = Path(self.input_img_fname).stem
        
        # check output_img_fname
        if output_img_fname is None:
            self.output_img_fname = f"galfit_{self.target_name}.fits"
        else:
            self.output_img_fname = output_img_fname

        # check sigma_img_fname
        if isinstance(sigma_img_fname, str):
            self.sigma_img_fname = sigma_img_fname
        elif sigma_img_fname is None:
            self.sigma_img_fname = 'none'
        else:
            raise ValueError("[astrokit.galfit] sigma_img_fname must be a string or None ('none' will be written in feedme)")
        
        # check mask_img_fname
        if isinstance(mask_img_fname, str):
            self.mask_img_fname = mask_img_fname
        elif mask_img_fname is None:
            self.mask_img_fname = 'none'
        else:
            raise ValueError("[astrokit.galfit] mask_img_fname must be a string or None ('none' will be written in feedme)")

        # check psf_img_fname
        if isinstance(psf_img_fname, str):
            self.psf_img_fname = psf_img_fname
        elif psf_img_fname is None:
            self.psf_img_fname = 'none'
        else:
            raise ValueError("[astrokit.galfit] psf_img_fname must be a string or None ('none' will be written in feedme)")
        
        # check constraints_fname
        if isinstance(constraints_fname, str):
            self.constraints_fname = constraints_fname
        elif constraints_fname is None:
            self.constraints_fname = 'none'
        else:
            raise ValueError("[astrokit.galfit] constraints_fname must be a string or None ('none' will be written in feedme)")

        # check zeropoint
        if isinstance(zeropoint, int) or isinstance(zeropoint, float):
            self.zeropoint = zeropoint
        else:
            raise ValueError("==> [astrokit.galfit] zeropoint must be specified")

        # check pix_scale
        if isinstance(pix_scale, int) or isinstance(pix_scale, float):
            self.pix_scale = pix_scale
        else:
            raise ValueError("==> [astrokit.galfit] pix_scale must be specified")

        self.sampling_factor = sampling_factor

        # check fit_region
        self.img = fits.getdata(self.dir_work / self.input_img_fname)  # 存入数据并且验证输入的input_img_fname正确
        # 若不指定拟合区域, 拟合区域将被设置为图像大小
        if fit_region is None: # 若不指定拟合区域, 拟合区域将被设置为图像大小
            (xmax, ymax) = self.img.shape
            self.fit_region = (1, xmax, 1, ymax)
        else:
            self.fit_region = fit_region
        
        self.convbox_size = convbox_size

        self.display_type = display_type
        self.out_option = out_option

        # ========== 返回值 ==========
        self.feedme_fname = None  # 制作feedme文件后, 将更改此变量, 之后执行终端命令时需要

        # ========== other settings ==========
        # check and remove existing output files
        if remove_existing_output:
            for fpath in list(self.dir_work.glob(f"galfit_{self.target_name}*")):
                fpath.unlink()
        
        if not silent:
            print("==> [astrokit.galfit] galfit settings:")
            print(f"    dir_work: {str(self.dir_work)}")
            print("    ------------------------------------")
            print(f"    input_img_fname: {self.input_img_fname}")
            print(f"    output_img_fname: {self.output_img_fname}")
            print(f"    sigma_img_fname: {self.sigma_img_fname}")
            print(f"    mask_img_fname: {self.mask_img_fname}")
            print(f"    psf_img_fname: {self.psf_img_fname}")
            print(f"    constraints_fname: {self.constraints_fname}")
            print("    ------------------------------------")
            print(f"    zeropoint: {self.zeropoint}")
            print(f"    pix_scale: {self.pix_scale}")
            print(f"    sampling_factor: {self.sampling_factor}")
            print(f"    fit_region: {self.fit_region}")
            print(f"    convbox_size: {self.convbox_size}")
            print(f"    display_type: {self.display_type}")
            print(f"    out_option: {self.out_option}")
            print("    ------------------------------------")


    def control_parameters(self) -> str:
        """整合传入的数据, 生成feedme control parameters部分的字符串"""
        value_space = 35
        content = "# IMAGE and GALFIT CONTROL PARAMETERS\n"
        content += f"A) {self.input_img_fname:<{value_space}} # Input data image (FITS file)\n"
        content += f"B) {self.output_img_fname:<{value_space}} # Output data image block\n"
        content += f"C) {self.sigma_img_fname:<{value_space}} # Sigma image name (made from data if blank or 'none')\n"
        content += f"D) {self.psf_img_fname:<{value_space}} # Input PSF image and (optional) diffusion kernel\n"
        content += f"E) {self.sampling_factor:<{value_space}} # PSF fine sampling factor relative to data\n"
        content += f"F) {self.mask_img_fname:<{value_space}} # Bad pixel mask (FITS image or ASCII coord list)\n"
        content += f"G) {self.constraints_fname:<{value_space}} # File with parameter constraints (ASCII file)\n"

        value = f"{self.fit_region[0]}  {self.fit_region[1]}  {self.fit_region[2]}  {self.fit_region[3]}"
        content += f"H) {value:<{value_space}} # Image region to fit (xmin xmax ymin ymax)\n"

        value_I = f"{self.convbox_size[0]}  {self.convbox_size[1]}"
        content += f"I) {value_I:<{value_space}} # Size of the convolution box (x y)\n"

        content += f"J) {self.zeropoint:<{value_space}} # Magnitude photometric zeropoint\n"

        value_K = f"{self.pix_scale}  {self.pix_scale}"
        content += f"K) {value_K:<{value_space}} # Plate scale (dx dy) [arcsec per pixel]\n"

        content += f"O) {self.display_type:<{value_space}} # Display type (regular, curses, both)\n"
        content += f"P) {self.out_option:<{value_space}} # Choose: 0=optimize, 1=model, 2=imgblock, 3=subcomps\n"

        return content

    def make_feedme(self, models, feedme_fname=None) -> None:
        """制作并输出galfit feedme文件"""
        # control parameters
        feedme = "="*80 + "\n"
        feedme += self.control_parameters()

        # model parameters
        feedme += "\n\n"
        feedme += f"# {'-'*78}\n"
        feedme += f"# {'par)'}  {'par value(s)'}  {'fit toggle(s)'}  {'# parameter description'}\n"
        feedme += f"# {'-'*78}\n"
        idx = 0
        for model in models:
            idx += 1
            feedme += f"\n# Component number: {idx}\n"
            feedme += model['feedme']
        
        # 将feedme文件输出
        if feedme_fname is None:
            fname = f"galfit_{self.target_name}.feedme"
        else:
            fname = feedme_fname
        path = self.dir_work / fname
        with open(path, 'w') as f:
            f.write(feedme)
        self.feedme_fname = fname
        
        return None
    
    def fit(self, time_out=30, show_prompt: bool = True) -> None:
        """
        执行galfit拟合

        Parameters
        ----------
        time_out: 超时时间
        show_prompt: 是否打印提示内容, True显示, False不显示

        Files
        -----
        [galfit_<xxx>.runlog]: 记录galfit运行时的终端打印结果
        [galfit_<xxx>.fits]: galfit结果文件
        [galfit_<xxx>.result]: galfit最终拟合参数, 原galfit.01
        """
        if show_prompt == True:
            st = time.time()
            print(f"==> [galfit] running...\r", end='')
            
        # 运行galfit
        process = subprocess.run(
            args=f"galfit {self.feedme_fname}", 
            cwd=self.dir_work, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            universal_newlines=True, 
            shell=True, timeout=time_out,
        )

        # 将命令行中的结果提示保存在xxx.runlog文件中
        with open(self.dir_work / f"galfit_{self.target_name}.runlog", 'w') as f:
            f.write(process.stdout)
        
        path = self.dir_work / "galfit.01"
        if path.exists():
            path_new = self.dir_work / f"galfit_{self.target_name}.result"
            os.rename(path, path_new)  # 重命名生成的galfit.01文件
            
            # 由结果生成子成分fitscube
            process = subprocess.run(
                args=f"galfit -o3 galfit_{self.target_name}.result", 
                cwd=self.dir_work, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                universal_newlines=True, 
                shell=True, timeout=time_out,
            )
            # 重命名子成分fits文件
            path = self.dir_work / "subcomps.fits"
            path_new = self.dir_work / f"galfit_{self.target_name}_subcomps.fits"
            os.rename(path, path_new)
        
            # 检查结果状态
            self.result = get_result(path_output_img_fname=self.dir_work / self.output_img_fname, if_show=False)
            if self.result['unreliable'] == []:
                flag = 'success'
            else:
                flag = 'unreliable'
        else:
            flag = 'failed'

        # 删除不需要的产生文件
        for path in list(self.dir_work.iterdir()):
            fname = path.name
            if fnmatch.fnmatch(name=fname, pat='galfit.*') or fname=='fit.log':
                Path(self.dir_work / fname).unlink()
            
        if show_prompt == True:
            et = time.time()
            print(f"==> [astrokit.galfit] {flag}: {self.input_img_fname}, cost time: {et-st:.2f} s")

        return None
    
    def get_result(self):
        """输出拟合结果到dict"""
        res = get_result(path_output_img_fname=self.dir_work / self.output_img_fname, if_show=True)
        return None

    def plot(self, path_save=None, vmin=None, vmax=None) -> None:
        """"预览galfit拟合结果"""
        plot(
            path_output_img_fname=self.dir_work / self.output_img_fname, 
            path_save=path_save, 
            vmin=vmin, 
            vmax=vmax
            )
        return None