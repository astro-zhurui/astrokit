import pandas as pd
import numpy as np

from scipy.interpolate import interp1d
from scipy.integrate import trapezoid


from specutils import Spectrum1D

from astropy.io import fits
from astropy.table import Table
from astropy import units as u
from astropy.nddata import StdDevUncertainty

from astrokit.phot import cal_pivot_wavelength
from astrokit.toolbox import flam_to_ABmag, flamErr_to_ABmagErr

__all__ = [
    'read_SDSS_spec',
    'show_emission_lines', 
    'norm_spec', 
    'phot_from_spec'
    ]

def read_SDSS_spec(path):
    """
    读取SDSS光谱数据

    Parameters
    ----------
    path : str or Path
        光谱文件路径

    Note
    ----
    * 读取方法参考: https://specutils.readthedocs.io/en/stable/manipulation.html#resampling
    * 光谱误差已转换成StdDevUncertainty
    """
    with fits.open(path) as hdul:
        header = hdul[0].header
        data = Table(hdul[1].data)
        # 光谱信息校验
        if header['BUNIT'] != '1E-17 erg/cm^2/s/Ang':
            raise ValueError("Spectrum unit error!")
        
    lamb = 10**data['loglam'] * u.AA
    flam = data['flux'] * 1e-17 * u.Unit('erg cm-2 s-1 AA-1')
    ivar = np.array(data['ivar'])
    ivar[ivar==0] = np.NaN
    error = StdDevUncertainty(1 / np.sqrt(ivar) * 1e-17 * u.Unit('erg cm-2 s-1 AA-1'))

    return Spectrum1D(spectral_axis=lamb, flux=flam, uncertainty=error)

def show_emission_lines(z=0) -> pd.DataFrame:
    """
    Notes
    -----
    infomation from https://classic.sdss.org/dr6/algorithms/linestable.php
    """
    # spectral lines
    qso_emission_lines = [
        [1215.24, r'Ly$\alpha$'], 
        [1240.81, r'NV'],
        [1549.48, r'CIV'], 
        [1908.73, r'CIII'], 
        [2799.117, r'MgII'], 
        [3727.092, r'[OII]'],
        [3729.875, r'[OII]'],
        [4102.89, r'H$\delta$'],
        [4341.68, r'H$\gamma$'],
        [4862.68, r'H$\beta$'],
        [4960.295, r'[OIII]'],
        [5008.240, r'[OIII]'],
        [6564.61, r'H$\alpha$'],
        [6718.29, r'[SII]'],
        [6732.67, r'[SII]'],
    ]
    df = pd.DataFrame(qso_emission_lines, columns=['rest_lamb', 'name'])
    if z != 0:
        df['obs_lamb'] = df['rest_lamb'] * (1+z)
    return df


def norm_spec(wave, flux, norm_wave=5500) -> tuple:
    """
    对给定的光谱进行归一化, 使用前请进行光谱数据检查

    Parameters
    ----------
    wave : array
        波长
    flux : array
        flux
    norm_wave : int or float or list
        归一化的波长，可以是一个数值(e.g. 5500)或者一个列表(e.g. [5000, 5200])

    Returns
    -------
    wave : array
        波长
    flux : array
        归一化后的flux
    """
    # 光谱检查与修正
    # idx = np.argsort(wave)
    # wave = wave[idx]
    # flux = flux[idx]
    f = interp1d(wave, flux, bounds_error=False)

    if isinstance(norm_wave, list):
        used_wave = wave[(wave > norm_wave[0]) & (wave < norm_wave[1])]  # 用于归一化的波长范围
        used_wave = np.unique(np.concatenate([used_wave, norm_wave]))
        used_flux = f(used_wave)
        coeff = trapezoid(used_flux, used_wave) / (used_wave.max() - used_wave.min())
    if isinstance(norm_wave, (int, float)):
        coeff = f(norm_wave)

    flux = flux / coeff
    return wave, flux

def phot_from_spec(flt_wave, flt_trans, 
                   spec_wave, spec_flam, spec_std):
    """
    给定响应曲线,计算一条光谱在该波段的平均流量及其误差
    注意: 
    1. 输入的响应曲线需要最大值归一化
    2. 输入的光谱需要覆盖响应曲线的波长范围, 否则会报错
    3. 输入的光谱的所有异常点需要提前标记成NAN, 如flam<=0, snr < 1等

    Parameters
    ----------
    flt_wave : array-like
        wavelength for filter transmission, unit: Angstrom
    flt_trans: array-like
        transmission for filter (norm maxium value = 1)
    spec_wave : array-like
        wavelength for spectrum, unit: Angstrom
    spec_flam : array-like
        flux for spectrum, unit: erg/s/cm^2/Angstrom
    spec_std : array-like
        std for spectrum, unit: erg/s/cm^2/Angstrom
    """
    flt_wave, flt_trans = np.array(flt_wave), np.array(flt_trans)
    spec_wave, spec_flam = np.array(spec_wave), np.array(spec_flam)

    # 去除光谱的NAN值
    mask_nan = np.isnan(spec_flam)
    spec_wave = spec_wave[~mask_nan]
    spec_flam = spec_flam[~mask_nan]

    # 计算响应曲线的特征量
    lc = cal_pivot_wavelength(wave=flt_wave, trans=flt_trans)
    threshold = 0.01
    left_edge = flt_wave[np.where(flt_trans > threshold)].min()
    right_edge = flt_wave[np.where(flt_trans > threshold)].max()

    # 检查光谱是否完全覆盖响应曲线
    if (left_edge < spec_wave.min()) or (right_edge > spec_wave.max()):
        mag, magerr = np.NaN, np.NaN
    else:
        # 获取用于计算的光谱片段
        index_used = (spec_wave >= left_edge) & (spec_wave <= right_edge)
        spec_wave = spec_wave[index_used]
        spec_flam = spec_flam[index_used]
        
        # 插值响应曲线
        flt_trans_interp = np.interp(x=spec_wave, xp=flt_wave, fp=flt_trans)

        # 计算该波段的平均flam
        up = np.trapz(y=spec_wave * flt_trans_interp * spec_flam, x=spec_wave)
        down = np.trapz(y=spec_wave * flt_trans_interp, x=spec_wave)
        flam_avg = up / down

        # 计算fnu
        mag = flam_to_ABmag(flam=flam_avg, lambda_c=lc)
        
        # 计算误差
        if spec_std is not None:
            spec_std = np.array(spec_std)
            spec_std = spec_std[~mask_nan]  # 去除NAN值
            spec_std = spec_std[index_used]  # 选择片段

            dlambda = np.diff(spec_wave)
            dlambda = np.append(np.diff(spec_wave), dlambda[-1])

            up = np.trapz(y=spec_wave * flt_trans_interp * spec_std, x=spec_wave)
            down = np.trapz(y=spec_wave * flt_trans_interp, x=spec_wave)
            flam_std = up / down

            magerr = flamErr_to_ABmagErr(flam=flam_avg, flam_err=flam_std, lambda_c=lc)
        else:
            magerr = np.nan

    return mag, magerr