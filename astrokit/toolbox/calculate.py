import numpy as np
from scipy import stats
from scipy import integrate

import astropy.units as u
import astropy.constants as const
from astropy.coordinates import SkyCoord

import KDEpy
from KDEpy.bw_selection import silvermans_rule

__all__ = [
    'fnu_to_flam', 'flam_to_fnu',
    'fnu_to_ABmag', 'ABmag_to_fnu',
    'flam_to_ABmag', 'ABmag_to_flam',
    'ABmagErr_to_fnuErr', 'fnuErr_to_ABmagErr',
    'fnuErr_to_flamErr', 'flamErr_to_fnuErr',
    'ABmagErr_to_flamErr', 'flamErr_to_ABmagErr',
    'crack', 
    'cal_filter_central_wavelength',
    'logify',
    'kde2D',
    'binned_stats',
    'NMAD', 
    'cal_min_dist'
]


# ========================== 单位换算 ==========================
def fnu_to_flam(fnu, lambda_c):
    """
    将单位为erg s-1 cm-2 Hz-1的fnu转换为erg s-1 cm-2 AA-1的flam

    Parameters
    ----------
    fnu: float or array-like or astropy.units.Quantity
        f_nu, 单位为erg s-1 cm-2 Hz-1
    lambda_c: float or astropy.units.Quantity
        filter的中心波长, 单位为AA
    
    Return
    ------
    flam: float
        flam, 单位为erg s-1 cm-2 AA-1
    """
    if not isinstance(fnu, u.Quantity):
        fnu = fnu * u.Unit("erg s-1 cm-2 Hz-1")
    if not isinstance(lambda_c, u.Quantity):
        lambda_c = lambda_c * u.AA
    flam = (const.c / lambda_c**2) * fnu
    flam = flam.to("erg s-1 cm-2 AA-1")
    return flam.value

def flam_to_fnu(flam, lambda_c):
    """
    将单位为erg s-1 cm-2 AA-1的flam转换为单位为erg s-1 cm-2 Hz-1的fnu

    Parameters
    ----------
    flam : float or array-like or astropy.units.Quantity
        flux density in unit of erg s-1 cm-2 AA-1
    lambda_c : float or astropy.units.Quantity
        central wavelength in unit of AA

    Returns
    -------
    fnu : float
        flux density in unit of erg s-1 cm-2 Hz-1
    """
    if not isinstance(flam, u.Quantity):
        flam = flam * u.Unit("erg s-1 cm-2 AA-1")
    if not isinstance(lambda_c, u.Quantity):
        lambda_c = lambda_c * u.AA
    f_nu = (lambda_c**2 / const.c) * flam
    f_nu = f_nu.to("erg s-1 cm-2 Hz-1")

    return f_nu.value

def fnu_to_ABmag(fnu, zeropoint=-48.6):
    """
    fnu转换成AB星等

    Parameters
    ----------
    fnu : float or array-like
        fnu
    zeropoint : float
        AB星等的零点, 默认值为-48.6
    """
    mag = -2.5 * np.log10(fnu) + zeropoint
    return mag

def ABmag_to_fnu(mag, zeropoint=-48.6):
    """
    AB星等转换成fnu

    Parameters
    ----------
    mag : float or array-like
        AB星等
    zeropoint : float
        AB星等的零点, 默认值为-48.6
    """
    fnu = 10**(0.4*(zeropoint-mag))
    return fnu

def flam_to_ABmag(flam, lambda_c):
    """
    flam转换成AB星等

    Parameters
    ----------
    flam : float or array-like
        flam
    lambda_c : float or astropy.units.Quantity
        filter的中心波长, 单位为AA
    """
    fnu = flam_to_fnu(flam, lambda_c)
    mag = fnu_to_ABmag(fnu)
    return mag

def ABmag_to_flam(mag, lambda_c):
    """
    AB星等转换成flam

    Parameters
    ----------
    mag : float or array-like
        AB星等
    lambda_c : float or astropy.units.Quantity
        filter的中心波长, 单位为AA
    """
    fnu = ABmag_to_fnu(mag)
    flam = fnu_to_flam(fnu, lambda_c)
    return flam

def ABmagErr_to_fnuErr(mag, mag_err, zeropoint=-48.6):
    """
    AB星等误差转换成fnu误差

    Parameters
    ----------
    mag : float or array-like
        AB星等
    mag_err : float or array-like
        AB星等误差
    zeropoint : float
        AB星等的零点, 默认值为-48.6
    """
    coeff = -0.4 * np.log(10) * 10**(0.4*(zeropoint-mag))
    fnu_err = np.abs(coeff * mag_err)
    return fnu_err

def fnuErr_to_ABmagErr(fnu, fnu_err):
    """
    fnu误差转换成AB星等误差

    Parameters
    ----------
    fnu : float or array-like
        fnu
    fnu_err : float or array-like
        fnu误差
    """
    coeff = -2.5 * (1 / (fnu * np.log(10)))
    ABmag_err = np.abs(coeff * fnu_err)
    return ABmag_err

def fnuErr_to_flamErr(fnu_err, lambda_c):
    """
    fnu的误差转换成flam的误差

    Parameters
    ----------
    fnu_err : float or array-like or astropy.units.Quantity
        fnu的误差
    lambda_c : float or astropy.units.Quantity
        filter的中心波长, 单位为AA
    """
    if not isinstance(fnu_err, u.Quantity):
        fnu_err = fnu_err * u.Unit("erg s-1 cm-2 Hz-1")
    if not isinstance(lambda_c, u.Quantity):
        lambda_c = lambda_c * u.AA
    coeff = (const.c / lambda_c**2)
    flam_err = np.abs(coeff * fnu_err)
    flam_err = flam_err.to(u.Unit("erg s-1 cm-2 AA-1"))
    return flam_err.value

def flamErr_to_fnuErr(flam_err, lambda_c):
    """
    flam的误差转换成fnu的误差

    Parameters
    ----------
    flam_err : float or array-like or astropy.units.Quantity
        flam的误差
    lambda_c : float or astropy.units.Quantity
        filter的中心波长, 单位为AA
    """
    if not isinstance(flam_err, u.Quantity):
        flam_err = flam_err * u.Unit("erg s-1 cm-2 AA-1")
    if not isinstance(lambda_c, u.Quantity):
        lambda_c = lambda_c * u.AA
    coeff = (const.c / lambda_c**2)
    fnu_err = np.abs(flam_err / coeff)
    fnu_err = fnu_err.to(u.Unit("erg s-1 cm-2 Hz-1"))
    return fnu_err.value

def ABmagErr_to_flamErr(mag, mag_err, lambda_c):
    """
    AB星等误差转换成flam误差

    Parameters
    ----------
    mag : float or array-like
        AB星等
    mag_err : float or array-like
        AB星等误差
    lambda_c : float or astropy.units.Quantity
        filter的中心波长, 单位为AA
    """
    fnu_err = ABmagErr_to_fnuErr(mag, mag_err)
    flam_err = fnuErr_to_flamErr(fnu_err, lambda_c)
    return flam_err

def flamErr_to_ABmagErr(flam, flam_err, lambda_c):
    """
    flam误差转换成AB星等误差

    Parameters
    ----------
    flam : float or array-like
        flam
    flam_err : float or array-like
        flam误差
    lambda_c : float or astropy.units.Quantity
        filter的中心波长, 单位为AA
    """
    fnu = flam_to_fnu(flam, lambda_c)
    fnu_err = flamErr_to_fnuErr(flam_err, lambda_c)
    mag_err = fnuErr_to_ABmagErr(fnu, fnu_err)
    return mag_err


# ========================== 小工具 ==========================
def crack(integer):
    """将一个整数(integer)分成两个相近整数的乘积(a*b)"""
    a = int(np.sqrt(integer))
    b = integer / a
    while int(b) != b:
        a += 1
        b = integer / a
    res = (int(a), int(b))
    return max(res), min(res)

def cal_filter_central_wavelength(wave, trans):
    """
    计算一个滤光片响应曲线的中心波长

    Parameters
    ----------
    wave : array-like
        unit: Angstrom
    trans : array-like
        程序自动对其进行归一化
    """
    wave = np.array(wave)
    trans = np.array(trans)
    trans = trans / integrate.trapz(trans, wave)  # normalize transmission curve

    return integrate.trapz(trans * wave, wave)

def logify(arr):
    """
    将输入数组的每个元素取log, 对于小于0的数, 取绝对值再取log再乘以-1
    """
    # 创建一个与输入数组相同形状的数组，用于存储结果
    result = np.zeros_like(arr, dtype=float)
    
    # 对于大于0的数，取log
    positive_mask = arr > 0
    result[positive_mask] = np.log(arr[positive_mask])
    
    # 对于小于0的数，对绝对值取log再乘以-1
    negative_mask = arr < 0
    result[negative_mask] = -np.log(np.abs(arr[negative_mask]))

    # 对于nan，结果也是nan
    nan_mask = np.isnan(arr)
    result[nan_mask] = np.nan
    
    return result

def kde2D(x_in, y_in, bw=None, grid_size=1024):
    """
    计算二维核密度估计

    Parameters
    ----------
    bw : float, optional
        bandwidth for KDE, default is None (使用Silverman's rule自动选择带宽)
    grid_size : int, optional
        size of the grid for evaluation, default is 1024
    """
    x_in = np.asarray(x_in)
    y_in = np.asarray(y_in)
    data = np.array([x_in, y_in]).T
    if bw is not None:
        kde = KDEpy.FFTKDE(bw=bw)
        grid, points = kde.fit(data).evaluate(grid_size)
    else:
        bw1 = silvermans_rule(data[:, [0]])
        bw2 = silvermans_rule(data[:, [1]])
        data_scaled = data / np.array([bw1, bw2])
        kde = KDEpy.FFTKDE(bw=1)
        x_scaled, y_scaled = kde.fit(data_scaled).evaluate((grid_size, grid_size))
        grid = x_scaled * np.array([bw1, bw2])
        points = y_scaled / (bw1 * bw2)
    x, y = np.unique(grid[:, 0]), np.unique(grid[:, 1])
    z = points.reshape(grid_size, grid_size).T
    return x, y, z



# ========================== 统计工具 ==========================
def binned_stats(x, y, bins=8):
    """
    calculate the binned statistic

    Parameters
    ----------
    x: the data list for binned
    y: values
    bins: the number of bins

    Return
    ------
    bin_median
    bin_std
    bin_centers
    """
    # calculate median of each bins
    bin_median, bin_edges, binnumber = stats.binned_statistic(
        x=x, values=y, 
        statistic='median', bins=bins
        )
    # calculate std of each bins
    bin_std, bin_edges, binnumber = stats.binned_statistic(
        x=x, values=y, 
        statistic='std', bins=bins
        )
    bin_width = (bin_edges[1] - bin_edges[0])
    bin_centers = bin_edges[1:] - bin_width/2
    return bin_median, bin_std, bin_centers

def NMAD(arr):
    return 1.4826 * np.nanmedian(np.abs(arr - np.nanmedian(arr)))

def cal_min_dist(point, line_start, line_end):
    """
    计算单个天球点到多条线段的最短角距离 (向量化版, 单位: arcsec)

    Parameters
    ----------
    point : SkyCoord
        单个点 (isscalar=True)
    line_start : SkyCoord
        线段起点，可为多条
    line_end : SkyCoord
        线段终点，可为多条

    Returns
    -------
    distances : np.ndarray
        shape (N,), 表示该点到每条线段的最短角距离 (arcsec)
    """
    if not point.isscalar:
        raise ValueError("point must be a single SkyCoord")
    if line_start.isscalar:
        line_start = SkyCoord([line_start])
    if line_end.isscalar:
        line_end = SkyCoord([line_end])

    # ---- 单位向量转换 ----
    def sph_to_vec(coord):
        ra, dec = np.deg2rad(coord.ra.deg), np.deg2rad(coord.dec.deg)
        x = np.cos(dec) * np.cos(ra)
        y = np.cos(dec) * np.sin(ra)
        z = np.sin(dec)
        return np.stack([x, y, z], axis=-1)

    P = sph_to_vec(point)[None, :]      # shape (1, 3)
    A = sph_to_vec(line_start)          # shape (N, 3)
    B = sph_to_vec(line_end)            # shape (N, 3)

    # ---- 线段所在大圆的法向量 ----
    N = np.cross(A, B)                  # shape (N, 3)
    N /= np.linalg.norm(N, axis=1, keepdims=True)

    # ---- 点到大圆的角距 ----
    dot_pn = np.abs(np.dot(N, P.T)).ravel()  # shape (N,)
    delta_gc = np.arcsin(dot_pn)             # 弧度

    # ---- 球面投影点 Q (逐条线段广播) ----
    proj = P - (N @ P.T) * N                 # shape (N, 3)
    proj /= np.linalg.norm(proj, axis=1, keepdims=True)
    Q = proj                                 # shape (N, 3)

    # ---- 各角度 ----
    dot_aq = np.einsum('ij,ij->i', A, Q)
    dot_qb = np.einsum('ij,ij->i', Q, B)
    dot_ab = np.einsum('ij,ij->i', A, B)
    ang_aq = np.arccos(np.clip(dot_aq, -1, 1))
    ang_qb = np.arccos(np.clip(dot_qb, -1, 1))
    ang_ab = np.arccos(np.clip(dot_ab, -1, 1))

    # ---- 判断投影点是否在线段弧内 ----
    on_arc = np.abs((ang_aq + ang_qb) - ang_ab) < 1e-8

    # ---- 点到端点角距 ----
    dot_pa = np.einsum('ij,j->i', A, P[0])
    dot_pb = np.einsum('ij,j->i', B, P[0])
    dist_pa = np.arccos(np.clip(dot_pa, -1, 1))
    dist_pb = np.arccos(np.clip(dot_pb, -1, 1))
    dist_end = np.minimum(dist_pa, dist_pb)

    # ---- 选择最终距离 ----
    dist = np.where(on_arc, delta_gc, dist_end)

    return np.degrees(dist) * 3600.0  # arcsec
