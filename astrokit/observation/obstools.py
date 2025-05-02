"""
Some useful tools for observation.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from pathlib import Path
import importlib

from io import BytesIO
from PIL import Image, ImageOps
import requests
from loguru import logger
import time

from scipy.interpolate import interp1d
from astropy.time import Time
from astropy.coordinates import Angle
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u

from astrokit.toolbox.plot import bold_axis

__all__ = {
    'cal_time',
    'cal_exptime',
    'get_ps1_image',
    'plot_finder',
    'ra_dec_to_hms_dms',
    'ra_dec_to_Jname',
    'XinglongObservation',
}


def cal_time(start_time, exptime):
    """
    输入开始观测时间(e.g. 20:08)和曝光时间(e.g. 30)，计算结束时间
    """
    if not isinstance(start_time, str):
        raise TypeError("start_time must be str, like: '20:08'")
    if not isinstance(exptime, int):
        raise TypeError("exptime must be int!")

    hour, min = start_time.split(":")
    hour = int(hour)
    min = int(min)

    lc_time = time.localtime()
    long_time = f"{lc_time.tm_year}-{lc_time.tm_mon}-{lc_time.tm_mday} {hour:02d}:{min:02d}"

    st_time = time.strptime(long_time, "%Y-%m-%d %H:%M")
    st_timestamp = time.mktime(st_time)
    end_timestamp = st_timestamp + exptime
    end_time = time.localtime(end_timestamp)
    end_time = time.strftime("%Y-%m-%d %H:%M", end_time)

    logger.success(f"Finish time will be: {end_time}")
    print(f"Start time: {long_time}")
    print(f"Expousure time: {exptime} s")
    
    return None

def cal_exptime(mag, SNR=12):
    """
    计算给定星等下的曝光时间

    Note: 
    - 仅证认: SNR10-12
    - CLQ: SNR16
    - 测量: SNR20
    CLQ:SNR16
    """
    time = (SNR/5)**2 * 3600 / (10**((20-mag)*0.4))
    # 将浮点数的时间转换为整数秒
    time = int(np.round(time/10**len(str(int(time))), 2) * 10**len(str(int(time))))
    return time

"""
证认图
"""
def get_ps1_image(ra, dec, cutout_size=10, filter='grizy', color=True, format='png'):
    """
    Parameters
    ----------
    ra, dec: float
        目标源坐标
    cutout_size: int
        image size in arcmin. 0.25 arcsec/pixel, 240 pixels = 1 arcmin
    filter: str
        - 下载的图像band, 可选'g', 'r', 'i', 'z', 'y', 此时color应设置为False, 即下载单色灰度图
        - 或者设置为'grizy', 此时color应设置为True, 即下载彩色图
    color: bool
        是否下载彩色图
    format: str
        图像格式, 可选'png', 'jpg'
    
    Returns
    -------
    PIL.Image
        图像对象
        
    Cite
    ----
    https://outerspace.stsci.edu/display/PANSTARRS/PS1+Image+Cutout+Service#PS1ImageCutoutService-PythonExampleScript
    """
    size = cutout_size * 240  # 1 arcmin = 240 pixels
    # 查询该源的PS1图像url表格
    ps1filename = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
    url = (f"{ps1filename}?ra={ra}&dec={dec}&size={size}"
        f"&format=fits&filters={filter}")
    response = requests.get(url, verify=True).text
    tbl = Table.read(response, format='ascii')

    # sort filters from red to blue
    flist = ["yzirg".find(x) for x in tbl['filter']]
    tbl = tbl[np.argsort(flist)]

    # 获取图像url
    url = (f"https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?"
        f"ra={ra}&dec={dec}&size={size}&format={format}")
    if color:
        if len(tbl) > 3:
            # pick 3 filters
            tbl = tbl[[0, len(tbl)//2, len(tbl)-1]]
        for i, param in enumerate(["red","green","blue"]):
            url = url + "&{}={}".format(param, tbl['filename'][i])
    else:
        urlbase = url + "&red="
        url = []
        for filename in tbl['filename']:
            url.append(urlbase+filename)
        url = url[0]

    # 获取图像
    r = requests.get(url, verify=True)
    img = Image.open(BytesIO(r.content))

    return ImageOps.invert(img)

def plot_finder(ra, dec, cutout_size=10, name=None):
    """
    绘制证认图

    Parameters
    ----------
    ra, dec: float
        目标源坐标
    cutout_size: int
        图像大小, 单位arcmin
    name: str
        目标源名称
    """
    img = get_ps1_image(ra, dec, cutout_size=cutout_size, 
                        filter='grizy', color=True, format='png')

    fig, ax = plt.subplots(figsize=(8, 8))
    center_x = img.width / 2
    center_y = img.height / 2

    ax.imshow(img, origin='upper')
    ax.axis('off')

    ax.plot(center_x, center_x, marker='o', markersize=8, markeredgewidth=0.5,
            markeredgecolor="r", fillstyle="none")
    ax.plot(center_x, center_x, marker='.', markersize=1,
            markeredgecolor="r", fillstyle="none")

    coord = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    ra_hms = coord.ra.to_string(unit=u.hourangle, sep=':', precision=2, pad=True)
    dec_dms = coord.dec.to_string(sep=':', precision=2, alwayssign=True, pad=True)

    pe = [PathEffects.withStroke(linewidth=5, foreground="w")]

    txt1 = ax.text(100, 100, name, c="r",fontsize=18, path_effects=pe)
    txt2 = ax.text(100, 180, f"RA: {ra_hms}", fontsize=18, path_effects=pe)
    txt3 = ax.text(100, 260, f"DEC: {dec_dms}", fontsize=18, path_effects=pe)

    if img.width != img.height:
        raise ValueError("The image is not square.")
    img_size = img.width

    ax.axhline(y=img_size - img_size/12, 
            xmin=1/12, xmax=1/12 + (60/0.25) / img_size, 
            c="r", path_effects=pe)
    ax.text(img_size / 12, 
            img_size - img_size / 12 - 22, 
            "1 arcmin",
            c="r",
            fontsize=16, path_effects=pe)
    
    return None

def ra_dec_to_hms_dms(ra, dec, precision=0):
    coord = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg))
    ra_hms, dec_dms = coord.to_string(style="hmsdms", precision=precision).split(" ")
    ra_hms = ra_hms.replace("h", ":").replace("m", ":").replace("s", "")
    dec_dms = dec_dms.replace("d", ":").replace("m", ":").replace("s", "")
    return ra_hms, dec_dms

def ra_dec_to_Jname(ra, dec):
    coord = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg))
    ra_hms, dec_dms = coord.to_string(style="hmsdms", precision=0).split(" ")
    ra_h = ra_hms.split("h")[0]
    ra_m = ra_hms.split("m")[0].split("h")[1]
    dec_d = dec_dms.split("d")[0]
    dec_m = dec_dms.split("m")[0].split("d")[1]
    Jname = f"J{ra_h}{ra_m}{dec_d}{dec_m}"
    return Jname

class XinglongObservation:
    def __init__(self):
        pass

    def cal_dSID_bound(self, dec, safe_margin=15, show=True):
        """
        指定dec, 计算dSID的上下限

        Parameters
        ----------
        dec : float
            declination
        safe_margin : float
            距离观测边界的安全余量, 单位min
        """
        dir_data = importlib.util.find_spec(
            'astrokit.data_center'
            ).submodule_search_locations[0]
        
        path = Path(dir_data) / 'observation' / 'xinglong_safe_boundary.csv'
        points_data = pd.read_csv(path, names=['DEC', 'dSID'])

        lower_bound = points_data.query("DEC < 89.2 and dSID <= 0")
        lower_bound = lower_bound.sort_values('DEC')
        lower_bound_DEC = lower_bound['DEC'].values
        lower_bound_dSID = lower_bound['dSID'].values

        upper_bound = points_data.query("DEC < 89.2 and dSID >= 0")
        upper_bound = upper_bound.sort_values('DEC')
        upper_bound_DEC = upper_bound['DEC'].values
        upper_bound_dSID = upper_bound['dSID'].values

        f_lower = interp1d(lower_bound_DEC, lower_bound_dSID, fill_value='extrapolate')
        f_upper = interp1d(upper_bound_DEC, upper_bound_dSID, fill_value='extrapolate')
        if safe_margin is not None:
            safe_margin = safe_margin / 60
        else:
            safe_margin = 0
        lower_dSID = float(f_lower(dec)) + safe_margin
        upper_dSID = float(f_upper(dec)) - safe_margin

        if show:
            fig, ax = plt.subplots(figsize=(6, 4))
            bold_axis(ax)
            ax.plot(lower_bound_dSID, lower_bound_DEC, c='k')
            ax.plot(upper_bound_dSID, upper_bound_DEC, c='k')
            ax.scatter([lower_dSID, upper_dSID], [dec, dec], c='r')
            ax.set_xlabel('dSID', fontsize=12)
            ax.set_ylabel('DEC', fontsize=12)

        return lower_dSID, upper_dSID
    
    def cal_obs_time(self, ra, dec, obs_date):
        """
        计算给定观测日期, 目标源的观测时间范围和观测峰值时刻

        Parameters
        ----------
        ra : float
            right ascension, unit: degree
        dec : float
            declination, unit: degree
        obs_date : str
            观测日期, 格式: 'yyyy-mm-dd'

        Returns
        -------
        dict: 
            obs_window - 观测时间窗口
            obs_peak_time - 观测峰值时刻
            obs_peak_height - 观测峰值地平高度
        """

        xinglong_location = ('117d34m27s', '40d23m36s')
        time_str = obs_date + ' 16:00:00'  # 假定观测起始时间
        t = Time(time_str, scale='utc', location=xinglong_location)
        sidereal_time = t.sidereal_time('apparent').value  # 当地恒星时, 即春分点的时角

        ra_hours = ra / 15  # 将赤经转换为小时

        # dt: 目标源相对原点的时角(顺时针为正)
        if ra <= 180:
            dt = sidereal_time - ra_hours
        else:
            dt = sidereal_time - ra_hours + 24

        obs_time_lower_dSID, obs_time_upper_dSID = self.cal_dSID_bound(
            dec=dec, 
            safe_margin=15, 
            show=False)
        obs_time_lower = obs_time_lower_dSID - dt
        obs_time_upper = obs_time_upper_dSID - dt
        obs_time_mid = -dt  # peak time

        # 转换到北京时间
        def hour2hm(hour):
            hour = hour % 24  # 确保小时在0到24之间
            h, remainder = divmod(hour, 1)  # 获取小时和余数
            m = remainder * 60  # 将余数转换为分钟
            return int(h), int(m)

        obs_time_lower_h, obs_time_lower_m = hour2hm(obs_time_lower)
        obs_time_upper_h, obs_time_upper_m = hour2hm(obs_time_upper)
        obs_time_mid_h, obs_time_mid_m = hour2hm(obs_time_mid)

        obs_window = f"{obs_time_lower_h:02.0f}:{obs_time_lower_m:02.0f}"
        obs_window += f"-{obs_time_upper_h:02.0f}:{obs_time_upper_m:02.0f}"

        obs_peak_time = f"{obs_time_mid_h:02.0f}:{obs_time_mid_m:02.0f}"

        xinglong_altitude = Angle(xinglong_location[1]).degree
        obs_peak_height = 90 - np.abs(dec - xinglong_altitude)

        output = {
            'obs_date': obs_date, 
            'obs_window': obs_window, 
            'obs_peak_time': obs_peak_time, 
            'obs_peak_height': float(obs_peak_height).__round__(2)
        }

        return output