"""
Manage Filters

@author: Rui Zhu  
@creation time: 2024-05-14
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astrokit import datasets

__all__ = [
    'cal_pivot_wavelength', 
    'cal_equivalent_width',
    'find_filter_edges',
    'EazyFilters'
]

def cal_pivot_wavelength(wave, trans):
    """
    计算pivot wavelength
    cite: https://pysynphot.readthedocs.io/en/latest/properties.html#pivot-wavelength
    """
    wave, trans = np.array(wave), np.array(trans)
    up = np.trapezoid(y=wave*trans, x=wave)
    down = np.trapezoid(y=trans/wave, x=wave)
    pivot_wavelength = np.sqrt(up / down)
    return pivot_wavelength

def cal_equivalent_width(wave, trans):
    """
    calculate the bandpass equivalent width
    cite: https://pysynphot.readthedocs.io/en/stable/properties.html
    """
    wave, trans = np.array(wave), np.array(trans)
    width = np.trapezoid(y=trans, x=wave)
    return width

def find_filter_edges(wave, trans, threshold=0.5):
    """
    输入归一化的响应曲线, 返回指定阈值处的边界波长
    """
    wave = np.array(wave)
    trans = np.array(trans)

    idx_left = np.argmax(trans > threshold)
    wave_inversed = wave[::-1]
    trans_inversed = trans[::-1]
    idx_right = np.argmax(trans_inversed > threshold)

    return wave[idx_left], wave_inversed[idx_right]


class EazyFilters:
    """A class to manage filters for EAZY."""

    def __init__(self, translate_dict=None):
        """
        Parameters
        ----------
        translate_dict : dict
            filter的简称到序号的映射
        """

        self.DIR_EAZY_FILTERS = Path(datasets.__path__[0]) / 'eazy_filters'
        self.filters_data = self._read_EAZY_filters()
        self.translate_dict = translate_dict
    
    def show_info_file(self):
        """
        显示FILTER.RES.latest.info文件的路径, 方便查看band和idx的对应关系
        """
        print(self.DIR_EAZY_FILTERS / "FILTER.RES.latest.info")

    def _read_EAZY_filters(self) -> dict:
        """
        解析EAZY的Filter文件'FILTER.RES.latest'成一个dict

        Note
        ----
        filters_data: dict
            idx: 该filter出现的序号, 从1开始
            line_idx: 找到该filter单元在文件中的行号(从0开始)
            num_of_lines: 该filter单元的数据行数
            description: filter的描述
            lambda_c: 该filter的中心波长(从文件中直接读出来的, 与lc有差别)
            lambda_min: 该filter的最小波长
            lambda_max: 该filter的最大波长
            data: DataFrame
                wavelength: 波长
                transmission: 透射率
        """
        with open(self.DIR_EAZY_FILTERS / 'FILTER.RES.latest', 'r') as f:
            lines = f.readlines()

        # 收集filter header info
        fitlers_header = []  # 从文件中提取filter单元的头信息
        for line in lines:
            if 'lambda_c' in line:
                fitlers_header.append(line.strip("\n"))

        # 解析头信息
        filters_data = dict()
        regex = re.compile(r"\s*(\d+)\s+(.*(?=lambda)).*((?<=lambda_c=)\s?\S+)")

        i = 1 # key是该filter出现的序号，从1开始
        for filter_header in fitlers_header:
            regex_res = regex.match(filter_header).groups()

            # 参数line_idx: 找到该filter单元所在的行号(从0开始)
            line_idx = lines.index(filter_header+'\n')
            
            # 参数num_of_lines: 该filter单元的数据行数
            num_of_lines = int(regex_res[0].strip())

            # 参数lambda_c: 该filter的中心波长(从文件中直接读出来的, 与lc有差别)
            lambda_c = regex_res[2].strip()
            if "um" in lambda_c:
                lambda_c = lambda_c.strip("um")
                lambda_c = float(lambda_c) * 10000
            else:
                lambda_c = float(lambda_c)

            # 收集wavelength和transmission
            data = lines[line_idx+1: line_idx + 1 + num_of_lines]
            wavelength, transmission = [], []
            for row in data:
                row = row.strip("\n").split()
                wavelength.append(float(row[1]))
                transmission.append(float(row[2]))
            # normalize transmission curve
            transmission = np.array(transmission) / np.max(transmission)

            df = pd.DataFrame({'wavelength': wavelength, 'transmission': transmission})
            df.sort_values(by='wavelength', inplace=True)

            filters_data[i] = dict()

            filters_data[i]['idx'] = i
            filters_data[i]['line_idx'] = line_idx
            filters_data[i]['num_of_lines'] = num_of_lines
            filters_data[i]['description'] = regex_res[1].strip()
            filters_data[i]['lambda_c'] = lambda_c

            filters_data[i]['lambda_min'] = min(wavelength)
            filters_data[i]['lambda_max'] = max(wavelength)
            filters_data[i]['lambda_width'] = max(wavelength) - min(wavelength)

            filters_data[i]['data'] = df

            i += 1
        return filters_data
    
    def get_data(self, idx=None, filter_name=None):
        if filter_name is not None:
            idx = self.translate_dict[filter_name]
        return self.filters_data[idx]['data']

    def info(self) -> pd.DataFrame:
        """
        arrange filters data into a table
        """
        all_info = []
        for flt_name in self.translate_dict.keys():

            flt_idx = self.translate_dict[flt_name]
            flt_info = self.filters_data[flt_idx]
            flt_data = flt_info['data']

            get_info = {}
            get_info['filter_idx'] = flt_idx
            get_info['filter_name'] = flt_name

            get_info['lambda_c'] = flt_info['lambda_c']
            get_info['lambda_min'] = flt_info['lambda_min']
            get_info['lambda_max'] = flt_info['lambda_max']
            get_info['lambda_width'] = flt_info['lambda_width']

            get_info['pivot_wavelength'] = cal_pivot_wavelength(flt_data['wavelength'], 
                                                                flt_data['transmission'])
            get_info['equivalent_width'] = cal_equivalent_width(flt_data['wavelength'],
                                                                flt_data['transmission'])
            left, right = find_filter_edges(flt_data['wavelength'], 
                                            flt_data['transmission'], 
                                            threshold=0.5)
            
            get_info['FWHM_left'] = left
            get_info['FWHM_right'] = right
            get_info['FWHM'] = right - left
            
            all_info.append(get_info)

        df = pd.DataFrame(all_info)
        df = df.sort_values(by="lambda_c")
        df.reset_index(drop=True, inplace=True)

        return df

    def show(self, idx):
        """
        查询filter的信息

        Parameters
        ----------
        filter_idx : int or None
            filter在FILTERS.RES.latest文件中的序号(从1开始)
        filter_name : str or None
            filter的简称, 需要在eazy.json中登记
        """
        # get filter info
        info = self.filters_data[idx]
        print(f"+--------- [astrokit] Filter Query ---------+")
        print(f"==> filter_idx: {idx}")
        print(f"==> description: {info['description']}")
        print(f"==> lambda_c: {info['lambda_c']} Angstrom")

        data = info['data']

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(data['wavelength'], data['transmission'], color='black')
        ax.axvline(info['lambda_c'], color='r', linestyle='--', 
                label=rf'$\lambda_c = ${info["lambda_c"]:.2f} $\AA$')
        ax.set_xlabel(r'$\lambda$ [$\AA$]', fontsize=12)
        ax.set_ylabel('Transmission', fontsize=12)
        ax.legend()

        return None
