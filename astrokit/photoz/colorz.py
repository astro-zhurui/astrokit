"""
Quick methods (e.g. Color-z & ML) for photometric redshift estimation.
For template fitting, see `astrokit.wrapper.eazy`.

@author: Rui Zhu
@creation time: 2024-04-17
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pickle

from loguru import logger

from scipy.signal import savgol_filter

import dask
from dask.diagnostics import ProgressBar

from astrokit.toolbox import crack
from .utils import plot_zspec_zphot

__all__ = ['ColorZ', 'run']

def _get_colorz_relation(cat_train, colors, z_min, z_max, z_step, n_threshold):
    """
    get the color-redshift relation for a dataframe

    Parameters
    ----------
    cat_train : pd.DataFrame
        input dataframe with columns of colors and redshift
    colors : list
        list of color column names used to calculate the color-redshift relation
    z_min : float, optional
        minimum redshift for sample bin, by default 0
    z_max : float, optional
        maximum redshift for sample bin, by default 4
    z_step : float, optional
        redshift bin width, by default 0.2
    n_threshold : int, optional
        minimum number of sources in each bin, by default 50

    
    Returns
    -------
    colorz_orig : pd.DataFrame
        the color-redshift relation dataframe for each bins
    """

    # 划分红移区间
    zbins = np.arange(z_min, z_max + z_step, z_step)
    df = cat_train.copy()
    df['zbins'] = pd.cut(df['z_spec'], bins=zbins)  # 匹配每个源到对应的红移区间

    zbin_group = df.groupby('zbins', observed=True)
    n_counts = zbin_group['z_spec'].count()  # 统计每个bin的数量

    # 遍历每个bin,如果数量小于n_threshold, 则合并到相邻的bin
    while n_counts.values.min() < n_threshold:
        for idx in n_counts.index:
            if n_counts[idx] < n_threshold:
                if idx == n_counts.index[-1]:
                    zbins = np.delete(zbins, np.where(np.abs(zbins - idx.left) < 1e-5))
                else:
                    zbins = np.delete(zbins, np.where(np.abs(zbins - idx.right) < 1e-5))
                break
        df['zbins'] = pd.cut(df['z_spec'], bins=zbins)  # 匹配每个源到对应的红移区间
        zbin_group = df.groupby('zbins', observed=True)
        n_counts = zbin_group['z_spec'].count()

        # bin statistics
        z_mean = zbin_group['z_spec'].mean()

    color_avg, color_err = {}, {}
    for color in colors:
        color_avg[f"{color}"] = zbin_group[color].mean()
        
        # 计算每个bin的误差, 包含两部分: color分布的std和实际测量的mc误差
        def monte_carlo(colors, err_colors, n_mc=1000):
            stds = []
            for _ in range(n_mc):
                sampled_colors = np.random.normal(colors, err_colors, size=len(colors))
                stds.append(sampled_colors.std())
            return np.mean(stds)
        error_mc = zbin_group.apply(lambda x: monte_carlo(x[color], x[f"err_{color}"]), 
                                    include_groups=False)
        color_err[f"err_{color}"] = np.sqrt(zbin_group[color].std()**2 + error_mc**2)

    colorz_orig = pd.DataFrame({'n': n_counts, 'z': z_mean, **color_avg, **color_err})
    colorz_orig.insert(0, 'zbins', colorz_orig.index)
    colorz_orig.reset_index(drop=True, inplace=True)
    return colorz_orig

def _plot_colorz_relation(cat_train, colorz_orig, colors, mincnt=1, title=""):
    """
    plot the color-redshift relation

    Parameters
    ----------
    cat_train : pd.DataFrame
        input dataframe with columns of colors and redshift for each source
    colorz_orig : pd.DataFrame
        color-redshift relation dataframe from `get_color_z_relation`
    colors : list
        list of color column names used to calculate the color-redshift relation
    mincnt : int
        the minimum count for the hexbin plot
    title : str
        the plot title
    
    Returns
    -------
    axs : np.ndarray
        return the color-redshift relation plot
    """
    n_colors = len(colors)
    n_cols, n_rows = crack(n_colors)
    if n_cols > 4:
        n_cols = 4
        n_rows = n_colors // n_cols + 1

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*4.5, n_rows*3), 
                            constrained_layout=True)
    axs = axs.flatten()
    # 隐藏多余的子图
    for i in range(n_colors, n_cols*n_rows):
        axs[i].axis('off')
    
    for i, color in enumerate(colors):
        axs[i].set_xlabel("Redshift", fontsize=15)
        axs[i].set_ylabel(f"{color}", fontsize=15)
        axs[i].hexbin(cat_train['z_spec'], cat_train[color], gridsize=300, 
                      cmap='hot', bins='log', mincnt=mincnt)
        axs[i].plot(colorz_orig['z'], colorz_orig[color], color='orange', 
                    linestyle='-', lw=2.5, marker='o', markersize=5)
        axs[i].plot(colorz_orig['z'], colorz_orig[color] + colorz_orig[f"err_{color}"], 
                    color='limegreen', lw=2.5, marker='o', markersize=5)
        axs[i].plot(colorz_orig['z'], colorz_orig[color] - colorz_orig[f"err_{color}"], 
                    color='limegreen', lw=2.5, marker='o', markersize=5)
        axs[i].set_xlim(0, colorz_orig['z'].max())
        for spine in axs[i].spines.values():
            spine.set_linewidth(1.5)
        axs[i].minorticks_on()
        axs[i].tick_params(axis='x', which='major', width=1.5, length=5, direction='in')
        axs[i].tick_params(axis='x', which='minor', width=1.5, length=3, direction='in')
        axs[i].tick_params(axis='y', which='major', width=1.5, length=5, direction='in')
        axs[i].tick_params(axis='y', which='minor', width=1.5, length=3, direction='in')
        # 取数据的10%和90%分位数作为y轴的范围
        axs[i].set_ylim(np.percentile(cat_train[color], 0.01), 
                        np.percentile(cat_train[color], 99.99))
    plt.suptitle(title, fontsize=15)
    
    return axs

def _resample_colorz_relation(colorz_orig, colors, accuracy):
    """
    resample color-z relation to a grid

    Parameters
    ----------
    colorz_orig : pd.DataFrame
        color-z relation
    colors : list
        colors to be resampled
    accuracy : float
        the accuracy of the resampled color-z relation

    Returns
    -------
    colorz_resampled: pd.DataFrame
        resampled color-z relation
    """

    zmin = colorz_orig['z'].min()
    zmax = colorz_orig['z'].max()
    n_z = int((zmax - zmin) / accuracy) + 1

    zgrid = np.linspace(zmin, zmax, n_z)

    colorz_resampled = {}
    colorz_resampled['z'] = zgrid
    z = colorz_orig['z'].values
    for color in colors:
        color_avg = colorz_orig[color].values
        color_err = colorz_orig[f'err_{color}'].values
        # 线性插值
        f_color_avg = np.interp(zgrid, z, color_avg)
        f_color_err = np.interp(zgrid, z, color_err)

        colorz_resampled[color] = f_color_avg
        colorz_resampled[f'err_{color}'] = f_color_err

    colorz_resampled = pd.DataFrame(colorz_resampled)

    return colorz_resampled

def _plot_resampled_colorz_relation(colorz_orig, colorz_resampled, colors):
    """
    plot the comparison between resampled color-z relation and color-z relation in bins

    Parameters
    ----------
    colorz_orig : pd.DataFrame
        color-z relation in bins
    colorz_resampled : pd.DataFrame
        resampled color-z relation
    colors : list
        colors to be plotted
    """
    n_cols, n_rows = crack(len(colors))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*4.5, n_rows*3), 
                            constrained_layout=True)
    axs = axs.flatten()
    for i, color in enumerate(colors):
        ax = axs[i]
        ax.set_xlabel("Redshift", fontsize=15)
        ax.set_ylabel(f"{color}", fontsize=15)
        ax.plot(colorz_resampled['z'], colorz_resampled[color], 'b-', label=f"resampled colorz")
        ax.fill_between(colorz_resampled['z'],
                        colorz_resampled[color] - colorz_resampled[f"err_{color}"],
                        colorz_resampled[color] + colorz_resampled[f"err_{color}"],
                        color='blue', alpha=0.2)
        ax.scatter(colorz_orig['z'], colorz_orig[color], c='red', alpha=0.5, label=f"colorz in bins")
        ax.legend()
        ax.set_xlim(0, colorz_resampled['z'].max())
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        ax.minorticks_on()
        ax.tick_params(axis='x', which='major', width=1.5, length=5, direction='in')
        ax.tick_params(axis='x', which='minor', width=1.5, length=3, direction='in')
        ax.tick_params(axis='y', which='major', width=1.5, length=5, direction='in')
        ax.tick_params(axis='y', which='minor', width=1.5, length=3, direction='in')
    return axs

def _fit(idx, cat_test, colors, colorz_resampled, min_colors=3):
    """
    对一个源执行color-z拟合

    Parameters
    ----------
    idx : int
        the index of the object in the test catalog
    cat_test : pd.DataFrame
        the test catalog
    colors : list
        the list of colors used for photoz
    colorz_resampled : pd.DataFrame
        the resampled color-z relation
    min_colors : int
        the minimum number of colors used for photoz
    """
    row_phot = cat_test.iloc[idx].to_dict()

    # 统计有效颜色
    colors_used = []
    for color in colors:
        if not np.isnan(row_phot[color]) and not np.isnan(row_phot[f'err_{color}']):
            colors_used.append(color)
    n_colors = len(colors_used)

    if n_colors >= min_colors:
        # 计算 chi2
        df = colorz_resampled.copy()
        def cal_chi2_at_z(row_colorz, row_phot, colors):
            chi2 = 0
            for color in colors:
                chi2_color = (row_phot[color] - row_colorz[color])**2
                chi2_color = chi2_color / (row_phot[f'err_{color}']**2 + row_colorz[f'err_{color}']**2)
                chi2 += chi2_color
            return chi2
        df['chi2'] = df.apply(cal_chi2_at_z, axis=1, args=(row_phot, colors_used))
        min_chi2 = df['chi2'].min()

        z = np.array(df['z'])
        pz_orig = np.exp(-0.5 * (df['chi2'] - min_chi2))
        df['pz_orig'] = pz_orig / np.trapz(pz_orig, z)

        df['pz_smooth'] = savgol_filter(df['pz_orig'], window_length=20, polyorder=2)
        zPDF = df[['z', 'chi2', 'pz_orig', 'pz_smooth']].copy()

        # find the best z
        z_peak = zPDF['z'][zPDF['pz_smooth'].idxmax()]

        res = {}
        res['id'] = row_phot['id']
        res['z_spec'] = row_phot['z_spec']
        res['z_peak'] = z_peak
        res['use_colors'] = colors_used
        res['n_colors'] = n_colors
        res['chi2'] = min_chi2
        res['chi2nu'] = min_chi2 / (n_colors - 1)
        res['zPDF'] = zPDF
    else:
        res = {
            'id': row_phot['id'],
            'z_spec': row_phot['z_spec'],
            'z_peak': np.nan,
            'use_colors': colors_used,
            'n_colors': n_colors,
            'chi2': np.nan,
            'chi2nu': np.nan,
            'zPDF': None
        }
    return res


class ColorZ:
    def __init__(self, cat_train, cat_test, colors, dir_output, 
                 fname_pkl="zPDF_colorz.pkl", 
                 fname_csv="res_colorz.csv", 
                 z_grid=[0, 4, 0.1]):
        """
        Input Parameters
        ----------
        cat_train : pd.DataFrame
            the DataFrame for calculating the color-redshift relation.
            This DataFrame should contain the following columns:
            - 'id': the unique identifier for each source
            - 'z_spec': the spectroscopic redshift
            - colors & errors of colors; e.g. 'color1', 'err_color1', 'color2', 'err_color2', ...

        cat_test : pd.DataFrame
            the same as cat_train, but for apply the color-redshift relation for redshift estimation.
        colors : list
            the list of colors used for photoz
        dir_output : str
            the output directory for saving the results
        fname_pkl : str
            the filename for saving all results
        fname_csv : str
            the filename for saving table of results
        z_grid : list
            制作colorz关系的红移初始网格[z_min, z_max, z_step]

        Other Parameters
        ----------
        colorz_orig : pd.DataFrame
            由数据直接计算得到的colorz关系
        colorz_resampled : pd.DataFrame
            通过差值得到的colorz关系
        """
        self.dir_output = Path(dir_output)
        self.dir_output.mkdir(parents=True, exist_ok=True)

        self.colors = colors

        self.cat_train = self._load_cat_train(cat_train, colors=colors)
        self.cat_test = self._load_cat_test(cat_test, colors=colors)
        
        self.fname_pkl = fname_pkl
        self.fname_csv = fname_csv

        self.z_grid = z_grid

    def _load_cat_train(self, cat_train, colors):
        """
        从输入的cat_train中提取id, z_spec, colors, err_colors用于训练color-z关系
        """
        cat_train = cat_train[['id', 'z_spec'] + colors + ['err_' + c for c in colors]]
        cat_train.reset_index(drop=True, inplace=True)
        return cat_train
    
    def _load_cat_test(self, cat_test, colors):
        """
        从输入的cat_test中提取id, z_spec, colors, err_colors
        """
        cat_test = cat_test[['id', 'z_spec'] + colors + ['err_' + c for c in colors]]
        cat_test.reset_index(drop=True, inplace=True)
        return cat_test

    def get_colorz_relation(self, n_threshold=10):
        """
        get the color-redshift relation for a dataframe

        Parameters
        ----------
        n_threshold : int, optional
            minimum number of sources in each bin, by default 10
        
        Returns
        -------
        colorz : pd.DataFrame
            if show is False, return the color-redshift relation dataframe
        """
        colorz_orig = _get_colorz_relation(self.cat_train, self.colors, 
                                           z_min=self.z_grid[0], 
                                           z_max=self.z_grid[1], 
                                           z_step=self.z_grid[2], 
                                           n_threshold=n_threshold)
        self.colorz_orig = colorz_orig
        return colorz_orig
    
    def plot_colorz_relation(self, mincnt=1, title="", return_axs=False):
        """
        plot the color-redshift relation

        Parameters
        ----------
        mincnt : int
            the minimum count for the hexbin plot
        title : str
            the plot title
        
        Returns
        -------
        axs : np.ndarray
            return the color-redshift relation plot
        """
        if not hasattr(self, 'colorz_orig'):
            self.get_colorz_relation()
        axs = _plot_colorz_relation(self.cat_train, self.colorz_orig, self.colors, 
                                    mincnt=mincnt, title=title)
        for ax in axs:
            ax.set_xlim(0, self.z_grid[1])

        if return_axs:
            return axs

    def resample_colorz_relation(self, accuracy=0.01):
        if not hasattr(self, 'colorz_orig'):
            self.get_colorz_relation()

        colorz_resampled = _resample_colorz_relation(self.colorz_orig, self.colors, accuracy)
        self.colorz_resampled = colorz_resampled
        return colorz_resampled
        
    def plot_resampled_colorz_relation(self, return_axs=False):
        if not hasattr(self, 'colorz_resampled'):
            self.resample_colorz_relation()
        axs = _plot_resampled_colorz_relation(self.colorz_orig, 
                                              self.colorz_resampled, 
                                              self.colors)
        for ax in axs:
            ax.set_xlim(0, self.z_grid[1])
        if return_axs:
            return axs
    
    def fit(self, idx):
        if not hasattr(self, 'colorz_resampled'):
            self.resample_colorz_relation()
        res = _fit(idx, self.cat_test, self.colors, self.colorz_resampled)
        return res

    def get_res(self):
        return pd.read_csv(self.dir_output / self.fname_csv)
    
    def plot_zspec_zphot(self, return_ax=False, **kwargs):
        res = self.get_res()
        ax = plot_zspec_zphot(res, return_ax=return_ax, **kwargs)
        return ax
    

def run(colorz_class, n_cpu=8):
    """
    以并行方式计算color-z
    """
    if not hasattr(colorz_class, 'colorz_resampled'):
        colorz_class.resample_colorz_relation()

    cat_test = colorz_class.cat_test
    colorz_resampled = colorz_class.colorz_resampled
    colors = colorz_class.colors
    N = len(cat_test)

    def _run_obj(idx):
        res = _fit(idx, cat_test, colors, colorz_resampled)
        return res

    client = dask.config.set(scheduler='processes', num_workers=n_cpu, 
                             threads_per_worker=1)

    # 并行计算zPDFs
    logger.info(f"running color-z for {N} sources...")

    delayed_tasks = [dask.delayed(_run_obj)(idx) for idx in range(N)]
    with ProgressBar(minimum=1.0):
        res = dask.compute(*delayed_tasks)

    # 保存结果到pkl
    path_pkl = colorz_class.dir_output / colorz_class.fname_pkl
    with open(path_pkl, 'wb') as f:
        pickle.dump(res, f)
    logger.success(f"==> All results saved to {path_pkl}")

    # 保存结果到csv
    need_keys = ['id', 'z_spec', 'z_peak', 'n_colors', 'chi2', 'chi2nu']
    res = [{k: d[k] for k in need_keys} for d in res]
    df = pd.DataFrame(res)
    path_csv = colorz_class.dir_output / colorz_class.fname_csv
    df.to_csv(path_csv, index=False)
    logger.success(f"==> Final table saved to {path_csv}")