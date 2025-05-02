"""
Useful functions for plot

@author: Rui Zhu 
@creation time: 2023-02-28
"""
import math
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cmasher
import warnings

from astropy.visualization import ImageNormalize, LogStretch
from astropy.stats import sigma_clipped_stats
from astropy.nddata import Cutout2D
from astropy import units as u
from astropy.visualization import wcsaxes
from astropy.wcs import WCS


from specutils import Spectrum1D
from specutils.manipulation import spectral_slab
from specutils.manipulation import gaussian_smooth

from photutils.detection import find_peaks
from photutils.aperture import CircularAperture

__all__ = [
    'show_colors',
    'add_text',
    'savefig',
    'bold_axis',
    'add_colorbar_ax',
    'plot_scatter_distribution',
    'plot_aperture_photometry_growth_curve',
    'plot_stamps', 
    'plot_SDSS_spectrum', 
    'imshow'
]

# **************************************************
# favorite colors
# colors from: https://color.adobe.com/zh/explore
# https://www.simplifiedsciencepublishing.com/resources/best-color-palettes-for-scientific-figures-and-data-visualizations

ray_colors = [
    '#FF0018', # red
    '#FFAF01', # orange
    '#004CAA', # blue
    '#02B370', # green
    '#7ABBDB',  # sky blue
    '#682487', # purple
    '#151F30',  
    '#103778', 
    '#0593A2', 
    '#FF7A48', 
    '#E3371E', 
    '#c1272d', 
    '#0000a7', 
    '#eecc16',
    '#008176',
    '#b3b3b3'
    ]

def show_colors(colors=[]):
    """
    show my favorite colors code
    see https://color.adobe.com/zh/explore
    """
    if len(colors) == 0:
        color_name = ray_colors
    else:
        color_name = colors
    n = len(color_name)
    fig, ax = plt.subplots(figsize=(2*n, 5))
    rects = ax.bar(range(len(color_name)), 0.8, color=color_name, label=color_name)
    ax.set_ylim(0, 1)
    ax.bar_label(rects, labels=color_name, padding=3)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.yaxis.set_visible(False)
    plt.show()

    print(color_name)

    return None
# **************************************************

def add_text(ax, text, fontsize=20, frameon=False, loc='upper right', **kwargs):
    """
    Add text to the figure's ax.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.AxesSubplot
        Axes object.
    text : str
        Text to add.
    fontsize : int, optional
        Font size of the text. The default is 20.
    frameon : bool, optional
        Whether to draw a frame around the text. The default is False.
    loc : str, optional
        Location of the text. The default is 'upper left'. Options are:
        'upper left', 'upper right', 'lower left', 'lower right',
        'center', 'center left', 'center right', 'lower center',
        'upper center', 'right', 'left', 'top', 'bottom'.
    kwargs : dict, optional
        Additional keyword arguments for the text function.

    Returns
    -------
    None.

    """
    at = AnchoredText(text, prop=dict(size=fontsize), frameon=frameon, 
                      loc=loc, **kwargs)
    ax.add_artist(at)
    return None


def savefig(fname='unname.png', 
            dir_save=Path("/Users/rui/Downloads/"), 
            dpi=300, bbox_inches="tight", **kwargs):
    """
    save the figure to the path_dir
    
    Parameters
    ----------
    fname : str
        the name of the figure
    path_dir : str or pathlib.Path, optional
        the path to save the figure. The default is Path("/Users/rui/Downloads/").
    dir_name : str, optional

    """
    if isinstance(dir_save, str):
        path = Path(dir_save)
    else:
        path = dir_save
    plt.savefig(path / fname, dpi=dpi, bbox_inches=bbox_inches, facecolor='w', **kwargs)

def bold_axis(ax, width=1.5):
    for spine in ax.spines.values():
        spine.set_linewidth(width)
    ax.minorticks_on()
    ax.tick_params(axis='x', which='major', width=width, length=5, direction='in')
    ax.tick_params(axis='x', which='minor', width=width, length=3, direction='in')
    ax.tick_params(axis='y', which='major', width=width, length=5, direction='in')
    ax.tick_params(axis='y', which='minor', width=width, length=3, direction='in')
    return None
    
def add_colorbar_ax(fig, ax, width=0.05, pad=0.07, location='right', fix=True):
    """
    Add a colorbar axis to a figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes._subplots.AxesSubplot
        Axes object or a list of axes objects.
    width : float, optional
        Width of the colorbar axis. The default is 0.05.
    pad : float, optional
        Padding between the colorbar axis and the axes. The default is 0.07.
    location : str, optional
        Location of the colorbar axis. The default is 'right'.
    fix : bool, optional
        Whether to fix the width and padding. The default is True.
    """
    from matplotlib.transforms import Bbox
    h = fig.properties()['figheight']
    w = fig.properties()['figwidth']

    if isinstance(ax, np.ndarray):
        x0, x1, y0, y1 = [], [], [], []
        for ax_sub in ax:
            x0.append(ax_sub.get_position().x0)
            x1.append(ax_sub.get_position().x1)
            y0.append(ax_sub.get_position().y0)
            y1.append(ax_sub.get_position().y1)
        ax_pos = Bbox.from_extents(min(x0), min(y0), max(x1), max(y1))
    else:
        ax_pos = ax.get_position()

    if location == 'right':
        if fix:
            width = width / w
            pad = pad / w
        cax_pos = Bbox.from_extents(
            ax_pos.x1 + pad,
            ax_pos.y0,
            ax_pos.x1 + pad + width,
            ax_pos.y1,
        )
    if location == 'bottom':
        if fix:
            width = width / h
            pad = pad / h
        cax_pos = Bbox.from_extents(
            ax_pos.x0,
            ax_pos.y0 - pad - width,
            ax_pos.x1,
            ax_pos.y0 - pad,
        )
    if location == 'left':
        if fix:
            width = width / w
            pad = pad / w
        cax_pos = Bbox.from_extents(
            ax_pos.x0 - pad - width,
            ax_pos.y0,
            ax_pos.x0 - pad,
            ax_pos.y1,
        )
    if location == 'top':
        if fix:
            width = width / h
            pad = pad / h
        cax_pos = Bbox.from_extents(
            ax_pos.x0,
            ax_pos.y1 + pad,
            ax_pos.x1,
            ax_pos.y1 + pad + width,
        )
    cax = fig.add_axes(cax_pos)
    return cax

def plot_scatter_distribution(
        data, x: str, y: str, xlabel=None, ylabel=None,
        marker_size=1, bins=50, color="#103778", 
        label=None,
        alpha=0.7, save_path=None
):
    """
    plot散点图及数量分布
    
    Parameters
    ----------
    data: 结构化数据表
    x: str, x label
    y: str, y label
    xlabel: None(optional); 指定显示的xlabel
    ylabel: None(optional); 指定显示的ylabel
    marker_size: 1(optinoal); 散点的大小
    bins: 50(optional); hist bins
    color: optional; 散点的颜色
    label: scatter的label
    alpha: 0.7(optional); scatter和hist的alpha
    sava_path: None(optinal); 图像保存路径

    Return
    ------
    ax: 主散点图的ax
    ax_hist_top: top hist ax
    ax_hist_right: right hist ax
    """
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(data[x], data[y], s=marker_size, c=color, alpha=alpha, label=label)
    # 创建两个ax, 用于绘制柱状图
    divider = make_axes_locatable(ax)
    ax_hist_top = divider.append_axes("top", 1.7, pad=0, sharex=ax)
    ax_hist_right = divider.append_axes("right", 1.7, pad=0, sharey=ax)
    # 关闭柱状图的标签
    ax_hist_top.xaxis.set_tick_params(labelbottom=False)
    ax_hist_right.yaxis.set_tick_params(labelleft=False)
    ax_hist_top.hist(data[x], bins=bins, histtype='bar', orientation='vertical', 
                    align='mid', color=color, alpha=alpha)
    ax_hist_right.hist(data[y], bins=bins, histtype='bar', orientation='horizontal', 
                    align='mid', color=color, alpha=alpha)
    # 细节配置
    xlabel=x if (xlabel is None) else xlabel
    ylabel=y if (ylabel is None) else ylabel
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax_hist_top.set_ylabel("N", fontsize=20)
    ax_hist_right.set_xlabel("N", fontsize=20)

    for spine in ax.spines.values():
        spine.set_linewidth(2)
    for spine in ax_hist_top.spines.values():
        spine.set_linewidth(2)
    for spine in ax_hist_right.spines.values():
        spine.set_linewidth(2)
    # ax的坐标轴设置
    ax.tick_params(axis='both', which='major', 
                    top=True, right=True, 
                    width=2, length=5, 
                    direction='in', labelsize=15)
    ax.tick_params(axis='both', which='minor', 
                    top=True, right=True, 
                    width=2, length=2.5, 
                    direction='in', labelsize=15)
    ax.minorticks_on()
    # ax_hist_top的坐标轴设置
    ax_hist_top.tick_params(axis='y', which='major', 
                            top=True, right=True, 
                            width=2, length=5, 
                            direction='in', labelsize=15)
    ax_hist_top.tick_params(axis='y', which='minor', 
                    top=True, right=True, 
                    width=2, length=2.5, 
                    direction='in', labelsize=15)
    ax_hist_top.minorticks_on()
    ax_hist_top.yaxis.get_major_ticks()[0].label1.set_visible(False)

    # ax_hist_right的坐标轴设置
    ax_hist_right.tick_params(axis='x', which='major', 
                            top=True, right=True, 
                            width=2, length=5, 
                            direction='in', labelsize=15)
    ax_hist_right.tick_params(axis='x', which='minor', 
                    top=True, right=True, 
                    width=2, length=2.5, 
                    direction='in', labelsize=15)
    ax_hist_right.minorticks_on()
    ax_hist_right.xaxis.get_major_ticks()[0].label1.set_visible(False)

    if save_path is not None:
        plt.savefig(save_path, dpi=500, bbox_inches='tight', facecolor='w')

    return ax, ax_hist_top, ax_hist_right


def plot_aperture_photometry_growth_curve(
        data, position=None, r_min=1, r_max=35, dr=1, zp=None, pixel_scale=0.06, 
        name=None, detect_threshold=None, 
        cmap=cmasher.chroma, vmin=None, vmax=None, save_path=None):
    """
    Plot the growth curve of aperture photometry.

    Parameters
    ----------
    data : 2D array
        The image data.
    position : tuple, optional
        The position of the center of the aperture. If not given, the peak position will be used.
    r_min : int, optional
        The minimum radius of the aperture. The default is 1.
    r_max : int, optional
        The maximum radius of the aperture. The default is 35.
    dr : int, optional
        The step of the radius. The default is 1.
    zp : float, optional
        The zero point of the image. The default is None.
    pixel_scale : float, optional
        The pixel scale of the image. The default is 0.06.
    name : str, optional
        The name of the image. The default is None.
    detect_threshold : float, optional
        The threshold of the detection. The default is None, which means the threshold will be calculated automatically using sigma clipping.
    cmap : str, optional
        The colormap of the image. The default is cmasher.chroma.
    vmin : float, optional
        The minimum value of the image. The default is None.
    vmax : float, optional
        The maximum value of the image. The default is None.
    save_path : str, optional
        The path to save the figure. The default is None.

    Returns
    -------
    None.
    """

    # decide the center of the aperture
    mean, median, std = sigma_clipped_stats(data, sigma=3)
    if detect_threshold is None:
        detect_threshold = median + 5*std
    else:
        detect_threshold = detect_threshold
    peak_position = find_peaks(data, threshold=detect_threshold)
    peak_position = peak_position.to_pandas()
    cx = data.shape[1]/2
    cy = data.shape[0]/2
    peak_position['dist'] = np.sqrt((peak_position['x_peak']-cx)**2 + (peak_position['y_peak']-cy)**2)
    peak_position = peak_position.loc[peak_position['dist'] < 10]
    peak_position.sort_values(by='peak_value', ascending=False, inplace=True)
    peak_position.reset_index(drop=True, inplace=True)

    # perform aperture photometry around the peak position with radius r
    if position is None:
        if len(peak_position) == 0:
            raise ValueError(f"No peak position found. Current detect_threshold is {detect_threshold}.")
        position = peak_position.loc[0, [ 'x_peak', 'y_peak']]

    # a for loop to perform aperture photometry with different radius
    res = dict()
    res['r'] = []
    res['aper_sum'] = []
    for r in np.arange(r_min, r_max, dr):
        res['r'].append(r)
        aperture = CircularAperture(position, r=r)
        from photutils.aperture import aperture_photometry
        phot_table = aperture_photometry(data, aperture)
        res['aper_sum'].append(phot_table['aperture_sum'][0])
    res = pd.DataFrame(res)

    fig, ax = plt.subplots(1, 2, width_ratios=[0.6, 0.4], figsize=(12, 5), layout='constrained')

    fontsize = 15

    l1, = ax[0].plot(res['r'], res['aper_sum']/res['aper_sum'].max(), c='gray', 
                    linewidth=2, label='enclosed flux curve')
    ax[0].set_xlabel("aperture radius [pixel]", fontsize=fontsize)
    ax[0].set_ylabel("normalized enclosed flux", fontsize=fontsize)

    if pixel_scale is not None:
        ax_0 = ax[0].twiny()
        ax_0.scatter(res['r']*pixel_scale, res['aper_sum']/res['aper_sum'].max(), c='black', s=1)
        ax_0.set_xlabel("aperture radius [arcsec]", fontsize=fontsize)
        ax_0.tick_params(axis='x', which='major', width=2, length=5, direction='in')

    if zp is not None:
        ax_1 = ax[0].twinx()
        res['mag'] = -2.5*np.log10(res['aper_sum']) + zp
        l2, = ax_1.plot(res['r'], res['mag'], c='blue', 
                        linewidth=2, label="magnitude curve")
        ax_1.set_ylabel("Magnitude", fontsize=fontsize)
        ax_1.minorticks_on()
        ax_1.tick_params(axis='y', which='major', width=2, length=5, direction='in')
        ax_1.tick_params(axis='y', which='minor', width=2, length=3, direction='in')
        ax[0].legend(loc='center right', handles=[l1, l2])

        # add annotation
        at = AnchoredText(f"mag={res['mag'].values[-1]:.3f} (zp={zp})", 
                          loc='lower right', 
                          prop=dict(size=12, color='white'), 
                          frameon=False)
        ax[1].add_artist(at)

    # setting the linewidth of the spines
    for spine in ax[0].spines.values():
        spine.set_linewidth(2)
    ax[0].minorticks_on()
    ax[0].tick_params(axis='x', which='major', width=2, length=5, direction='in')
    ax[0].tick_params(axis='x', which='minor', width=2, length=3, direction='in')
    ax[0].tick_params(axis='y', which='major', width=2, length=5, direction='in')
    ax[0].tick_params(axis='y', which='minor', width=2, length=3, direction='in')

    vmin = np.min(data) if vmin is None else vmin
    vmax = np.max(data) if vmax is None else vmax
    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LogStretch())
    ax[1].imshow(data, norm=norm, cmap=cmap, origin='lower')
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), 
                        ax=ax[1], location='right', ticks=[vmin, 0, vmax], 
                        pad=0.05, shrink=0.8)
    for spine in ax[1].spines.values():
        spine.set_linewidth(1)
    ax[1].tick_params(axis='both', which='major', width=1, length=5, direction='out')

    # plot the aperture
    x = aperture.positions[0]
    y = aperture.positions[1]
    ax[1].scatter(x, y, marker='+', s=100, c='red', 
                label=f"aperture center ({x:.1f}, {y:.1f})")
    aperture.plot(color='white', lw=2, ax=ax[1], 
                label=f"maximal aperture (r = {aperture.r})")
    ax[1].legend(loc='upper left', fontsize=10, frameon=False, labelcolor='white')
    if name is not None:
        ax[1].set_title(name, fontsize=15, fontweight='bold')

    # save the figure
    if save_path is not None:
        save_path = Path(save_path)
        title = save_path.stem
        fig.savefig(save_path, dpi=500, bbox_inches='tight', facecolor='w')

    return None


def plot_stamps(coordinates, wcs, large_image_data, 
                names=None, texts=None, cutout_size=100, save_fname=None,
                nx=None, ny=None, cmap=cmasher.chroma, vmin=None, vmax=None):
    """
    Plot stamps of sources on a large image.

    Parameters
    ----------
    coordinates : list of tuples
        List of coordinates of sources.
    wcs : astropy.wcs.wcs.WCS
        WCS of the large image.
    large_image_data : numpy.ndarray
        Data of the large image.
    names : list of str, optional
        Names of sources. The default is None.
    texts : list of str, optional
        Texts of sources. The default is None.
    cutout_size : int, optional
        Size of the cutout. The default is 100.
    save_fname : str, optional
        File name to save the figure. The default is None.
    nx : int, optional
        Number of columns of the figure. The default is None.
    ny : int, optional
        Number of rows of the figure. The default is None.
    cmap : matplotlib.colors.LinearSegmentedColormap, optional
        Colormap of the large image. The default is cmasher.chroma.

    """
    from astrokit.toolbox.calculate import crack

    num = len(coordinates)
    if (nx is None) & (ny is None):
        nx = max(crack(num))
        ny = min(crack(num))
        if (nx / ny) > 5 or (nx > 10):
            ny = 5
            nx = math.ceil(num / ny)
    else:
        pass

    # set figure axes
    fig, axs = plt.subplots(ny, nx, figsize=(2*nx, 2*ny))
    ax = axs.ravel()

    # 超出子图设置为空
    if nx*ny > num:
        null = list(range(num, nx*ny))
        for i in null:
            ax[i].axis("off")

    # plot each source
    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LogStretch())
    for i in range(num):
        (ra, dec) = coordinates[i]
        (x, y) = wcs.world_to_pixel_values(ra, dec)
        x, y = float(x), float(y)

        cutout_obj = Cutout2D(large_image_data, position=(x, y), size=cutout_size)
        ax[i].imshow(cutout_obj.data, norm=norm, origin='lower', cmap=cmap)
        if names is not None:
            at = AnchoredText(names[i], prop=dict(size=12, color='white'), borderpad=0.05, 
                            frameon=False, loc='upper left')
            ax[i].add_artist(at)
        if texts is not None:
            at = AnchoredText(texts[i], prop=dict(size=10, color='white'), borderpad=0.05, 
                    frameon=False, loc='lower right')
            ax[i].add_artist(at)

        ax[i].set_xticks([])
        ax[i].set_yticks([])

    cbar = add_colorbar_ax(fig, ax=ax, width=0.1, pad=0.1)
    fig.colorbar(mappable=ScalarMappable(norm=norm, cmap=cmap), 
                cax=cbar, orientation='vertical', ticks=[norm.vmin, 0, norm.vmax])
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    if save_fname is not None:
        savefig(f"{save_fname}")
    return None

def plot_SDSS_spectrum(path, source_name, redshift=None, obs_lamb_range=None, 
                       y_range_percentile=99.9, y_range_scale=1.5):
    """
    Plot SDSS spectrum.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the SDSS spectrum file.
    source_name : str
        Name of the source.
    redshift : float, optional
        Redshift of the source.
    obs_lamb_range : tuple, optional
        Observed wavelength range to plot.
    y_range_percentile : float, optional
        Percentile of the flux to set the y range.
    y_range_scale : float, optional
        Scale factor of the y range.
    """
    from astrokit.spec import show_emission_lines

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # load SDSS spectrum
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec = Spectrum1D.read(path, format="SDSS-III/IV spec")
    if redshift is not None:
        spec.set_redshift_to(redshift)

    # 用于绘图的光谱范围
    if obs_lamb_range is not None:
        (lambda_min, lambda_max) = obs_lamb_range
        spec = spectral_slab(spec, lambda_min*u.AA, lambda_max*u.AA)
    else:
        (lambda_min, lambda_max) = (spec.spectral_axis.min().value, 
                                    spec.spectral_axis.max().value)

    spec_smooth = gaussian_smooth(spec, stddev=3)
    spec_smooth.set_redshift_to(spec.redshift)

    # plot
    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(spec.spectral_axis, spec.flux, lw=1, color='grey')
    ax.plot(spec_smooth.spectral_axis, spec_smooth.flux, lw=2, color='black')

    ax.set_xlabel(r"Observed wavelength ($\mathrm{\AA}$)", fontsize=15)
    ax.set_ylabel(r"$f_{\lambda}$ ($\mathrm{10^{-17}\ erg\ s^{-1}\ cm^{-2}\ \AA^{-1}}$)", fontsize=15)

    # x范围
    ax.set_xlim(lambda_min, lambda_max)

    # y范围
    flux = spec_smooth.flux.data
    y_min = np.percentile(flux, 100-y_range_percentile) * y_range_scale
    y_max = np.percentile(flux, y_range_percentile) * y_range_scale
    ax.set_ylim(y_min, y_max)

    # 刻度线细节设置
    ax.minorticks_on()
    ax.tick_params(axis='y', which='major', width=1, length=5, direction='in')
    ax.tick_params(axis='y', which='minor', width=0.5, length=3, direction='in')
    ax.tick_params(axis='x', which='major', width=1, length=5, direction='in')
    ax.tick_params(axis='x', which='minor', width=0.5, length=3, direction='in')

    if redshift is not None:
        ax2 = ax.twiny()
        # 上刻度
        ax2.plot(spec_smooth.spectral_axis.to_rest(), 
                spec_smooth.flux, 
                lw=0.5, color='None')
        ax2.set_xlim(spec_smooth.spectral_axis.to_rest().value.min(), 
                    spec_smooth.spectral_axis.to_rest().value.max())
        ax2.set_ylim(y_min, y_max)
        ax2.set_xlabel(r"Rest-frame wavelength ($\mathrm{\AA}$)", fontsize=15)

        ax2.minorticks_on()
        ax2.tick_params(axis='x', which='major', width=1, length=5, direction='in')
        ax2.tick_params(axis='x', which='minor', width=0.5, length=3, direction='in')

        # 添加发射线注释线
        emission_lines = show_emission_lines()
        min = spec_smooth.spectral_axis.to_rest().min().value
        max = spec_smooth.spectral_axis.to_rest().max().value

        for index, row in emission_lines.iterrows():
            if (row['rest_lamb'] > min) and (row['rest_lamb'] < max):
                ax2.axvline(row['rest_lamb'], ls='--', color='blue', alpha=0.5, lw=1)
                ax2.text(row['rest_lamb'], ax2.get_ylim()[0], 
                        row['name'], 
                        rotation=90, va='bottom', ha='right', 
                        fontsize=10, color='blue')
        # 添加信息
        content = f"{source_name} at Z = {redshift:.3f}"
    else:
        content = f"{source_name}"
    ax.text(0.8, 0.8, content, transform=ax.transAxes, fontsize=25, 
            color="red", weight='bold', ha='right')
    return None

def imshow(data, wcs=None, ax=None, 
           vmin=None, vmax=None, 
           cmap=cmasher.chroma, 
           colorbar=True, 
           scalebar_loc='bottom right',
           scalebar_length=2, 
           scalebar_text_size=10, 
           scalebar_width=0.5, 
           scalebar_pad=1, 
           scalebar_color='white'):
    """
    Quick look a data of image.

    Parameters
    ----------
    data : 2D array
        The image data.
    wcs : `~astropy.wcs.WCS` instance
        The WCS information.
    ax : `~matplotlib.axes.Axes` instance
        The axes to plot the image.
    vmin : float
        The minimum value of the image.
    vmax : float
        The maximum value of the image.
    cmap : `~matplotlib.colors.Colormap` instance
        The colormap of the image.
    colorbar : bool
        Whether to show the colorbar.
    scalebar_loc : str
        The location of the scalebar.
        Acceptable values are:
        'left', 'right', 'top', 'bottom', 
        'top left', 'top right', 'bottom left', 'bottom right' (default)
    scalebar_length : float
        The length of the scalebar in arcsec.
    scalebar_text_size : int
        The size of the text of the scalebar.
    scalebar_width : float
        The width of the scalebar.
    scalebar_pad : float
        The pad of the scalebar.
    scalebar_color : str
        The color of the scalebar.
    """
    if vmin is None:
        vmin = np.min(data)
    if vmax is None:
        vmax = np.max(data)
    ticks= [vmin, 0, vmax]

    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LogStretch())
    
    if ax is None:
        if isinstance(wcs, WCS):
            fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={'projection': wcs})
            # xy坐标轴等比例
            ax.set_xlabel('RA', labelpad=1, fontsize=12)
            ax.set_ylabel('Dec', labelpad=1, fontsize=12)
        else:
            fig, ax = plt.subplots(figsize=(5, 5))

    # Hide ticks and labels separately for each axis
    ax.tick_params(axis='x', bottom=True, labelbottom=True, top=False)
    ax.tick_params(axis='y', left=True, labelleft=True, right=False)

    ax.imshow(data, origin='lower', norm=norm, cmap=cmap)

    if isinstance(wcs, WCS):
        # 添加scalebar
        wcsaxes.add_scalebar(
            ax=ax, 
            corner=scalebar_loc, 
            length=scalebar_length*u.arcsec, 
            label=f'{scalebar_length}"', 
            color=scalebar_color,
            label_top=True, 
            pad=scalebar_pad, 
            size_vertical=scalebar_width, 
            fontproperties={'size': scalebar_text_size}
            )
    if colorbar:
        fig.colorbar(
            ScalarMappable(norm=norm, cmap=cmap), 
            ax=ax, location='right', ticks=ticks, 
            pad=0.05, shrink=0.83
            )
    return None