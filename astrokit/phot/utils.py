import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.convolution import convolve
from astropy.visualization import ImageNormalize, LogStretch
from astropy.coordinates import SkyCoord


try:
    from photutils.segmentation import detect_threshold, SourceCatalog, SourceFinder
    from photutils.segmentation import make_2dgaussian_kernel
except:
    print("==> Can not import photutils. [astrokit.image.utils]")


__all__ = [
    'search_source', 
    'check_header', 
    'cutout', 
    'source_detect', 
    'make_mask_image', 
    'edge_detection', 
    'edge_process'
]


def search_source(ra, dec, catalog, sep=1, col_names=('ra', 'dec'), silence=False):
    """
    Search for the nearest source in the catalog.
    
    Parameters
    ----------
    ra : float
        Right ascension of the target source.
    dec : float
        Declination of the target source.
    catalog : pandas.DataFrame
        The catalog to search for the nearest source.
    sep: float, unit in arcsec
        threshold of query
    """
    c = SkyCoord(ra=ra, dec=dec, unit='deg')
    catalog_coords = SkyCoord(ra=catalog[col_names[0]], dec=catalog[col_names[1]], unit='deg')

    idx, d2d, d3d = c.match_to_catalog_sky(catalog_coords)
    d2d = d2d[0].to("arcsec").value

    if d2d < 1:
        obj = catalog[catalog.index == idx].copy()
        if not silence:
            print(f"separation distance: {d2d:.6f} arcsec")
    else:
        obj = None
        if not silence:
            print(f"separation distance: {d2d:.6f} arcsec is larger than the sep: {sep} arcsec.")

    return obj


def check_header(header, keywords=["EXPTIME", "GAIN", "NCOMBINE", "BUNIT"]):
    """
    check some important keywords in the header.
    """
    for kw in keywords:
        if kw in list(header.keys()):
            print(f"{kw}: {header[kw]} | {header.comments[kw]}")
        else:
            print(f"{kw}: not in header.")

    print("-"*10+' about TIME '+"-"*10)
    for kw in list(header.keys()):
        if ('TIME' in kw) & (kw != 'EXPTIME'):
            print(f"{kw}: {header[kw]} | {header.comments[kw]}")

    print("-"*10+' about GAIN '+"-"*10)
    for kw in list(header.keys()):
        if ('GAIN' in kw) & (kw != 'GAIN'):
            print(f"{kw}: {header[kw]} | {header.comments[kw]}")
    return None

def cutout(ra, dec, large_image_data, wcs, size=100):
    position = wcs.world_to_pixel_values(ra, dec)
    cutout_obj = Cutout2D(data=large_image_data, position=position, size=size, copy=True)
    hdu = fits.PrimaryHDU(data=cutout_obj.data)
    return hdu


def source_detect(data, kernel_fwhm=3, kernel_size=21, 
                  detect_threshold_nsigma=3, 
                  npixels=16, show=True, 
                  vmin=None, vmax=None, 
                  return_segment_map=False):
    """
    using photutils to detect sources.
    
    Parameters
    ----------
    data : 2d array
        The 2D array of the image.
    kernel_fwhm : float, optional
        The full width at half maximum (FWHM) of the Gaussian kernel in pixels.
        The default is 3.
    kernel_size : int, optional
        The kernel size in pixels. The default is 21.
    detect_threshold_nsigma : float, optional
        The number of standard deviations per pixel above the background
        for which to consider a pixel as possibly being part of a source.
        The default is 5.
    npixels : int, optional
        The minimum number of connected pixels, each greater than
        ``detect_threshold``, that an object must have to be detected.
        The default is 16.
    show : bool, optional
        show the image. The default is False.
    vmin : float, optional
        The minimum value of the image. The default is None.
    vmax : float, optional
        The maximum value of the image. The default is None.

    Returns
    -------
    cat : pandas.DataFrame
        The catalog of detected sources.
    segment_map : 2d array
        In case of 'return_segment_map=True', the return will also include the 
        segmentation map of the image.
    """
    # ---------- step1: make smoothing kernel and convolve with data
    kernel = make_2dgaussian_kernel(fwhm=kernel_fwhm, size=kernel_size)
    data_convolved = convolve(data, kernel)

    # ---------- step2: source detect and deblend
    threshold = detect_threshold(data=data_convolved, nsigma=detect_threshold_nsigma)
    finder = SourceFinder(
        npixels=npixels, # the minimun number of connected pixels
        nlevels=32, # The number of multi-thresholding levels to use for deblending.
        contrast=0.01, 
        progress_bar=False, nproc=1
        )
    segment_map = finder(data_convolved, threshold)

    # ---------- step3: make catalog
    segment_cat = SourceCatalog(data=data, 
                                segment_img=segment_map, 
                                convolved_data=data_convolved)
    params = [
        # 坐标
        'label', 'xcentroid', 'ycentroid',
        # 测光
        'segment_flux', 'kron_flux', 
        # 形态参数
        'semimajor_sigma', 'semiminor_sigma', 'orientation', 'ellipticity', 
        're', 'kron_radius', 
        'fwhm', 'gini', 
        # 特殊像素值
        'min_value', 'max_value', 
        # 占据空间
        'area', 'bbox_xmin', 'bbox_xmax', 'bbox_ymin', 'bbox_ymax',  
    ]
    segment_cat.fluxfrac_radius(fluxfrac=0.5, name='re', overwrite=True)
    cat = segment_cat.to_table(columns=params).to_pandas()

    cat['axis_ratio'] = 1 - cat['ellipticity']
    cat.loc[cat['orientation']>=0, 'PA'] =  cat.loc[cat['orientation']>=0, 'orientation'] - 90
    cat.loc[cat['orientation']< 0, 'PA'] =  cat.loc[cat['orientation']< 0, 'orientation'] + 90

    cat.rename(columns={"xcentroid": "x", 
                        "ycentroid": "y",
                        "semimajor_sigma": "a", 
                        "semiminor_sigma": "b", 
                        }, inplace=True)
    params = [
        # position
        'label', 'x', 'y', 
        # flux
        'segment_flux', 'kron_flux', 
        # morphological parameters
        'a', 'b', 're', 'kron_radius', 'axis_ratio', 'PA', 
        'fwhm', 'gini', 
        # stats
        'min_value', 'max_value', 
        # bounding box
        'area', 'bbox_xmin', 'bbox_xmax', 'bbox_ymin', 'bbox_ymax', 
    ]
    cat = cat[params]
    cat.insert(0, 'class', None)

    # ---------- step4: label each object
    center = 0.5 * data.shape[0]
    cat['Dc'] = ((cat['x']-center)**2 + (cat['y']-center)**2)**0.5

    # 最中心的源设定为目标源
    index_target = cat['Dc'].idxmin() 
    cat.loc[index_target, 'class'] = 'target'

    target = cat.iloc[index_target]
    mask_zero = np.zeros(shape=data.shape)
    mask_target = mask_zero.copy()  # 创建target bbox的mask图像
    mask_target[target['bbox_ymin']: target['bbox_ymax'], target['bbox_xmin']: target['bbox_xmax']] = 1

    for idx, row in cat.iterrows():
        if row['class'] == None:
            mask_obj = mask_zero.copy()  # 待分类目标的bbox mask
            mask_obj[row['bbox_ymin']: row['bbox_ymax'], row['bbox_xmin']: row['bbox_xmax']] = 1
            if 1 in (mask_target * mask_obj):
                cat.loc[idx, 'class'] = 'overlap'
            else:
                cat.loc[idx, 'class'] = 'near'
    
    # ---------- step5: plot
    if show:
        fig, ax = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        import cmasher as cmr
        import matplotlib as mpl
        cmap = cmr.chroma
        if vmin is None:
            vmin = np.min(data)
        if vmax is None:
            vmax = np.max(data)
        norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LogStretch())
        ax[0].set_title("detection image", fontsize=15, fontweight='bold')
        ax[0].imshow(data, origin='lower', norm=norm, cmap=cmap)
        segment_cat.plot_kron_apertures(ax=ax[0], color='white')

        ax[1].set_title("smoothed image", fontsize=15, fontweight='bold')
        ax[1].imshow(data_convolved, origin='lower', norm=norm, cmap=cmap)

        ax[2].set_title("segmentation map", fontsize=15, fontweight='bold')
        segment_map.imshow(ax[2], cmap=None)
        for label in segment_map.labels:
            bbox = segment_map.bbox[segment_map.get_index(label=label)]
            center_x, center_y = bbox.center[1], bbox.center[0]
            ax[2].text(center_x, center_y, str(label), 
                    backgroundcolor='white')
            ax_bbox = bbox.plot(ax[2])
            ax_bbox.set_color("green")
    
    if return_segment_map:
        return cat, segment_map
    else:
        return cat
    


def make_mask_image(segment_map, mask_labels, show=False, 
                    fname=None, save_path="/Users/rui/Downloads"):
    """
    make mask image from segment map

    Parameters
    ----------
    segment_map : 2d array
        segment map
    mask_labels : list
        labels to be masked
    show : bool, optional
        show mask image, by default False
    fname : str, optional
        file name to save mask image, by default None
    save_path : str, optional
        path to save mask image, by default "/Users/rui/Downloads"
    """
    mask_img = segment_map.copy()
    mask_img.keep_labels(labels=mask_labels)
    mask_img = mask_img.data
    mask_img[mask_img!=0] = 1

    if fname is not None:
        hdu = fits.PrimaryHDU(mask_img)
        path = Path(save_path) / fname
        hdu.writeto(path, overwrite=True)

    if show is True:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(mask_img, origin='lower', cmap='gray')
        ax.set_title("mask image", fontsize=15, fontweight='bold')
    return mask_img


def edge_detection(data, wcs=None):
    """
    scan the image to find the (x, y) position of the edge points.

    Parameters
    ----------
    data : 2D array
        The image data.
    wcs : WCS object, optional
        The WCS object of the image. The default is None.
        If wcs is provided, it will return the edge points' ra and dec.

    Returns
    -------
    edge : dict
        The edge points' position.
        ['up', 'down', 'left', 'right'] are defined in x-y plane or ra-dec plane.
    """

    # transform the data to binary for detecting the edge
    print(f"==> creating edge detection map ...")
    data[data != 0] = True
    data[data == 0] = False

    # detect the edge points
    position_x_small = []
    position_x_big = []
    for i in tqdm(range(len(data)), desc="==> scanning y axis: "):
        x_true = np.where(data[i]==1)[0]
        if len(x_true) > 1:
            position_x_small.append((x_true[0], i))
            position_x_big.append((x_true[-1], i))

    position_y_small = []
    position_y_big = []
    for i in tqdm(range(len(data.T)), desc="==> scanning x axis: "):
        y_true = np.where(data.T[i]==1)[0]
        if len(y_true) > 1:
            position_y_small.append((i, y_true[0]))
            position_y_big.append((i, y_true[-1]))

    # collect the edge points' position into a dictionary
    edge = dict()
    edge[0] = pd.DataFrame(position_x_small, columns=['x', 'y'])
    edge[1] = pd.DataFrame(position_x_big, columns=['x', 'y'])
    edge[2] = pd.DataFrame(position_y_big, columns=['x', 'y'])
    edge[3] = pd.DataFrame(position_y_small, columns=['x', 'y'])

    if wcs is None:
        edge['up'] = edge.pop(2)
        edge['down'] = edge.pop(3)
        edge['left'] = edge.pop(0)
        edge['right'] = edge.pop(1)

        edge['left'].sort_values(by='y', ascending=False, inplace=True, ignore_index=True)
        edge['down'].sort_values(by='x', ascending=True, inplace=True, ignore_index=True)
        edge['right'].sort_values(by='y', ascending=True, inplace=True, ignore_index=True)
        edge['up'].sort_values(by='x', ascending=False, inplace=True, ignore_index=True)
        
    else:
        df_mean_coord = pd.DataFrame()  # store the mean coordinate of each edge
        for key in edge.keys():
            edge[key]['ra'], edge[key]['dec'] = wcs.pixel_to_world_values(edge[key]['x'], edge[key]['y'])
            df_mean_coord.loc[key, 'ra'] = edge[key]['ra'].mean()
            df_mean_coord.loc[key, 'dec'] = edge[key]['dec'].mean()
        # identify the directions
        df_mean_coord.loc[df_mean_coord['ra']==df_mean_coord['ra'].min(), 'new_key'] = 'left'
        df_mean_coord.loc[df_mean_coord['ra']==df_mean_coord['ra'].max(), 'new_key'] = 'right'
        df_mean_coord.loc[df_mean_coord['dec']==df_mean_coord['dec'].min(), 'new_key'] = 'down'
        df_mean_coord.loc[df_mean_coord['dec']==df_mean_coord['dec'].max(), 'new_key'] = 'up'
        # rename the keys
        for key in df_mean_coord.index:
            edge[df_mean_coord.loc[key, 'new_key']] = edge.pop(key)
        # sort the edge points
        edge['left'].sort_values(by='dec', ascending=False, inplace=True, ignore_index=True)
        edge['down'].sort_values(by='ra', ascending=True, inplace=True, ignore_index=True)
        edge['right'].sort_values(by='dec', ascending=True, inplace=True, ignore_index=True)
        edge['up'].sort_values(by='ra', ascending=False, inplace=True, ignore_index=True)
    
    return edge

def edge_process(edge, left_cut=None, right_cut=None, show=False):
    """
    process the edge detection result for plotting

    Parameters
    ----------
    edge : dict
        the result of edge detection (from function `edge_detection`)
    left_cut : float, optional
        the left cut of the edge for splitting, by default None
    right_cut : float, optional
        the right cut of the edge for splitting, by default None
    show : bool, optional
        whether to show the result, by default False
    """
    if (left_cut is None) & (right_cut is None):
        new_left = edge['left'].copy()
        new_right = edge['right'].copy()
        new_up = edge['up'].copy()
        new_down = edge['down'].copy()
        if 'ra' in edge['left'].columns:
            xlabel = 'ra'
            ylabel = 'dec'
        else:
            xlabel = 'x'
            ylabel = 'y'
    else:
        if 'ra' in edge['left'].columns:
            new_left = edge['left'].query("ra < @left_cut")
            new_right = edge['right'].query("ra > @right_cut")
            new_up = edge['up'].query("@left_cut < ra < @right_cut")
            new_down = edge['down'].query("@left_cut < ra < @right_cut")
            xlabel = 'ra'
            ylabel = 'dec'
        else:
            new_left = edge['left'].query("x < @left_cut")
            new_right = edge['right'].query("x > @right_cut")
            new_up = edge['up'].query("@left_cut < x < @right_cut")
            new_down = edge['down'].query("@left_cut < x < @right_cut")
            xlabel = 'x'
            ylabel = 'y'
    df = pd.concat([new_left, new_down, new_right, new_up])

    if show:
        fig, axs = plt.subplots(1, 2, figsize=(18, 5))
        axs[0].set_xlabel(xlabel)
        axs[0].set_ylabel(ylabel)
        axs[0].scatter(edge['left'][xlabel], edge['left'][ylabel], s=1, c='r')
        axs[0].scatter(edge['right'][xlabel], edge['right'][ylabel], s=1, c='b')
        axs[0].scatter(edge['up'][xlabel], edge['up'][ylabel], s=1, c='g')
        axs[0].scatter(edge['down'][xlabel], edge['down'][ylabel], s=1, c='y')
        if left_cut is not None:
            axs[0].axvline(left_cut, c='k', ls='--')
        if right_cut is not None:
            axs[0].axvline(right_cut, c='k', ls='--')
        axs[1].set_xlabel(xlabel)
        axs[1].set_ylabel(ylabel)
        axs[1].fill(df[xlabel], df[ylabel], edgecolor='black', lw=2)
    
    return df
