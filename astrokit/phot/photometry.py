"""
Tools for photometry.

@author: Rui Zhu
@creation time: 2023-04-03
"""
import numpy as np
from photutils.aperture import CircularAperture
from photutils.detection import find_peaks
from astropy.stats import sigma_clipped_stats
import matplotlib.pyplot as plt

from astrokit.toolbox import imshow

__all__ = ['aperture_photometry']


def aperture_photometry(data, header, zp, r, position=None, vmin=None, vmax=None, 
                        if_prompt=True, if_plot=False):
    """
    Perform aperture photometry.

    Parameters
    ----------
    data : numpy.ndarray
        The image data.
    header : astropy.io.fits.header.Header
        The image header for obtaining the pixel unit.
    zp : float
        The zeropoint.
    r : float
        The aperture radius in pixels
    position : tuple, optional
        The position of the aperture center in pixel. If None, the peak position will be used.
    vmin : float, optional
        The minimum value of the image for plotting. If None, the minimum value of the image will be used.
    vmax : float, optional
        The maximum value of the image for plotting. If None, the maximum value of the image will be used.
    if_prompt : bool, optional
        If True, the aperture photometry result will be printed.
    if_plot : bool, optional
        If True, the image will be plotted with the aperture and the peak position.

    Returns
    -------
    mag : float
        The magnitude of the aperture photometry.
    """
    # find the peak position
    mean, median, std = sigma_clipped_stats(data, sigma=3)
    peak_position = find_peaks(data, threshold=median+5*std)
    peak_position = peak_position.to_pandas()
    cx = data.shape[1]/2
    cy = data.shape[0]/2
    peak_position['dist'] = np.sqrt((peak_position['x_peak']-cx)**2 + (peak_position['y_peak']-cy)**2)
    peak_position = peak_position.loc[peak_position['dist'] < 10]
    peak_position.sort_values(by='peak_value', ascending=False, inplace=True)
    peak_position.reset_index(drop=True, inplace=True)
    
    # perform aperture photometry around the peak position with radius r
    if position is None:
        position = peak_position.loc[0, [ 'x_peak', 'y_peak']]

    aperture = CircularAperture(positions=position, r=r)
    from photutils.aperture import aperture_photometry
    res = aperture_photometry(data, aperture, method='exact')
    mag = -2.5*np.log10(res['aperture_sum'][0]) + zp

    # plot
    if if_plot == True:
        vmin = np.min(data) if vmin is None else vmin
        vmax = np.max(data) if vmax is None else vmax
        fig, ax = plt.subplots(figsize=(5, 5))
        imshow(data, ax=ax, vmin=vmin, vmax=vmax)
        ax.scatter(peak_position['x_peak'][0], peak_position['y_peak'][0], c='red', 
                label=f"peak ({peak_position['x_peak'][0]:.1f}, {peak_position['y_peak'][0]:.1f})",
                marker='+', s=35)
        aperture.plot(color='white', lw=2, alpha=1, ax=ax, label=f'aperture (r={r})')
        ax.scatter(aperture.positions[0], aperture.positions[1], c='red', 
                label=f'center ({aperture.positions[0]:.1f}, {aperture.positions[1]:.1f})',
                marker='o', s=5)
        ax.legend();
    
    if if_prompt == True:
        unit = header['BUNIT']
        print(f"==> pixel unit: {unit}")
        print(f"==> peak position: ({peak_position['x_peak'][0]:.1f}, {peak_position['y_peak'][0]:.1f})")
        print(f"==> aperture center: ({aperture.positions[0]:.1f}, {aperture.positions[1]:.1f})")
        print(f"==> aperture radius: {r}")
        print(f"==> aperture sum: {res['aperture_sum'][0]:.2f} {unit}")
        print(f"==> zero point: {zp}")
        print(f"==> magnitude: {mag}")
        
    return mag
