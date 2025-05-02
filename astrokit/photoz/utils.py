import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

__all__ = [
    'cal_RMSE',
    'cal_NMAD',
    'cal_bias',
    'cal_outlier_fraction',
    'cal_R2',
    'plot_zspec_zphot',
]



def cal_RMSE(z_spec: np.ndarray, z_phot: np.ndarray) -> float:
    """
    Calculate the root mean square error (RMSE)
    """
    dz = z_spec - z_phot
    rmse = np.sqrt(np.mean(dz**2))
    return float(rmse)

def cal_NMAD(z_spec: np.ndarray, z_phot: np.ndarray) -> float:
    """
    Calculate the normalized median absolute deviation (NMAD)
    """
    dz = z_spec - z_phot
    norm_dz = dz / (1 + z_spec)
    nmad = 1.4826 * np.median(np.abs(norm_dz))
    return float(nmad)

def cal_bias(z_spec: np.ndarray, z_phot: np.ndarray) -> float:
    """
    Calculate the bias: mean(z_spec - z_phot)
    """
    dz = z_spec - z_phot
    bias = np.mean(dz)
    return float(bias)

def cal_outlier_fraction(z_spec: np.ndarray, z_phot: np.ndarray, 
                         outlier_threshold=0.15) -> float:
    """
    Calculate the outlier fraction

    Parameters
    ----------
    outlier_threshold : float
        The threshold of the outlier, default is 0.15
    """
    dz = z_spec - z_phot
    norm_dz = dz / (1 + z_spec)
    N = len(dz)
    N_out = (np.abs(norm_dz) > outlier_threshold).sum()
    f_out = N_out / N
    return float(f_out)

def cal_R2(z_spec: np.ndarray, z_phot: np.ndarray) -> float:
    """
    Calculate the R2 score
    """
    R2 = 1 - np.sum((z_spec - z_phot)**2) / np.sum((z_spec - np.mean(z_spec))**2)
    return float(R2)

def plot_zspec_zphot(
        z_spec: np.ndarray, z_phot: np.ndarray,
        outlier_threshold: float = 0.15,
        z_range: list = None,
        bins1=[200, 200], bins2=[200, 200], 
        color_norm='log', 
        cmap1=None, 
        cmap2=None, 
        title=None,
        fig=None, 
        axes=None, 
        return_fig=False
        ):

    dz = z_spec - z_phot
    norm_dz = dz / (1 + z_spec)
    bias = cal_bias(z_spec, z_phot)
    NMAD = cal_NMAD(z_spec, z_phot)
    RMSE = cal_RMSE(z_spec, z_phot)
    outlier_fraction = cal_outlier_fraction(z_spec, z_phot, outlier_threshold)
    R2 = cal_R2(z_spec, z_phot)

    if z_range is None:
        z_range = [0, np.max([z_spec, z_phot])]
    
    colors = ["#000000", "#13235d", "#e0000f", "#f7f9b9"]
    cmap = LinearSegmentedColormap.from_list(name="mycmap", colors=colors)
    if cmap1 is None:
        cmap1 = cmap
    if cmap2 is None:
        cmap2 = cmap

    if (fig is None) and (axes is None):
        fig, axes = plt.subplots(
            2, 1, figsize=(5, 5), sharex=True, 
            height_ratios=[3, 1], gridspec_kw={'hspace': 0.01}, 
            constrained_layout=True
            )

    ax = axes[0]
    ax.set_xlim(z_range)
    ax.set_ylim(z_range)
    ax.set_ylabel(r'$z_{\rm{phot}}$', fontsize=15, usetex=True)

    N, xedges, yedges = np.histogram2d(z_spec, z_phot, bins=bins1, density=False)
    x_bin_centers = 0.5 * (xedges[:-1] + xedges[1:])
    y_bin_centers = 0.5 * (yedges[:-1] + yedges[1:])
    X, Y = np.meshgrid(x_bin_centers, y_bin_centers)
    N = N.T
    pcm = ax.pcolormesh(X, Y, N, cmap=cmap1, norm=color_norm)
    cbar = fig.colorbar(pcm, ax=ax, orientation='vertical')
    cbar.ax.yaxis.set_label_text('counts/pixel', fontsize=12, usetex=True)


    ax = axes[1]
    ymax = np.quantile(np.abs(norm_dz), 0.99)
    ax.set_ylim(-ymax, ymax)
    ax.set_xlabel(r'$z_{\rm{spec}}$', fontsize=15, usetex=True)
    ax.set_ylabel(r'$\delta z_{\rm{norm}}$', fontsize=15, usetex=True)

    N, xedges, yedges = np.histogram2d(z_spec, norm_dz, bins=bins2, density=False)
    x_bin_centers = 0.5 * (xedges[:-1] + xedges[1:])
    y_bin_centers = 0.5 * (yedges[:-1] + yedges[1:])
    X, Y = np.meshgrid(x_bin_centers, y_bin_centers)
    N = N.T
    pcm = ax.pcolormesh(X, Y, N, cmap=cmap2, norm=color_norm)
    cbar = fig.colorbar(pcm, ax=ax, orientation='vertical')
    cbar.ax.yaxis.set_label_text('counts/pixel', fontsize=12, usetex=True)

    # 刻度线设置
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        ax.minorticks_on()
        ax.tick_params(axis='x', which='major', top=True, width=1.5, length=5, direction='in')
        ax.tick_params(axis='x', which='minor', top=True, width=1, length=3, direction='in')
        ax.tick_params(axis='y', which='major', right=True, width=1.5, length=5, direction='in')
        ax.tick_params(axis='y', which='minor', right=True, width=1, length=3, direction='in')

    # 辅助线
    ax = axes[0]
    ax.plot(z_range, z_range, ls='--', lw=1.5, c='gray')
    def lines_y_list(outlier_threshold, x_range):
        """
        line: z_p = z_s +- dz(1+z_s)
        """
        y_range = [x + outlier_threshold*(1+x) for x in x_range]
        return y_range
    ax.plot(z_range, lines_y_list(outlier_threshold, z_range), ls='--', c='black', lw=1)
    ax.plot(z_range, lines_y_list(-outlier_threshold, z_range), ls='--', c='black', lw=1)

    ax = axes[1]
    ax.axhline(0, ls='--', lw=1.5, c='gray')
    ax.axhline(outlier_threshold, ls='--', lw=1, c='black')
    ax.axhline(-outlier_threshold, ls='--', lw=1, c='black')

    # info text
    ax = axes[0]
    text = (
        f"N = {len(dz):,}\n"
        r"$\mu$"+f" = {bias:.6f}\n"
        r"$R^2$" + f" = {R2:.4f}\n"
        f"RMSE = {RMSE:.4f}\n"
        r"$\sigma_{\rm{NMAD}}$"+f" = {NMAD:.4f}\n"
        r"$f_{\rm{c}}$"+f" = {outlier_fraction*100:.2f}" + r"\%"
        )
    ax.text(0.05, 0.95, text, ha='left', va='top', 
            transform=ax.transAxes, fontsize=12, usetex=True)
    ax.set_title(title, fontsize=15)

    if return_fig:
        return fig