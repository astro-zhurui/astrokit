"""
Tools for working with the Legacy Survey (LS10) datasets.

@Author: Rui Zhu
@Date: 2025-10-10
"""
import numpy as np
from pathlib import Path
from typing import Sequence, Union
ArrayLike = Union[np.ndarray, Sequence[int]]
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib import rcParams
rcParams['font.family'] = 'Times New Roman'
from tqdm import tqdm
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import requests
from loguru import logger
import fitsio
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.spatial import cKDTree

from astrokit.toolbox import cal_min_dist
from astrokit.toolbox import sec_to_hms
from astrokit.wrapper import stilts

__all__ = ['LegacySurvey']

def _find_bricks_static(i, ra_x, dec_x, search_radius, bricksinfo):
    search_radius = _search_radius_to_arcsec(search_radius)
    df = bricksinfo.copy()
    ra, dec = ra_x[i], dec_x[i]

    p = SkyCoord(ra=ra*u.degree, dec=dec*u.degree)
    p1 = SkyCoord(ra=df['ra1'].values*u.degree, dec=df['dec1'].values*u.degree)
    p2 = SkyCoord(ra=df['ra2'].values*u.degree, dec=df['dec1'].values*u.degree)
    p3 = SkyCoord(ra=df['ra2'].values*u.degree, dec=df['dec2'].values*u.degree)
    p4 = SkyCoord(ra=df['ra1'].values*u.degree, dec=df['dec2'].values*u.degree)

    dist1 = cal_min_dist(p, p1, p2)
    dist2 = cal_min_dist(p, p2, p3)
    dist3 = cal_min_dist(p, p3, p4)
    dist4 = cal_min_dist(p, p4, p1)

    df['min_dist'] = np.minimum.reduce([dist1, dist2, dist3, dist4])
    df['in_brick'] = (ra >= df['ra1']) & (ra <= df['ra2']) & (dec >= df['dec1']) & (dec <= df['dec2'])

    res = df[(df['in_brick']) | (df['min_dist'] < search_radius)].copy()
    return res

def _search_radius_to_arcsec(search_radius):
    """
    Convert a search radius to arcsec.

    Numeric values keep the historical LegacySurvey convention: arcsec.
    Astropy quantities and strings such as "5 arcmin" are also accepted.
    """
    if isinstance(search_radius, u.Quantity):
        return float(search_radius.to_value(u.arcsec))
    if isinstance(search_radius, str):
        return float(u.Quantity(search_radius).to_value(u.arcsec))
    return float(search_radius)

def _radec_to_unit_vector(ra, dec):
    ra_rad = np.deg2rad(np.asarray(ra, dtype=float) % 360.0)
    dec_rad = np.deg2rad(np.asarray(dec, dtype=float))
    cos_dec = np.cos(dec_rad)
    return np.column_stack((
        cos_dec * np.cos(ra_rad),
        cos_dec * np.sin(ra_rad),
        np.sin(dec_rad),
    ))

def _angular_delta_deg(angle, origin):
    return (np.asarray(angle, dtype=float) - origin + 180.0) % 360.0 - 180.0

def _prepare_brick_geometry(bricksinfo):
    ra_center = bricksinfo['ra'].values.astype(float) % 360.0
    dec_center = bricksinfo['dec'].values.astype(float)
    ra1 = bricksinfo['ra1'].values.astype(float)
    ra2 = bricksinfo['ra2'].values.astype(float)
    dec1 = bricksinfo['dec1'].values.astype(float)
    dec2 = bricksinfo['dec2'].values.astype(float)

    half_width = np.maximum(
        np.abs(_angular_delta_deg(ra1, ra_center)),
        np.abs(_angular_delta_deg(ra2, ra_center)),
    )
    half_height = np.maximum(np.abs(dec1 - dec_center), np.abs(dec2 - dec_center))
    return {
        'ra_center': ra_center,
        'dec_center': dec_center,
        'half_width': half_width,
        'half_height': half_height,
    }

def _max_brick_half_diagonal_arcsec_fast(brick_geometry):
    dec_abs_min = np.maximum(np.abs(brick_geometry['dec_center']) - brick_geometry['half_height'], 0.0)
    dra = brick_geometry['half_width'] * np.cos(np.deg2rad(dec_abs_min))
    ddec = brick_geometry['half_height']
    return np.sqrt(dra*dra + ddec*ddec).max() * 3600.0 + 1.0

def _filter_candidate_bricks_fast(ra, dec, candidate_idx, search_radius, brick_geometry):
    if len(candidate_idx) == 0:
        return np.array([], dtype=int)

    idx = np.asarray(candidate_idx, dtype=int)
    ra_delta = _angular_delta_deg(ra, brick_geometry['ra_center'][idx])
    dec_delta = dec - brick_geometry['dec_center'][idx]

    outside_ra = np.maximum(np.abs(ra_delta) - brick_geometry['half_width'][idx], 0.0)
    outside_dec = np.maximum(np.abs(dec_delta) - brick_geometry['half_height'][idx], 0.0)
    cos_dec = np.cos(np.deg2rad(dec))
    dist_arcsec = np.sqrt((outside_ra * cos_dec)**2 + outside_dec**2) * 3600.0

    return idx[dist_arcsec <= search_radius]

def _filter_candidate_bricks_chunk(args):
    ra, dec, candidate_lists, search_radius, brick_geometry = args
    return [
        _filter_candidate_bricks_fast(ra_i, dec_i, candidate_idx, search_radius, brick_geometry)
        for ra_i, dec_i, candidate_idx in zip(ra, dec, candidate_lists)
    ]

def _unwrap_ra_for_plot(ra_values, ra_center):
    return ra_center + _angular_delta_deg(ra_values, ra_center)

def _auto_figure_size(x_span, y_span, min_size=5.0, max_size=9.0):
    if not np.isfinite(x_span) or not np.isfinite(y_span) or x_span <= 0 or y_span <= 0:
        return (6.0, 6.0)
    ratio = np.clip(x_span / y_span, 0.6, 1.8)
    base = 6.0 if max(x_span, y_span) < 1.0 else 7.0
    if ratio >= 1:
        return (min(max_size, base*ratio), max(min_size, base))
    return (max(min_size, base), min(max_size, base/ratio))

def _plot_bricks(ax, bricks, ra_center):
    release_style = {
        'dr9': {
            'edgecolor': '0.15',
            'facecolor': '0.65',
            'hatch': '///',
            'linestyle': '-',
            'label': 'DR9',
        },
        'dr10': {
            'edgecolor': '0.15',
            'facecolor': '0.85',
            'hatch': '\\\\\\',
            'linestyle': '--',
            'label': 'DR10',
        },
    }
    default_style = {
        'edgecolor': '0.25',
        'facecolor': '0.25',
        'hatch': '',
        'linestyle': '-',
        'label': 'Other',
    }

    grouped = bricks.groupby(['brickname', 'ra1', 'ra2', 'dec1', 'dec2'], sort=False)
    text_items = []
    for _, group in grouped:
        brick = group.iloc[0]
        x = _unwrap_ra_for_plot(
            np.array([brick['ra1'], brick['ra2'], brick['ra2'], brick['ra1'], brick['ra1']]),
            ra_center
        )
        y = np.array([brick['dec1'], brick['dec1'], brick['dec2'], brick['dec2'], brick['dec1']])
        releases = group['release'].astype(str).tolist()
        for offset, (_, row) in enumerate(group.iterrows()):
            style = release_style.get(str(row['release']), default_style)
            ax.fill(
                x, y,
                facecolor=style['facecolor'],
                edgecolor=style['edgecolor'],
                alpha=0.18,
                linewidth=1.2,
                linestyle=style['linestyle'],
                hatch=style['hatch'],
                zorder=2 + offset,
            )
        label = f"{brick['brickname']} ({'/'.join(releases)})" if len(releases) > 1 else brick['brickname']
        text_items.append((x.min(), y.max(), label))

    for i, (x, y, label) in enumerate(text_items):
        ax.text(
            x,
            y,
            label,
            color='0.1',
            fontsize=8.5,
            ha='left',
            va='top',
            zorder=12,
        )

    handles = [
        Patch(
            facecolor=style['facecolor'],
            edgecolor=style['edgecolor'],
            hatch=style['hatch'],
            alpha=0.18,
            linewidth=1.2,
            linestyle=style['linestyle'],
            label=style['label'],
        )
        for release, style in release_style.items()
        if release in set(bricks['release'].astype(str))
    ]
    return handles

class LegacySurvey:
    """
    A class to handle Legacy Survey datasets.
    """
    def __init__(self, dir_legacysurvey):
        """
        Parameters:
        -----------
        dir_legacysurvey : pathlib.Path
            Directory containing the Legacy Survey data, organized as follows:
            - dr9_north/survey-bricks-dr9-north.fits.gz
            - dr10_south/survey-bricks-dr10-south.fits.gz
            - dr9_north/tractor/000/tractor-0001p000.fits
            - dr10_south/tractor/000/tractor-0001p000.fits
        """
        self.dir_data = Path(dir_legacysurvey)
        self.bricksinfo = self._load_bricksinfo()

    def _load_bricksinfo(self):
        """
        Create a combined bricksinfo DataFrame from DR9 and DR10 datasets.
        """
        path = self.dir_data / 'legacysurvey_bricksinfo.parquet'
        if path.exists():
            bricksinfo = pd.read_parquet(path)
        else:
            logger.info(f"Making bricksinfo to {path}")
            bricksinfo_dr9 = Table.read(
                self.dir_data / 'dr9_north' / 'survey-bricks-dr9-north.fits.gz',
                character_as_bytes=False
            )
            bricksinfo_dr10 = Table.read(
                self.dir_data / 'dr10_south' / 'survey-bricks-dr10-south.fits.gz',
                character_as_bytes=False
            )
            cols = ['brickname', 'ra', 'dec', 'ra1', 'ra2', 'dec1', 'dec2']
            df_dr9 = bricksinfo_dr9[cols].to_pandas()
            df_dr10 = bricksinfo_dr10[cols].to_pandas()
            df_dr9.insert(0, 'release', 'dr9')
            df_dr10.insert(0, 'release', 'dr10')
            bricksinfo = pd.concat([df_dr9, df_dr10], ignore_index=True)
            bricksinfo.insert(1, 'AAA', bricksinfo['brickname'].str[:3])
            bricksinfo.to_parquet(path, index=False)
        return bricksinfo

    def find_tractor_file(self, release, brickname, silent=False):
        """
        Given a release and brickname, return the path to the corresponding tractor file.
        """
        dir_tractor = {
            'dr9': self.dir_data / 'dr9_north' / 'tractor',
            'dr10': self.dir_data / 'dr10_south' / 'tractor'
        }
        path = dir_tractor[release] / brickname[:3] / f'tractor-{brickname}.fits'
        if path.exists():
            return path
        else:
            if not silent:
                logger.error(f"Tractor file for brick {brickname} in release {release} not found.")
            return None

    def load_tractor_catalog(self, brickname, release, columns=None) -> pd.DataFrame:
        """
        Read the tractor catalog for a given brickname and release from DIR_DATA.

        NOTE:
        -----
        The multi-parameter columns are split into separate columns with suffixes _1, _2, etc.
        !!! 1-based indexing is used for the suffixes.
        
        Parameters:
        -----------
        brickname : str
            The name of the brick (e.g., '0001p000').
        release : str
            The data release ('dr9' or 'dr10').
        columns : list, optional
            List of columns to read from the tractor catalog. If None, all columns are read.
        """
        path = self.find_tractor_file(release=release, brickname=brickname, silent=False)
        arr = fitsio.read(path, ext=1, columns=columns)
        arr = arr.astype(arr.dtype.newbyteorder("="), copy=False)

        data = {}
        for name in arr.dtype.names:
            if arr[name].ndim == 1:
                data[name] = arr[name]
            else:
                n_params = arr[name].shape[1]
                for i in range(n_params):
                    data[f'{name}_{i+1}'] = arr[name][:, i]
        return pd.DataFrame(data)


    def find_brickname(self, ra, dec):
        """
        Given RA and Dec, return a list of (release, brickname) tuples for bricks containing the coordinates.
        """
        df = self.bricksinfo
        m = (df["ra1"].values <= ra) & (ra <= df["ra2"].values) & \
            (df["dec1"].values <= dec) & (dec <= df["dec2"].values)
        return df[m].set_index("release")["brickname"].to_dict()
    
    def find_bricks(self, ra, dec, search_radius=60, show=False, silent=False):
        """
        Find bricks that contain or are within a certain distance from the given RA and Dec.

        Parameters:
        -----------
        ra : float
            Right Ascension in degrees.
        dec : float
            Declination in degrees.
        search_radius : float
            Search radius in arcseconds.
        show : bool
            Whether to plot the search area and bricks.

        Returns:
        --------
        pd.DataFrame
            DataFrame of bricks that contain or are within the search radius from the given coordinates.
        """
        radius_arcsec = _search_radius_to_arcsec(search_radius)
        res = _find_bricks_static(
            i=0, ra_x=[ra], dec_x=[dec],
            search_radius=radius_arcsec,
            bricksinfo=self.bricksinfo
        )
        if (len(res) == 0) and (not silent):
            logger.warning(f"No bricks found within {radius_arcsec} arcsec of (RA, Dec)=({ra}, {dec})")
        else:
            # check if tractor file exists
            for idx, row in res.iterrows():
                path = self.find_tractor_file(row['release'], row['brickname'], silent=True)
                res.loc[idx, 'file_is_ready'] = (path is not None) and path.exists()
            if show:
                with plt.rc_context({
                    'font.family': 'serif',
                    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
                    'mathtext.fontset': 'stix',
                    'axes.linewidth': 1.0,
                    'xtick.direction': 'in',
                    'ytick.direction': 'in',
                    'xtick.top': True,
                    'ytick.right': True,
                    'xtick.major.size': 5,
                    'ytick.major.size': 5,
                    'xtick.minor.size': 3,
                    'ytick.minor.size': 3,
                }):
                    p = SkyCoord(ra=ra*u.degree, dec=dec*u.degree)
                    circle = p.directional_offset_by(
                        position_angle=np.linspace(0, 360, 200)*u.degree,
                        separation=radius_arcsec*u.arcsec
                    )
                    circle_ra = _unwrap_ra_for_plot(circle.ra.degree, ra)
                    circle_dec = circle.dec.degree
                    target_ra = _unwrap_ra_for_plot(ra, ra)

                    brick_ra_values = np.r_[res['ra1'].values, res['ra2'].values]
                    brick_dec_values = np.r_[res['dec1'].values, res['dec2'].values]
                    brick_ra_plot = _unwrap_ra_for_plot(brick_ra_values, ra)
                    x_values = np.r_[circle_ra, target_ra, brick_ra_plot]
                    y_values = np.r_[circle_dec, dec, brick_dec_values]
                    x_span = np.nanmax(x_values) - np.nanmin(x_values)
                    y_span = np.nanmax(y_values) - np.nanmin(y_values)
                    pad = max(radius_arcsec / 3600.0 * 0.35, x_span, y_span) * 0.10
                    pad = max(pad, radius_arcsec / 3600.0 * 0.20, 0.015)

                    fig, ax = plt.subplots(1, 1, figsize=_auto_figure_size(x_span + 2*pad, y_span + 2*pad))
                    ax.set_aspect('equal', adjustable='box')
                    brick_handles = _plot_bricks(ax, res, ra)

                    ax.plot(circle_ra, circle_dec, color='0.05', lw=1.4, zorder=20)
                    ax.plot(target_ra, dec, marker='x', color='0.05', ms=8, mew=1.8, linestyle='None', zorder=25)
                    target_handles = [
                        Line2D([], [], color='0.05', lw=1.4, label='Search radius'),
                        Line2D([], [], marker='x', color='0.05', linestyle='None', ms=7, mew=1.8, label='Target'),
                    ]
                    fig.subplots_adjust(right=0.78)
                    ax.legend(
                        handles=brick_handles + target_handles,
                        loc='center left',
                        bbox_to_anchor=(1.02, 0.5),
                        frameon=False,
                        fontsize=9,
                        borderaxespad=0.0,
                        handlelength=2.2,
                    )

                    ax.set_xlim(np.nanmin(x_values) - pad, np.nanmax(x_values) + pad)
                    ax.set_ylim(np.nanmin(y_values) - pad, np.nanmax(y_values) + pad)
                    ax.minorticks_on()
                    ax.grid(alpha=0.12, lw=0.5)

                    ax.set_xlabel('RA [deg]', fontsize=13)
                    ax.set_ylabel('Dec [deg]', fontsize=13)
                    ax.set_title(
                        f"RA={ra:.4f}, Dec={dec:.4f}, "
                        f"$r={radius_arcsec:.0f}$ arcsec",
                        fontsize=12
                    )
                    plt.show()
        return res

    def find_bricks_from_list(
        self,
        ra,
        dec,
        search_radius=60,
        max_workers=1,
        chunksize=10000,
        return_per_source=False,
        check_files=False,
        show_progress=True,
    ):
        """
        Find all Legacy Survey bricks overlapping circles around a list of sky positions.

        The returned table has the same columns as ``self.bricksinfo`` and is
        de-duplicated, so each overlapping brick appears only once.
        Candidate bricks are selected on the sphere with a KD-tree, then filtered
        with a fast local-plane circle-rectangle overlap approximation.

        Parameters
        ----------
        ra, dec : array-like
            Source coordinates in degrees.
        search_radius : float, str, or astropy.units.Quantity
            Search radius. Numeric values are interpreted as arcsec for
            consistency with ``find_bricks``. Strings such as ``"5 arcmin"``
            and astropy quantities are also accepted.
        max_workers : int
            Number of parallel workers used for exact candidate filtering.
            ``1`` disables parallel execution.
        chunksize : int
            Number of sources per parallel task.
        return_per_source : bool
            If True, also return a list of brick row indices for each input
            source. Indices are positional row indices in ``self.bricksinfo``.
        check_files : bool
            If True, add a ``file_is_ready`` column indicating whether the
            local tractor file exists.
        show_progress : bool
            If True, show a progress bar for the exact filtering step.

        Returns
        -------
        pd.DataFrame or tuple
            ``bricks`` if ``return_per_source=False``; otherwise
            ``(bricks, per_source_brick_indices)``.
        """
        radius_arcsec = _search_radius_to_arcsec(search_radius)
        if radius_arcsec < 0:
            raise ValueError("search_radius must be non-negative.")

        ra = np.asarray(ra, dtype=float)
        dec = np.asarray(dec, dtype=float)
        if ra.ndim != 1 or dec.ndim != 1:
            raise ValueError("ra and dec must be 1D arrays.")
        if ra.size != dec.size:
            raise ValueError("ra and dec must have the same length.")

        n_sources = ra.size
        per_source = [np.array([], dtype=int) for _ in range(n_sources)]
        valid = np.isfinite(ra) & np.isfinite(dec)
        if not np.any(valid):
            res = self.bricksinfo.iloc[[]].copy()
            return (res, per_source) if return_per_source else res

        bricksinfo = self.bricksinfo.reset_index(drop=True)
        brick_geometry = _prepare_brick_geometry(bricksinfo)
        brick_vectors = _radec_to_unit_vector(bricksinfo['ra'].values, bricksinfo['dec'].values)
        tree = cKDTree(brick_vectors)

        max_half_diagonal = _max_brick_half_diagonal_arcsec_fast(brick_geometry)
        query_radius_rad = np.deg2rad((radius_arcsec + max_half_diagonal) / 3600.0)
        query_radius_chord = 2.0 * np.sin(query_radius_rad / 2.0)

        valid_idx = np.flatnonzero(valid)
        ra_valid = ra[valid]
        dec_valid = dec[valid]
        source_vectors = _radec_to_unit_vector(ra_valid, dec_valid)
        candidate_lists = tree.query_ball_point(source_vectors, query_radius_chord)

        chunks = []
        for start in range(0, len(ra_valid), chunksize):
            end = min(start + chunksize, len(ra_valid))
            chunks.append((
                ra_valid[start:end],
                dec_valid[start:end],
                candidate_lists[start:end],
                radius_arcsec,
                brick_geometry,
            ))

        if max_workers is None or max_workers <= 1 or len(chunks) == 1:
            iterator = chunks
            if show_progress:
                iterator = tqdm(iterator, total=len(chunks), desc="Filtering bricks")
            chunk_results = [_filter_candidate_bricks_chunk(chunk) for chunk in iterator]
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                iterator = executor.map(_filter_candidate_bricks_chunk, chunks)
                if show_progress:
                    iterator = tqdm(iterator, total=len(chunks), desc="Filtering bricks")
                chunk_results = list(iterator)

        filtered = [idx for chunk in chunk_results for idx in chunk]
        for source_idx, brick_idx in zip(valid_idx, filtered):
            per_source[source_idx] = brick_idx

        nonempty = [idx for idx in filtered if len(idx) > 0]
        if nonempty:
            unique_idx = np.unique(np.concatenate(nonempty))
            res = bricksinfo.iloc[unique_idx].copy().reset_index(drop=True)
        else:
            res = bricksinfo.iloc[[]].copy()

        if check_files and len(res) > 0:
            res['file_is_ready'] = [
                (path is not None) and path.exists()
                for path in (
                    self.find_tractor_file(row['release'], row['brickname'], silent=True)
                    for _, row in res.iterrows()
                )
            ]

        if return_per_source:
            return res, per_source
        return res
    
    def query_catalogs_around_source(
        self, input_cat, output_dir, task_name, 
        search_radius, col_ra, col_dec, max_workers=32
        ):

        """
        Retrieve all source catalogs within search_radius around the input source 
        coordinates from the Legacy Survey database.

        NOTE: Both DR9 and DR10 are queried, respectively.

        Parameters
        ----------
        input_cat : pd.DataFrame
            Input source catalog containing RA and Dec columns.
        output_dir : pathlib.Path
            Directory to save output files.
        task_name : str
            Main task name for output files.
        search_radius : float
            Search radius in arcseconds.
        col_ra : str
            Column name for Right Ascension in input_cat.
        col_dec : str
            Column name for Declination in input_cat.
        max_workers : int
            Maximum number of parallel workers for querying bricks info.

        Outputs
        -------
        1. input source catalog in FITS format: <task_name>_input_catalog.fits
        2. Brickinfo for each input source: <task_name>_bricksinfo.pkl
        3. Combined LS catalogs for relevant bricks: <task_name>_dr9_all.fits and <task_name>_dr10_all.fits
        4. Final matched catalogs for all input sources: <task_name>_dr9.fits and <task_name>_dr10.fits
        """
        st_all = time.time()
        output_dir.mkdir(parents=True, exist_ok=True)

        fname_bricksinfo = f"{task_name}_bricksinfo.pkl"
        fname_input_catalog = f"{task_name}_input_catalog.fits"
        path_output_bricksinfo = output_dir / fname_bricksinfo

        # step1: read input source coordinates
        if not isinstance(input_cat, pd.DataFrame):
            raise ValueError("input_cat must be a pandas DataFrame.")
        else:
            df_x = input_cat.copy()
            Table.from_pandas(df_x).write(output_dir / fname_input_catalog, overwrite=True)
        N_x = len(df_x)
        ra_x, dec_x = df_x[col_ra].values, df_x[col_dec].values

        # step2: collect bricks info for all sources
        if path_output_bricksinfo.exists():
            logger.info(f"Bricks info file {path_output_bricksinfo.name} exists. Loading ...")
            df_x = pd.read_pickle(path_output_bricksinfo)
            logger.info("Bricks info loaded.")
        else:
            logger.info(f"Collecting bricks info for {N_x} sources ...")
            func = partial(_find_bricks_static, ra_x=ra_x, dec_x=dec_x, search_radius=search_radius, bricksinfo=self.bricksinfo)
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                bricks = list(tqdm(executor.map(func, range(N_x)), total=N_x))
            df_x['n_bricks'] = [len(b) for b in bricks]
            df_x['bricks'] = [df for df in bricks]
            df_x.to_pickle(path_output_bricksinfo)
            logger.info(f"Bricks info saved to {path_output_bricksinfo.name}")

        # step3: combine DR9 and DR10 catalogs paths for all relevant bricks
        bricks = pd.concat(df_x['bricks'].to_list(), ignore_index=True)
        brick_names = {
            "dr9": set(bricks[bricks['release']=='dr9']['brickname']),
            "dr10": set(bricks[bricks['release']=='dr10']['brickname']),
        }
        for release in ['dr9', 'dr10']:
            N_bricks = len(brick_names[release])
            if N_bricks == 0:
                logger.warning(f"No bricks found for LS {release.upper()}. Skipping ...")
                continue
            logger.info(f"Combining LS {release.upper()} catalogs for {N_bricks} bricks ...")
            st = time.time()
            path_tractors = []
            for brickname in brick_names[release]:
                path_file = self.find_tractor_file(brickname=brickname, release=release, silent=True)
                if path_file is None:
                    logger.warning(f"Tractor file for brick {brickname} (LS {release.upper()}) not found. Skipping ...")
                    continue
                path_tractors.append(path_file)
            path = output_dir / f"{task_name}_{release}_all.fits"
            stilts.tcat(
                path_in=path_tractors, path_out=path,
                stilts_flags='', uloccol='from', silent=True
            )
            if path.exists():
                logger.info(f"Combined LS {release.upper()} catalog saved to {path.name}")
                logger.info(f"Combining completed in {sec_to_hms(time.time() - st)}.")
            else:
                raise FileNotFoundError(f"Failed to combine LS {release.upper()} catalogs!")

        # step4: cross-match input sources with combined LS catalogs
        for release in ['dr9', 'dr10']:
            N_bricks = len(brick_names[release])
            if N_bricks != 0:
                path_ls_all = output_dir / f"{task_name}_{release}_all.fits"
                path_out = output_dir / f"{task_name}_{release}.fits"
                logger.info(f"Cross-matching {fname_input_catalog} with {path_ls_all.name} ...")
                st = time.time()
                stilts.tskymatch2(
                    path_cat_left=output_dir / fname_input_catalog,
                    path_cat_right=path_ls_all,
                    path_cat_output=path_out,
                    coord_name_left=(col_ra, col_dec),
                    coord_name_right=('ra', 'dec'), 
                    sep=search_radius, 
                    find='all', 
                    join='1and2', 
                    silent=True, 
                )
                if path_out.exists():
                    logger.info(f"Matched catalog saved to {path_out.name}")
                else:
                    raise FileNotFoundError(f"Failed to save matched catalog for LS {release.upper()}!")
                logger.info(f"Cross-matching completed in {sec_to_hms(time.time() - st)}.")
        logger.success(f"All Done in {sec_to_hms(time.time() - st_all)}.")

    def make_ls_id(self, release: ArrayLike, brickid: ArrayLike, objid: ArrayLike) -> np.ndarray:
        """
        Make the Unique Legacy Survey Identifier for given release, brickid, and objid.
        The format is: "{release}_{brickid}_{objid}"
        """

        r = np.asarray(release, dtype=np.int64)
        b = np.asarray(brickid, dtype=np.int64)
        o = np.asarray(objid,   dtype=np.int64)

        if r.ndim != 1 or b.ndim != 1 or o.ndim != 1:
            raise ValueError("release/brickid/objid must be 1D arrays.")
        if not (r.size == b.size == o.size):
            raise ValueError("release/brickid/objid must have the same length.")

        r = r.astype(str)
        b = b.astype(str)
        o = o.astype(str)

        out = np.char.add(r, "_")
        out = np.char.add(out, b)
        out = np.char.add(out, "_")
        out = np.char.add(out, o)
        return out
    

    def download_image(self, release, brickname, band, dir_output, silent=False):
        """
        release: str, 'dr9' or 'dr10'
        band: g, r, i, z, W1, W2, W3, W4
        """
        fname_fz = f"legacysurvey-{brickname}-image-{band}.fits.fz"

        if release == 'dr10':
            url_coadd = "https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr10/south/coadd"
        if release == 'dr9':
            url_coadd = "https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr9/north/coadd"
        url = f"{url_coadd}/{brickname[:3]}/{brickname}/{fname_fz}"

        fname_output = f"{brickname}-{band}.fits.fz"
        path_output = dir_output / fname_output
        if not path_output.exists():
            timeout = 60  # seconds
            try:
                with requests.get(url, stream=True, timeout=timeout) as r:
                    r.raise_for_status()
                    with open(path_output, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)
                if not silent:
                    logger.success(f"legacysurvey image: {brickname} ({band}) | DONE")
            except Exception as e:
                logger.error(f"legacysurvey image: {brickname} ({band}) | ERROR | {e}")
        else:
            logger.info(f"legacysurvey image: {brickname} ({band}) | EXISTS")
        return None
