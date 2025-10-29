"""
Tools for working with the Legacy Survey (LS) datasets.

@Author: Rui Zhu
@Date: 2025-10-10
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import rcParams
rcParams['font.family'] = 'Times New Roman'
from tqdm import tqdm
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from loguru import logger
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u

from astrokit import DIR_DATA, DIR_datasets
from astrokit.toolbox import cal_min_dist
from astrokit.toolbox import sec_to_hms
from astrokit.wrapper import stilts

__all__ = ['LegacySurvey', '_find_bricks_static']

def _find_bricks_static(i, ra_x, dec_x, search_radius, bricksinfo):
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

class LegacySurvey:
    """
    A class to handle Legacy Survey datasets.
    """
    def __init__(self):
        self.dir_data = DIR_DATA / 'legacysurvey'
        if not self.dir_data.exists():
            raise FileNotFoundError(f"Data directory {self.dir_data} does not exist.")
        self.bricksinfo = self._load_bricksinfo()

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

    def _load_bricksinfo(self):
        """
        Create a combined bricksinfo DataFrame from DR9 and DR10 datasets.
        """
        fname = 'legacysurvey_bricksinfo.parquet'
        dir_save = DIR_datasets / 'legacysurvey'
        dir_save.mkdir(parents=True, exist_ok=True)
        path = dir_save / fname
        if path.exists():
            bricksinfo = pd.read_parquet(path)
        else:
            logger.info(f"Making bricksinfo to {path}")
            bricksinfo_dr9 = Table.read(
                DIR_DATA / 'legacysurvey' / 'dr9_north' / 'survey-bricks-dr9-north.fits.gz',
                character_as_bytes=False
            )
            bricksinfo_dr10 = Table.read(
                DIR_DATA / 'legacysurvey' / 'dr10_south' / 'survey-bricks-dr10-south.fits.gz',
                character_as_bytes=False
            )
            cols = ['brickname', 'ra', 'dec', 'ra1', 'ra2', 'dec1', 'dec2']
            df_dr9 = bricksinfo_dr9[cols].to_pandas()
            df_dr10 = bricksinfo_dr10[cols].to_pandas()
            df_dr9.insert(0, 'release', 'dr9')
            df_dr10.insert(0, 'release', 'dr10')
            bricksinfo = pd.concat([df_dr9, df_dr10], ignore_index=True)
            bricksinfo.to_parquet(path, index=False)
        return bricksinfo
    
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
        res = _find_bricks_static(
            i=0, ra_x=[ra], dec_x=[dec],
            search_radius=search_radius,
            bricksinfo=self.bricksinfo
        )
        if (len(res) == 0) and (not silent):
            logger.warning(f"No bricks found within {search_radius} arcsec of (RA, Dec)=({ra}, {dec})")
        else:
            # check if tractor file exists
            for idx, row in res.iterrows():
                path = self.find_tractor_file(row['release'], row['brickname'])
                res.loc[idx, 'file_is_ready'] = path.exists()
            if show:
                fig, ax = plt.subplots(1,1, figsize=(6,6))
                ax.set_aspect('equal')
                p = SkyCoord(ra=ra*u.degree, dec=dec*u.degree)
                circle = p.directional_offset_by(
                    position_angle=np.linspace(0, 360, 200)*u.degree,
                    separation=search_radius*u.arcsec
                )
                ax.plot(circle.ra.degree, circle.dec.degree, 'r-', lw=1)
                ax.plot(ra, dec, 'rx')

                colors = plt.cm.berlin(np.linspace(0, 1, len(res)))
                for idx, (i, brick) in enumerate(res.iterrows()):
                    rect_x = [brick['ra1'], brick['ra2'], brick['ra2'], brick['ra1'], brick['ra1']]
                    rect_y = [brick['dec1'], brick['dec1'], brick['dec2'], brick['dec2'], brick['dec1']]
                    # 在每个矩形的左上角标注砖块ID
                    ax.text(brick['ra1'], brick['dec2'], brick['brickname'], color='k', fontsize=15, ha='left', va='top', )
                    ax.fill(rect_x, rect_y, color=colors[idx], alpha=0.3)

                ax.set_xlabel('RA [deg]', fontsize=15)
                ax.set_ylabel('Dec [deg]', fontsize=15)
                ax.set_title(f"Target: RA={ra:.4f}, Dec={dec:.4f}, search_radius={search_radius}'' ", fontsize=12)
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
        1. input source catalog in FITS format: input_catalog.fits
        2. Brickinfo for each input source: <main_output>_bricksinfo.pkl
        3. Combined LS catalogs for relevant bricks: <main_output>_ALL_LS_DR9.fits and <main_output>_ALL_LS_DR10.fits
        4. Final matched catalogs for all input sources: <main_output>_LS_DR9.fits and <main_output>_LS_DR10.fits
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