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

from loguru import logger
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u

from astrokit import DIR_DATA, DIR_datasets
from astrokit.toolbox import cal_min_dist

__all__ = ['LegacySurvey']

class LegacySurvey:
    """
    A class to handle Legacy Survey datasets.
    """
    def __init__(self):
        self.dir_data = DIR_DATA / 'legacysurvey'
        if not self.dir_data.exists():
            raise FileNotFoundError(f"Data directory {self.dir_data} does not exist.")
        self.bricksinfo = self._load_bricksinfo()

    def find_tractor_file(self, release, brickname):
        """
        Given a release and brickname, return the path to the corresponding tractor file.
        """
        dir_tractor = {
            'dr9': self.dir_data / 'dr9_north' / 'tractor',
            'dr10': self.dir_data / 'dr10_south' / 'tractor'
        }
        path = dir_tractor[release] / brickname[:3] / f'tractor-{brickname}.fits'
        if not path.exists():
            logger.warning(f"Tractor file {path} does not exist.")
        return path
        
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
        df = self.bricksinfo.copy()
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
        df['in_brick'] = False
        df.loc[(ra >= df['ra1']) & (ra <= df['ra2']) &
            (dec >= df['dec1']) & (dec <= df['dec2']), 'in_brick'] = True
        res = df[(df['in_brick'] == True) | (df['min_dist'] < search_radius)].copy()
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
                circle = Circle((ra, dec), search_radius/3600, edgecolor='r', 
                                facecolor='none', lw=1)
                ax.add_patch(circle)
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