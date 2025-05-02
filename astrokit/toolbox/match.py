import pandas as pd
import numpy as np
import time

from loguru import logger
from astropy.coordinates import SkyCoord
from astropy import units as u

__all__ = [
    'MatchCatalog'
]

class MatchCatalog:
    """A class for matching two catalogs like TOPCAT."""

    def __init__(self, cat_left, cat_right, 
                 coord_name_left=('ra', 'dec'), 
                 coord_name_right=('ra', 'dec'),
                 remove_dup=True,
                 sep=1,
                 keep_coord='left', 
                 silent=False):
        """
        Match two catalogs.

        Parameters
        ----------
        cat_left: DataFrame
            The left catalog as the primary catalog.
        cat_right: DataFrame
            The right catalog to be matched.
        coord_name_left: tuple (optional, default=('ra', 'dec'))
            The column names for the coordinates in the left catalog.
        coord_name_right: tuple (optional, default=('ra', 'dec'))
            The column names for the coordinates in the right catalog.
        remove_dup: bool (optional, default=True)
            Whether to remove the duplicated sources (n-to-1 cases) in the left catalog.
        sep: float (optional, default=1)
            The matching threshold in arcsec.
        keep_coord: str (optional, default='left')
            The catalog to keep the coordinates. 'left' or 'right' or 'both'.
        silent: bool (optional, default=False)
            Whether to print the information.
        """
        self.coord_name_left = coord_name_left
        self.coord_name_right = coord_name_right

        self.cat_left, self.cols_left = self._load_catalog(
            cat_left, coord_name_left)
        self.cat_right, self.cols_right = self._load_catalog(
            cat_right, coord_name_right)
        self.cat_left.rename(columns={'ra': 'ra_left', 'dec': 'dec_left'}, inplace=True)
        self.cat_right.rename(columns={'ra': 'ra_right', 'dec': 'dec_right'}, inplace=True)
        
        # 两表重复列名预警
        dup_cols = set(self.cols_left) & set(self.cols_right)
        if bool(dup_cols):
            raise ValueError(f"Duplicate columns {dup_cols} in two catalogs")

        self.sep = sep
        self.remove_dup = remove_dup
        self.keep_coord = keep_coord
        self.silent = silent

        self._result_all = None
        self.result = None

    def _load_catalog(self, cat, coord_name):
        df = cat.copy()
        df.reset_index(inplace=True, drop=True)
        cols = df.columns.tolist()

        # 检查coord_name是否在df的columns中
        if not all([i in cols for i in coord_name]):
            raise ValueError(f"coord_name {coord_name} not in columns {cols}")
        
        df.rename(columns={coord_name[0]: 'ra', coord_name[1]: 'dec'}, inplace=True)

        # 收集非坐标列的列名
        cols_other = [i for i in cols if i not in coord_name]
        return df, cols_other
    
    def _drop_duplicated(self, df):
        """
        Drop sources which have the same matched source in the right catalog, 
        and keep the nearest one.

        Note:
        -----
        left表格中的一些源可能本身就离的很近, 这时就会匹配到同一个right表格中的源.
        I call this situation "n-to-1".

        Return:
        -------
        去除结果中重复指向源的较远的sources, 返回去重后的结果表格

        """
        df_dup = df.loc[df['idx'].duplicated(keep=False)]
        df_dup = df_dup.sort_values('idx')

        dup_index = list(df_dup.index)
        safe_index = list(df.index.difference(dup_index))

        # left表匹配right表中的同一个source时，只保留最近的那个
        keep_index = list(df_dup.groupby('idx')['d2d'].idxmin().values)
        drop_index = list(df_dup.index.difference(keep_index))

        df = df.loc[safe_index + keep_index]
        num_count = {
            "n_safe": len(safe_index),
            "n_dup": len(dup_index), 
            "n_keep": len(keep_index),
            "n_drop": len(drop_index),
        }

        return df, num_count
    
    def run(self):

        st = time.time()

        ra_left = self.cat_left['ra_left'].values
        dec_left = self.cat_left['dec_left'].values

        ra_right = self.cat_right['ra_right'].values
        dec_right = self.cat_right['dec_right'].values

        coord_left = SkyCoord(ra=ra_left*u.degree, dec=dec_left*u.degree)
        coord_right = SkyCoord(ra=ra_right*u.degree, dec=dec_right*u.degree)

        if not self.silent:
            logger.info("Start matching two catalogs...")

        # 遍历coord_left, 找到coord_right中距离coord_left中每个源最近的源的索引idx等信息
        idx, d2d, d3d = coord_left.match_to_catalog_sky(coord_right, nthneighbor=1)

        df_left = self.cat_left.copy()
        df_left.insert(0, 'idx', idx)
        df_left.insert(0, 'd2d', d2d.to('arcsec').value)
        df_left.insert(0, 'sep_constraint', d2d < self.sep*u.arcsec)

        df_right = self.cat_right.copy()
        df_right.insert(0, 'idx', df_right.index)

        df = pd.merge(df_left, df_right, on='idx')
        del df_left, df_right

        self._result_all = df.copy()  # 保留全部信息的结果表

        # 整理最终结果
        df = df[df['sep_constraint'] == True]

        # 统计匹配结果
        info = f"+ --- Match Result ({self.sep} arcsec) --- +\n"
        info += f"Left Catalog: {len(self.cat_left)} sources\n"
        info += f"Right Catalog: {len(self.cat_right)} sources\n"
        info += f"Matched: {len(df)} sources\n"
        info += f"+ ------------------------------- +"

        if self.remove_dup:
            df, ns = self._drop_duplicated(df)
            info += "\n"
            info += f"n-to-1 cases: {ns['n_dup']}\n"
            info += f"keep souces in n-to-1 cases: {ns['n_keep']}\n"
            info += f"drop souces in n-to-1 cases: {ns['n_drop']}\n"
            info += f"+ ------------------------------- +\n"
            info += f"Final Result: {len(df)} sources\n"
        else:
            info += f"+ ------------------------------- +\n"
            info += f"Final Result: {len(df)} sources"
        if not self.silent:
            print(info)

        # 格式整理
        df.drop(columns=['sep_constraint', 'd2d', 'idx'], inplace=True)

        if self.keep_coord == 'left':
            df.drop(columns=['ra_right', 'dec_right'], inplace=True)
            df.rename(columns={'ra_left': 'ra', 'dec_left': 'dec'}, 
                      inplace=True)
            df = df[['ra', 'dec'] + self.cols_left + self.cols_right]
        elif self.keep_coord == 'right':
            df.drop(columns=['ra_left', 'dec_left'], inplace=True)
            df.rename(columns={'ra_right': 'ra', 'dec_right': 'dec'}, 
                      inplace=True)
            df = df[['ra', 'dec'] + self.cols_left + self.cols_right]
        elif self.keep_coord == 'both':
            df = df[
                ['ra_left', 'dec_left', 
                 'ra_right', 'dec_right'] + self.cols_left + self.cols_right
                 ]
        else:
            raise ValueError(f"Invalid keep_coord {self.keep_coord}. You should choose from 'left', 'right', 'both'.")
        
        df.reset_index(drop=True, inplace=True)
        self.result = df.copy()
        del df
        
        if not self.silent:
            logger.success(f"Finished in {time.time()-st:.2f} s")

        return None
