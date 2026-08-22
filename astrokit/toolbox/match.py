import pandas as pd
import numpy as np
import time

from loguru import logger
from astropy.coordinates import SkyCoord
from astropy import units as u
from scipy.spatial import cKDTree

__all__ = [
    'fast_match', 'MatchCatalog'
]

def fast_match(
    ra_1,
    dec_1,
    ra_2,
    dec_2,
    radius_arcsec=1.0,
    *,
    id_1=None,
    id_2=None,
    mode='nearest',
    workers=-1,
):
    """
    Fast fixed-radius spherical crossmatch.

    Parameters
    ----------
    ra_1, dec_1, ra_2, dec_2 : array-like
        Coordinates in degrees.
    radius_arcsec : float
        Matching radius in arcseconds.
    id_1, id_2 : array-like or None
        Optional object identifiers. If omitted, positional indices
        ``0 .. N-1`` are used.
    mode : {'nearest', 'all', 'best'}
        ``nearest``: one nearest neighbour per source in catalogue 1.
        ``all``: every pair within the radius.
        ``best``: greedy one-to-one pairs ranked by separation.
    workers : int
        Number of workers for SciPy's KD-tree query. ``-1`` uses all cores.

    Returns
    -------
    pd.DataFrame
        Columns: ``id_1``, ``ra_1``, ``dec_1``, ``id_2``, ``ra_2``, ``dec_2``,
        ``sep`` (arcsec).
    """
    columns = ['id_1', 'ra_1', 'dec_1', 'id_2', 'ra_2', 'dec_2', 'sep']
    if mode not in {'all', 'nearest', 'best'}:
        raise ValueError("mode must be one of: 'all', 'nearest', 'best'")
    if not np.isfinite(radius_arcsec) or radius_arcsec <= 0:
        raise ValueError('radius_arcsec must be a positive finite number')

    def _as_coord(values, label):
        arr = np.asarray(values, dtype=float)
        if arr.ndim != 1:
            raise ValueError(f'{label} must be a 1D array')
        if not np.all(np.isfinite(arr)):
            raise ValueError(f'{label} must be finite')
        return arr

    def _as_id(values, n, label):
        if values is None:
            return np.arange(n, dtype=np.intp)
        arr = np.asarray(values)
        if arr.ndim != 1:
            raise ValueError(f'{label} must be a 1D array')
        if arr.shape[0] != n:
            raise ValueError(f'{label} must have length {n}')
        return arr

    def to_unit_vectors(ra, dec):
        ra_rad = np.deg2rad(ra)
        dec_rad = np.deg2rad(dec)
        cos_dec = np.cos(dec_rad)
        return np.column_stack((
            cos_dec * np.cos(ra_rad),
            cos_dec * np.sin(ra_rad),
            np.sin(dec_rad),
        ))

    ra_values_1 = _as_coord(ra_1, 'ra_1')
    dec_values_1 = _as_coord(dec_1, 'dec_1')
    ra_values_2 = _as_coord(ra_2, 'ra_2')
    dec_values_2 = _as_coord(dec_2, 'dec_2')
    if ra_values_1.shape[0] != dec_values_1.shape[0]:
        raise ValueError('ra_1 and dec_1 must have the same length')
    if ra_values_2.shape[0] != dec_values_2.shape[0]:
        raise ValueError('ra_2 and dec_2 must have the same length')
    if np.any((dec_values_1 < -90) | (dec_values_1 > 90)):
        raise ValueError('dec_1 must lie in [-90, 90] degrees')
    if np.any((dec_values_2 < -90) | (dec_values_2 > 90)):
        raise ValueError('dec_2 must lie in [-90, 90] degrees')

    n_1 = ra_values_1.shape[0]
    n_2 = ra_values_2.shape[0]
    ids_1 = _as_id(id_1, n_1, 'id_1')
    ids_2 = _as_id(id_2, n_2, 'id_2')
    if n_1 == 0 or n_2 == 0:
        return pd.DataFrame(columns=columns)

    xyz_1 = to_unit_vectors(ra_values_1, dec_values_1)
    xyz_2 = to_unit_vectors(ra_values_2, dec_values_2)
    radius_chord = 2 * np.sin(np.deg2rad(radius_arcsec / 3600) / 2)
    tree = cKDTree(xyz_2, compact_nodes=True, balanced_tree=True)
    if mode == 'nearest':
        _, idx_2 = tree.query(xyz_1, k=1, distance_upper_bound=radius_chord, workers=workers)
        idx_1 = np.flatnonzero(idx_2 < n_2)
        idx_2 = idx_2[idx_1].astype(np.intp, copy=False)
    else:
        neighbours = tree.query_ball_point(xyz_1, r=radius_chord, workers=workers)
        counts = np.fromiter((len(x) for x in neighbours), dtype=np.intp, count=len(neighbours))
        idx_1 = np.repeat(np.arange(n_1, dtype=np.intp), counts)
        if len(idx_1) == 0:
            return pd.DataFrame(columns=columns)
        idx_2 = np.concatenate(neighbours).astype(np.intp, copy=False)

    chord = np.linalg.norm(xyz_1[idx_1] - xyz_2[idx_2], axis=1)
    sep = np.rad2deg(2 * np.arcsin(np.clip(chord / 2, 0, 1))) * 3600
    if mode == 'best':
        order = np.lexsort((idx_2, idx_1, sep))
        used_1, used_2 = np.zeros(n_1, bool), np.zeros(n_2, bool)
        keep = np.zeros(len(idx_1), bool)
        for candidate in order:
            i, j = idx_1[candidate], idx_2[candidate]
            if not used_1[i] and not used_2[j]:
                keep[candidate] = True
                used_1[i] = used_2[j] = True
        idx_1, idx_2, sep = idx_1[keep], idx_2[keep], sep[keep]

    return pd.DataFrame({
        'id_1': ids_1[idx_1], 'ra_1': ra_values_1[idx_1], 'dec_1': dec_values_1[idx_1],
        'id_2': ids_2[idx_2], 'ra_2': ra_values_2[idx_2], 'dec_2': dec_values_2[idx_2],
        'sep': sep,
    }, columns=columns)

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
