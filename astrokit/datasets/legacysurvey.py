"""
Tools for working with the Legacy Survey DR11 datasets.

@Author: Rui Zhu
@Date: 2025-10-10
@Update: 2026-09-03, switch to LS DR11 north/south.
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import astropy.units as u
import fitsio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from astropy.coordinates import SkyCoord
from astropy.table import Table
from loguru import logger
from matplotlib import rcParams
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.spatial import cKDTree

from astrokit.toolbox import sec_to_hms
from astrokit.toolbox.match import fast_match

rcParams['font.family'] = 'Times New Roman'

__all__ = ['LegacySurvey']

REGIONS = ('north', 'south')
DIR_REGION = {
    'north': Path('dr11_north'),
    'south': Path('dr11_south'),
}
URL_COADD = {
    'north': 'https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr11/north/coadd',
    'south': 'https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr11/south/coadd',
}
_BRICK_STYLE = {
    'north': dict(edgecolor='0.15', facecolor='0.65', hatch='///', linestyle='-', label='North'),
    'south': dict(edgecolor='0.15', facecolor='0.85', hatch='\\\\\\', linestyle='--', label='South'),
}


def _to_arcsec(search_radius):
    if isinstance(search_radius, u.Quantity):
        return float(search_radius.to_value(u.arcsec))
    if isinstance(search_radius, str):
        return float(u.Quantity(search_radius).to_value(u.arcsec))
    return float(search_radius)


def _xyz(ra, dec):
    ra = np.deg2rad(np.asarray(ra, dtype=float) % 360.0)
    dec = np.deg2rad(np.asarray(dec, dtype=float))
    cos_dec = np.cos(dec)
    return np.column_stack((cos_dec * np.cos(ra), cos_dec * np.sin(ra), np.sin(dec)))


def _dlon(lon, origin):
    return (np.asarray(lon, dtype=float) - origin + 180.0) % 360.0 - 180.0


def _read_tractor(path, columns=None, rows=None) -> pd.DataFrame:
    kwargs = {'ext': 1}
    if columns is not None:
        available = set(fitsio.FITS(path)[1].get_colnames())
        resolved = []
        for name in dict.fromkeys(columns):
            if name in available:
                resolved.append(name)
            else:
                base, sep, suffix = name.rpartition('_')
                if sep and suffix.isdigit() and base in available:
                    resolved.append(base)
        kwargs['columns'] = resolved
    if rows is not None:
        kwargs['rows'] = np.asarray(rows, dtype=np.intp)
    arr = fitsio.read(path, **kwargs)
    arr = arr.astype(arr.dtype.newbyteorder('='), copy=False)
    data = {}
    for name in arr.dtype.names:
        col = arr[name]
        if col.ndim == 1:
            data[name] = col
        else:
            for i in range(col.shape[1]):
                data[f'{name}_{i+1}'] = col[:, i]
    return pd.DataFrame(data)


def _progress(iterable, *, total, desc, show=True, unit='brick', min_total=1):
    if (not show) or (total < min_total):
        return iterable

    def _report():
        start = last = time.monotonic()
        for count, item in enumerate(iterable, start=1):
            now = time.monotonic()
            if count == total or now - last >= 1.0:
                elapsed = now - start
                rate = count / elapsed if elapsed else 0.0
                remaining = (total - count) / rate if rate else float('inf')
                eta = '--' if not np.isfinite(remaining) else sec_to_hms(remaining)
                print(f"\r{desc}: {count:,}/{total:,} {unit}s | {rate:.2f} {unit}/s | ETA {eta:<20}",
                      end='', file=sys.stdout, flush=True)
                last = now
            yield item
        print(file=sys.stdout, flush=True)

    return _report()


class LegacySurvey:
    """Handle local Legacy Survey DR11 north/south tractor catalogs."""

    def __init__(self, dir_legacysurvey):
        """
        Expected layout::

            dir_legacysurvey/
                dr11_north/survey-bricks-dr11-north.fits.gz
                dr11_north/tractor/<AAA>/tractor-<brickname>.fits
                dr11_south/survey-bricks-dr11-south.fits.gz
                dr11_south/tractor/<AAA>/tractor-<brickname>.fits
        """
        self.dir_data = Path(dir_legacysurvey)
        self.bricksinfo = self._load_bricksinfo()
        self._set_brick_index(self.bricksinfo)

    def _load_bricksinfo(self):
        path = self.dir_data / 'legacysurvey_bricksinfo.parquet'
        cols = ['brickname', 'ra', 'dec', 'ra1', 'ra2', 'dec1', 'dec2', 'survey_primary', 'in_desi']
        if path.exists():
            df = pd.read_parquet(path)
            if 'region' in df.columns and set(df['region'].astype(str).unique()) <= set(REGIONS):
                return df
            logger.info(f'Rebuilding DR11 bricksinfo at {path}')

        logger.info(f'Making bricksinfo to {path}')
        frames = []
        for region in REGIONS:
            table = Table.read(
                self.dir_data / DIR_REGION[region] / f'survey-bricks-dr11-{region}.fits.gz',
                character_as_bytes=False,
            )
            df = table[cols].to_pandas()
            df.insert(0, 'region', region)
            frames.append(df)
        bricksinfo = pd.concat(frames, ignore_index=True)
        bricksinfo.insert(1, 'AAA', bricksinfo['brickname'].str[:3])
        bricksinfo.to_parquet(path, index=False)
        return bricksinfo

    def _set_brick_index(self, bricks):
        ra_c = bricks['ra'].to_numpy(float) % 360.0
        dec_c = bricks['dec'].to_numpy(float)
        half_w = np.maximum(
            np.abs(_dlon(bricks['ra1'].to_numpy(float), ra_c)),
            np.abs(_dlon(bricks['ra2'].to_numpy(float), ra_c)),
        )
        half_h = np.maximum(
            np.abs(bricks['dec1'].to_numpy(float) - dec_c),
            np.abs(bricks['dec2'].to_numpy(float) - dec_c),
        )
        self._ra_c, self._dec_c = ra_c, dec_c
        self._half_w, self._half_h = half_w, half_h
        self._tree = cKDTree(_xyz(ra_c, dec_c))
        dec_abs_min = np.maximum(np.abs(dec_c) - half_h, 0.0)
        dra = half_w * np.cos(np.deg2rad(dec_abs_min))
        self._max_half_diag = np.sqrt(dra * dra + half_h * half_h).max() * 3600.0 + 1.0

    @staticmethod
    def _check_region(region):
        if region not in REGIONS:
            raise ValueError(f"region must be 'north' or 'south', got {region!r}")
        return region

    def tractor_path(self, region, brickname):
        region = self._check_region(region)
        return self.dir_data / DIR_REGION[region] / 'tractor' / brickname[:3] / f'tractor-{brickname}.fits'

    def find_tractor_file(self, region, brickname, silent=False):
        path = self.tractor_path(region, brickname)
        if path.exists():
            return path
        if not silent:
            logger.error(f'Tractor file for brick {brickname} in {region} not found.')
        return None

    def load_tractor_catalog(self, brickname, region, columns=None) -> pd.DataFrame:
        """
        Read a tractor catalog. Multi-parameter FITS columns are split with
        1-based suffixes (``flux_ivar_1``, ``flux_ivar_2``, ...).
        """
        path = self.find_tractor_file(region=region, brickname=brickname)
        return _read_tractor(path, columns=columns)

    def find_brickname(self, ra, dec):
        """Return ``{'north': brickname, 'south': brickname}`` for bricks containing the coordinate."""
        df = self.bricksinfo
        m = ((df['ra1'].to_numpy() <= ra) & (ra <= df['ra2'].to_numpy())
             & (df['dec1'].to_numpy() <= dec) & (dec <= df['dec2'].to_numpy()))
        return df.loc[m].set_index('region')['brickname'].to_dict()

    def _nearby_idx(self, ra, dec, cand, radius_arcsec, ra_c=None, dec_c=None, half_w=None, half_h=None):
        if len(cand) == 0:
            return np.array([], dtype=int)
        idx = np.asarray(cand, dtype=int)
        ra_c = self._ra_c if ra_c is None else ra_c
        dec_c = self._dec_c if dec_c is None else dec_c
        half_w = self._half_w if half_w is None else half_w
        half_h = self._half_h if half_h is None else half_h
        outside_ra = np.maximum(np.abs(_dlon(ra, ra_c[idx])) - half_w[idx], 0.0)
        outside_dec = np.maximum(np.abs(dec - dec_c[idx]) - half_h[idx], 0.0)
        dist = np.sqrt((outside_ra * np.cos(np.deg2rad(dec))) ** 2 + outside_dec ** 2) * 3600.0
        return idx[dist <= radius_arcsec]

    def find_bricks_from_list(
        self, ra, dec, search_radius=60, max_workers=1,
        check_file_exists=False, show_progress=True,
    ):
        """
        Find unique bricks overlapping circles around a list of positions.

        ``search_radius`` is in arcsec if numeric; strings such as ``"5 arcmin"``
        and astropy quantities are also accepted.
        """
        radius_arcsec = _to_arcsec(search_radius)
        if radius_arcsec < 0:
            raise ValueError('search_radius must be non-negative.')

        ra = np.asarray(ra, dtype=float)
        dec = np.asarray(dec, dtype=float)
        if ra.ndim != 1 or dec.ndim != 1 or ra.size != dec.size:
            raise ValueError('ra and dec must be 1D arrays of the same length.')

        valid = np.isfinite(ra) & np.isfinite(dec)
        if not np.any(valid):
            return self.bricksinfo.iloc[[]].copy()

        ra_v, dec_v = ra[valid], dec[valid]
        chord = 2.0 * np.sin(np.deg2rad((radius_arcsec + self._max_half_diag) / 3600.0) / 2.0)
        cands = self._tree.query_ball_point(_xyz(ra_v, dec_v), chord)

        n = len(ra_v)
        workers = 1 if (max_workers is None or max_workers <= 1) else int(max_workers)
        chunksize = min(10000, n) if workers <= 1 else max(500, min(20000, (n + workers * 4 - 1) // (workers * 4)))
        chunks = [
            (ra_v[i:j], dec_v[i:j], cands[i:j])
            for i in range(0, n, chunksize)
            for j in [min(i + chunksize, n)]
        ]

        def _filter_chunk(chunk):
            ras, decs, cand_lists = chunk
            return [self._nearby_idx(r, d, c, radius_arcsec) for r, d, c in zip(ras, decs, cand_lists)]

        if workers <= 1 or len(chunks) == 1:
            results = [_filter_chunk(c) for c in _progress(
                chunks, total=len(chunks), desc='Assigning source batches',
                show=show_progress, unit='chunk', min_total=5,
            )]
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                results = list(_progress(
                    ex.map(_filter_chunk, chunks), total=len(chunks),
                    desc='Assigning source batches', show=show_progress, unit='chunk', min_total=5,
                ))

        nonempty = [idx for chunk in results for idx in chunk if len(idx) > 0]
        if nonempty:
            res = self.bricksinfo.iloc[np.unique(np.concatenate(nonempty))].copy().reset_index(drop=True)
        else:
            res = self.bricksinfo.iloc[[]].copy()

        if check_file_exists and len(res) > 0:
            res['file_is_ready'] = [
                self.tractor_path(row.region, row.brickname).exists()
                for row in res.itertuples(index=False)
            ]
        return res

    def find_bricks(self, ra, dec, search_radius=60, show=False, silent=False):
        """Find bricks overlapping a circle around one sky position."""
        radius_arcsec = _to_arcsec(search_radius)
        res = self.find_bricks_from_list(
            [ra], [dec], search_radius=radius_arcsec,
            check_file_exists=True, show_progress=False,
        )
        if len(res) == 0:
            if not silent:
                logger.warning(f'No bricks found within {radius_arcsec} arcsec of (RA, Dec)=({ra}, {dec})')
            return res
        if show:
            self._show_bricks(ra, dec, radius_arcsec, res)
        return res

    def _show_bricks(self, ra, dec, radius_arcsec, bricks):
        p = SkyCoord(ra=ra * u.degree, dec=dec * u.degree)
        circle = p.directional_offset_by(
            position_angle=np.linspace(0, 360, 200) * u.degree,
            separation=radius_arcsec * u.arcsec,
        )
        def unwrap(x):
            return ra + _dlon(x, ra)
        circle_ra, circle_dec = unwrap(circle.ra.degree), circle.dec.degree
        brick_ra = unwrap(np.r_[bricks['ra1'].to_numpy(), bricks['ra2'].to_numpy()])
        brick_dec = np.r_[bricks['dec1'].to_numpy(), bricks['dec2'].to_numpy()]
        x = np.r_[circle_ra, unwrap(ra), brick_ra]
        y = np.r_[circle_dec, dec, brick_dec]
        x_span = np.nanmax(x) - np.nanmin(x)
        y_span = np.nanmax(y) - np.nanmin(y)
        pad = max(radius_arcsec / 3600.0 * 0.20, 0.015, 0.10 * max(x_span, y_span, radius_arcsec / 3600.0 * 0.35))
        if not np.isfinite(x_span) or x_span <= 0 or y_span <= 0:
            figsize = (6.0, 6.0)
        else:
            ratio = np.clip(x_span / y_span, 0.6, 1.8)
            base = 6.0 if max(x_span, y_span) < 1.0 else 7.0
            figsize = (min(9.0, base * ratio), max(5.0, base)) if ratio >= 1 else (max(5.0, base), min(9.0, base / ratio))

        with plt.rc_context({
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
            'mathtext.fontset': 'stix',
            'axes.linewidth': 1.0,
            'xtick.direction': 'in', 'ytick.direction': 'in',
            'xtick.top': True, 'ytick.right': True,
            'xtick.major.size': 5, 'ytick.major.size': 5,
            'xtick.minor.size': 3, 'ytick.minor.size': 3,
        }):
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            ax.set_aspect('equal', adjustable='box')
            brick_handles = []
            seen = set()
            grouped = bricks.groupby(['brickname', 'ra1', 'ra2', 'dec1', 'dec2'], sort=False)
            for _, group in grouped:
                brick = group.iloc[0]
                bx = unwrap(np.array([brick['ra1'], brick['ra2'], brick['ra2'], brick['ra1'], brick['ra1']]))
                by = np.array([brick['dec1'], brick['dec1'], brick['dec2'], brick['dec2'], brick['dec1']])
                names = group['region'].astype(str).tolist()
                for offset, (_, row) in enumerate(group.iterrows()):
                    style = _BRICK_STYLE[str(row['region'])]
                    ax.fill(bx, by, facecolor=style['facecolor'], edgecolor=style['edgecolor'],
                            alpha=0.18, linewidth=1.2, linestyle=style['linestyle'],
                            hatch=style['hatch'], zorder=2 + offset)
                    if row['region'] not in seen:
                        seen.add(row['region'])
                        brick_handles.append(Patch(
                            facecolor=style['facecolor'], edgecolor=style['edgecolor'],
                            hatch=style['hatch'], alpha=0.18, linewidth=1.2,
                            linestyle=style['linestyle'], label=style['label'],
                        ))
                label = f"{brick['brickname']} ({'/'.join(names)})" if len(names) > 1 else brick['brickname']
                ax.text(bx.min(), by.max(), label, color='0.1', fontsize=8.5, ha='left', va='top', zorder=12)

            ax.plot(circle_ra, circle_dec, color='0.05', lw=1.4, zorder=20)
            ax.plot(unwrap(ra), dec, marker='x', color='0.05', ms=8, mew=1.8, linestyle='None', zorder=25)
            fig.subplots_adjust(right=0.78)
            ax.legend(
                handles=brick_handles + [
                    Line2D([], [], color='0.05', lw=1.4, label='Search radius'),
                    Line2D([], [], marker='x', color='0.05', linestyle='None', ms=7, mew=1.8, label='Target'),
                ],
                loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=9,
            )
            ax.set_xlim(np.nanmin(x) - pad, np.nanmax(x) + pad)
            ax.set_ylim(np.nanmin(y) - pad, np.nanmax(y) + pad)
            ax.minorticks_on()
            ax.grid(alpha=0.12, lw=0.5)
            ax.set_xlabel('RA [deg]', fontsize=13)
            ax.set_ylabel('Dec [deg]', fontsize=13)
            ax.set_title(f'RA={ra:.4f}, Dec={dec:.4f}, $r={radius_arcsec:.0f}$ arcsec', fontsize=12)
            plt.show()

    def _source_idx_per_brick(self, bricks, ra, dec, radius_arcsec):
        n_bricks = len(bricks)
        if n_bricks == 0 or len(ra) == 0:
            return [np.array([], dtype=int) for _ in range(n_bricks)]

        ra_c = bricks['ra'].to_numpy(float) % 360.0
        dec_c = bricks['dec'].to_numpy(float)
        half_w = np.maximum(np.abs(_dlon(bricks['ra1'].to_numpy(float), ra_c)),
                            np.abs(_dlon(bricks['ra2'].to_numpy(float), ra_c)))
        half_h = np.maximum(np.abs(bricks['dec1'].to_numpy(float) - dec_c),
                            np.abs(bricks['dec2'].to_numpy(float) - dec_c))
        dec_abs_min = np.maximum(np.abs(dec_c) - half_h, 0.0)
        half_diag = np.sqrt((half_w * np.cos(np.deg2rad(dec_abs_min))) ** 2 + half_h ** 2) * 3600.0 + 1.0
        chord = 2.0 * np.sin(np.deg2rad((radius_arcsec + half_diag) / 3600.0) / 2.0)
        cands = cKDTree(_xyz(ra, dec)).query_ball_point(_xyz(ra_c, dec_c), chord)

        out = []
        for i, cand in enumerate(cands):
            if len(cand) == 0:
                out.append(np.array([], dtype=int))
                continue
            idx = np.asarray(cand, dtype=int)
            outside_ra = np.maximum(np.abs(_dlon(ra[idx], ra_c[i])) - half_w[i], 0.0)
            outside_dec = np.maximum(np.abs(dec[idx] - dec_c[i]) - half_h[i], 0.0)
            dist = np.sqrt((outside_ra * np.cos(np.deg2rad(dec[idx]))) ** 2 + outside_dec ** 2) * 3600.0
            out.append(idx[dist <= (radius_arcsec + 2.0)])
        return out

    def _match_one_brick(self, task):
        region, brickname, ra, dec, ids, radius_arcsec, collect_sources, load_columns = task
        path = self.tractor_path(region, brickname)
        if not path.exists():
            return 'missing', region, brickname, None
        if len(ra) == 0:
            return 'ok', region, brickname, None
        try:
            coords = _read_tractor(path, columns=['release', 'brickid', 'objid', 'ra', 'dec'])
            if coords.empty:
                return 'ok', region, brickname, None
            row_id = np.arange(len(coords), dtype=np.intp)
            if collect_sources:
                pairs = fast_match(
                    coords['ra'].to_numpy(), coords['dec'].to_numpy(), ra, dec,
                    radius_arcsec=radius_arcsec, id_1=row_id, mode='nearest', workers=1,
                )
            else:
                pairs = fast_match(
                    ra, dec, coords['ra'].to_numpy(), coords['dec'].to_numpy(),
                    radius_arcsec=radius_arcsec, id_1=ids, id_2=row_id, mode='all', workers=1,
                )
            if pairs.empty:
                return 'ok', region, brickname, None

            row_idx = np.asarray(pairs['id_1' if collect_sources else 'id_2'], dtype=np.intp)
            unique_rows = np.unique(row_idx)
            tractor = _read_tractor(path, columns=load_columns, rows=unique_rows)
            if collect_sources:
                return 'ok', region, brickname, tractor
            matched = tractor.iloc[np.searchsorted(unique_rows, row_idx)].reset_index(drop=True)
            matched.insert(0, 'id', np.asarray(pairs['id_1']))
            matched.insert(1, 'sep', np.asarray(pairs['sep']))
            return 'ok', region, brickname, matched
        except Exception as exc:
            return 'error', region, brickname, str(exc)

    def collect_matches(
        self, ra, dec, search_radius=60, id=None, columns=None,
        max_workers=None, show_progress=True, filter_primary=False, quiet=False,
    ):
        """
        Collect tractor sources within ``search_radius`` of the input coordinates.

        Returns ``{'north': DataFrame, 'south': DataFrame}``.
        """
        st_all = time.time()
        radius_arcsec = _to_arcsec(search_radius)
        if radius_arcsec <= 0:
            raise ValueError('search_radius must be positive.')

        ra = np.asarray(ra, dtype=float)
        dec = np.asarray(dec, dtype=float)
        if ra.ndim != 1 or dec.ndim != 1 or ra.size != dec.size:
            raise ValueError('ra and dec must be 1D arrays of the same length.')

        n_sources = ra.size
        if id is None:
            ids = np.arange(n_sources, dtype=np.intp)
        else:
            ids = np.asarray(id)
            if ids.ndim != 1 or ids.shape[0] != n_sources:
                raise ValueError('id must be a 1D array with the same length as ra and dec.')

        if columns is None:
            output_columns = None
        else:
            output_columns = [name for name in dict.fromkeys(columns) if name not in {'id', 'sep', 'ls_id_dr11'}]
            if any(not isinstance(name, str) for name in output_columns):
                raise TypeError('columns must contain only strings.')
        load_columns = None if output_columns is None else list(dict.fromkeys(
            [*output_columns, 'ls_id_dr11', 'ra', 'dec']
            + (['type', 'brick_primary'] if filter_primary else [])
        ))

        workers = (os.cpu_count() or 1) if max_workers is None else int(max_workers)
        if workers < 1:
            raise ValueError('max_workers must be >= 1.')

        if not quiet:
            logger.info(f'Collecting LS matches | {n_sources:,} sources | r = {radius_arcsec:g} arcsec')
        bricks = self.find_bricks_from_list(
            ra=ra, dec=dec, search_radius=radius_arcsec,
            max_workers=workers, show_progress=show_progress,
        )
        n_north = int((bricks['region'] == 'north').sum()) if len(bricks) else 0
        n_south = int((bricks['region'] == 'south').sum()) if len(bricks) else 0
        if not quiet:
            logger.info(f'Overlapping bricks | {len(bricks):,} total (north {n_north:,}, south {n_south:,})')

        source_idx = self._source_idx_per_brick(bricks, ra, dec, radius_arcsec)
        empty = pd.DataFrame(columns=['id', 'sep', 'ls_id_dr11'])
        results = {region: empty.copy() for region in REGIONS}

        for region in REGIONS:
            rel_mask = bricks['region'].to_numpy() == region if len(bricks) else np.array([], dtype=bool)
            bricks_rel = bricks.loc[rel_mask]
            n_bricks = len(bricks_rel)
            if n_bricks == 0:
                logger.warning(f'No bricks found for {region}.')
                continue

            tasks = []
            for row, brick_i in zip(bricks_rel.itertuples(index=False), np.flatnonzero(rel_mask)):
                if filter_primary and (not bool(row.survey_primary)):
                    continue
                src = source_idx[brick_i]
                if len(src) == 0:
                    continue
                tasks.append((region, row.brickname, ra[src], dec[src], ids[src],
                              radius_arcsec, filter_primary, load_columns))
            if not tasks:
                logger.info(f'{region} | no nearby sources to match')
                continue

            matched_frames, n_missing, n_error = [], 0, 0

            def _consume(status, sv, brickname, payload):
                nonlocal n_missing, n_error
                if status == 'missing':
                    n_missing += 1
                    logger.warning(f'Missing tractor | {sv}-{brickname}')
                    return
                if status != 'ok':
                    n_error += 1
                    logger.warning(f'Failed to read | {sv}-{brickname} | {payload}')
                    return
                if payload is None:
                    return
                if filter_primary:
                    payload = payload.loc[
                        (payload['type'].astype(str) != 'DUP') & payload['brick_primary'].astype(bool)
                    ].drop(columns=['id', 'sep'], errors='ignore').drop_duplicates(subset=['ls_id_dr11'], keep='first')
                    if payload.empty:
                        return
                if output_columns is not None:
                    payload = payload.loc[:, ['ls_id_dr11'] + [n for n in output_columns if n in payload.columns]]
                matched_frames.append(payload)

            n_workers = max(1, min(workers, len(tasks)))
            if n_workers <= 1:
                for task in _progress(tasks, total=len(tasks), desc=region, show=show_progress):
                    _consume(*self._match_one_brick(task))
            else:
                with ThreadPoolExecutor(max_workers=n_workers) as ex:
                    for item in _progress(ex.map(self._match_one_brick, tasks),
                                          total=len(tasks), desc=region, show=show_progress):
                        _consume(*item)

            if matched_frames:
                results[region] = pd.concat(matched_frames, ignore_index=True)
            extra = []
            if n_missing:
                extra.append(f'{n_missing:,} missing')
            if n_error:
                extra.append(f'{n_error:,} errors')
            extra_txt = f" | {', '.join(extra)}" if extra else ''
            if not quiet:
                logger.info(f'{region} | {len(results[region]):,} matches | {n_bricks:,} bricks{extra_txt}')

        if not quiet:
            logger.success(f'Done | {sec_to_hms(time.time() - st_all)}')
        return results

    def download_image(self, region, brickname, band, dir_output, silent=False):
        """Download one coadd image. ``region`` is ``'north'`` or ``'south'``; band is g/r/i/z/W1–W4."""
        region = self._check_region(region)
        fname = f'legacysurvey-{brickname}-image-{band}.fits.fz'
        url = f"{URL_COADD[region]}/{brickname[:3]}/{brickname}/{fname}"
        path_output = Path(dir_output) / f'{brickname}-{band}.fits.fz'
        if path_output.exists():
            logger.info(f'legacysurvey image: {brickname} ({band}) | EXISTS')
            return None
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(path_output, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            if not silent:
                logger.success(f'legacysurvey image: {brickname} ({band}) | DONE')
        except Exception as e:
            logger.error(f'legacysurvey image: {brickname} ({band}) | ERROR | {e}')
        return None
