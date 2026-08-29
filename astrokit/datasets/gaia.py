"""Local Gaia DR3 ``gaia_source`` parquet catalogue."""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import re
import time

import astropy.units as u
import healpy as hp
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger
from tqdm import tqdm

from astrokit.toolbox import sec_to_hms
from astrokit.toolbox.match import fast_match

__all__ = ["Gaia"]

_FILE_RE = re.compile(r"^GaiaSource_(?P<start>\d{6})-(?P<end>\d{6})\.parquet$")
_NSIDE = 256


def _to_arcsec(search_radius):
    if isinstance(search_radius, u.Quantity):
        value = search_radius.to_value(u.arcsec)
    elif isinstance(search_radius, str):
        value = u.Quantity(search_radius).to_value(u.arcsec)
    else:
        value = float(search_radius)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("search_radius must be a positive finite angle.")
    return float(value)


def _match_file(args):
    """Read overlapping row groups of one parquet file and match nearby sources."""
    path, ra, dec, radius_arcsec, columns = args
    try:
        pf = pq.ParquetFile(path, memory_map=True)
        radius_rad = np.deg2rad(radius_arcsec / 3600.0)
        hpx8 = _query_hpx8(ra, dec, radius_rad)
        groups = _row_groups(pf, hpx8)
        table = pf.read_row_groups(groups, columns=columns, use_threads=False)
        if table.num_rows == 0:
            return None

        ra_cat = table.column("ra").to_numpy()
        dec_cat = table.column("dec").to_numpy()
        valid = np.isfinite(ra_cat) & np.isfinite(dec_cat)
        if not np.any(valid):
            return None

        rows = np.flatnonzero(valid)
        pairs = fast_match(
            ra, dec, ra_cat[rows], dec_cat[rows],
            radius_arcsec=radius_arcsec,
            id_2=rows, mode="all", workers=1,
        )
        if pairs.empty:
            return None

        idx = np.unique(np.asarray(pairs["id_2"], dtype=np.int64))
        return table.take(pa.array(idx)).to_pandas(self_destruct=True)
    except Exception as exc:
        logger.warning(f"{Path(path).name}  {exc}")
        return None


def _query_hpx8(ra, dec, radius_rad):
    ra = np.atleast_1d(np.asarray(ra, dtype=float))
    dec = np.atleast_1d(np.asarray(dec, dtype=float))
    pixels = []
    for r, d in zip(ra, dec):
        if not (np.isfinite(r) and np.isfinite(d)):
            continue
        vec = hp.ang2vec(np.deg2rad(90.0 - d), np.deg2rad(r % 360.0))
        pixels.append(hp.query_disc(_NSIDE, vec, radius_rad, inclusive=True, fact=4, nest=True))
    if not pixels:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(pixels))


def _row_groups(pf, hpx8):
    n_groups = pf.num_row_groups
    if n_groups == 0 or hpx8.size == 0 or "source_id" not in pf.schema.names:
        return list(range(n_groups))
    sid = pf.schema.names.index("source_id")
    pixels = np.unique(np.asarray(hpx8, dtype=np.int64))
    keep = []
    for i in range(n_groups):
        stats = pf.metadata.row_group(i).column(sid).statistics
        if stats is None or stats.min is None or stats.max is None:
            keep.append(i)
            continue
        lo = int(stats.min) >> 43
        hi = int(stats.max) >> 43
        j = np.searchsorted(pixels, lo)
        if j < pixels.size and pixels[j] <= hi:
            keep.append(i)
    return keep or list(range(n_groups))


class Gaia:
    """Match cones against local Gaia DR3 ``gaia_source`` parquet files."""

    def __init__(self, dir_gaia):
        self.dir_data = (Path(dir_gaia) / "gaia_source").expanduser()
        if not self.dir_data.is_dir():
            raise FileNotFoundError(f"not found: {self.dir_data}")

        files = []
        for path in self.dir_data.iterdir():
            match = _FILE_RE.match(path.name)
            if match is not None:
                files.append((int(match["start"]), int(match["end"]), path))
        if not files:
            raise FileNotFoundError(f"no GaiaSource_XXXXXX-XXXXXX.parquet in {self.dir_data}")

        files.sort(key=lambda item: item[0])
        self._hpx8_start = np.array([item[0] for item in files], dtype=np.int32)
        self._hpx8_end = np.array([item[1] for item in files], dtype=np.int32)
        self._paths = np.array([item[2] for item in files], dtype=object)
        if np.any(self._hpx8_start[1:] <= self._hpx8_end[:-1]):
            raise ValueError("overlapping HEALPix ranges in gaia_source")

    def _file_ids_for_cone(self, ra, dec, radius_rad):
        if not (np.isfinite(ra) and np.isfinite(dec)):
            return np.empty(0, dtype=np.intp)
        if not -90.0 <= dec <= 90.0:
            raise ValueError("dec must be within [-90, 90] degrees.")
        hpx8 = hp.query_disc(
            _NSIDE,
            hp.ang2vec(np.deg2rad(90.0 - dec), np.deg2rad(ra % 360.0)),
            radius_rad, inclusive=True, fact=4, nest=True,
        )
        index = np.searchsorted(self._hpx8_start, hpx8, side="right") - 1
        ok = (index >= 0) & (hpx8 <= self._hpx8_end[index])
        return np.unique(index[ok]).astype(np.intp, copy=False) if np.any(ok) else np.empty(0, dtype=np.intp)

    def _files_per_source(self, ra, dec, radius_rad, max_workers, show_progress):
        coords = list(zip(ra, dec))
        n = ra.size
        bar = dict(total=n, desc="files", unit="src", disable=not show_progress)
        if max_workers == 1 or n < 2048:
            return [self._file_ids_for_cone(r, d, radius_rad) for r, d in tqdm(coords, **bar)]
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            mapped = pool.map(
                lambda xy: self._file_ids_for_cone(xy[0], xy[1], radius_rad),
                coords,
                chunksize=max(1, n // (max_workers * 8)),
            )
            return list(tqdm(mapped, **bar))

    def find_source_files(self, ra, dec, search_radius=60, max_workers=None, show_progress=True):
        """Filenames whose HEALPix-8 range overlaps the search cones."""
        radius_rad = np.deg2rad(_to_arcsec(search_radius) / 3600.0)
        ra, dec = np.atleast_1d(np.asarray(ra, dtype=float)), np.atleast_1d(np.asarray(dec, dtype=float))
        if ra.size != dec.size:
            raise ValueError("ra and dec must have the same length.")
        workers = max(1, min(32, os.cpu_count() or 1) if max_workers is None else int(max_workers))
        used = {i for ids in self._files_per_source(ra, dec, radius_rad, workers, show_progress) for i in ids}
        return [path.name for i, path in enumerate(self._paths) if i in used]

    def collect_matches(
        self,
        ra,
        dec,
        search_radius=60,
        columns=None,
        max_workers=None,
        show_progress=True,
    ):
        """Unique Gaia ``gaia_source`` rows within ``search_radius`` of the inputs.

        ``search_radius`` is in arcsec if numeric. ``columns`` is the output
        schema: ``None`` keeps every parquet column, otherwise the table
        contains exactly those fields. Rows are de-duplicated by ``source_id``.
        """
        t0 = time.time()
        radius_arcsec = _to_arcsec(search_radius)
        radius_rad = np.deg2rad(radius_arcsec / 3600.0)
        ra, dec = np.atleast_1d(np.asarray(ra, dtype=float)), np.atleast_1d(np.asarray(dec, dtype=float))
        if ra.size != dec.size:
            raise ValueError("ra and dec must have the same length.")

        n = ra.size
        if columns is None:
            out_cols = None
            load_cols = None
        else:
            out_cols = list(columns)
            load_cols = list(dict.fromkeys([*out_cols, "ra", "dec", "source_id"]))

        workers = max(1, min(32, os.cpu_count() or 1) if max_workers is None else int(max_workers))
        per_file = defaultdict(list)
        for i, file_ids in enumerate(self._files_per_source(ra, dec, radius_rad, workers, show_progress=False)):
            for j in file_ids:
                per_file[int(j)].append(i)

        tasks = [
            (self._paths[j], ra[np.asarray(src, np.intp)], dec[np.asarray(src, np.intp)],
             radius_arcsec, load_cols)
            for j, src in per_file.items()
        ]
        if isinstance(search_radius, str):
            r_txt = search_radius
        elif isinstance(search_radius, u.Quantity):
            r_txt = f"{search_radius}"
        else:
            r_txt = f"{radius_arcsec:g} arcsec"

        logger.info(f"Gaia DR3 | {n:,} input sources | r = {r_txt}")
        logger.info(f"Parquet  | {len(tasks):,} files to read")
        empty = pd.DataFrame(columns=out_cols) if out_cols is not None else pd.DataFrame()
        if not tasks:
            logger.warning("No overlapping Gaia files.")
            return empty

        n_workers = min(workers, len(tasks))
        bar = dict(total=len(tasks), desc="Reading", unit="file", disable=not show_progress)
        if n_workers == 1:
            frames = [_match_file(task) for task in tqdm(tasks, **bar)]
        else:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                frames = list(tqdm(
                    pool.map(_match_file, tasks, chunksize=max(1, len(tasks) // (n_workers * 4))),
                    **bar,
                ))
        frames = [df for df in frames if df is not None]
        if not frames:
            logger.success(f"Done | 0 unique Gaia sources | {sec_to_hms(time.time() - t0)}")
            return empty

        result = pd.concat(frames, ignore_index=True)
        result = result.drop_duplicates(subset=["source_id"], keep="first")
        if out_cols is not None:
            result = result.loc[:, out_cols]
        result = result.reset_index(drop=True)
        logger.success(f"Done | {len(result):,} unique Gaia sources | {sec_to_hms(time.time() - t0)}")
        return result
