from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from astrokit.toolbox.utils import read


def _make_test_fits(path: Path):
    cols = [
        fits.Column(name="flux", format="E", array=np.array([1.0, 2.0, 3.0], dtype=np.float32)),
        fits.Column(name="objid", format="20A", array=np.array(["EPF_J1", "EPF_J2", "EPF_J3"])),
        fits.Column(name="flag", format="L", array=np.array([True, False, True])),
    ]
    hdu = fits.BinTableHDU.from_columns(cols)
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path)


def test_read_decodes_strings_and_preserves_other_column_types(tmp_path):
    path = tmp_path / "sample.fits"
    _make_test_fits(path)

    table = read(path)

    assert table.colnames == ["flux", "objid", "flag"]
    assert table["objid"].tolist() == ["EPF_J1", "EPF_J2", "EPF_J3"]
    assert all(isinstance(value, str) for value in table["objid"])
    assert table["flag"].tolist() == [True, False, True]
    assert np.allclose(table["flux"], [1.0, 2.0, 3.0])


def test_read_respects_requested_columns_and_rows(tmp_path):
    path = tmp_path / "sample.fits"
    _make_test_fits(path)

    table = read(path, n_rows=2, columns=["objid", "flux"])

    assert table.colnames == ["objid", "flux"]
    assert len(table) == 2
    assert table["objid"].tolist() == ["EPF_J1", "EPF_J2"]
    assert np.allclose(table["flux"], [1.0, 2.0])


def test_read_fill_missing_columns_with_masked_column(tmp_path):
    path = tmp_path / "sample.fits"
    _make_test_fits(path)

    table = read(path, columns=["objid", "missing"], fill=True)

    assert table.colnames == ["objid", "missing"]
    assert table["missing"].mask.tolist() == [True, True, True]


def test_read_raises_for_missing_columns_without_fill(tmp_path):
    path = tmp_path / "sample.fits"
    _make_test_fits(path)

    with pytest.raises(KeyError, match="Unknown column"):
        read(path, columns=["missing"])


def test_read_handles_zero_rows(tmp_path):
    path = tmp_path / "sample.fits"
    _make_test_fits(path)

    table = read(path, n_rows=0)

    assert len(table) == 0
    assert table.colnames == ["flux", "objid", "flag"]
