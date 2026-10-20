import warnings
import gzip
import io

import numpy as np
import pandas as pd
import tempfile
import unittest
from pathlib import Path
from astropy.io import fits

from astrokit.toolbox.utils import fits2df, read


def make_table(path):
    hdu = fits.BinTableHDU.from_columns([
        fits.Column(name='id', format='K', array=[2**60 + 1, 2**60 + 2, 2**60 + 3]),
        fits.Column(name='name', format='8A', array=['EPF1', 'EPF2', '']),
        fits.Column(name='flux', format='E', unit='nanomaggy', array=[1., np.nan, 3.]),
        fits.Column(name='density', format='D', unit='1/deg^2', array=[2., 4., 6.]),
        fits.Column(name='flag', format='L', array=[True, False, True]),
        fits.Column(name='count', format='J', null=-999, array=[1, -999, 3]),
        fits.Column(name='vector', format='2E', array=[[1., 2.], [3., 4.], [5., 6.]]),
    ], name='SOURCES')
    buffer = io.BytesIO()
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(buffer)
    payload = bytearray(buffer.getvalue())
    with fits.open(io.BytesIO(payload)) as hdul:
        offset = hdul[1].fileinfo()['datLoc']
        row_size = hdul[1].header['NAXIS1']
        flag_offset = hdul[1].columns.dtype.fields['flag'][1]
    payload[offset + 2 * row_size + flag_offset] = 0
    if str(path).endswith('.gz'):
        with gzip.open(path, 'wb') as stream:
            stream.write(payload)
    else:
        path.write_bytes(payload)


class FitsReaderTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_values_and_nulls_without_unit_warnings(self):
        for suffix in ['.fits', '.fits.gz']:
            with self.subTest(suffix=suffix):
                path = self.root / ('sources' + suffix)
                make_table(path)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    df = fits2df(path, columns=['id', 'name', 'flux', 'density', 'flag', 'count'])
                self.assertEqual(caught, [], [str(item.message) for item in caught])
                self.assertEqual(df.id.tolist(), [2**60 + 1, 2**60 + 2, 2**60 + 3])
                self.assertEqual(df.name.iloc[:2].tolist(), ['EPF1', 'EPF2'])
                self.assertTrue(pd.isna(df.name.iloc[2]) and pd.isna(df.flux.iloc[1]))
                self.assertEqual(df.density.tolist(), [2., 4., 6.])
                self.assertEqual(str(df.flag.dtype), 'boolean')
                self.assertEqual(df.flag.iloc[:2].tolist(), [True, False])
                self.assertTrue(pd.isna(df.flag.iloc[2]) and pd.isna(df['count'].iloc[1]))

    def test_columns_rows_and_extension(self):
        path = self.root / 'sources.fits'
        make_table(path)
        df = fits2df(path, columns=['density', 'id'], n_rows=2, hdu_name='SOURCES')
        self.assertEqual(df.columns.tolist(), ['density', 'id'])
        self.assertEqual(df.density.tolist(), [2., 4.])
        self.assertEqual(len(fits2df(path, columns='id', n_rows=0)), 0)
        with self.assertRaisesRegex(KeyError, 'Unknown column'):
            fits2df(path, columns=['absent'])
        with self.assertRaisesRegex(ValueError, 'n_rows'):
            fits2df(path, columns=['id'], n_rows=4)

    def test_read_preserves_undefined_logical_values(self):
        path = self.root / 'sources.fits'
        make_table(path)
        table = read(path, columns=['flag'])
        self.assertEqual(table['flag'].mask.tolist(), [False, False, True])
        self.assertEqual(table['flag'].data[:2].tolist(), [True, False])

    def test_existing_read_options(self):
        path = self.root / 'sources.fits'
        make_table(path)
        table = read(path, n_rows=2, columns=['name', 'flux'])
        self.assertEqual(table.colnames, ['name', 'flux'])
        self.assertEqual(table['name'].tolist(), ['EPF1', 'EPF2'])
        np.testing.assert_allclose(table['flux'], [1., np.nan], equal_nan=True)
        table = read(path, columns=['name', 'missing'], fill=True)
        self.assertEqual(table['missing'].mask.tolist(), [True, True, True])
        self.assertEqual(len(read(path, n_rows=0)), 0)
        with self.assertRaisesRegex(KeyError, 'Unknown column'):
            read(path, columns=['absent'])

    def test_unrelated_warnings_are_not_filtered(self):
        path = self.root / 'sources.fits'
        make_table(path)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            fits2df(path, columns=['id'])
            warnings.warn('unrelated warning', UserWarning)
        self.assertEqual([str(item.message) for item in caught], ['unrelated warning'])


if __name__ == '__main__':
    unittest.main()
