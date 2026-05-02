"""
Extinction correction utilities.

@ Author: Rui Zhu
@ Date: 2025-04-09
"""
import tarfile

import extinction
import numpy as np
import requests

from astrokit import DIR_datasets
from astrokit.externals import sfdmap

__all__ = ["ExtinctionCorrection"]


class ExtinctionCorrection:
    def __init__(self, scaling=0.86, Rv=3.1):
        """
        Correct Galactic extinction with SFD maps and the FM07 curve.

        Parameters
        ----------
        scaling : float, optional
            Scale factor applied to the SFD E(B-V) map. The default 0.86
            follows the Schlafly & Finkbeiner (2011) recalibration of SFD98.
        Rv : float, optional
            Total-to-selective extinction ratio used as A_V/E(B-V). The
            default 3.1 matches the FM07 Milky Way average curve.
        """
        self.dir_dustmap = DIR_datasets / "dustmaps"
        self.dir_dustmap.mkdir(parents=True, exist_ok=True)
        self.dir_sfdmap = self.dir_dustmap / "sfddata-master"

        if not self.dir_sfdmap.exists():
            self.download_sfdmap()

        self.scaling = scaling
        self.Rv = Rv
        self.sfd = sfdmap.SFDMap(mapdir=self.dir_sfdmap, scaling=self.scaling)

    def download_sfdmap(self, timeout=60):
        """Download and extract the SFD dust maps if they are not available."""
        url = "https://github.com/kbarbary/sfddata/archive/master.tar.gz"
        path_sfdmaps_gz = self.dir_dustmap / "master.tar.gz"

        print(f"==> Downloading from {url}")
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()

        print(f"==> Saving to {path_sfdmaps_gz}")
        with open(path_sfdmaps_gz, "wb") as f:
            f.write(response.content)

        print(f"==> Extracting {path_sfdmaps_gz}")
        with tarfile.open(path_sfdmaps_gz, "r:gz") as tar:
            tar.extractall(path=self.dir_dustmap, filter="tar")

        print(f"==> Removing {path_sfdmaps_gz}")
        path_sfdmaps_gz.unlink()
        print("==> SFD maps downloaded!")

        return None

    def cal_ebv(self, ra, dec):
        """
        Calculate SFD E(B-V) at ICRS coordinates.

        Parameters
        ----------
        ra, dec : float or array-like
            ICRS coordinates in degrees.
        """
        return self.sfd.ebv(ra, dec, frame="icrs", unit="degree")

    def cal_extinction_coeff(self, filter_waves):
        """
        Calculate monochromatic FM07 coefficients A_lambda / E(B-V).

        Parameters
        ----------
        filter_waves : float or array-like
            Wavelengths in Angstrom.

        Returns
        -------
        ndarray
            Extinction coefficients evaluated at the input wavelengths.
        """
        waves = np.atleast_1d(np.asarray(filter_waves, dtype=float))
        if np.any(~np.isfinite(waves)) or np.any(waves <= 0):
            raise ValueError("filter_waves must contain finite positive wavelengths")

        return extinction.fm07(waves, self.Rv)

    def cal_extinction(self, ra, dec, filter_waves):
        """
        Calculate A_lambda from SFD E(B-V) and FM07 coefficients.

        Parameters
        ----------
        ra : float or array-like
            ICRS right ascension in degrees.
        dec : float or array-like
            ICRS declination in degrees.
        filter_waves : float or array-like
            Filter central wavelengths in Angstrom.

        Returns
        -------
        list
            Each element is A_lambda for one input wavelength. This preserves
            the historical API.
        """
        ebv = np.atleast_1d(self.cal_ebv(ra, dec))
        coeff = self.cal_extinction_coeff(filter_waves)
        Ax = ebv.reshape(-1, 1) * coeff.reshape(1, -1)
        return [Ax[:, i] for i in range(Ax.shape[1])]

    def cal_bandpass_coeff(self, waves, response):
        """
        Calculate broadband extinction coefficient A_band / E(B-V).

        The integration assumes an AB-system source with flat f_nu, so the
        effective weights are response / wavelength.

        Parameters
        ----------
        waves : array-like
            Wavelength grid in Angstrom.
        response : array-like
            Dimensionless filter throughput sampled on ``waves``.
        """
        waves = np.asarray(waves, dtype=float)
        response = np.asarray(response, dtype=float)

        if waves.shape != response.shape:
            raise ValueError("waves and response must have the same shape")
        if waves.ndim != 1:
            raise ValueError("waves and response must be one-dimensional")
        if len(waves) < 2:
            raise ValueError("waves and response must contain at least two samples")
        if np.any(~np.isfinite(waves)) or np.any(waves <= 0):
            raise ValueError("waves must contain finite positive values")
        if np.any(~np.isfinite(response)) or np.any(response < 0):
            raise ValueError("response must contain finite non-negative values")

        order = np.argsort(waves)
        waves = waves[order]
        response = response[order]
        weights = response / waves
        denominator = np.trapz(weights, waves)
        if denominator <= 0:
            raise ValueError("response must have positive integrated throughput")

        a_lambda = self.cal_extinction_coeff(waves)
        numerator = np.trapz(weights * 10 ** (-0.4 * a_lambda), waves)
        if numerator <= 0:
            raise ValueError("attenuated response has non-positive integral")

        return -2.5 * np.log10(numerator / denominator)

    def cal_bandpass_extinction(self, ra, dec, bandpasses):
        """
        Calculate broadband extinction for multiple filters.

        Parameters
        ----------
        ra, dec : float or array-like
            ICRS coordinates in degrees.
        bandpasses : dict
            Mapping ``name -> (waves, response)``.

        Returns
        -------
        dict
            Mapping ``name -> A_band`` arrays, one array per filter.
        """
        ebv = np.atleast_1d(self.cal_ebv(ra, dec))
        output = {}
        for name, (waves, response) in bandpasses.items():
            output[name] = ebv * self.cal_bandpass_coeff(waves, response)
        return output
