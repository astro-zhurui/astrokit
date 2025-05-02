import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u

from astroquery.ipac.ned import Ned

__all__ = [
    'query_NED'
]


def query_NED(ra, dec, sep=3, raw=False):
    """
    query NED for the object with given ra and dec

    Parameters
    ----------
    ra : float
        right ascension in degree
    dec : float
        declination in degree
    sep : float
        search radius in arcsec. Default is 3 arcsec
    raw : bool
        return the raw result from NED. Default is False
    """
    coord = SkyCoord(ra=ra, dec=dec, unit='deg')
    res_NED = Ned.query_region(coord, radius=sep*u.arcsec)
    if raw:
        res = res_NED
    else:
        res_NED = res_NED.to_pandas()
        if len(res_NED) == 0:
            res = {'NED_num': 0, 
                'NED_redshift': np.nan, 
                'NED_type': []}
        else:
            res = {'NED_num': len(res_NED)}

        # 获取光谱红移
        spec_z = list(res_NED['Redshift'])
        # 获取光谱类型
        spec_type = list(res_NED['Type'])

        # 去除NAN
        spec_z = [z for z in spec_z if not np.isnan(z)]
        spec_type = [t for t in spec_type if isinstance(t, str)]

        if len(spec_z) == 0:
            spec_z = np.nan
        else:
            spec_z = float(np.mean(spec_z))

        res['NED_redshift'] = spec_z
        res['NED_type'] = spec_type
    return res
